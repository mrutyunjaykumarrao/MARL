"""
Jamming Disruption Module
=========================

This module implements the FSPL-based jamming disruption logic.
Jammers disrupt communication links between enemy drones based on:
    1. Jamming power received at the link midpoint
    2. Frequency band matching (jammer must use same band as enemy)

Reference: PROJECT_MASTER_GUIDE_v2.md Sections 3.3, 4.4

Jamming Condition:
    Link (i,j) is disrupted if for ANY jammer k:
        P_jam(k, midpoint_ij) >= P_jam_thresh  AND  band_k == band_enemy

Key Insight:
    Wrong band = ZERO disruption even at close range.
    This forces agents to learn intelligent frequency selection.

Author: MARL Jammer Team
"""

import numpy as np
from typing import Tuple, Optional, Union

from .fspl import (
    received_power_watts,
    SPEED_OF_LIGHT,
    FREQUENCY_BANDS,
)


# =============================================================================
# MIDPOINT COMPUTATION
# =============================================================================

def compute_midpoints(positions: np.ndarray) -> np.ndarray:
    """
    Compute midpoints of all potential communication links.
    
    The midpoint of link (i,j) is where we evaluate jamming power.
    This is a simplification - in reality, jamming affects the entire link,
    but the midpoint is a reasonable approximation.
    
    Args:
        positions: Enemy drone positions, shape (N, 2)
        
    Returns:
        Midpoints array of shape (N, N, 2) where midpoints[i,j] is
        the (x, y) midpoint of the link between drones i and j.
        
    Note:
        midpoints[i,j] == midpoints[j,i] (symmetric)
        midpoints[i,i] = positions[i] (self-midpoint is just position)
        
    Implementation:
        Uses NumPy broadcasting for O(N^2) vectorized computation.
        m_ij = (positions[i] + positions[j]) / 2
    """
    N = positions.shape[0]
    
    if N == 0:
        return np.array([]).reshape(0, 0, 2)
    
    # Broadcasting: positions[:, None, :] has shape (N, 1, 2)
    #               positions[None, :, :] has shape (1, N, 2)
    # Result shape: (N, N, 2)
    midpoints = (positions[:, None, :] + positions[None, :, :]) / 2
    
    return midpoints


def compute_distances_to_midpoints(
    jammer_positions: np.ndarray,
    midpoints: np.ndarray
) -> np.ndarray:
    """
    Compute distances from each jammer to each link midpoint.
    
    Args:
        jammer_positions: Jammer positions, shape (M, 2)
        midpoints: Midpoints array, shape (N, N, 2)
        
    Returns:
        Distance array of shape (M, N, N) where result[k, i, j] is
        the distance from jammer k to the midpoint of link (i, j).
        
    Implementation (Section 7.1):
        Uses NumPy broadcasting: no Python loops.
        jammer_positions[:, None, None, :] shape: (M, 1, 1, 2)
        midpoints[None, :, :, :]           shape: (1, N, N, 2)
        diff = jammer - midpoint           shape: (M, N, N, 2)
        distance = ||diff||                shape: (M, N, N)
    """
    M = jammer_positions.shape[0]
    N = midpoints.shape[0]
    
    if M == 0 or N == 0:
        return np.array([]).reshape(M, N, N)
    
    # Broadcast computation
    # jammer_positions shape: (M, 2) -> (M, 1, 1, 2)
    # midpoints shape: (N, N, 2) -> (1, N, N, 2)
    diff = jammer_positions[:, None, None, :] - midpoints[None, :, :, :]
    
    # Euclidean distance along last axis
    distances = np.sqrt(np.sum(diff ** 2, axis=3))
    
    return distances


# =============================================================================
# JAMMING POWER COMPUTATION
# =============================================================================

def compute_jamming_power(
    distances: np.ndarray,
    jammer_power_watts: float,
    frequency: float,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Compute jamming power received at each link midpoint from each jammer.
    
    Uses the FSPL model: P_jam = P_jammer * (c / (4*pi*f*d))^2
    
    Args:
        distances: Jammer-to-midpoint distances, shape (M, N, N)
        jammer_power_watts: Jammer transmit power in Watts
        frequency: Jammer transmission frequency in Hz
        eps: Small value to prevent division by zero
        
    Returns:
        Jamming power array of shape (M, N, N) in Watts.
        P_jam[k, i, j] = jamming power from jammer k at midpoint of link (i,j)
        
    Note:
        For distances near zero, jamming power approaches infinity.
        We clip using eps to prevent numerical issues.
    """
    # Use FSPL formula for received power
    P_jam = received_power_watts(jammer_power_watts, distances, frequency, eps)
    
    return P_jam


# =============================================================================
# LINK DISRUPTION LOGIC
# =============================================================================

def compute_disrupted_links(
    jammer_positions: np.ndarray,
    jammer_bands: np.ndarray,
    enemy_positions: np.ndarray,
    enemy_band: int,
    jammer_power_watts: float,
    jam_threshold_watts: float,
    frequency_bands: dict = FREQUENCY_BANDS,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Determine which enemy communication links are disrupted by jammers.
    
    A link (i,j) is disrupted if there exists ANY jammer k such that:
        1. P_jam(k, midpoint_ij) >= P_jam_thresh
        2. band_k == band_enemy
    
    This is the core implementation of Section 4.4.
    
    Args:
        jammer_positions: Jammer positions, shape (M, 2)
        jammer_bands: Jammer frequency band indices, shape (M,), values in {0,1,2,3}
        enemy_positions: Enemy drone positions, shape (N, 2)
        enemy_band: Enemy swarm frequency band index
        jammer_power_watts: Jammer transmit power in Watts (default: 1W = 30dBm)
        jam_threshold_watts: Jamming disruption threshold in Watts (default: 1e-10 = -70dBm)
        frequency_bands: Dict mapping band index to frequency in Hz
        eps: Small value for numerical stability
        
    Returns:
        Boolean array of shape (N, N) where True indicates link is jammed.
        
    Example:
        >>> jammer_pos = np.array([[50, 50]])  # 1 jammer at (50, 50)
        >>> jammer_bands = np.array([2])       # Using 2.4 GHz
        >>> enemy_pos = np.array([[40, 50], [60, 50]])  # 2 enemies
        >>> enemy_band = 2                      # Enemy also using 2.4 GHz
        >>> jammed = compute_disrupted_links(
        ...     jammer_pos, jammer_bands, enemy_pos, enemy_band, 1.0, 1e-10
        ... )
        >>> jammed[0, 1]  # Is link between enemy 0 and 1 jammed?
        True  # Yes, jammer is at midpoint with matching band
        
    Critical Design:
        If jammer_bands[k] != enemy_band, that jammer has ZERO effect,
        regardless of how close it is. This incentivizes frequency learning.
    """
    M = jammer_positions.shape[0]
    N = enemy_positions.shape[0]
    
    # Handle edge cases
    if M == 0:
        # No jammers -> no links disrupted
        return np.zeros((N, N), dtype=bool)
    
    if N <= 1:
        # 0 or 1 enemies -> no links to disrupt
        return np.zeros((N, N), dtype=bool)
    
    # Get enemy frequency
    enemy_frequency = frequency_bands[enemy_band]
    
    # Step 1: Compute midpoints of all enemy links
    # Shape: (N, N, 2)
    midpoints = compute_midpoints(enemy_positions)
    
    # Step 2: Compute distance from each jammer to each midpoint
    # Shape: (M, N, N)
    distances = compute_distances_to_midpoints(jammer_positions, midpoints)
    
    # Step 3: Compute jamming power at each midpoint from each jammer
    # Note: We use enemy_frequency for FSPL calculation (same band assumption)
    # Shape: (M, N, N)
    P_jam = compute_jamming_power(distances, jammer_power_watts, enemy_frequency, eps)
    
    # Step 4: Check if jamming power exceeds threshold
    # Shape: (M, N, N)
    power_sufficient = P_jam >= jam_threshold_watts
    
    # Step 5: Check band matching for each jammer
    # jammer_bands shape: (M,)
    # We need shape (M, 1, 1) to broadcast with (M, N, N)
    band_matches = (jammer_bands == enemy_band)[:, None, None]
    
    # Step 6: Link is jammed if BOTH conditions met
    # Shape: (M, N, N)
    effective_jamming = power_sufficient & band_matches
    
    # Step 7: Any jammer can disrupt the link
    # Reduce along jammer axis (axis=0)
    # Shape: (N, N)
    jammed_links = np.any(effective_jamming, axis=0)
    
    # Ensure diagonal is False (no self-links)
    np.fill_diagonal(jammed_links, False)
    
    return jammed_links


def compute_disrupted_links_per_jammer(
    jammer_positions: np.ndarray,
    jammer_bands: np.ndarray,
    enemy_positions: np.ndarray,
    enemy_band: int,
    jammer_power_watts: float,
    jam_threshold_watts: float,
    frequency_bands: dict = FREQUENCY_BANDS,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Compute which links each individual jammer disrupts.
    
    Similar to compute_disrupted_links but returns per-jammer breakdown.
    Useful for credit assignment and analysis.
    
    Args:
        Same as compute_disrupted_links
        
    Returns:
        Boolean array of shape (M, N, N) where result[k, i, j] indicates
        whether jammer k disrupts link (i, j).
    """
    M = jammer_positions.shape[0]
    N = enemy_positions.shape[0]
    
    if M == 0:
        return np.zeros((0, N, N), dtype=bool)
    
    if N <= 1:
        return np.zeros((M, N, N), dtype=bool)
    
    enemy_frequency = frequency_bands[enemy_band]
    
    # Compute midpoints and distances
    midpoints = compute_midpoints(enemy_positions)
    distances = compute_distances_to_midpoints(jammer_positions, midpoints)
    
    # Compute jamming power
    P_jam = compute_jamming_power(distances, jammer_power_watts, enemy_frequency, eps)
    
    # Check conditions
    power_sufficient = P_jam >= jam_threshold_watts
    band_matches = (jammer_bands == enemy_band)[:, None, None]
    
    # Per-jammer jamming
    effective_jamming = power_sufficient & band_matches
    
    # Clear diagonals
    for k in range(M):
        np.fill_diagonal(effective_jamming[k], False)
    
    return effective_jamming


# =============================================================================
# COMBINED ADJACENCY WITH JAMMING
# =============================================================================

def apply_jamming_to_adjacency(
    adjacency: np.ndarray,
    jammed_links: np.ndarray
) -> np.ndarray:
    """
    Apply jamming disruption to an adjacency matrix.
    
    Sets A[i,j] = 0 where jammed_links[i,j] = True.
    
    Args:
        adjacency: Base adjacency matrix, shape (N, N)
        jammed_links: Boolean jamming mask, shape (N, N)
        
    Returns:
        Modified adjacency matrix with jammed links removed
        
    Note:
        This is the final step in computing the "effective" communication
        graph under jamming. The resulting matrix can be used for
        Laplacian and lambda-2 computation.
    """
    A_jammed = adjacency.copy()
    A_jammed[jammed_links] = 0
    return A_jammed


def compute_effective_adjacency(
    enemy_positions: np.ndarray,
    jammer_positions: np.ndarray,
    jammer_bands: np.ndarray,
    enemy_band: int,
    tx_power_watts: float = 0.1,
    sensitivity_watts: float = 1e-12,
    jammer_power_watts: float = 1.0,
    jam_threshold_watts: float = 1e-10,
    frequency_bands: dict = FREQUENCY_BANDS,
    eps: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the effective adjacency matrix considering both FSPL communication
    and jamming disruption.
    
    This is the main function combining communication graph construction
    and jamming logic. It implements the full pipeline from Section 4.4.
    
    Args:
        enemy_positions: Enemy drone positions, shape (N, 2)
        jammer_positions: Jammer positions, shape (M, 2)
        jammer_bands: Jammer frequency bands, shape (M,)
        enemy_band: Enemy swarm frequency band index
        tx_power_watts: Enemy drone transmit power (default: 100mW)
        sensitivity_watts: Receiver sensitivity (default: 1pW = -90dBm)
        jammer_power_watts: Jammer transmit power (default: 1W)
        jam_threshold_watts: Jam disruption threshold (default: 100pW = -70dBm)
        frequency_bands: Band index to frequency mapping
        eps: Numerical stability constant
        
    Returns:
        Tuple of:
            - A_unjammed: Adjacency matrix without jamming, shape (N, N)
            - jammed_links: Boolean mask of jammed links, shape (N, N)
            - A_jammed: Effective adjacency matrix with jamming, shape (N, N)
            
    Example:
        >>> # 4 enemies in a square, 1 jammer in center
        >>> enemies = np.array([[0,0], [60,0], [60,60], [0,60]])
        >>> jammers = np.array([[30, 30]])
        >>> jammer_bands = np.array([2])  # 2.4 GHz
        >>> enemy_band = 2
        >>> A_orig, jammed, A_eff = compute_effective_adjacency(
        ...     enemies, jammers, jammer_bands, enemy_band
        ... )
        >>> print(f"Original edges: {np.sum(A_orig)/2}")
        >>> print(f"Jammed edges: {np.sum(jammed)/2}")
        >>> print(f"Remaining edges: {np.sum(A_eff)/2}")
    """
    from .communication_graph import compute_adjacency_matrix
    
    N = enemy_positions.shape[0]
    enemy_frequency = frequency_bands[enemy_band]
    
    # Step 1: Compute base adjacency (without jamming)
    A_unjammed = compute_adjacency_matrix(
        enemy_positions, tx_power_watts, sensitivity_watts, enemy_frequency
    )
    
    # Step 2: Compute jammed links
    jammed_links = compute_disrupted_links(
        jammer_positions, jammer_bands, enemy_positions, enemy_band,
        jammer_power_watts, jam_threshold_watts, frequency_bands, eps
    )
    
    # Step 3: Apply jamming to adjacency
    A_jammed = apply_jamming_to_adjacency(A_unjammed, jammed_links)
    
    return A_unjammed, jammed_links, A_jammed


# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

def count_jammed_links(jammed_links: np.ndarray) -> int:
    """
    Count the number of jammed links.
    
    Args:
        jammed_links: Boolean array (N, N)
        
    Returns:
        Number of jammed links (undirected, so divide by 2)
    """
    return int(np.sum(jammed_links) / 2)


def compute_jamming_effectiveness(
    A_original: np.ndarray,
    A_jammed: np.ndarray
) -> dict:
    """
    Compute metrics about jamming effectiveness.
    
    Args:
        A_original: Adjacency matrix before jamming
        A_jammed: Adjacency matrix after jamming
        
    Returns:
        Dictionary with:
            - original_edges: Number of edges before jamming
            - remaining_edges: Number of edges after jamming
            - jammed_edges: Number of edges disrupted
            - disruption_rate: Fraction of edges disrupted
    """
    original_edges = int(np.sum(A_original) / 2)
    remaining_edges = int(np.sum(A_jammed) / 2)
    jammed_edges = original_edges - remaining_edges
    
    disruption_rate = jammed_edges / original_edges if original_edges > 0 else 0.0
    
    return {
        "original_edges": original_edges,
        "remaining_edges": remaining_edges,
        "jammed_edges": jammed_edges,
        "disruption_rate": disruption_rate,
    }


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def verify_jamming_logic() -> dict:
    """
    Run verification tests on jamming logic.
    
    Returns:
        Dictionary with test results
    """
    results = {}
    
    # Parameters
    P_tx = 0.1        # 20 dBm
    P_sens = 1e-12    # -90 dBm
    P_jam = 1.0       # 30 dBm
    P_jam_thresh = 1e-10  # -70 dBm
    
    # Test 1: Jammer at midpoint with correct band - should jam
    enemy_positions = np.array([[0, 0], [80, 0]])  # 80m apart (connected at 2.4GHz)
    jammer_positions = np.array([[40, 0]])  # Exactly at midpoint
    jammer_bands = np.array([2])  # 2.4 GHz
    enemy_band = 2
    
    jammed = compute_disrupted_links(
        jammer_positions, jammer_bands, enemy_positions, enemy_band,
        P_jam, P_jam_thresh
    )
    
    results["test_midpoint_correct_band"] = {
        "scenario": "Jammer at midpoint, matching band",
        "link_01_jammed": bool(jammed[0, 1]),
        "expected": True,
        "pass": jammed[0, 1] == True
    }
    
    # Test 2: Jammer at midpoint with WRONG band - should NOT jam
    jammer_bands_wrong = np.array([0])  # 433 MHz (wrong)
    
    jammed_wrong_band = compute_disrupted_links(
        jammer_positions, jammer_bands_wrong, enemy_positions, enemy_band,
        P_jam, P_jam_thresh
    )
    
    results["test_midpoint_wrong_band"] = {
        "scenario": "Jammer at midpoint, wrong band",
        "link_01_jammed": bool(jammed_wrong_band[0, 1]),
        "expected": False,
        "pass": jammed_wrong_band[0, 1] == False
    }
    
    # Test 3: Jammer too far away - should NOT jam
    jammer_positions_far = np.array([[40, 100]])  # 100m from midpoint
    
    jammed_far = compute_disrupted_links(
        jammer_positions_far, jammer_bands, enemy_positions, enemy_band,
        P_jam, P_jam_thresh
    )
    
    results["test_jammer_too_far"] = {
        "scenario": "Jammer 100m from midpoint, correct band",
        "link_01_jammed": bool(jammed_far[0, 1]),
        "expected": False,  # 100m > ~43m jamming range at 2.4GHz
        "pass": jammed_far[0, 1] == False
    }
    
    # Test 4: Multiple jammers, only some on correct band
    enemy_positions = np.array([[0, 0], [60, 0], [120, 0]])  # 3 drones
    jammer_positions = np.array([
        [30, 0],   # Near midpoint of 0-1
        [90, 10],  # Near midpoint of 1-2
    ])
    jammer_bands = np.array([2, 0])  # First correct, second wrong
    
    jammed_multi = compute_disrupted_links(
        jammer_positions, jammer_bands, enemy_positions, enemy_band,
        P_jam, P_jam_thresh
    )
    
    results["test_multiple_jammers"] = {
        "scenario": "2 jammers, mixed bands",
        "link_01_jammed": bool(jammed_multi[0, 1]),  # Should be jammed (correct band)
        "link_12_jammed": bool(jammed_multi[1, 2]),  # Should NOT be jammed (wrong band)
        "pass": jammed_multi[0, 1] == True and jammed_multi[1, 2] == False
    }
    
    # Test 5: Full pipeline - compute lambda-2 reduction
    from .communication_graph import compute_laplacian, compute_lambda2
    
    enemy_positions = np.array([
        [0, 0], [50, 0], [100, 0], [50, 50]
    ])  # 4 drones, all within comm range
    
    # Without jamming
    A_orig, _, A_jammed = compute_effective_adjacency(
        enemy_positions,
        np.array([[50, 25]]),  # 1 jammer
        np.array([2]),         # Correct band
        2,
        P_tx, P_sens, P_jam, P_jam_thresh
    )
    
    L_orig = compute_laplacian(A_orig)
    L_jammed = compute_laplacian(A_jammed)
    
    lambda2_orig = compute_lambda2(L_orig)
    lambda2_jammed = compute_lambda2(L_jammed)
    
    reduction = (lambda2_orig - lambda2_jammed) / lambda2_orig if lambda2_orig > 0 else 0
    
    results["test_lambda2_reduction"] = {
        "scenario": "4 drones, 1 jammer with correct band",
        "original_edges": int(np.sum(A_orig) / 2),
        "remaining_edges": int(np.sum(A_jammed) / 2),
        "lambda2_original": lambda2_orig,
        "lambda2_jammed": lambda2_jammed,
        "reduction_percent": reduction * 100,
        "pass": lambda2_jammed < lambda2_orig  # Jamming should reduce connectivity
    }
    
    return results


# =============================================================================
# MAIN (for standalone testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Jamming Disruption Module Verification")
    print("=" * 60)
    
    results = verify_jamming_logic()
    
    all_passed = True
    for test_name, test_result in results.items():
        print(f"\n{test_name}:")
        if isinstance(test_result, dict):
            for key, val in test_result.items():
                if isinstance(val, float):
                    print(f"  {key}: {val:.4f}")
                else:
                    print(f"  {key}: {val}")
            if "pass" in test_result:
                status = "PASS" if test_result["pass"] else "FAIL"
                print(f"  STATUS: {status}")
                if not test_result["pass"]:
                    all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All jamming logic tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
