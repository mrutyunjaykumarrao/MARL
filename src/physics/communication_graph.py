"""
Communication Graph Module
==========================

This module constructs the communication graph of the enemy drone swarm
and computes the Graph Laplacian eigenvalues, specifically lambda-2 (Fiedler value).

Reference: PROJECT_MASTER_GUIDE_v2.md Sections 3.1, 3.2, 3.4, 3.5

Key Concepts:
    - G = (V, E) where V = enemy drones, E = communication links
    - Edge (i,j) exists if received power P_R(i,j) >= P_sens
    - Adjacency matrix A[i,j] = 1 if link exists (and not jammed)
    - Degree matrix D[i,i] = sum of row i of A
    - Laplacian L = D - A
    - Lambda-2 (Fiedler value) = second smallest eigenvalue of L
    
Theoretical Foundation (Proposition 1):
    - Lambda-2 > 0 iff graph is connected
    - Lambda-2 = 0 iff graph is disconnected
    - Minimizing lambda-2 to 0 guarantees swarm fragmentation

Author: MARL Jammer Team
"""

import numpy as np
from scipy import linalg
from typing import Tuple, Optional

from .fspl import (
    received_power_watts,
    compute_pairwise_received_power,
    SPEED_OF_LIGHT,
)


# =============================================================================
# ADJACENCY MATRIX CONSTRUCTION
# =============================================================================

def compute_adjacency_matrix(
    positions: np.ndarray,
    tx_power_watts: float,
    sensitivity_watts: float,
    frequency: float,
    jammed_links: Optional[np.ndarray] = None,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Compute the adjacency matrix of the enemy communication graph.
    
    An edge (i,j) exists if:
        1. P_R(i,j) >= P_sens (received signal above sensitivity threshold)
        2. Link is not jammed (jammed_links[i,j] == False)
    
    This implements Section 3.2 of the project guide using FSPL-based
    link determination rather than crude distance thresholds.
    
    Args:
        positions: Enemy drone positions, shape (N, 2)
        tx_power_watts: Transmit power in Watts (default: 0.1W = 20dBm)
        sensitivity_watts: Receiver sensitivity in Watts (default: 1e-12W = -90dBm)
        frequency: Carrier frequency in Hz
        jammed_links: Boolean array of shape (N, N), True if link is jammed.
                      If None, no links are jammed.
        eps: Small value for numerical stability
        
    Returns:
        Adjacency matrix A of shape (N, N), binary values {0, 1}
        
    Example:
        >>> positions = np.array([[0, 0], [50, 0], [150, 0]])  # 3 drones
        >>> A = compute_adjacency_matrix(positions, 0.1, 1e-12, 2.4e9)
        >>> # Drone 0 and 1 are 50m apart (connected)
        >>> # Drone 1 and 2 are 100m apart (may not be connected at 2.4GHz)
        
    Physical Interpretation:
        - A[i,j] = 1 means drone i can communicate with drone j
        - The graph is undirected (A is symmetric)
        - Self-loops excluded (diagonal is 0)
    """
    N = positions.shape[0]
    
    if N == 0:
        return np.array([]).reshape(0, 0)
    
    if N == 1:
        return np.array([[0]])
    
    # Compute received power matrix (N, N)
    P_R = compute_pairwise_received_power(positions, tx_power_watts, frequency, eps)
    
    # Links exist where received power >= sensitivity threshold
    # Note: P_R diagonal is inf, so A diagonal will be True
    A = (P_R >= sensitivity_watts).astype(np.float64)
    
    # Remove self-loops (diagonal = 0)
    np.fill_diagonal(A, 0)
    
    # Apply jamming mask if provided
    if jammed_links is not None:
        # Where jammed_links is True, set A to 0
        A = A * (~jammed_links).astype(np.float64)
    
    return A


def compute_adjacency_from_distances(
    distances: np.ndarray,
    tx_power_watts: float,
    sensitivity_watts: float,
    frequency: float,
    jammed_links: Optional[np.ndarray] = None,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Compute adjacency matrix from pre-computed distance matrix.
    
    This is an optimization when distances are already computed
    (e.g., for use with jamming module).
    
    Args:
        distances: Pairwise distance matrix, shape (N, N)
        tx_power_watts: Transmit power in Watts
        sensitivity_watts: Receiver sensitivity in Watts
        frequency: Carrier frequency in Hz
        jammed_links: Boolean array (N, N), True if link is jammed
        eps: Small value for numerical stability
        
    Returns:
        Adjacency matrix A of shape (N, N)
    """
    N = distances.shape[0]
    
    if N == 0:
        return np.array([]).reshape(0, 0)
    
    # Compute received power using FSPL
    P_R = received_power_watts(tx_power_watts, distances, frequency, eps)
    
    # Links exist where received power >= sensitivity
    A = (P_R >= sensitivity_watts).astype(np.float64)
    
    # Remove self-loops
    np.fill_diagonal(A, 0)
    
    # Apply jamming
    if jammed_links is not None:
        A = A * (~jammed_links).astype(np.float64)
    
    return A


# =============================================================================
# DEGREE MATRIX AND LAPLACIAN
# =============================================================================

def compute_degree_matrix(A: np.ndarray) -> np.ndarray:
    """
    Compute the degree matrix D from adjacency matrix A.
    
    D is a diagonal matrix where D[i,i] = sum of row i of A
    (i.e., the number of neighbors of node i).
    
    Reference: Section 3.4
    
    Args:
        A: Adjacency matrix of shape (N, N)
        
    Returns:
        Degree matrix D of shape (N, N), diagonal
        
    Example:
        >>> A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])  # Triangle graph
        >>> D = compute_degree_matrix(A)
        >>> np.diag(D)
        array([2., 2., 2.])  # Each node has degree 2
    """
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    return D


def compute_laplacian(A: np.ndarray) -> np.ndarray:
    """
    Compute the Graph Laplacian L = D - A.
    
    Properties of the Laplacian:
        - L is symmetric positive semi-definite
        - Row sums are zero: L @ 1 = 0
        - Smallest eigenvalue is always 0
        - Number of zero eigenvalues = number of connected components
    
    Reference: Section 3.4
    
    Args:
        A: Adjacency matrix of shape (N, N)
        
    Returns:
        Laplacian matrix L of shape (N, N)
        
    Example:
        >>> A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # Path graph
        >>> L = compute_laplacian(A)
        >>> L
        array([[ 1, -1,  0],
               [-1,  2, -1],
               [ 0, -1,  1]])
    """
    D = compute_degree_matrix(A)
    L = D - A
    return L


# =============================================================================
# LAMBDA-2 (FIEDLER VALUE) COMPUTATION
# =============================================================================

def compute_lambda2(
    L: np.ndarray,
    use_sparse: bool = False
) -> float:
    """
    Compute lambda-2 (Fiedler value / algebraic connectivity) of the Laplacian.
    
    Lambda-2 is the second smallest eigenvalue of L.
    
    Key Properties (from Section 3.5 and Proposition 1):
        - Lambda-2 > 0 iff graph is connected
        - Lambda-2 = 0 iff graph is disconnected  
        - Larger lambda-2 = more fault-tolerant topology
        - Minimizing lambda-2 to 0 guarantees swarm fragmentation
    
    Implementation uses subset_by_index=[0,1] to compute only the two
    smallest eigenvalues, reducing complexity from O(N^3) to O(N^2).
    Reference: Section 4.5
    
    Args:
        L: Laplacian matrix of shape (N, N)
        use_sparse: If True, use sparse eigensolver (for N > 200)
        
    Returns:
        Lambda-2 value (float). Returns 0.0 for single-node graphs.
        
    Example:
        >>> # Complete graph K_4 (fully connected, 4 nodes)
        >>> A = np.ones((4, 4)) - np.eye(4)
        >>> L = compute_laplacian(A)
        >>> lambda2 = compute_lambda2(L)
        >>> lambda2
        4.0  # For K_n, lambda_2 = n
        
        >>> # Disconnected graph (two isolated nodes)
        >>> A = np.zeros((2, 2))
        >>> L = compute_laplacian(A)
        >>> lambda2 = compute_lambda2(L)
        >>> lambda2
        0.0  # Graph is disconnected
    """
    N = L.shape[0]
    
    # Edge cases
    if N == 0:
        return 0.0
    if N == 1:
        return 0.0  # Single node is trivially "connected" but has no edges
    
    if use_sparse and N > 50:
        # Use sparse eigensolver for large graphs
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        
        L_sparse = csr_matrix(L)
        # Compute 2 smallest eigenvalues
        # which='SM' means smallest magnitude
        eigenvalues = eigsh(L_sparse, k=2, which='SM', return_eigenvectors=False)
        eigenvalues = np.sort(eigenvalues)
        return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    else:
        # Use dense eigensolver with subset for efficiency
        # subset_by_index=[0,1] computes only eigenvalues 0 and 1
        eigenvalues = linalg.eigh(L, subset_by_index=[0, 1], eigvals_only=True)
        
        # eigenvalues are returned in ascending order
        # eigenvalues[0] should be ~0, eigenvalues[1] is lambda-2
        return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0


def compute_all_eigenvalues(L: np.ndarray) -> np.ndarray:
    """
    Compute all eigenvalues of the Laplacian (for analysis/visualization).
    
    Args:
        L: Laplacian matrix of shape (N, N)
        
    Returns:
        Array of all eigenvalues in ascending order
    """
    if L.shape[0] == 0:
        return np.array([])
    
    eigenvalues = linalg.eigh(L, eigvals_only=True)
    return np.sort(eigenvalues)


def compute_lambda2_from_positions(
    positions: np.ndarray,
    tx_power_watts: float,
    sensitivity_watts: float,
    frequency: float,
    jammed_links: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Convenience function to compute lambda-2 directly from positions.
    
    This is the main function used during simulation.
    
    Args:
        positions: Enemy drone positions, shape (N, 2)
        tx_power_watts: Transmit power in Watts
        sensitivity_watts: Receiver sensitivity in Watts
        frequency: Carrier frequency in Hz
        jammed_links: Boolean array (N, N) of jammed links
        
    Returns:
        Tuple of (lambda_2, adjacency_matrix, laplacian_matrix)
        
    Example:
        >>> positions = np.random.rand(10, 2) * 100  # 10 drones in 100x100
        >>> lambda2, A, L = compute_lambda2_from_positions(
        ...     positions, 0.1, 1e-12, 2.4e9
        ... )
        >>> print(f"Algebraic connectivity: {lambda2:.4f}")
    """
    A = compute_adjacency_matrix(
        positions, tx_power_watts, sensitivity_watts, frequency, jammed_links
    )
    L = compute_laplacian(A)
    lambda2 = compute_lambda2(L)
    
    return lambda2, A, L


# =============================================================================
# GRAPH ANALYSIS UTILITIES
# =============================================================================

def is_graph_connected(A: np.ndarray) -> bool:
    """
    Check if the graph represented by adjacency matrix A is connected.
    
    Uses lambda-2 > 0 as the connectivity criterion (Fiedler's theorem).
    
    Args:
        A: Adjacency matrix of shape (N, N)
        
    Returns:
        True if graph is connected, False otherwise
        
    Note:
        A single-node graph is considered connected.
        An empty graph is considered disconnected.
    """
    N = A.shape[0]
    
    if N == 0:
        return False
    if N == 1:
        return True
    
    L = compute_laplacian(A)
    lambda2 = compute_lambda2(L)
    
    # Use small epsilon for numerical tolerance
    return lambda2 > 1e-10


def count_connected_components(A: np.ndarray) -> int:
    """
    Count the number of connected components in the graph.
    
    Uses the multiplicity of zero eigenvalues of the Laplacian.
    
    Args:
        A: Adjacency matrix of shape (N, N)
        
    Returns:
        Number of connected components
        
    Example:
        >>> # Two isolated nodes
        >>> A = np.zeros((2, 2))
        >>> count_connected_components(A)
        2
        
        >>> # Triangle (one component)
        >>> A = np.array([[0,1,1], [1,0,1], [1,1,0]])
        >>> count_connected_components(A)
        1
    """
    N = A.shape[0]
    
    if N == 0:
        return 0
    if N == 1:
        return 1
    
    L = compute_laplacian(A)
    eigenvalues = compute_all_eigenvalues(L)
    
    # Count eigenvalues that are approximately zero
    num_zero = np.sum(np.abs(eigenvalues) < 1e-10)
    
    return int(num_zero)


def get_edge_count(A: np.ndarray) -> int:
    """
    Get the number of edges in the graph.
    
    Args:
        A: Adjacency matrix of shape (N, N)
        
    Returns:
        Number of edges (undirected graph, so we divide by 2)
    """
    return int(np.sum(A) / 2)


def compute_graph_density(A: np.ndarray) -> float:
    """
    Compute the density of the graph.
    
    Density = (number of edges) / (maximum possible edges)
            = 2|E| / (N * (N-1))
    
    Args:
        A: Adjacency matrix of shape (N, N)
        
    Returns:
        Graph density in [0, 1]
    """
    N = A.shape[0]
    
    if N <= 1:
        return 0.0
    
    num_edges = get_edge_count(A)
    max_edges = N * (N - 1) / 2
    
    return num_edges / max_edges


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def verify_laplacian_properties(L: np.ndarray) -> dict:
    """
    Verify that the Laplacian has expected mathematical properties.
    
    Properties checked:
        1. Symmetric: L = L^T
        2. Row sums are zero: L @ 1 = 0
        3. Positive semi-definite: all eigenvalues >= 0
        4. Smallest eigenvalue is 0
    
    Args:
        L: Laplacian matrix
        
    Returns:
        Dictionary with verification results
    """
    N = L.shape[0]
    
    if N == 0:
        return {"valid": True, "message": "Empty graph"}
    
    results = {}
    
    # Check symmetry
    results["symmetric"] = np.allclose(L, L.T)
    
    # Check row sums
    row_sums = np.sum(L, axis=1)
    results["row_sums_zero"] = np.allclose(row_sums, 0)
    
    # Check eigenvalues
    eigenvalues = compute_all_eigenvalues(L)
    results["all_eigenvalues_nonnegative"] = np.all(eigenvalues >= -1e-10)
    results["smallest_eigenvalue_zero"] = np.abs(eigenvalues[0]) < 1e-10
    
    # Overall validity
    results["valid"] = all([
        results["symmetric"],
        results["row_sums_zero"],
        results["all_eigenvalues_nonnegative"],
        results["smallest_eigenvalue_zero"],
    ])
    
    return results


def verify_communication_graph() -> dict:
    """
    Run verification tests on communication graph construction.
    
    Tests various graph configurations to ensure correctness.
    
    Returns:
        Dictionary with test results
    """
    results = {}
    
    # Physical parameters
    P_tx = 0.1       # 20 dBm
    P_sens = 1e-12   # -90 dBm
    f = 2.4e9        # 2.4 GHz
    
    # Test 1: Two drones close together (should be connected)
    positions = np.array([[0, 0], [50, 0]])  # 50m apart
    A = compute_adjacency_matrix(positions, P_tx, P_sens, f)
    L = compute_laplacian(A)
    lambda2 = compute_lambda2(L)
    
    results["test_close_drones"] = {
        "positions": "50m apart",
        "adjacency": A.tolist(),
        "lambda2": lambda2,
        "connected": lambda2 > 0,
        "expected_connected": True,
        "pass": lambda2 > 0
    }
    
    # Test 2: Two drones far apart (should be disconnected at 2.4GHz)
    positions = np.array([[0, 0], [150, 0]])  # 150m apart (> 86m R_comm)
    A = compute_adjacency_matrix(positions, P_tx, P_sens, f)
    L = compute_laplacian(A)
    lambda2 = compute_lambda2(L)
    
    results["test_far_drones"] = {
        "positions": "150m apart",
        "adjacency": A.tolist(),
        "lambda2": lambda2,
        "connected": lambda2 > 0,
        "expected_connected": False,
        "pass": lambda2 < 1e-10
    }
    
    # Test 3: Complete graph (4 drones very close)
    positions = np.array([[0, 0], [5, 0], [0, 5], [5, 5]])  # 5m apart
    A = compute_adjacency_matrix(positions, P_tx, P_sens, f)
    L = compute_laplacian(A)
    lambda2 = compute_lambda2(L)
    
    expected_edges = 6  # K_4 has 6 edges
    actual_edges = get_edge_count(A)
    
    results["test_complete_graph"] = {
        "positions": "4 drones, 5m grid",
        "expected_edges": expected_edges,
        "actual_edges": actual_edges,
        "lambda2": lambda2,
        "expected_lambda2": 4.0,  # For K_n, lambda_2 = n
        "pass": actual_edges == expected_edges and lambda2 > 3.5
    }
    
    # Test 4: Laplacian properties
    results["laplacian_properties"] = verify_laplacian_properties(L)
    
    # Test 5: Line graph (3 drones in line, middle connects both ends)
    positions = np.array([[0, 0], [40, 0], [80, 0]])  # Each pair <86m
    A = compute_adjacency_matrix(positions, P_tx, P_sens, f)
    L = compute_laplacian(A)
    lambda2 = compute_lambda2(L)
    
    results["test_line_graph"] = {
        "positions": "3 drones in line, 40m spacing",
        "adjacency": A.tolist(),
        "lambda2": lambda2,
        "connected": is_graph_connected(A),
        "num_components": count_connected_components(A),
        "pass": is_graph_connected(A)
    }
    
    return results


# =============================================================================
# MAIN (for standalone testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Communication Graph Module Verification")
    print("=" * 60)
    
    results = verify_communication_graph()
    
    all_passed = True
    for test_name, test_result in results.items():
        print(f"\n{test_name}:")
        if isinstance(test_result, dict):
            for key, val in test_result.items():
                print(f"  {key}: {val}")
            if "pass" in test_result:
                status = "PASS" if test_result["pass"] else "FAIL"
                print(f"  STATUS: {status}")
                if not test_result["pass"]:
                    all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All communication graph tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
