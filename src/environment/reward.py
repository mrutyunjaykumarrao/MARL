"""
Reward Calculator Module
========================

Computes the 5-term reward function for jammer agents.

Reference: PROJECT_MASTER_GUIDE_v2.md Section 3.7

Reward Terms:
    1. Lambda-2 reduction (primary objective)
    2. Band match reward
    3. Proximity to assigned centroid
    4. Energy penalty
    5. Overlap penalty

R(t) = omega_1 * [1 - lambda_2(t)/lambda_2(0)]
     + omega_2 * (1/M) * sum 1[band_k = band_enemy]
     + omega_3 * (1/M) * sum exp(-kappa * d(centroid, p_k))
     - omega_4 * (1/M) * sum ||v_k||^2 / v_max^2
     - omega_5 * overlap_penalty

Author: MARL Jammer Team
"""

import numpy as np
from typing import Dict, Optional, Tuple, NamedTuple


class RewardComponents(NamedTuple):
    """Container for individual reward components."""
    lambda2_reduction: float
    band_match: float
    proximity: float
    energy_penalty: float
    overlap_penalty: float
    total: float


class RewardCalculator:
    """
    Calculates reward for jammer agents.
    
    The reward function is designed to:
        1. Maximize swarm connectivity disruption (lambda-2 reduction)
        2. Encourage correct frequency band selection
        3. Keep jammers near target clusters
        4. Minimize energy consumption
        5. Encourage spatial distribution (avoid overlap)
    
    Attributes:
        omega_1: Weight for lambda-2 reduction (default: 1.0)
        omega_2: Weight for band match (default: 0.3)
        omega_3: Weight for proximity (default: 0.2)
        omega_4: Weight for energy penalty (default: 0.1)
        omega_5: Weight for overlap penalty (default: 0.2)
        
    Example:
        >>> calculator = RewardCalculator()
        >>> reward, components = calculator.compute(
        ...     lambda2_current=0.5,
        ...     lambda2_initial=2.0,
        ...     jammer_bands=np.array([2, 2]),
        ...     enemy_band=2,
        ...     jammer_positions=np.array([[50, 50], [150, 150]]),
        ...     centroids={0: np.array([50, 50]), 1: np.array([150, 150])},
        ...     velocities=np.array([[1, 0], [0, 1]])
        ... )
    """
    
    def __init__(
        self,
        omega_1: float = 1.0,
        omega_2: float = 0.3,
        omega_3: float = 0.2,
        omega_4: float = 0.1,
        omega_5: float = 0.2,
        kappa: float = 0.05,
        v_max: float = 5.0,
        R_jam: float = 43.0,
        arena_size: float = 200.0
    ):
        """
        Initialize reward calculator.
        
        Args:
            omega_1: Weight for lambda-2 reduction
            omega_2: Weight for band match
            omega_3: Weight for proximity
            omega_4: Weight for energy penalty
            omega_5: Weight for overlap penalty
            kappa: Decay rate for proximity reward
            v_max: Maximum velocity for energy normalization
            R_jam: Jamming radius for overlap calculation
            arena_size: Arena size
        """
        self.omega_1 = omega_1
        self.omega_2 = omega_2
        self.omega_3 = omega_3
        self.omega_4 = omega_4
        self.omega_5 = omega_5
        self.kappa = kappa
        self.v_max = v_max
        self.R_jam = R_jam
        self.arena_size = arena_size
    
    def compute(
        self,
        lambda2_current: float,
        lambda2_initial: float,
        jammer_bands: np.ndarray,
        enemy_band: int,
        jammer_positions: np.ndarray,
        centroids: Dict[int, np.ndarray],
        velocities: np.ndarray,
        jammer_assignments: Optional[Dict[int, list]] = None
    ) -> Tuple[float, RewardComponents]:
        """
        Compute total reward and breakdown.
        
        Args:
            lambda2_current: Current lambda-2 value
            lambda2_initial: Initial lambda-2 at episode start
            jammer_bands: Band selections, shape (M,)
            enemy_band: Enemy swarm's operating band
            jammer_positions: Jammer positions, shape (M, 2)
            centroids: Dict of cluster centroids
            velocities: Velocity actions taken, shape (M, 2)
            jammer_assignments: Optional dict of cluster -> jammer list
            
        Returns:
            Tuple of (total_reward, RewardComponents)
        """
        M = jammer_positions.shape[0]
        
        if M == 0:
            return 0.0, RewardComponents(0, 0, 0, 0, 0, 0)
        
        # Term 1: Lambda-2 reduction
        r_lambda2 = self._compute_lambda2_reward(lambda2_current, lambda2_initial)
        
        # Term 2: Band match
        r_band = self._compute_band_match_reward(jammer_bands, enemy_band)
        
        # Term 3: Proximity to centroids
        r_proximity = self._compute_proximity_reward(
            jammer_positions, centroids, jammer_assignments
        )
        
        # Term 4: Energy penalty
        r_energy = self._compute_energy_penalty(velocities)
        
        # Term 5: Overlap penalty
        r_overlap = self._compute_overlap_penalty(jammer_positions)
        
        # Weighted sum
        total = (
            self.omega_1 * r_lambda2
            + self.omega_2 * r_band
            + self.omega_3 * r_proximity
            - self.omega_4 * r_energy
            - self.omega_5 * r_overlap
        )
        
        components = RewardComponents(
            lambda2_reduction=r_lambda2,
            band_match=r_band,
            proximity=r_proximity,
            energy_penalty=r_energy,
            overlap_penalty=r_overlap,
            total=total
        )
        
        return total, components
    
    def _compute_lambda2_reward(
        self,
        lambda2_current: float,
        lambda2_initial: float
    ) -> float:
        """
        Compute lambda-2 reduction reward.
        
        Returns value in [0, 1] where 1 = complete disconnection.
        """
        if lambda2_initial <= 0:
            # Already disconnected at start
            return 0.0
        
        # Clamp lambda2_current to be non-negative
        lambda2_current = max(0, lambda2_current)
        
        # Reduction ratio
        reduction = 1.0 - (lambda2_current / lambda2_initial)
        
        # Clamp to [0, 1]
        return np.clip(reduction, 0.0, 1.0)
    
    def _compute_band_match_reward(
        self,
        jammer_bands: np.ndarray,
        enemy_band: int
    ) -> float:
        """
        Compute band match reward.
        
        Returns fraction of jammers with correct band.
        """
        M = len(jammer_bands)
        if M == 0:
            return 0.0
        
        matches = np.sum(jammer_bands == enemy_band)
        return float(matches) / M
    
    def _compute_proximity_reward(
        self,
        jammer_positions: np.ndarray,
        centroids: Dict[int, np.ndarray],
        jammer_assignments: Optional[Dict[int, list]] = None
    ) -> float:
        """
        Compute proximity to assigned centroids.
        
        Uses exponential decay: exp(-kappa * distance)
        """
        M = jammer_positions.shape[0]
        
        if M == 0 or len(centroids) == 0:
            return 0.0
        
        total_proximity = 0.0
        
        for j in range(M):
            p_j = jammer_positions[j]
            
            # Find assigned centroid
            assigned_centroid = None
            if jammer_assignments is not None:
                for cluster_id, jammer_list in jammer_assignments.items():
                    if j in jammer_list and cluster_id in centroids:
                        assigned_centroid = centroids[cluster_id]
                        break
            
            # If no assignment, use nearest centroid
            if assigned_centroid is None:
                min_dist = float('inf')
                for centroid in centroids.values():
                    dist = np.linalg.norm(p_j - centroid)
                    if dist < min_dist:
                        min_dist = dist
                        assigned_centroid = centroid
            
            if assigned_centroid is not None:
                dist = np.linalg.norm(p_j - assigned_centroid)
                total_proximity += np.exp(-self.kappa * dist)
        
        return total_proximity / M
    
    def _compute_energy_penalty(
        self,
        velocities: np.ndarray
    ) -> float:
        """
        Compute energy penalty based on velocity magnitude.
        
        Returns mean normalized velocity squared.
        """
        M = velocities.shape[0]
        
        if M == 0:
            return 0.0
        
        # ||v||^2 / v_max^2
        speed_sq = np.sum(velocities ** 2, axis=1)  # Shape: (M,)
        normalized = speed_sq / (self.v_max ** 2)
        
        return float(np.mean(normalized))
    
    def _compute_overlap_penalty(
        self,
        jammer_positions: np.ndarray
    ) -> float:
        """
        Compute overlap penalty.
        
        Returns fraction of jammer pairs within 2*R_jam.
        """
        M = jammer_positions.shape[0]
        
        if M <= 1:
            return 0.0
        
        overlap_threshold = 2 * self.R_jam
        total_pairs = M * (M - 1) / 2
        overlapping = 0
        
        for i in range(M):
            for j in range(i + 1, M):
                dist = np.linalg.norm(jammer_positions[i] - jammer_positions[j])
                if dist < overlap_threshold:
                    overlapping += 1
        
        return float(overlapping) / total_pairs
    
    def compute_per_agent(
        self,
        lambda2_current: float,
        lambda2_initial: float,
        jammer_bands: np.ndarray,
        enemy_band: int,
        jammer_positions: np.ndarray,
        centroids: Dict[int, np.ndarray],
        velocities: np.ndarray,
        jammer_assignments: Optional[Dict[int, list]] = None
    ) -> np.ndarray:
        """
        Compute per-agent rewards (for individual learning).
        
        Most terms are shared, but proximity and energy are agent-specific.
        
        Returns:
            Per-agent rewards, shape (M,)
        """
        M = jammer_positions.shape[0]
        
        if M == 0:
            return np.array([])
        
        # Shared terms
        r_lambda2 = self._compute_lambda2_reward(lambda2_current, lambda2_initial)
        r_overlap = self._compute_overlap_penalty(jammer_positions)
        
        # Per-agent
        rewards = np.zeros(M, dtype=np.float32)
        
        for j in range(M):
            # Band match for this agent
            r_band_j = 1.0 if jammer_bands[j] == enemy_band else 0.0
            
            # Proximity for this agent
            r_prox_j = self._compute_single_proximity(
                j, jammer_positions, centroids, jammer_assignments
            )
            
            # Energy for this agent
            r_energy_j = np.sum(velocities[j] ** 2) / (self.v_max ** 2)
            
            # Combine
            rewards[j] = (
                self.omega_1 * r_lambda2
                + self.omega_2 * r_band_j
                + self.omega_3 * r_prox_j
                - self.omega_4 * r_energy_j
                - self.omega_5 * r_overlap
            )
        
        return rewards
    
    def _compute_single_proximity(
        self,
        agent_idx: int,
        jammer_positions: np.ndarray,
        centroids: Dict[int, np.ndarray],
        jammer_assignments: Optional[Dict[int, list]]
    ) -> float:
        """Compute proximity reward for single agent."""
        if len(centroids) == 0:
            return 0.0
        
        p_j = jammer_positions[agent_idx]
        
        # Find assigned centroid
        assigned_centroid = None
        if jammer_assignments is not None:
            for cluster_id, jammer_list in jammer_assignments.items():
                if agent_idx in jammer_list and cluster_id in centroids:
                    assigned_centroid = centroids[cluster_id]
                    break
        
        # Use nearest if not assigned
        if assigned_centroid is None:
            min_dist = float('inf')
            for centroid in centroids.values():
                dist = np.linalg.norm(p_j - centroid)
                if dist < min_dist:
                    min_dist = dist
                    assigned_centroid = centroid
        
        if assigned_centroid is None:
            return 0.0
        
        dist = np.linalg.norm(p_j - assigned_centroid)
        return np.exp(-self.kappa * dist)
    
    def get_reward_breakdown(
        self,
        components: RewardComponents
    ) -> Dict[str, float]:
        """
        Get human-readable reward breakdown.
        
        Returns dict with component names and weighted values.
        """
        return {
            "lambda2_reduction": self.omega_1 * components.lambda2_reduction,
            "band_match": self.omega_2 * components.band_match,
            "proximity": self.omega_3 * components.proximity,
            "energy_penalty": -self.omega_4 * components.energy_penalty,
            "overlap_penalty": -self.omega_5 * components.overlap_penalty,
            "total": components.total
        }


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_reward_calculator() -> dict:
    """Run verification tests."""
    results = {}
    
    calculator = RewardCalculator(
        omega_1=1.0, omega_2=0.3, omega_3=0.2, omega_4=0.1, omega_5=0.2
    )
    
    # Test 1: Perfect scenario - lambda2 = 0, all bands match
    reward, components = calculator.compute(
        lambda2_current=0.0,
        lambda2_initial=2.0,
        jammer_bands=np.array([2, 2]),
        enemy_band=2,
        jammer_positions=np.array([[50, 50], [150, 150]]),
        centroids={0: np.array([50, 50]), 1: np.array([150, 150])},
        velocities=np.array([[0, 0], [0, 0]])
    )
    
    results["test_perfect_scenario"] = {
        "lambda2_reduction": components.lambda2_reduction,
        "band_match": components.band_match,
        "proximity": components.proximity,
        "energy_penalty": components.energy_penalty,
        "total": components.total,
        "pass": (
            components.lambda2_reduction == 1.0 and
            components.band_match == 1.0 and
            components.energy_penalty == 0.0
        )
    }
    
    # Test 2: No disruption
    reward_none, comp_none = calculator.compute(
        lambda2_current=2.0,
        lambda2_initial=2.0,
        jammer_bands=np.array([0, 0]),  # Wrong bands
        enemy_band=2,
        jammer_positions=np.array([[50, 50], [150, 150]]),
        centroids={0: np.array([50, 50]), 1: np.array([150, 150])},
        velocities=np.array([[5, 5], [5, 5]])  # Max velocity
    )
    
    results["test_no_disruption"] = {
        "lambda2_reduction": comp_none.lambda2_reduction,
        "band_match": comp_none.band_match,
        "pass": comp_none.lambda2_reduction == 0.0 and comp_none.band_match == 0.0
    }
    
    # Test 3: High overlap
    reward_overlap, comp_overlap = calculator.compute(
        lambda2_current=1.0,
        lambda2_initial=2.0,
        jammer_bands=np.array([2, 2]),
        enemy_band=2,
        jammer_positions=np.array([[100, 100], [100, 100]]),  # Same position
        centroids={0: np.array([100, 100])},
        velocities=np.array([[0, 0], [0, 0]])
    )
    
    results["test_high_overlap"] = {
        "overlap_penalty": comp_overlap.overlap_penalty,
        "pass": comp_overlap.overlap_penalty == 1.0
    }
    
    # Test 4: Lambda2 already zero
    reward_already, comp_already = calculator.compute(
        lambda2_current=0.0,
        lambda2_initial=0.0,  # Already disconnected
        jammer_bands=np.array([2]),
        enemy_band=2,
        jammer_positions=np.array([[100, 100]]),
        centroids={0: np.array([100, 100])},
        velocities=np.array([[0, 0]])
    )
    
    results["test_already_disconnected"] = {
        "lambda2_reduction": comp_already.lambda2_reduction,
        "pass": comp_already.lambda2_reduction == 0.0
    }
    
    # Test 5: Per-agent rewards
    per_agent = calculator.compute_per_agent(
        lambda2_current=1.0,
        lambda2_initial=2.0,
        jammer_bands=np.array([2, 0]),  # First matches, second doesn't
        enemy_band=2,
        jammer_positions=np.array([[50, 50], [150, 150]]),
        centroids={0: np.array([50, 50]), 1: np.array([150, 150])},
        velocities=np.array([[0, 0], [5, 5]])
    )
    
    results["test_per_agent"] = {
        "shape": per_agent.shape,
        "agent_0_higher": per_agent[0] > per_agent[1],
        "pass": per_agent.shape == (2,) and per_agent[0] > per_agent[1]
    }
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Reward Calculator Verification")
    print("=" * 60)
    
    results = verify_reward_calculator()
    
    all_passed = True
    for test_name, result in results.items():
        print(f"\n{test_name}:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        if not result.get("pass", False):
            all_passed = False
    
    print("\n" + "=" * 60)
    print("PASSED" if all_passed else "FAILED")
    print("=" * 60)
