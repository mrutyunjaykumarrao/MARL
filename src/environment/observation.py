"""
Observation Builder Module
==========================

Builds the 5-dimensional normalized observation vector for each jammer agent.

Reference: PROJECT_MASTER_GUIDE_v2.md Section 5

Observation Vector (per agent j):
    [0] dist_to_centroid: min_k ||p_j - mu_k(t)||_2 / arena_size
    [1] cluster_density: |C_assigned| / N  
    [2] dist_to_others: mean_{k!=j} ||p_j - p_k||_2 / arena_size
    [3] coverage_overlap: pairs within 2*R_jam / total_pairs
    [4] band_match: 1[b_j == b_enemy]

All features normalized to [0, 1].

Author: MARL Jammer Team
"""

import numpy as np
from typing import Dict, Optional, Tuple


class ObservationBuilder:
    """
    Builds observation vectors for all jammer agents.
    
    Attributes:
        arena_size: Size of arena for normalization
        R_jam: Effective jamming radius for overlap calculation
        obs_dim: Dimensionality of observation (5)
        
    Example:
        >>> builder = ObservationBuilder(arena_size=200.0, R_jam=43.0)
        >>> obs = builder.build(jammer_positions, jammer_bands, 
        ...                     centroids, cluster_sizes, enemy_band, N_enemies)
        >>> print(obs.shape)  # (M, 5)
    """
    
    OBS_DIM = 5  # Fixed observation dimensionality
    
    def __init__(
        self,
        arena_size: float = 200.0,
        R_jam: float = 43.0,
        distance_clip: float = 1.0
    ):
        """
        Initialize observation builder.
        
        Args:
            arena_size: Arena size for normalization (meters)
            R_jam: Effective jamming radius for overlap calculation
            distance_clip: Maximum normalized distance (default 1.0)
        """
        self.arena_size = arena_size
        self.R_jam = R_jam
        self.distance_clip = distance_clip
        self.obs_dim = self.OBS_DIM
    
    def build(
        self,
        jammer_positions: np.ndarray,
        jammer_bands: np.ndarray,
        centroids: Dict[int, np.ndarray],
        cluster_sizes: Dict[int, int],
        enemy_band: int,
        N_enemies: int,
        jammer_assignments: Optional[Dict[int, list]] = None
    ) -> np.ndarray:
        """
        Build observations for all jammers.
        
        Args:
            jammer_positions: Jammer positions, shape (M, 2)
            jammer_bands: Jammer band selections, shape (M,)
            centroids: Dict of cluster_id -> centroid position
            cluster_sizes: Dict of cluster_id -> number of enemies
            enemy_band: Enemy swarm's operating band (0-3)
            N_enemies: Total number of enemies
            jammer_assignments: Optional dict of cluster_id -> list of jammer indices
            
        Returns:
            Observations for all agents, shape (M, 5)
        """
        M = jammer_positions.shape[0]
        obs = np.zeros((M, self.OBS_DIM), dtype=np.float32)
        
        # Compute per-agent observations
        for j in range(M):
            obs[j] = self._build_single(
                j,
                jammer_positions,
                jammer_bands,
                centroids,
                cluster_sizes,
                enemy_band,
                N_enemies,
                jammer_assignments
            )
        
        return obs
    
    def _build_single(
        self,
        agent_idx: int,
        jammer_positions: np.ndarray,
        jammer_bands: np.ndarray,
        centroids: Dict[int, np.ndarray],
        cluster_sizes: Dict[int, int],
        enemy_band: int,
        N_enemies: int,
        jammer_assignments: Optional[Dict[int, list]] = None
    ) -> np.ndarray:
        """
        Build observation for a single agent.
        
        Returns:
            5-dimensional observation vector
        """
        M = jammer_positions.shape[0]
        p_j = jammer_positions[agent_idx]
        
        # Feature 0: Distance to nearest centroid
        dist_to_centroid = self._compute_dist_to_centroid(
            p_j, centroids, jammer_assignments, agent_idx
        )
        
        # Feature 1: Cluster density
        cluster_density = self._compute_cluster_density(
            agent_idx, cluster_sizes, N_enemies, jammer_assignments
        )
        
        # Feature 2: Mean distance to other jammers
        dist_to_others = self._compute_dist_to_others(
            agent_idx, jammer_positions
        )
        
        # Feature 3: Coverage overlap
        coverage_overlap = self._compute_coverage_overlap(
            jammer_positions
        )
        
        # Feature 4: Band match
        band_match = float(jammer_bands[agent_idx] == enemy_band)
        
        # Normalize and clip
        obs = np.array([
            np.clip(dist_to_centroid / self.arena_size, 0, self.distance_clip),
            np.clip(cluster_density, 0, 1.0),
            np.clip(dist_to_others / self.arena_size, 0, self.distance_clip),
            np.clip(coverage_overlap, 0, 1.0),
            band_match
        ], dtype=np.float32)
        
        return obs
    
    def _compute_dist_to_centroid(
        self,
        position: np.ndarray,
        centroids: Dict[int, np.ndarray],
        jammer_assignments: Optional[Dict[int, list]],
        agent_idx: int
    ) -> float:
        """
        Compute distance to assigned or nearest centroid.
        
        If jammer_assignments provided, use assigned cluster's centroid.
        Otherwise, find nearest centroid.
        """
        if len(centroids) == 0:
            return self.arena_size  # Max distance sentinel
        
        # Get assigned centroid if available
        if jammer_assignments is not None:
            for cluster_id, jammer_list in jammer_assignments.items():
                if agent_idx in jammer_list and cluster_id in centroids:
                    return float(np.linalg.norm(position - centroids[cluster_id]))
        
        # Otherwise find nearest centroid
        min_dist = float('inf')
        for centroid in centroids.values():
            dist = np.linalg.norm(position - centroid)
            min_dist = min(min_dist, dist)
        
        return min_dist if min_dist < float('inf') else self.arena_size
    
    def _compute_cluster_density(
        self,
        agent_idx: int,
        cluster_sizes: Dict[int, int],
        N_enemies: int,
        jammer_assignments: Optional[Dict[int, list]]
    ) -> float:
        """
        Compute density of assigned cluster (|C| / N).
        """
        if N_enemies == 0:
            return 0.0
        
        # Find assigned cluster
        if jammer_assignments is not None:
            for cluster_id, jammer_list in jammer_assignments.items():
                if agent_idx in jammer_list:
                    size = cluster_sizes.get(cluster_id, 0)
                    return size / N_enemies
        
        # If no assignment, return average density
        if len(cluster_sizes) == 0:
            return 0.0
        
        return sum(cluster_sizes.values()) / (len(cluster_sizes) * N_enemies)
    
    def _compute_dist_to_others(
        self,
        agent_idx: int,
        jammer_positions: np.ndarray
    ) -> float:
        """
        Compute mean distance to other jammers.
        """
        M = jammer_positions.shape[0]
        
        if M <= 1:
            return 0.5 * self.arena_size  # Sentinel for single agent
        
        p_j = jammer_positions[agent_idx]
        distances = []
        
        for k in range(M):
            if k != agent_idx:
                dist = np.linalg.norm(p_j - jammer_positions[k])
                distances.append(dist)
        
        return np.mean(distances)
    
    def _compute_coverage_overlap(
        self,
        jammer_positions: np.ndarray
    ) -> float:
        """
        Compute fraction of jammer pairs within 2*R_jam.
        
        High overlap indicates inefficient spatial distribution.
        """
        M = jammer_positions.shape[0]
        
        if M <= 1:
            return 0.0
        
        overlap_threshold = 2 * self.R_jam
        total_pairs = M * (M - 1) / 2
        overlapping_pairs = 0
        
        for i in range(M):
            for j in range(i + 1, M):
                dist = np.linalg.norm(jammer_positions[i] - jammer_positions[j])
                if dist < overlap_threshold:
                    overlapping_pairs += 1
        
        return overlapping_pairs / total_pairs
    
    def build_vectorized(
        self,
        jammer_positions: np.ndarray,
        jammer_bands: np.ndarray,
        centroids: Dict[int, np.ndarray],
        cluster_sizes: Dict[int, int],
        enemy_band: int,
        N_enemies: int,
        jammer_assignments: Optional[Dict[int, list]] = None
    ) -> np.ndarray:
        """
        Vectorized observation building for better performance.
        
        Same interface as build(), but uses NumPy operations.
        """
        M = jammer_positions.shape[0]
        obs = np.zeros((M, self.OBS_DIM), dtype=np.float32)
        
        if M == 0:
            return obs
        
        # Convert centroids to array
        if len(centroids) > 0:
            centroid_ids = list(centroids.keys())
            centroid_array = np.array([centroids[c] for c in centroid_ids])
        else:
            centroid_array = np.array([[self.arena_size/2, self.arena_size/2]])
            centroid_ids = [-1]
        
        # Feature 0: Distance to nearest centroid (vectorized)
        # Shape: (M, num_centroids)
        dists_to_centroids = np.linalg.norm(
            jammer_positions[:, np.newaxis, :] - centroid_array[np.newaxis, :, :],
            axis=2
        )
        min_dists = np.min(dists_to_centroids, axis=1)
        obs[:, 0] = np.clip(min_dists / self.arena_size, 0, self.distance_clip)
        
        # Feature 1: Cluster density
        # For simplicity, use nearest cluster's size
        if N_enemies > 0 and len(cluster_sizes) > 0:
            nearest_cluster_idx = np.argmin(dists_to_centroids, axis=1)
            for j in range(M):
                cluster_id = centroid_ids[nearest_cluster_idx[j]]
                obs[j, 1] = cluster_sizes.get(cluster_id, 0) / N_enemies
        
        # Feature 2: Mean distance to other jammers (vectorized)
        if M > 1:
            # Pairwise distances
            pairwise = np.linalg.norm(
                jammer_positions[:, np.newaxis, :] - jammer_positions[np.newaxis, :, :],
                axis=2
            )
            # Zero out diagonal
            np.fill_diagonal(pairwise, 0)
            # Mean excluding self
            mean_dists = pairwise.sum(axis=1) / (M - 1)
            obs[:, 2] = np.clip(mean_dists / self.arena_size, 0, self.distance_clip)
        else:
            obs[:, 2] = 0.5
        
        # Feature 3: Coverage overlap (shared across agents)
        overlap = self._compute_coverage_overlap(jammer_positions)
        obs[:, 3] = overlap
        
        # Feature 4: Band match
        obs[:, 4] = (jammer_bands == enemy_band).astype(np.float32)
        
        return obs


def build_global_observation(
    observations: np.ndarray
) -> np.ndarray:
    """
    Build mean-pooled global observation for critic.
    
    Args:
        observations: Per-agent observations, shape (M, 5)
        
    Returns:
        Mean-pooled observation, shape (5,)
    """
    if observations.shape[0] == 0:
        return np.zeros(ObservationBuilder.OBS_DIM, dtype=np.float32)
    
    return np.mean(observations, axis=0).astype(np.float32)


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_observation_builder() -> dict:
    """Run verification tests."""
    results = {}
    
    # Test 1: Basic observation building
    builder = ObservationBuilder(arena_size=200.0, R_jam=43.0)
    
    jammer_pos = np.array([[50, 50], [150, 150]])
    jammer_bands = np.array([2, 2])
    centroids = {0: np.array([50, 50]), 1: np.array([150, 150])}
    cluster_sizes = {0: 5, 1: 5}
    
    obs = builder.build(jammer_pos, jammer_bands, centroids, cluster_sizes, 2, 10)
    
    results["test_basic_obs"] = {
        "shape": obs.shape,
        "expected_shape": (2, 5),
        "in_range": np.all((obs >= 0) & (obs <= 1)),
        "pass": obs.shape == (2, 5) and np.all((obs >= 0) & (obs <= 1))
    }
    
    # Test 2: Band mismatch
    obs_mismatch = builder.build(jammer_pos, jammer_bands, centroids, cluster_sizes, 0, 10)
    
    results["test_band_mismatch"] = {
        "band_match_feature": obs_mismatch[:, 4].tolist(),
        "expected": [0.0, 0.0],
        "pass": np.allclose(obs_mismatch[:, 4], [0.0, 0.0])
    }
    
    # Test 3: Single agent
    obs_single = builder.build(
        np.array([[100, 100]]),
        np.array([2]),
        centroids,
        cluster_sizes,
        2, 10
    )
    
    results["test_single_agent"] = {
        "shape": obs_single.shape,
        "dist_to_others": obs_single[0, 2],
        "coverage_overlap": obs_single[0, 3],
        "pass": obs_single.shape == (1, 5)
    }
    
    # Test 4: Vectorized matches non-vectorized
    obs_vec = builder.build_vectorized(jammer_pos, jammer_bands, centroids, cluster_sizes, 2, 10)
    
    results["test_vectorized"] = {
        "shapes_match": obs_vec.shape == obs.shape,
        "values_close": np.allclose(obs_vec, obs, atol=0.1),
        "pass": obs_vec.shape == obs.shape
    }
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Observation Builder Verification")
    print("=" * 60)
    
    results = verify_observation_builder()
    
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
