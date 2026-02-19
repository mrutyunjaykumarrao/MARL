"""
DBSCAN Clustering Module
========================

Spatial clustering of enemy drones using DBSCAN algorithm.
Clusters are used to intelligently deploy jammer drones near enemy concentrations.

Reference: PROJECT_MASTER_GUIDE_v2.md Section 4.2

DBSCAN Parameters:
    - eps: 30m — neighborhood radius for core point determination
    - min_samples: 2 — minimum drones to form a cluster

Key Functions:
    - compute_clusters: Run DBSCAN on enemy positions
    - compute_centroids: Calculate cluster centers
    - assign_jammers_to_clusters: Distribute jammers across clusters

Author: MARL Jammer Team
"""

import numpy as np
from sklearn.cluster import DBSCAN
from typing import Tuple, List, Optional, Dict


class DBSCANClusterer:
    """
    DBSCAN-based clustering for enemy drone positions.
    
    This class handles:
        1. Running DBSCAN clustering on enemy positions
        2. Computing cluster centroids
        3. Tracking cluster changes over time
        4. Assigning jammers to clusters
    
    Attributes:
        eps: Neighborhood radius (meters)
        min_samples: Minimum points to form a cluster
        labels: Current cluster labels for each enemy
        centroids: Dictionary of cluster_id -> centroid position
        n_clusters: Number of clusters (excluding noise)
    
    Example:
        >>> clusterer = DBSCANClusterer(eps=30.0, min_samples=2)
        >>> positions = np.random.rand(10, 2) * 100
        >>> labels, centroids = clusterer.fit(positions)
        >>> print(f"Found {clusterer.n_clusters} clusters")
    """
    
    def __init__(
        self,
        eps: float = 30.0,
        min_samples: int = 2,
        arena_size: float = 200.0
    ):
        """
        Initialize DBSCAN clusterer.
        
        Args:
            eps: Maximum distance between two samples to be considered
                 in the same neighborhood (meters). Default: 30m
            min_samples: Minimum number of samples in a neighborhood
                         to form a core point. Default: 2
            arena_size: Size of the arena (for fallback centroid). Default: 200m
        """
        self.eps = eps
        self.min_samples = min_samples
        self.arena_size = arena_size
        
        # Internal state
        self._dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels: Optional[np.ndarray] = None
        self.centroids: Dict[int, np.ndarray] = {}
        self.n_clusters: int = 0
        self._prev_centroids: Dict[int, np.ndarray] = {}
    
    def fit(self, positions: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Run DBSCAN clustering on enemy positions.
        
        Args:
            positions: Enemy drone positions, shape (N, 2)
            
        Returns:
            Tuple of:
                - labels: Cluster label for each enemy, shape (N,)
                          Label -1 indicates noise (no cluster)
                - centroids: Dict mapping cluster_id to centroid position
                
        Example:
            >>> positions = np.array([[0, 0], [5, 5], [100, 100], [105, 105]])
            >>> labels, centroids = clusterer.fit(positions)
            >>> # Drones 0,1 form cluster 0; drones 2,3 form cluster 1
        """
        N = positions.shape[0]
        
        # Edge case: no enemies
        if N == 0:
            self.labels = np.array([], dtype=int)
            self.centroids = {}
            self.n_clusters = 0
            return self.labels, self.centroids
        
        # Edge case: single enemy
        if N == 1:
            self.labels = np.array([-1])  # Single point is noise
            self.centroids = {-1: positions[0].copy()}  # Treat as singleton
            self.n_clusters = 0
            return self.labels, self.centroids
        
        # Run DBSCAN
        self.labels = self._dbscan.fit_predict(positions)
        
        # Compute centroids
        self._prev_centroids = self.centroids.copy()
        self.centroids = compute_centroids(positions, self.labels)
        
        # Count clusters (excluding noise labeled -1)
        unique_labels = set(self.labels)
        self.n_clusters = len([l for l in unique_labels if l >= 0])
        
        # Handle edge case: no clusters found (all noise)
        if self.n_clusters == 0:
            # Place fallback centroid at arena center
            self.centroids = {-1: np.array([self.arena_size / 2, self.arena_size / 2])}
        
        return self.labels, self.centroids
    
    def get_cluster_sizes(self) -> Dict[int, int]:
        """
        Get the number of enemies in each cluster.
        
        Returns:
            Dict mapping cluster_id to number of enemies
        """
        if self.labels is None:
            return {}
        
        sizes = {}
        unique_labels = set(self.labels)
        for label in unique_labels:
            sizes[label] = int(np.sum(self.labels == label))
        return sizes
    
    def get_cluster_positions(self, cluster_id: int, positions: np.ndarray) -> np.ndarray:
        """
        Get positions of all enemies in a specific cluster.
        
        Args:
            cluster_id: Cluster label
            positions: All enemy positions
            
        Returns:
            Positions of enemies in this cluster, shape (K, 2)
        """
        if self.labels is None:
            return np.array([]).reshape(0, 2)
        
        mask = self.labels == cluster_id
        return positions[mask]
    
    def compute_centroid_drift(self) -> Dict[int, float]:
        """
        Compute how much each centroid has moved since last fit.
        
        Used to detect significant cluster structure changes
        that may require jammer reassignment.
        
        Returns:
            Dict mapping cluster_id to drift distance (meters)
        """
        drift = {}
        
        for cluster_id, centroid in self.centroids.items():
            if cluster_id in self._prev_centroids:
                prev = self._prev_centroids[cluster_id]
                drift[cluster_id] = float(np.linalg.norm(centroid - prev))
            else:
                drift[cluster_id] = float('inf')  # New cluster
        
        return drift
    
    def should_reassign_jammers(self, drift_threshold: float = 30.0) -> bool:
        """
        Check if cluster structure has changed enough to warrant jammer reassignment.
        
        Args:
            drift_threshold: Maximum acceptable centroid drift (meters)
            
        Returns:
            True if any centroid drifted more than threshold
        """
        drift = self.compute_centroid_drift()
        
        for cluster_id, d in drift.items():
            if d > drift_threshold:
                return True
        
        # Also check if number of clusters changed
        if len(self.centroids) != len(self._prev_centroids):
            return True
        
        return False


def compute_clusters(
    positions: np.ndarray,
    eps: float = 30.0,
    min_samples: int = 2
) -> np.ndarray:
    """
    Convenience function to run DBSCAN and get labels.
    
    Args:
        positions: Enemy positions, shape (N, 2)
        eps: Neighborhood radius
        min_samples: Minimum points for core point
        
    Returns:
        Cluster labels, shape (N,). Label -1 is noise.
    """
    if positions.shape[0] < min_samples:
        return np.full(positions.shape[0], -1, dtype=int)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(positions)


def compute_centroids(
    positions: np.ndarray,
    labels: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Compute centroid of each cluster.
    
    The centroid is the mean position of all drones in a cluster.
    
    Args:
        positions: Enemy positions, shape (N, 2)
        labels: Cluster labels from DBSCAN, shape (N,)
        
    Returns:
        Dict mapping cluster_id to centroid position (x, y)
        
    Note:
        Noise points (label = -1) are grouped together with centroid
        at their mean position if they exist.
    """
    centroids = {}
    unique_labels = set(labels)
    
    for label in unique_labels:
        mask = labels == label
        cluster_positions = positions[mask]
        
        if len(cluster_positions) > 0:
            centroids[label] = np.mean(cluster_positions, axis=0)
    
    return centroids


def assign_jammers_to_clusters(
    num_jammers: int,
    centroids: Dict[int, np.ndarray],
    cluster_sizes: Optional[Dict[int, int]] = None,
    strategy: str = "proportional"
) -> Dict[int, List[int]]:
    """
    Assign jammers to clusters.
    
    Determines which cluster each jammer should target.
    
    Args:
        num_jammers: Total number of jammers (M)
        centroids: Dict of cluster_id -> centroid position
        cluster_sizes: Dict of cluster_id -> number of enemies (optional)
        strategy: Assignment strategy:
            - "proportional": Assign more jammers to larger clusters
            - "uniform": Assign equal jammers to each cluster
            - "largest_first": Prioritize largest clusters
            
    Returns:
        Dict mapping cluster_id -> list of jammer indices
        
    Example:
        >>> centroids = {0: np.array([25, 25]), 1: np.array([75, 75])}
        >>> sizes = {0: 6, 1: 4}  # 60% in cluster 0
        >>> assignments = assign_jammers_to_clusters(4, centroids, sizes)
        >>> # assignments might be {0: [0, 1, 2], 1: [3]} 
    """
    if num_jammers == 0 or len(centroids) == 0:
        return {}
    
    # Filter out noise cluster for assignment (keep only valid clusters)
    valid_clusters = [c for c in centroids.keys() if c >= 0]
    
    # If all points are noise, treat noise as a single cluster
    if len(valid_clusters) == 0:
        valid_clusters = list(centroids.keys())
    
    n_clusters = len(valid_clusters)
    
    if n_clusters == 0:
        return {}
    
    if strategy == "uniform":
        # Equal distribution
        base_count = num_jammers // n_clusters
        remainder = num_jammers % n_clusters
        
        assignments = {}
        jammer_idx = 0
        for i, cluster_id in enumerate(valid_clusters):
            count = base_count + (1 if i < remainder else 0)
            assignments[cluster_id] = list(range(jammer_idx, jammer_idx + count))
            jammer_idx += count
            
    elif strategy == "proportional" and cluster_sizes is not None:
        # Proportional to cluster size
        total_enemies = sum(cluster_sizes.get(c, 1) for c in valid_clusters)
        
        if total_enemies == 0:
            total_enemies = n_clusters  # Fallback
        
        assignments = {}
        jammer_idx = 0
        remaining = num_jammers
        
        for i, cluster_id in enumerate(valid_clusters):
            size = cluster_sizes.get(cluster_id, 1)
            
            if i == len(valid_clusters) - 1:
                # Last cluster gets remaining jammers
                count = remaining
            else:
                # Proportional allocation
                count = max(1, int(num_jammers * size / total_enemies))
                count = min(count, remaining)
            
            assignments[cluster_id] = list(range(jammer_idx, jammer_idx + count))
            jammer_idx += count
            remaining -= count
            
    elif strategy == "largest_first" and cluster_sizes is not None:
        # Sort clusters by size (descending)
        sorted_clusters = sorted(
            valid_clusters,
            key=lambda c: cluster_sizes.get(c, 0),
            reverse=True
        )
        
        # Assign one jammer per cluster in order, then repeat
        assignments = {c: [] for c in valid_clusters}
        for j in range(num_jammers):
            cluster = sorted_clusters[j % n_clusters]
            assignments[cluster].append(j)
    else:
        # Default: uniform
        return assign_jammers_to_clusters(num_jammers, centroids, None, "uniform")
    
    return assignments


def get_jammer_initial_positions(
    num_jammers: int,
    centroids: Dict[int, np.ndarray],
    assignments: Dict[int, List[int]],
    spread: float = 10.0,
    arena_size: float = 200.0
) -> np.ndarray:
    """
    Generate initial jammer positions near assigned cluster centroids.
    
    Args:
        num_jammers: Number of jammers
        centroids: Dict of cluster_id -> centroid position
        assignments: Dict of cluster_id -> list of jammer indices
        spread: Random spread around centroid (meters)
        arena_size: Arena boundary for clipping
        
    Returns:
        Initial jammer positions, shape (M, 2)
        
    Note:
        Jammers are placed near their assigned cluster centroid
        with small random offset to avoid exact overlap.
    """
    positions = np.zeros((num_jammers, 2))
    
    for cluster_id, jammer_indices in assignments.items():
        if cluster_id not in centroids:
            continue
            
        centroid = centroids[cluster_id]
        
        for j in jammer_indices:
            # Random offset from centroid
            offset = np.random.uniform(-spread, spread, size=2)
            pos = centroid + offset
            
            # Clip to arena bounds
            pos = np.clip(pos, 0, arena_size)
            positions[j] = pos
    
    return positions


def get_assigned_centroid(
    jammer_index: int,
    assignments: Dict[int, List[int]],
    centroids: Dict[int, np.ndarray],
    default_centroid: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Get the centroid assigned to a specific jammer.
    
    Args:
        jammer_index: Index of the jammer
        assignments: Cluster assignments
        centroids: Cluster centroids
        default_centroid: Fallback if jammer not assigned
        
    Returns:
        Centroid position (x, y)
    """
    for cluster_id, jammer_list in assignments.items():
        if jammer_index in jammer_list:
            return centroids.get(cluster_id, default_centroid)
    
    # Jammer not found in any assignment
    if default_centroid is not None:
        return default_centroid
    
    # Ultimate fallback: arena center
    return np.array([100.0, 100.0])


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def verify_clustering() -> dict:
    """
    Run verification tests on DBSCAN clustering.
    
    Returns:
        Dictionary with test results
    """
    results = {}
    
    # Test 1: Two distinct clusters
    positions = np.array([
        [10, 10], [15, 15], [12, 18],  # Cluster 0
        [80, 80], [85, 85], [82, 88],  # Cluster 1
    ])
    
    clusterer = DBSCANClusterer(eps=30.0, min_samples=2)
    labels, centroids = clusterer.fit(positions)
    
    results["test_two_clusters"] = {
        "n_clusters": clusterer.n_clusters,
        "expected": 2,
        "labels": labels.tolist(),
        "pass": clusterer.n_clusters == 2
    }
    
    # Test 2: Single cluster (all close)
    positions = np.array([
        [50, 50], [55, 55], [52, 48], [48, 52]
    ])
    
    labels, centroids = clusterer.fit(positions)
    
    results["test_single_cluster"] = {
        "n_clusters": clusterer.n_clusters,
        "expected": 1,
        "pass": clusterer.n_clusters == 1
    }
    
    # Test 3: All noise (all far apart)
    positions = np.array([
        [0, 0], [100, 0], [0, 100], [100, 100]
    ])
    
    labels, centroids = clusterer.fit(positions)
    
    results["test_all_noise"] = {
        "n_clusters": clusterer.n_clusters,
        "expected": 0,
        "labels": labels.tolist(),
        "has_fallback_centroid": len(centroids) > 0,
        "pass": clusterer.n_clusters == 0 and len(centroids) > 0
    }
    
    # Test 4: Centroid computation
    positions = np.array([[0, 0], [10, 0], [5, 10]])  # Triangle
    labels = np.array([0, 0, 0])  # All in cluster 0
    centroids = compute_centroids(positions, labels)
    
    expected_centroid = np.array([5.0, 10/3])
    actual_centroid = centroids[0]
    
    results["test_centroid_computation"] = {
        "expected": expected_centroid.tolist(),
        "actual": actual_centroid.tolist(),
        "pass": np.allclose(actual_centroid, expected_centroid)
    }
    
    # Test 5: Jammer assignment (proportional)
    centroids = {0: np.array([25, 25]), 1: np.array([75, 75])}
    sizes = {0: 6, 1: 4}  # 60% in cluster 0
    
    assignments = assign_jammers_to_clusters(4, centroids, sizes, "proportional")
    total_assigned = sum(len(v) for v in assignments.values())
    
    results["test_jammer_assignment"] = {
        "total_assigned": total_assigned,
        "expected": 4,
        "assignments": {k: len(v) for k, v in assignments.items()},
        "pass": total_assigned == 4
    }
    
    return results


# =============================================================================
# MAIN (for standalone testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DBSCAN Clustering Module Verification")
    print("=" * 60)
    
    results = verify_clustering()
    
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
        print("All clustering tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
