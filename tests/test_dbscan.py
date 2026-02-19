"""
Test Suite for DBSCAN Clustering Module
=======================================

Tests for src/clustering/dbscan_clustering.py

Test Categories:
    1. DBSCANClusterer class functionality
    2. Cluster computation edge cases
    3. Centroid computation
    4. Jammer assignment strategies
    5. Initial position generation
    6. Integration tests

Author: MARL Jammer Team
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from clustering.dbscan_clustering import (
    DBSCANClusterer,
    compute_clusters,
    compute_centroids,
    assign_jammers_to_clusters,
    get_jammer_initial_positions,
    get_assigned_centroid,
)


# =============================================================================
# DBSCANClusterer Class Tests
# =============================================================================

class TestDBSCANClusterer:
    """Tests for DBSCANClusterer class."""
    
    def test_initialization_default_params(self):
        """Test default parameter initialization."""
        clusterer = DBSCANClusterer()
        
        assert clusterer.eps == 30.0
        assert clusterer.min_samples == 2
        assert clusterer.arena_size == 200.0
        assert clusterer.labels is None
        assert clusterer.n_clusters == 0
    
    def test_initialization_custom_params(self):
        """Test custom parameter initialization."""
        clusterer = DBSCANClusterer(eps=50.0, min_samples=3, arena_size=300.0)
        
        assert clusterer.eps == 50.0
        assert clusterer.min_samples == 3
        assert clusterer.arena_size == 300.0
    
    def test_fit_two_distinct_clusters(self):
        """Test clustering two well-separated groups."""
        positions = np.array([
            [10, 10], [15, 15], [12, 18],    # Cluster A
            [180, 180], [185, 185], [182, 178]  # Cluster B
        ])
        
        clusterer = DBSCANClusterer(eps=30.0, min_samples=2)
        labels, centroids = clusterer.fit(positions)
        
        assert clusterer.n_clusters == 2
        assert len(labels) == 6
        assert len(centroids) == 2
        
        # First 3 should be in same cluster
        assert labels[0] == labels[1] == labels[2]
        # Last 3 should be in same cluster
        assert labels[3] == labels[4] == labels[5]
        # But different from first cluster
        assert labels[0] != labels[3]
    
    def test_fit_single_cluster(self):
        """Test all points forming one cluster."""
        positions = np.array([
            [50, 50], [55, 55], [52, 48], [48, 52], [53, 51]
        ])
        
        clusterer = DBSCANClusterer(eps=30.0, min_samples=2)
        labels, centroids = clusterer.fit(positions)
        
        assert clusterer.n_clusters == 1
        assert len(set(labels)) == 1  # All same label
        assert labels[0] >= 0  # Not noise
    
    def test_fit_all_noise(self):
        """Test when all points are noise (too far apart)."""
        positions = np.array([
            [0, 0], [100, 0], [0, 100], [100, 100]
        ])
        
        clusterer = DBSCANClusterer(eps=30.0, min_samples=2)
        labels, centroids = clusterer.fit(positions)
        
        assert clusterer.n_clusters == 0
        assert all(l == -1 for l in labels)
        # Should have fallback centroid at arena center
        assert len(centroids) > 0
    
    def test_fit_empty_array(self):
        """Test empty input."""
        positions = np.array([]).reshape(0, 2)
        
        clusterer = DBSCANClusterer()
        labels, centroids = clusterer.fit(positions)
        
        assert len(labels) == 0
        assert clusterer.n_clusters == 0
    
    def test_fit_single_point(self):
        """Test single point (treated as noise)."""
        positions = np.array([[50, 50]])
        
        clusterer = DBSCANClusterer()
        labels, centroids = clusterer.fit(positions)
        
        assert len(labels) == 1
        assert labels[0] == -1  # Single point is noise
        assert clusterer.n_clusters == 0
    
    def test_get_cluster_sizes(self):
        """Test cluster size computation."""
        positions = np.array([
            [10, 10], [15, 15], [12, 18],    # 3 points
            [180, 180], [185, 185]           # 2 points
        ])
        
        clusterer = DBSCANClusterer(eps=30.0, min_samples=2)
        clusterer.fit(positions)
        
        sizes = clusterer.get_cluster_sizes()
        
        assert sum(sizes.values()) == 5
        size_values = sorted(sizes.values())
        assert size_values == [2, 3]
    
    def test_get_cluster_positions(self):
        """Test extracting positions of specific cluster."""
        positions = np.array([
            [10, 10], [15, 15],       # Cluster 0
            [180, 180], [185, 185]    # Cluster 1
        ])
        
        clusterer = DBSCANClusterer(eps=30.0, min_samples=2)
        labels, _ = clusterer.fit(positions)
        
        cluster0_id = labels[0]
        cluster0_pos = clusterer.get_cluster_positions(cluster0_id, positions)
        
        assert cluster0_pos.shape[0] == 2  # 2 points in this cluster
    
    def test_centroid_drift_detection(self):
        """Test centroid drift computation."""
        clusterer = DBSCANClusterer(eps=30.0, min_samples=2)
        
        # First fit
        positions1 = np.array([
            [50, 50], [55, 55], [52, 52]
        ])
        clusterer.fit(positions1)
        
        # Second fit - shifted positions
        positions2 = np.array([
            [60, 60], [65, 65], [62, 62]  # Shifted by ~10m
        ])
        clusterer.fit(positions2)
        
        drift = clusterer.compute_centroid_drift()
        
        # Should have drift around 10-14m (diagonal shift)
        for cluster_id, d in drift.items():
            if cluster_id >= 0:
                assert 10 < d < 20
    
    def test_should_reassign_jammers(self):
        """Test jammer reassignment trigger."""
        clusterer = DBSCANClusterer(eps=30.0, min_samples=2)
        
        # First fit
        positions1 = np.array([
            [50, 50], [55, 55], [52, 52]
        ])
        clusterer.fit(positions1)
        
        # Second fit - large shift
        positions2 = np.array([
            [100, 100], [105, 105], [102, 102]  # Shifted by ~70m
        ])
        clusterer.fit(positions2)
        
        assert clusterer.should_reassign_jammers(drift_threshold=30.0) == True
    
    def test_no_reassignment_for_small_drift(self):
        """Test no reassignment for small movements."""
        clusterer = DBSCANClusterer(eps=30.0, min_samples=2)
        
        # First fit
        positions1 = np.array([
            [50, 50], [55, 55], [52, 52]
        ])
        clusterer.fit(positions1)
        
        # Second fit - small shift
        positions2 = np.array([
            [51, 51], [56, 56], [53, 53]  # Shifted by ~1.4m
        ])
        clustered = clusterer.fit(positions2)
        
        assert clusterer.should_reassign_jammers(drift_threshold=30.0) == False


# =============================================================================
# Compute Clusters Function Tests
# =============================================================================

class TestComputeClusters:
    """Tests for compute_clusters function."""
    
    def test_basic_clustering(self):
        """Test basic clustering functionality."""
        positions = np.array([
            [0, 0], [5, 5], [3, 3],
            [100, 100], [105, 105]
        ])
        
        labels = compute_clusters(positions, eps=30.0, min_samples=2)
        
        assert len(labels) == 5
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4]
    
    def test_insufficient_points(self):
        """Test with fewer points than min_samples."""
        positions = np.array([[0, 0]])
        
        labels = compute_clusters(positions, eps=30.0, min_samples=2)
        
        assert len(labels) == 1
        assert labels[0] == -1  # Should be noise


# =============================================================================
# Compute Centroids Tests
# =============================================================================

class TestComputeCentroids:
    """Tests for compute_centroids function."""
    
    def test_single_cluster_centroid(self):
        """Test centroid of single cluster."""
        positions = np.array([
            [0, 0], [10, 0], [5, 10]  # Triangle
        ])
        labels = np.array([0, 0, 0])
        
        centroids = compute_centroids(positions, labels)
        
        expected = np.array([5.0, 10/3])
        assert np.allclose(centroids[0], expected)
    
    def test_multiple_clusters_centroids(self):
        """Test centroids of multiple clusters."""
        positions = np.array([
            [0, 0], [10, 0],          # Cluster 0
            [100, 100], [110, 100]    # Cluster 1
        ])
        labels = np.array([0, 0, 1, 1])
        
        centroids = compute_centroids(positions, labels)
        
        assert np.allclose(centroids[0], [5, 0])
        assert np.allclose(centroids[1], [105, 100])
    
    def test_noise_centroid(self):
        """Test centroid computation includes noise points."""
        positions = np.array([
            [0, 0], [10, 0],    # Cluster 0
            [50, 50]           # Noise
        ])
        labels = np.array([0, 0, -1])
        
        centroids = compute_centroids(positions, labels)
        
        assert 0 in centroids
        assert -1 in centroids  # Noise has its own centroid
        assert np.allclose(centroids[-1], [50, 50])


# =============================================================================
# Jammer Assignment Tests
# =============================================================================

class TestJammerAssignment:
    """Tests for jammer assignment strategies."""
    
    def test_uniform_assignment(self):
        """Test uniform jammer distribution."""
        centroids = {0: np.array([25, 25]), 1: np.array([75, 75])}
        
        assignments = assign_jammers_to_clusters(4, centroids, strategy="uniform")
        
        # Should distribute evenly
        total = sum(len(v) for v in assignments.values())
        assert total == 4
        
        # Each cluster should get 2
        for cluster_id, jammers in assignments.items():
            assert len(jammers) == 2
    
    def test_proportional_assignment(self):
        """Test proportional jammer distribution."""
        centroids = {0: np.array([25, 25]), 1: np.array([75, 75])}
        sizes = {0: 8, 1: 2}  # 80% vs 20%
        
        assignments = assign_jammers_to_clusters(
            10, centroids, sizes, strategy="proportional"
        )
        
        total = sum(len(v) for v in assignments.values())
        assert total == 10
        
        # Cluster 0 should get more jammers
        assert len(assignments[0]) >= len(assignments[1])
    
    def test_largest_first_assignment(self):
        """Test largest-first jammer distribution."""
        centroids = {0: np.array([25, 25]), 1: np.array([75, 75])}
        sizes = {0: 10, 1: 5}
        
        assignments = assign_jammers_to_clusters(
            3, centroids, sizes, strategy="largest_first"
        )
        
        total = sum(len(v) for v in assignments.values())
        assert total == 3
        
        # Largest cluster (0) should get first jammer
        assert 0 in assignments[0]
    
    def test_zero_jammers(self):
        """Test assignment with zero jammers."""
        centroids = {0: np.array([25, 25])}
        
        assignments = assign_jammers_to_clusters(0, centroids)
        
        assert assignments == {}
    
    def test_zero_clusters(self):
        """Test assignment with zero clusters."""
        assignments = assign_jammers_to_clusters(4, {})
        
        assert assignments == {}
    
    def test_more_clusters_than_jammers(self):
        """Test when there are more clusters than jammers."""
        centroids = {
            0: np.array([25, 25]),
            1: np.array([75, 75]),
            2: np.array([125, 125]),
            3: np.array([175, 175])
        }
        
        assignments = assign_jammers_to_clusters(2, centroids, strategy="uniform")
        
        total = sum(len(v) for v in assignments.values())
        assert total == 2


# =============================================================================
# Initial Position Generation Tests
# =============================================================================

class TestInitialPositions:
    """Tests for jammer initial position generation."""
    
    def test_jammer_positions_near_centroids(self):
        """Test jammers start near their assigned centroids."""
        centroids = {0: np.array([50, 50]), 1: np.array([150, 150])}
        assignments = {0: [0, 1], 1: [2, 3]}
        
        positions = get_jammer_initial_positions(
            4, centroids, assignments, spread=10.0, arena_size=200.0
        )
        
        assert positions.shape == (4, 2)
        
        # Check jammers 0, 1 are near centroid 0
        for j in [0, 1]:
            dist = np.linalg.norm(positions[j] - centroids[0])
            assert dist < 30  # Within spread + tolerance
        
        # Check jammers 2, 3 are near centroid 1
        for j in [2, 3]:
            dist = np.linalg.norm(positions[j] - centroids[1])
            assert dist < 30
    
    def test_positions_within_arena(self):
        """Test all positions are clipped to arena bounds."""
        # Centroid near edge
        centroids = {0: np.array([195, 195])}
        assignments = {0: [0, 1, 2]}
        
        positions = get_jammer_initial_positions(
            3, centroids, assignments, spread=20.0, arena_size=200.0
        )
        
        assert np.all(positions >= 0)
        assert np.all(positions <= 200)


# =============================================================================
# Get Assigned Centroid Tests
# =============================================================================

class TestGetAssignedCentroid:
    """Tests for get_assigned_centroid function."""
    
    def test_get_assigned_centroid(self):
        """Test retrieving centroid for given jammer."""
        centroids = {0: np.array([50, 50]), 1: np.array([150, 150])}
        assignments = {0: [0, 1], 1: [2, 3]}
        
        centroid = get_assigned_centroid(0, assignments, centroids)
        assert np.allclose(centroid, [50, 50])
        
        centroid = get_assigned_centroid(2, assignments, centroids)
        assert np.allclose(centroid, [150, 150])
    
    def test_unassigned_jammer_uses_default(self):
        """Test fallback for unassigned jammer."""
        centroids = {0: np.array([50, 50])}
        assignments = {0: [0]}
        
        centroid = get_assigned_centroid(
            5, assignments, centroids,
            default_centroid=np.array([100, 100])
        )
        
        assert np.allclose(centroid, [100, 100])
    
    def test_unassigned_no_default_uses_arena_center(self):
        """Test ultimate fallback to arena center."""
        centroid = get_assigned_centroid(0, {}, {})
        
        assert np.allclose(centroid, [100, 100])


# =============================================================================
# Integration Tests
# =============================================================================

class TestClusteringIntegration:
    """Integration tests for clustering workflow."""
    
    def test_full_clustering_workflow(self):
        """Test complete clustering -> assignment -> positioning workflow."""
        np.random.seed(42)
        
        # Create enemy positions in two clusters
        enemies = np.vstack([
            np.random.randn(10, 2) * 10 + [30, 30],  # Cluster near (30, 30)
            np.random.randn(10, 2) * 10 + [170, 170]  # Cluster near (170, 170)
        ])
        
        # Cluster enemies
        clusterer = DBSCANClusterer(eps=30.0, min_samples=2)
        labels, centroids = clusterer.fit(enemies)
        
        assert clusterer.n_clusters == 2
        
        # Assign jammers
        sizes = clusterer.get_cluster_sizes()
        assignments = assign_jammers_to_clusters(
            4, centroids, sizes, strategy="proportional"
        )
        
        assert sum(len(v) for v in assignments.values()) == 4
        
        # Generate jammer positions
        positions = get_jammer_initial_positions(
            4, centroids, assignments, spread=10.0
        )
        
        assert positions.shape == (4, 2)
    
    def test_reclustering_on_movement(self):
        """Test re-clustering after enemy movement."""
        clusterer = DBSCANClusterer(eps=30.0, min_samples=2)
        
        # Initial positions - two clusters
        enemies_t0 = np.array([
            [30, 30], [35, 35],
            [170, 170], [175, 175]
        ])
        clusterer.fit(enemies_t0)
        
        assert clusterer.n_clusters == 2
        
        # After movement - clusters merge
        enemies_t1 = np.array([
            [100, 100], [105, 105],
            [110, 110], [115, 115]
        ])
        clusterer.fit(enemies_t1)
        
        assert clusterer.n_clusters == 1  # Merged into one cluster
        assert clusterer.should_reassign_jammers(drift_threshold=30.0) == True


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
