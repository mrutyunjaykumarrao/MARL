"""
Test Suite for Observation Builder Module
=========================================

Tests for src/environment/observation.py

Author: MARL Jammer Team
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.observation import ObservationBuilder, build_global_observation


class TestObservationBuilder:
    """Tests for ObservationBuilder class."""
    
    def test_initialization(self):
        """Test default initialization."""
        builder = ObservationBuilder()
        
        assert builder.arena_size == 200.0
        assert builder.R_jam == 43.0
        assert builder.obs_dim == 5
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        builder = ObservationBuilder(arena_size=300.0, R_jam=50.0)
        
        assert builder.arena_size == 300.0
        assert builder.R_jam == 50.0
    
    def test_build_shape(self):
        """Test observation shape is correct."""
        builder = ObservationBuilder()
        
        jammer_pos = np.random.rand(4, 2) * 200
        jammer_bands = np.array([2, 2, 2, 2])
        centroids = {0: np.array([50, 50]), 1: np.array([150, 150])}
        cluster_sizes = {0: 5, 1: 5}
        
        obs = builder.build(jammer_pos, jammer_bands, centroids, cluster_sizes, 2, 10)
        
        assert obs.shape == (4, 5)
        assert obs.dtype == np.float32
    
    def test_observations_normalized(self):
        """Test all observations are in [0, 1]."""
        builder = ObservationBuilder()
        
        for _ in range(10):
            jammer_pos = np.random.rand(4, 2) * 200
            jammer_bands = np.random.randint(0, 4, size=4)
            centroids = {0: np.array([50, 50]), 1: np.array([150, 150])}
            cluster_sizes = {0: 5, 1: 5}
            enemy_band = np.random.randint(0, 4)
            
            obs = builder.build(jammer_pos, jammer_bands, centroids, cluster_sizes, enemy_band, 10)
            
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)
    
    def test_band_match_feature(self):
        """Test band match feature is binary."""
        builder = ObservationBuilder()
        
        jammer_pos = np.array([[100, 100], [100, 100]])
        jammer_bands = np.array([2, 0])  # First matches, second doesn't
        centroids = {0: np.array([100, 100])}
        cluster_sizes = {0: 10}
        
        obs = builder.build(jammer_pos, jammer_bands, centroids, cluster_sizes, 2, 10)
        
        assert obs[0, 4] == 1.0  # Band match
        assert obs[1, 4] == 0.0  # No match
    
    def test_dist_to_centroid_at_centroid(self):
        """Test distance is ~0 when at centroid."""
        builder = ObservationBuilder()
        
        jammer_pos = np.array([[50.0, 50.0]])
        jammer_bands = np.array([2])
        centroids = {0: np.array([50.0, 50.0])}
        cluster_sizes = {0: 10}
        
        obs = builder.build(jammer_pos, jammer_bands, centroids, cluster_sizes, 2, 10)
        
        assert obs[0, 0] < 0.01  # Very close to 0
    
    def test_coverage_overlap_same_position(self):
        """Test overlap is 1 when all at same position."""
        builder = ObservationBuilder(R_jam=50.0)
        
        jammer_pos = np.array([[100, 100], [100, 100], [100, 100]])
        jammer_bands = np.array([2, 2, 2])
        centroids = {0: np.array([100, 100])}
        cluster_sizes = {0: 10}
        
        obs = builder.build(jammer_pos, jammer_bands, centroids, cluster_sizes, 2, 10)
        
        assert obs[0, 3] == 1.0  # Full overlap
    
    def test_coverage_overlap_far_apart(self):
        """Test overlap is 0 when jammers far apart."""
        builder = ObservationBuilder(R_jam=10.0)  # Small R_jam
        
        jammer_pos = np.array([[0, 0], [200, 200]])  # Far corners
        jammer_bands = np.array([2, 2])
        centroids = {0: np.array([100, 100])}
        cluster_sizes = {0: 10}
        
        obs = builder.build(jammer_pos, jammer_bands, centroids, cluster_sizes, 2, 10)
        
        assert obs[0, 3] == 0.0  # No overlap
    
    def test_single_agent(self):
        """Test single agent observations."""
        builder = ObservationBuilder()
        
        jammer_pos = np.array([[100, 100]])
        jammer_bands = np.array([2])
        centroids = {0: np.array([100, 100])}
        cluster_sizes = {0: 10}
        
        obs = builder.build(jammer_pos, jammer_bands, centroids, cluster_sizes, 2, 10)
        
        assert obs.shape == (1, 5)
        assert obs[0, 2] == 0.5  # dist_to_others sentinel
        assert obs[0, 3] == 0.0  # No overlap for single agent
    
    def test_empty_centroids(self):
        """Test with no centroids."""
        builder = ObservationBuilder()
        
        jammer_pos = np.array([[100, 100]])
        jammer_bands = np.array([2])
        centroids = {}
        cluster_sizes = {}
        
        obs = builder.build(jammer_pos, jammer_bands, centroids, cluster_sizes, 2, 10)
        
        assert obs.shape == (1, 5)
        assert obs[0, 0] == 1.0  # Max distance sentinel
    
    def test_vectorized_matches_regular(self):
        """Test vectorized version produces similar results."""
        builder = ObservationBuilder()
        
        np.random.seed(42)
        jammer_pos = np.random.rand(4, 2) * 200
        jammer_bands = np.array([2, 2, 1, 0])
        centroids = {0: np.array([50, 50]), 1: np.array([150, 150])}
        cluster_sizes = {0: 6, 1: 4}
        
        obs_regular = builder.build(jammer_pos, jammer_bands, centroids, cluster_sizes, 2, 10)
        obs_vec = builder.build_vectorized(jammer_pos, jammer_bands, centroids, cluster_sizes, 2, 10)
        
        # Band match should be identical
        assert np.allclose(obs_regular[:, 4], obs_vec[:, 4])
        
        # Other features should be close
        assert np.allclose(obs_regular, obs_vec, atol=0.2)


class TestBuildGlobalObservation:
    """Tests for build_global_observation function."""
    
    def test_mean_pooling(self):
        """Test global observation is mean of per-agent obs."""
        obs = np.array([
            [0.2, 0.4, 0.6, 0.8, 1.0],
            [0.4, 0.6, 0.8, 1.0, 0.0]
        ], dtype=np.float32)
        
        global_obs = build_global_observation(obs)
        
        expected = np.array([0.3, 0.5, 0.7, 0.9, 0.5])
        assert np.allclose(global_obs, expected)
    
    def test_empty_input(self):
        """Test empty observation array."""
        obs = np.zeros((0, 5), dtype=np.float32)
        
        global_obs = build_global_observation(obs)
        
        assert global_obs.shape == (5,)
        assert np.allclose(global_obs, np.zeros(5))
    
    def test_single_agent(self):
        """Test single agent equals self."""
        obs = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32)
        
        global_obs = build_global_observation(obs)
        
        assert np.allclose(global_obs, obs[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
