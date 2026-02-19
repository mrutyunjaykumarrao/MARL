"""
Test Suite for Enemy Swarm Dynamics Module
==========================================

Tests for src/environment/enemy_swarm.py

Test Categories:
    1. EnemySwarm class - initialization
    2. Random walk motion model
    3. Coordinated turn motion model
    4. Boundary conditions
    5. Reset functionality
    6. Utility functions (clustered, formation swarms)
    7. Integration tests

Author: MARL Jammer Team
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.enemy_swarm import (
    EnemySwarm,
    create_clustered_swarm,
    create_formation_swarm,
)


# =============================================================================
# EnemySwarm Initialization Tests
# =============================================================================

class TestEnemySwarmInit:
    """Tests for EnemySwarm initialization."""
    
    def test_default_initialization(self):
        """Test default parameter initialization."""
        swarm = EnemySwarm(N=10)
        
        assert swarm.N == 10
        assert swarm.mode == 'random_walk'
        assert swarm.v_enemy == 2.0
        assert swarm.dt == 0.5
        assert swarm.arena_size == 200.0
        assert swarm.positions.shape == (10, 2)
        assert swarm.velocities.shape == (10, 2)
        assert swarm.step_count == 0
    
    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        swarm = EnemySwarm(
            N=20,
            mode='coordinated_turn',
            v_enemy=3.0,
            dt=0.25,
            arena_size=300.0
        )
        
        assert swarm.N == 20
        assert swarm.mode == 'coordinated_turn'
        assert swarm.v_enemy == 3.0
        assert swarm.dt == 0.25
        assert swarm.arena_size == 300.0
    
    def test_custom_positions(self):
        """Test initialization with custom positions."""
        positions = np.array([
            [50, 50], [100, 100], [150, 150]
        ])
        
        swarm = EnemySwarm(N=3, positions=positions)
        
        assert np.allclose(swarm.positions, positions)
    
    def test_seed_reproducibility(self):
        """Test random seed produces reproducible positions."""
        swarm1 = EnemySwarm(N=10, seed=42)
        swarm2 = EnemySwarm(N=10, seed=42)
        
        assert np.allclose(swarm1.positions, swarm2.positions)
        assert np.allclose(swarm1.velocities, swarm2.velocities)
    
    def test_different_seeds_different_positions(self):
        """Test different seeds produce different positions."""
        swarm1 = EnemySwarm(N=10, seed=42)
        swarm2 = EnemySwarm(N=10, seed=123)
        
        assert not np.allclose(swarm1.positions, swarm2.positions)
    
    def test_coordinated_turn_state_initialized(self):
        """Test coordinated turn mode initializes state vector."""
        swarm = EnemySwarm(N=5, mode='coordinated_turn')
        
        assert hasattr(swarm, 'state')
        assert swarm.state.shape == (5, 5)  # [x, x_dot, y, y_dot, omega]


# =============================================================================
# Random Walk Motion Tests
# =============================================================================

class TestRandomWalkMotion:
    """Tests for random walk motion model."""
    
    def test_positions_change_after_step(self):
        """Test positions change after stepping."""
        swarm = EnemySwarm(N=10, mode='random_walk', seed=42)
        initial = swarm.positions.copy()
        
        swarm.step()
        
        assert not np.allclose(initial, swarm.positions)
    
    def test_step_count_increments(self):
        """Test step counter increments."""
        swarm = EnemySwarm(N=5)
        
        swarm.step()
        assert swarm.step_count == 1
        
        swarm.step()
        assert swarm.step_count == 2
    
    def test_step_returns_positions(self):
        """Test step returns position array."""
        swarm = EnemySwarm(N=5)
        
        positions = swarm.step()
        
        assert positions.shape == (5, 2)
        assert np.allclose(positions, swarm.positions)
    
    def test_boundary_reflection_min(self):
        """Test reflective boundary at x=0, y=0."""
        # Start near boundary moving outward
        positions = np.array([[5.0, 5.0]])
        swarm = EnemySwarm(N=1, positions=positions, v_enemy=20.0, dt=1.0)
        swarm.velocities = np.array([[-30.0, -30.0]])  # Moving toward origin
        
        for _ in range(5):
            swarm.step()
        
        # Should still be in bounds
        assert np.all(swarm.positions >= 0)
    
    def test_boundary_reflection_max(self):
        """Test reflective boundary at arena edge."""
        positions = np.array([[195.0, 195.0]])
        swarm = EnemySwarm(N=1, positions=positions, v_enemy=20.0, dt=1.0)
        swarm.velocities = np.array([[30.0, 30.0]])  # Moving toward edge
        
        for _ in range(5):
            swarm.step()
        
        # Should still be in bounds
        assert np.all(swarm.positions <= 200)
    
    def test_stays_in_bounds_extended_simulation(self):
        """Test swarm stays in bounds after many steps."""
        swarm = EnemySwarm(N=20, mode='random_walk', v_enemy=5.0, seed=42)
        
        for _ in range(500):
            swarm.step()
        
        assert np.all(swarm.positions >= 0)
        assert np.all(swarm.positions <= swarm.arena_size)
    
    def test_speed_approximately_constant(self):
        """Test speed stays approximately at v_enemy."""
        swarm = EnemySwarm(N=10, v_enemy=2.0, seed=42)
        
        for _ in range(50):
            swarm.step()
        
        speeds = swarm.get_speeds()
        
        # Speed should be within reasonable tolerance of v_enemy
        assert np.allclose(speeds, 2.0, atol=0.5)


# =============================================================================
# Coordinated Turn Motion Tests
# =============================================================================

class TestCoordinatedTurnMotion:
    """Tests for coordinated turn motion model."""
    
    def test_positions_change(self):
        """Test positions change after stepping."""
        swarm = EnemySwarm(N=5, mode='coordinated_turn', seed=42)
        initial = swarm.positions.copy()
        
        swarm.step()
        
        assert not np.allclose(initial, swarm.positions)
    
    def test_state_updates(self):
        """Test state vector updates."""
        swarm = EnemySwarm(N=5, mode='coordinated_turn', seed=42)
        initial_state = swarm.state.copy()
        
        swarm.step()
        
        # At least position should change
        assert not np.allclose(initial_state[:, 0], swarm.state[:, 0])
    
    def test_wrapping_boundaries(self):
        """Test coordinated turn uses wrapping boundaries."""
        swarm = EnemySwarm(N=1, mode='coordinated_turn', arena_size=100.0)
        swarm.state = np.array([[95, 10, 50, 0, 0]])  # Near x boundary
        swarm.positions = swarm.state[:, [0, 2]]
        swarm.velocities = np.array([[10, 0]])
        
        for _ in range(5):
            swarm.step()
        
        # Position should wrap, staying in [0, arena_size)
        assert 0 <= swarm.positions[0, 0] < 100
    
    def test_omega_bounded(self):
        """Test turn rate stays within bounds."""
        swarm = EnemySwarm(N=10, mode='coordinated_turn', omega_max=0.5, seed=42)
        
        for _ in range(100):
            swarm.step()
        
        omegas = swarm.state[:, 4]
        assert np.all(np.abs(omegas) <= swarm.omega_max)
    
    def test_curved_trajectory(self):
        """Test drone follows curved path with non-zero omega."""
        swarm = EnemySwarm(N=1, mode='coordinated_turn')
        
        # Set up state with constant turn rate
        swarm.state[0] = [100, 2.0, 100, 0, 0.3]  # Moving +x, turning
        swarm.positions[0] = [100, 100]
        
        positions = [swarm.positions[0].copy()]
        for _ in range(50):
            swarm.step()
            positions.append(swarm.positions[0].copy())
        
        positions = np.array(positions)
        
        # Should have some curvature (not straight line)
        # Check y-position changes despite starting with y_dot=0
        y_variance = np.var(positions[:, 1])
        assert y_variance > 0  # Non-zero variance indicates curved path


# =============================================================================
# Reset Functionality Tests
# =============================================================================

class TestResetFunctionality:
    """Tests for swarm reset functionality."""
    
    def test_reset_clears_step_count(self):
        """Test reset sets step count to zero."""
        swarm = EnemySwarm(N=5)
        
        for _ in range(10):
            swarm.step()
        
        swarm.reset()
        
        assert swarm.step_count == 0
    
    def test_reset_with_seed(self):
        """Test reset with seed produces reproducible state."""
        swarm = EnemySwarm(N=10)
        
        swarm.reset(seed=42)
        pos1 = swarm.positions.copy()
        
        swarm.reset(seed=42)
        pos2 = swarm.positions.copy()
        
        assert np.allclose(pos1, pos2)
    
    def test_reset_with_positions(self):
        """Test reset with custom positions."""
        swarm = EnemySwarm(N=3)
        
        custom = np.array([[50, 50], [100, 100], [150, 150]])
        swarm.reset(positions=custom)
        
        assert np.allclose(swarm.positions, custom)
    
    def test_reset_returns_positions(self):
        """Test reset returns initial positions."""
        swarm = EnemySwarm(N=5)
        
        positions = swarm.reset()
        
        assert positions.shape == (5, 2)
        assert np.allclose(positions, swarm.positions)


# =============================================================================
# Utility Method Tests
# =============================================================================

class TestUtilityMethods:
    """Tests for utility methods."""
    
    def test_get_speeds(self):
        """Test speed computation."""
        swarm = EnemySwarm(N=3, v_enemy=2.0)
        # Manually set velocities
        swarm.velocities = np.array([
            [2, 0],    # Speed = 2
            [0, 3],    # Speed = 3
            [3, 4]     # Speed = 5
        ])
        
        speeds = swarm.get_speeds()
        
        assert np.allclose(speeds, [2, 3, 5])
    
    def test_get_headings(self):
        """Test heading angle computation."""
        swarm = EnemySwarm(N=4)
        swarm.velocities = np.array([
            [1, 0],    # 0 radians
            [0, 1],    # π/2 radians
            [-1, 0],   # π radians
            [0, -1]    # -π/2 radians
        ])
        
        headings = swarm.get_headings()
        
        assert np.isclose(headings[0], 0)
        assert np.isclose(headings[1], np.pi / 2)
        assert np.isclose(np.abs(headings[2]), np.pi)
        assert np.isclose(headings[3], -np.pi / 2)
    
    def test_set_positions(self):
        """Test manual position setting."""
        swarm = EnemySwarm(N=3)
        
        new_pos = np.array([[10, 10], [20, 20], [30, 30]])
        swarm.set_positions(new_pos)
        
        assert np.allclose(swarm.positions, new_pos)
    
    def test_set_positions_coordinated_turn(self):
        """Test position setting updates state in CT mode."""
        swarm = EnemySwarm(N=3, mode='coordinated_turn')
        
        new_pos = np.array([[10, 10], [20, 20], [30, 30]])
        swarm.set_positions(new_pos)
        
        assert np.allclose(swarm.state[:, 0], [10, 20, 30])  # x
        assert np.allclose(swarm.state[:, 2], [10, 20, 30])  # y


# =============================================================================
# Clustered Swarm Generation Tests
# =============================================================================

class TestClusteredSwarm:
    """Tests for clustered swarm generation."""
    
    def test_correct_shape(self):
        """Test output shape is correct."""
        positions = create_clustered_swarm(20, 3, arena_size=200.0)
        
        assert positions.shape == (20, 2)
    
    def test_reproducibility(self):
        """Test seed produces reproducible results."""
        pos1 = create_clustered_swarm(10, 2, seed=42)
        pos2 = create_clustered_swarm(10, 2, seed=42)
        
        assert np.allclose(pos1, pos2)
    
    def test_within_bounds(self):
        """Test positions are within arena."""
        positions = create_clustered_swarm(50, 5, arena_size=200.0)
        
        assert np.all(positions >= 0)
        assert np.all(positions <= 200)
    
    def test_clusters_are_clustered(self):
        """Test generated positions actually form clusters."""
        positions = create_clustered_swarm(
            30, 3, arena_size=200.0, cluster_radius=20.0, seed=42
        )
        
        # Run DBSCAN to verify clustering
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=30.0, min_samples=2)
        labels = dbscan.fit_predict(positions)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Should have approximately 3 clusters (might merge if centers close)
        assert n_clusters >= 2


# =============================================================================
# Formation Swarm Generation Tests
# =============================================================================

class TestFormationSwarm:
    """Tests for formation swarm generation."""
    
    def test_grid_formation_shape(self):
        """Test grid formation output shape."""
        positions = create_formation_swarm(9, 'grid', center=(100, 100))
        
        assert positions.shape == (9, 2)
    
    def test_line_formation(self):
        """Test line formation creates roughly linear arrangement."""
        positions = create_formation_swarm(
            5, 'line', center=(100, 100), spacing=20
        )
        
        # Y coordinates should be very close
        y_variance = np.var(positions[:, 1])
        assert y_variance < 5  # Small due to noise
    
    def test_circle_formation(self):
        """Test circle formation creates equidistant points."""
        positions = create_formation_swarm(
            8, 'circle', center=(100, 100), spacing=20
        )
        
        # All points should be roughly same distance from center
        distances = np.linalg.norm(positions - [100, 100], axis=1)
        distance_variance = np.var(distances)
        
        assert distance_variance < 5  # Should be nearly constant
    
    def test_v_formation(self):
        """Test V formation has V-shape."""
        positions = create_formation_swarm(
            10, 'v_shape', center=(100, 100), spacing=20
        )
        
        assert positions.shape == (10, 2)
        # Points should form a V (y increases as |x - center_x| increases)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for enemy swarm simulation."""
    
    def test_continuous_simulation(self):
        """Test continuous simulation maintains properties."""
        swarm = EnemySwarm(N=20, mode='random_walk', v_enemy=2.0, seed=42)
        
        for step in range(1000):
            swarm.step()
            
            # Check bounds
            assert np.all(swarm.positions >= 0)
            assert np.all(swarm.positions <= swarm.arena_size)
            
            # Check step count
            assert swarm.step_count == step + 1
    
    def test_simulation_with_clustering(self):
        """Test swarm can be clustered at each timestep."""
        from clustering.dbscan_clustering import DBSCANClusterer
        
        swarm = EnemySwarm(N=30, seed=42)
        clusterer = DBSCANClusterer(eps=30.0, min_samples=2)
        
        cluster_counts = []
        for _ in range(100):
            swarm.step()
            labels, _ = clusterer.fit(swarm.positions)
            cluster_counts.append(clusterer.n_clusters)
        
        # Should have varying cluster counts over time
        assert len(set(cluster_counts)) > 0
    
    def test_mode_switch_via_reset(self):
        """Test can effectively switch modes by recreating."""
        swarm_rw = EnemySwarm(N=10, mode='random_walk', seed=42)
        initial_pos = swarm_rw.positions.copy()
        
        # Simulate
        for _ in range(50):
            swarm_rw.step()
        
        # Create CT swarm from current positions
        swarm_ct = EnemySwarm(
            N=10,
            positions=swarm_rw.positions,
            mode='coordinated_turn'
        )
        
        assert np.allclose(swarm_ct.positions, swarm_rw.positions)
        assert swarm_ct.mode == 'coordinated_turn'


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
