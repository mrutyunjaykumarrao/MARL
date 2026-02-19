"""
Test Suite for Reward Calculator Module
========================================

Tests for src/environment/reward.py

Author: MARL Jammer Team
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.reward import RewardCalculator, RewardComponents


class TestRewardCalculator:
    """Tests for RewardCalculator class."""
    
    def test_initialization_default_weights(self):
        """Test default weight initialization."""
        reward = RewardCalculator()
        
        assert reward.omega_1 == 1.0   # lambda2 reduction
        assert reward.omega_2 == 0.3   # band match
        assert reward.omega_3 == 0.2   # proximity
        assert reward.omega_4 == 0.1   # energy penalty
        assert reward.omega_5 == 0.2   # overlap penalty
    
    def test_initialization_custom_weights(self):
        """Test custom weight initialization."""
        reward = RewardCalculator(
            omega_1=2.0,
            omega_2=0.5,
            omega_3=0.3,
            omega_4=0.2,
            omega_5=0.1
        )
        
        assert reward.omega_1 == 2.0
        assert reward.omega_2 == 0.5
        assert reward.omega_3 == 0.3
        assert reward.omega_4 == 0.2
        assert reward.omega_5 == 0.1
    
    def test_lambda2_reduction_reward_positive(self):
        """Test positive reward when lambda2 decreases."""
        reward = RewardCalculator()
        
        # 50% reduction in lambda2
        total, components = reward.compute(
            lambda2_current=0.5,
            lambda2_initial=1.0,
            jammer_bands=np.array([2, 2, 2, 2]),
            enemy_band=2,
            jammer_positions=np.zeros((4, 2)),
            centroids={0: np.array([0.0, 0.0])},
            velocities=np.zeros((4, 2))
        )
        
        assert total > 0
        assert components.lambda2_reduction > 0
    
    def test_lambda2_reduction_clamped(self):
        """Test lambda2 reduction is clamped to [0, 1]."""
        reward = RewardCalculator()
        
        # Lambda2 increased from initial
        _, components = reward.compute(
            lambda2_current=1.0,
            lambda2_initial=0.5,  # Started lower
            jammer_bands=np.array([2, 2, 2, 2]),
            enemy_band=2,
            jammer_positions=np.zeros((4, 2)),
            centroids={0: np.array([0.0, 0.0])},
            velocities=np.zeros((4, 2))
        )
        
        # Lambda2 reduction should be clamped to 0
        assert components.lambda2_reduction == 0.0
    
    def test_band_match_bonus(self):
        """Test band match provides bonus."""
        reward = RewardCalculator()
        
        # All jammers match enemy band
        all_match, comp_match = reward.compute(
            lambda2_current=1.0,
            lambda2_initial=1.0,
            jammer_bands=np.array([2, 2, 2, 2]),
            enemy_band=2,
            jammer_positions=np.zeros((4, 2)),
            centroids={0: np.array([0.0, 0.0])},
            velocities=np.zeros((4, 2))
        )
        
        # No jammers match
        no_match, comp_no = reward.compute(
            lambda2_current=1.0,
            lambda2_initial=1.0,
            jammer_bands=np.array([0, 0, 0, 0]),
            enemy_band=2,
            jammer_positions=np.zeros((4, 2)),
            centroids={0: np.array([0.0, 0.0])},
            velocities=np.zeros((4, 2))
        )
        
        assert comp_match.band_match > comp_no.band_match
        assert comp_match.band_match == 1.0  # All match
        assert comp_no.band_match == 0.0  # None match
    
    def test_proximity_reward(self):
        """Test proximity to centroids is rewarded."""
        reward = RewardCalculator()
        
        # Close to centroid
        close_r, comp_close = reward.compute(
            lambda2_current=1.0,
            lambda2_initial=1.0,
            jammer_bands=np.array([0, 0, 0, 0]),
            enemy_band=2,
            jammer_positions=np.array([[10, 10], [10, 10], [10, 10], [10, 10]], dtype=float),
            centroids={0: np.array([10.0, 10.0])},
            velocities=np.zeros((4, 2))
        )
        
        # Far from centroid
        far_r, comp_far = reward.compute(
            lambda2_current=1.0,
            lambda2_initial=1.0,
            jammer_bands=np.array([0, 0, 0, 0]),
            enemy_band=2,
            jammer_positions=np.array([[100, 100], [100, 100], [100, 100], [100, 100]], dtype=float),
            centroids={0: np.array([10.0, 10.0])},
            velocities=np.zeros((4, 2))
        )
        
        assert comp_close.proximity > comp_far.proximity
    
    def test_energy_penalty(self):
        """Test high velocity incurs energy penalty."""
        reward = RewardCalculator()
        
        # Stationary
        stationary_r, comp_stat = reward.compute(
            lambda2_current=1.0,
            lambda2_initial=1.0,
            jammer_bands=np.array([2, 2, 2, 2]),
            enemy_band=2,
            jammer_positions=np.zeros((4, 2)),
            centroids={0: np.array([0.0, 0.0])},
            velocities=np.zeros((4, 2))
        )
        
        # Moving fast
        fast_r, comp_fast = reward.compute(
            lambda2_current=1.0,
            lambda2_initial=1.0,
            jammer_bands=np.array([2, 2, 2, 2]),
            enemy_band=2,
            jammer_positions=np.zeros((4, 2)),
            centroids={0: np.array([0.0, 0.0])},
            velocities=np.array([[5, 5], [5, 5], [5, 5], [5, 5]], dtype=float)
        )
        
        assert comp_stat.energy_penalty == 0.0
        assert comp_fast.energy_penalty > 0
        assert stationary_r > fast_r
    
    def test_overlap_penalty(self):
        """Test overlapping jammers incurs penalty."""
        reward = RewardCalculator(R_jam=20.0)
        
        # Spread out (>2*R_jam=40m apart)
        spread_r, comp_spread = reward.compute(
            lambda2_current=1.0,
            lambda2_initial=1.0,
            jammer_bands=np.array([2, 2, 2, 2]),
            enemy_band=2,
            jammer_positions=np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=float),
            centroids={0: np.array([50.0, 50.0])},
            velocities=np.zeros((4, 2))
        )
        
        # All stacked at same point
        stacked_r, comp_stacked = reward.compute(
            lambda2_current=1.0,
            lambda2_initial=1.0,
            jammer_bands=np.array([2, 2, 2, 2]),
            enemy_band=2,
            jammer_positions=np.array([[50, 50], [50, 50], [50, 50], [50, 50]], dtype=float),
            centroids={0: np.array([50.0, 50.0])},
            velocities=np.zeros((4, 2))
        )
        
        # Spread should have no overlap, stacked has full overlap
        assert comp_spread.overlap_penalty == 0.0
        assert comp_stacked.overlap_penalty == 1.0  # All pairs overlap
    
    def test_reward_components_type(self):
        """Test reward returns RewardComponents tuple."""
        reward = RewardCalculator()
        
        total, components = reward.compute(
            lambda2_current=0.5,
            lambda2_initial=1.0,
            jammer_bands=np.array([2, 2, 2, 2]),
            enemy_band=2,
            jammer_positions=np.zeros((4, 2)),
            centroids={0: np.array([0.0, 0.0])},
            velocities=np.zeros((4, 2))
        )
        
        assert isinstance(components, RewardComponents)
        assert hasattr(components, 'lambda2_reduction')
        assert hasattr(components, 'band_match')
        assert hasattr(components, 'proximity')
        assert hasattr(components, 'energy_penalty')
        assert hasattr(components, 'overlap_penalty')
        assert hasattr(components, 'total')


class TestPerAgentReward:
    """Tests for per-agent reward computation."""
    
    def test_per_agent_shape(self):
        """Test per-agent rewards have correct shape."""
        reward = RewardCalculator()
        
        per_agent = reward.compute_per_agent(
            lambda2_current=0.5,
            lambda2_initial=1.0,
            jammer_bands=np.array([2, 2, 2, 2]),
            enemy_band=2,
            jammer_positions=np.zeros((4, 2)),
            centroids={0: np.array([0.0, 0.0])},
            velocities=np.zeros((4, 2))
        )
        
        assert per_agent.shape == (4,)
    
    def test_per_agent_band_match_difference(self):
        """Test agents with band match get higher reward."""
        reward = RewardCalculator()
        
        # Agents 0, 2 match enemy band (2), agents 1, 3 don't
        per_agent = reward.compute_per_agent(
            lambda2_current=1.0,
            lambda2_initial=1.0,
            jammer_bands=np.array([2, 0, 2, 1]),
            enemy_band=2,
            jammer_positions=np.zeros((4, 2)),
            centroids={0: np.array([0.0, 0.0])},
            velocities=np.zeros((4, 2))
        )
        
        # Agents 0, 2 match band, should have higher rewards
        assert per_agent[0] > per_agent[1]
        assert per_agent[2] > per_agent[3]


class TestEdgeCases:
    """Edge case tests for reward computation."""
    
    def test_zero_lambda2_gives_max_reward(self):
        """Test when lambda2 becomes zero (disconnected)."""
        reward = RewardCalculator()
        
        total, components = reward.compute(
            lambda2_current=0.0,  # Fully disconnected
            lambda2_initial=1.0,
            jammer_bands=np.array([2, 2, 2, 2]),
            enemy_band=2,
            jammer_positions=np.zeros((4, 2)),
            centroids={0: np.array([0.0, 0.0])},
            velocities=np.zeros((4, 2))
        )
        
        # Should have maximum lambda2_reduction reward (1.0)
        assert components.lambda2_reduction == 1.0
    
    def test_single_jammer(self):
        """Test with single jammer."""
        reward = RewardCalculator()
        
        total, comp = reward.compute(
            lambda2_current=0.5,
            lambda2_initial=1.0,
            jammer_bands=np.array([2]),
            enemy_band=2,
            jammer_positions=np.array([[50, 50]], dtype=float),
            centroids={0: np.array([50.0, 50.0])},
            velocities=np.zeros((1, 2))
        )
        
        assert np.isfinite(total)
        # Single jammer has no overlap (needs >=2)
        assert comp.overlap_penalty == 0.0
    
    def test_no_centroids(self):
        """Test when there are no clusters/centroids."""
        reward = RewardCalculator()
        
        total, comp = reward.compute(
            lambda2_current=0.5,
            lambda2_initial=1.0,
            jammer_bands=np.array([2, 2, 2, 2]),
            enemy_band=2,
            jammer_positions=np.zeros((4, 2)),
            centroids={},  # No centroids
            velocities=np.zeros((4, 2))
        )
        
        # Should still return a valid reward
        assert np.isfinite(total)
        # Proximity is 0 when no centroids
        assert comp.proximity == 0.0
    
    def test_zero_initial_lambda2(self):
        """Test with zero initial lambda2."""
        reward = RewardCalculator()
        
        total, comp = reward.compute(
            lambda2_current=0.0,
            lambda2_initial=0.0,  # Already disconnected
            jammer_bands=np.array([2, 2, 2, 2]),
            enemy_band=2,
            jammer_positions=np.zeros((4, 2)),
            centroids={0: np.array([0.0, 0.0])},
            velocities=np.zeros((4, 2))
        )
        
        # Lambda2 reward should be 0 (no improvement possible)
        assert comp.lambda2_reduction == 0.0
    
    def test_empty_jammers(self):
        """Test with zero jammers."""
        reward = RewardCalculator()
        
        total, comp = reward.compute(
            lambda2_current=1.0,
            lambda2_initial=1.0,
            jammer_bands=np.array([], dtype=int),
            enemy_band=2,
            jammer_positions=np.zeros((0, 2)),
            centroids={0: np.array([50.0, 50.0])},
            velocities=np.zeros((0, 2))
        )
        
        assert total == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
