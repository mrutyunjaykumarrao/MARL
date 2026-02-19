"""
Test Suite for Action Space Handler Module
==========================================

Tests for src/environment/action_space.py

Author: MARL Jammer Team
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.action_space import ActionHandler


class TestActionHandler:
    """Tests for ActionHandler class."""
    
    def test_initialization(self):
        """Test default initialization."""
        handler = ActionHandler()
        
        assert handler.v_max == 5.0
        assert handler.num_bands == 4
        assert handler.arena_size == 200.0
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        handler = ActionHandler(v_max=10.0, num_bands=6, arena_size=300.0)
        
        assert handler.v_max == 10.0
        assert handler.num_bands == 6
        assert handler.arena_size == 300.0
    
    def test_sample_shape(self):
        """Test sampled actions have correct shape."""
        handler = ActionHandler()
        
        actions = handler.sample(M=4)
        
        assert actions.shape == (4, 3)
    
    def test_sample_velocity_range(self):
        """Test sampled velocities are in valid range."""
        handler = ActionHandler(v_max=5.0)
        
        for _ in range(10):
            actions = handler.sample(M=10)
            
            assert np.all(actions[:, :2] >= -5.0)
            assert np.all(actions[:, :2] <= 5.0)
    
    def test_sample_band_range(self):
        """Test sampled bands are valid."""
        handler = ActionHandler(num_bands=4)
        
        for _ in range(10):
            actions = handler.sample(M=10)
            
            assert np.all(actions[:, 2] >= 0)
            assert np.all(actions[:, 2] < 4)
    
    def test_sample_reproducibility(self):
        """Test seeded sampling is reproducible."""
        handler = ActionHandler()
        
        actions1 = handler.sample(M=4, seed=42)
        actions2 = handler.sample(M=4, seed=42)
        
        assert np.allclose(actions1, actions2)
    
    def test_split_action(self):
        """Test splitting combined action."""
        handler = ActionHandler()
        
        actions = np.array([
            [1.0, 2.0, 2],
            [3.0, 4.0, 0]
        ], dtype=np.float32)
        
        velocity, bands = handler.split_action(actions)
        
        assert velocity.shape == (2, 2)
        assert bands.shape == (2,)
        assert np.allclose(velocity, [[1, 2], [3, 4]])
        assert np.array_equal(bands, [2, 0])
    
    def test_combine_action(self):
        """Test combining velocity and band."""
        handler = ActionHandler()
        
        velocity = np.array([[1.0, 2.0], [3.0, 4.0]])
        bands = np.array([2, 0])
        
        actions = handler.combine_action(velocity, bands)
        
        assert actions.shape == (2, 3)
        assert np.allclose(actions[:, :2], velocity)
        assert np.allclose(actions[:, 2], bands)
    
    def test_clip_velocity(self):
        """Test velocity clipping."""
        handler = ActionHandler(v_max=5.0)
        
        velocity = np.array([[10.0, -10.0], [3.0, 3.0]])
        
        clipped = handler.clip_velocity(velocity)
        
        assert np.allclose(clipped, [[5.0, -5.0], [3.0, 3.0]])
    
    def test_clip_bands(self):
        """Test band clipping."""
        handler = ActionHandler(num_bands=4)
        
        bands = np.array([-1, 0, 3, 5])
        
        clipped = handler.clip_bands(bands)
        
        assert np.array_equal(clipped, [0, 0, 3, 3])
    
    def test_apply_action_basic(self):
        """Test basic position update."""
        handler = ActionHandler(v_max=5.0, arena_size=200.0)
        
        positions = np.array([[100.0, 100.0]])
        velocity = np.array([[5.0, 0.0]])
        
        new_pos = handler.apply_action(positions, velocity, dt=1.0)
        
        assert np.allclose(new_pos, [[105.0, 100.0]])
    
    def test_apply_action_boundary_clip(self):
        """Test position clipping at boundaries."""
        handler = ActionHandler(v_max=10.0, arena_size=200.0)
        
        positions = np.array([[195.0, 5.0]])
        velocity = np.array([[10.0, -10.0]])
        
        new_pos = handler.apply_action(positions, velocity, dt=1.0)
        
        assert np.allclose(new_pos, [[200.0, 0.0]])
    
    def test_apply_action_velocity_clipped(self):
        """Test velocity is clipped before applying."""
        handler = ActionHandler(v_max=5.0)
        
        positions = np.array([[100.0, 100.0]])
        velocity = np.array([[50.0, 50.0]])  # Way over v_max
        
        new_pos = handler.apply_action(positions, velocity, dt=1.0)
        
        # Should move at most 5m
        assert new_pos[0, 0] <= 105.0
        assert new_pos[0, 1] <= 105.0
    
    def test_get_band_frequency(self):
        """Test band to frequency conversion."""
        handler = ActionHandler()
        
        assert handler.get_band_frequency(0) == 433e6
        assert handler.get_band_frequency(1) == 915e6
        assert handler.get_band_frequency(2) == 2.4e9
        assert handler.get_band_frequency(3) == 5.8e9
    
    def test_get_band_name(self):
        """Test band to name conversion."""
        handler = ActionHandler()
        
        assert handler.get_band_name(0) == "433 MHz"
        assert handler.get_band_name(2) == "2.4 GHz"


class TestNetworkOutputConversion:
    """Tests for network output to action conversion."""
    
    def test_deterministic_velocity(self):
        """Test deterministic mode uses mean velocity."""
        handler = ActionHandler()
        
        mu = np.array([[1.0, 2.0], [3.0, 4.0]])
        log_sigma = np.array([[-1.0, -1.0], [-1.0, -1.0]])
        logits = np.array([[10.0, 0, 0, 0], [0, 0, 10.0, 0]])
        
        actions, log_probs, entropy = handler.from_network_output(
            mu, log_sigma, logits, deterministic=True
        )
        
        assert np.allclose(actions[:, :2], mu)
    
    def test_deterministic_band(self):
        """Test deterministic mode uses argmax band."""
        handler = ActionHandler()
        
        mu = np.zeros((2, 2))
        log_sigma = np.zeros((2, 2))
        logits = np.array([
            [0.0, 10.0, 0.0, 0.0],  # Band 1
            [0.0, 0.0, 0.0, 10.0]   # Band 3
        ])
        
        actions, _, _ = handler.from_network_output(
            mu, log_sigma, logits, deterministic=True
        )
        
        assert np.array_equal(actions[:, 2], [1, 3])
    
    def test_stochastic_samples_vary(self):
        """Test stochastic mode produces varying samples."""
        handler = ActionHandler()
        
        mu = np.zeros((4, 2))
        log_sigma = np.zeros((4, 2))  # sigma = 1
        logits = np.ones((4, 4)) * 0.25  # Uniform
        
        samples = []
        for seed in range(5):
            actions, _, _ = handler.from_network_output(
                mu, log_sigma, logits, deterministic=False,
                rng=np.random.RandomState(seed)
            )
            samples.append(actions)
        
        # Not all samples should be identical
        all_same = all(np.allclose(samples[0], s) for s in samples[1:])
        assert not all_same
    
    def test_log_probs_finite(self):
        """Test log probs are finite."""
        handler = ActionHandler()
        
        mu = np.random.randn(4, 2)
        log_sigma = np.random.randn(4, 2) * 0.5
        logits = np.random.randn(4, 4)
        
        _, log_probs, _ = handler.from_network_output(
            mu, log_sigma, logits, deterministic=False,
            rng=np.random.RandomState(42)
        )
        
        assert np.all(np.isfinite(log_probs))
    
    def test_entropy_positive(self):
        """Test entropy is positive for stochastic distributions."""
        handler = ActionHandler()
        
        mu = np.zeros((4, 2))
        log_sigma = np.zeros((4, 2))  # sigma = 1
        logits = np.zeros((4, 4))  # Uniform categorical
        
        _, _, entropy = handler.from_network_output(
            mu, log_sigma, logits, deterministic=False,
            rng=np.random.RandomState(42)
        )
        
        assert np.all(entropy > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
