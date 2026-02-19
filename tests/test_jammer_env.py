"""
Test Suite for Jammer Environment Module
=========================================

Tests for src/environment/jammer_env.py

Author: MARL Jammer Team
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.jammer_env import JammerEnv


class TestJammerEnvInit:
    """Tests for JammerEnv initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        env = JammerEnv()
        
        assert env.M == 4  # Number of jammers
        assert env.N == 20  # Number of enemies
        assert env.arena_size == 200.0
        assert env.dt == 0.5
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        env = JammerEnv(
            M=8, N=30,
            arena_size=300.0,
            dt=0.2
        )
        
        assert env.M == 8
        assert env.N == 30
        assert env.arena_size == 300.0
        assert env.dt == 0.2
    
    def test_observation_space_shape(self):
        """Test observation space has correct dimensions."""
        env = JammerEnv(M=4)
        
        obs_shape = env.observation_space.shape
        
        # Each agent has 5-dim observation
        assert obs_shape == (4, 5)
    
    def test_action_space_structure(self):
        """Test action space has velocity and band components."""
        env = JammerEnv(M=4)
        
        # Action space should be a Dict
        assert 'velocity' in env.action_space.spaces
        assert 'band' in env.action_space.spaces
        
        # Velocity is Box (M, 2)
        assert env.action_space['velocity'].shape == (4, 2)
        
        # Band is MultiDiscrete with 4 options per agent
        assert env.action_space['band'].nvec.shape == (4,)


class TestJammerEnvReset:
    """Tests for environment reset."""
    
    def test_reset_returns_observation(self):
        """Test reset returns observation."""
        env = JammerEnv(M=4)
        
        obs, info = env.reset()
        
        assert obs is not None
        assert isinstance(obs, np.ndarray)
    
    def test_reset_observation_shape(self):
        """Test reset observation shape."""
        env = JammerEnv(M=4)
        
        obs, _ = env.reset()
        
        assert obs.shape == (4, 5)
    
    def test_reset_observation_normalized(self):
        """Test reset observations are normalized to [0, 1]."""
        env = JammerEnv(M=4)
        
        obs, _ = env.reset()
        
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)
    
    def test_reset_info_contains_lambda2(self):
        """Test info dict contains lambda2 values."""
        env = JammerEnv()
        
        _, info = env.reset()
        
        assert "lambda2_initial" in info
        assert "lambda2_current" in info
        assert isinstance(info["lambda2_initial"], (int, float))
    
    def test_reset_info_contains_enemy_band(self):
        """Test info dict contains enemy band."""
        env = JammerEnv()
        
        _, info = env.reset()
        
        assert "enemy_band" in info
        assert 0 <= info["enemy_band"] <= 3
    
    def test_reset_reproducibility(self):
        """Test seeded reset is reproducible."""
        env = JammerEnv(seed=42)
        obs1, info1 = env.reset(seed=42)
        
        env2 = JammerEnv(seed=42)
        obs2, info2 = env2.reset(seed=42)
        
        assert np.allclose(obs1, obs2)
        assert info1["enemy_band"] == info2["enemy_band"]
    
    def test_reset_initializes_jammers(self):
        """Test reset initializes jammer positions."""
        env = JammerEnv(M=4, arena_size=200.0)
        env.reset()
        
        assert env.jammer_positions.shape == (4, 2)
        assert np.all(env.jammer_positions >= 0)
        assert np.all(env.jammer_positions <= 200)


class TestJammerEnvStep:
    """Tests for environment step function."""
    
    def test_step_returns_tuple(self):
        """Test step returns 5-tuple (gym v26+ API)."""
        env = JammerEnv()
        env.reset()
        
        action = {
            'velocity': np.zeros((env.M, 2), dtype=np.float32),
            'band': np.zeros(env.M, dtype=int)
        }
        result = env.step(action)
        
        assert len(result) == 5  # obs, reward, terminated, truncated, info
    
    def test_step_observation_shape(self):
        """Test step returns correct observation shape."""
        env = JammerEnv(M=4)
        env.reset()
        
        action = {
            'velocity': np.zeros((4, 2), dtype=np.float32),
            'band': np.zeros(4, dtype=int)
        }
        obs, _, _, _, _ = env.step(action)
        
        assert obs.shape == (4, 5)
    
    def test_step_reward_scalar(self):
        """Test step returns scalar reward."""
        env = JammerEnv()
        env.reset()
        
        action = {
            'velocity': np.zeros((env.M, 2), dtype=np.float32),
            'band': np.full(env.M, env.enemy_band)
        }
        _, reward, _, _, _ = env.step(action)
        
        assert isinstance(reward, (int, float))
    
    def test_step_updates_positions(self):
        """Test step updates jammer positions with velocity."""
        env = JammerEnv(M=4, arena_size=200.0)
        env.reset()
        
        # Set jammers to center
        env.jammer_positions = np.full((4, 2), 100.0)
        old_positions = env.jammer_positions.copy()
        
        # Move in positive direction
        action = {
            'velocity': np.full((4, 2), 5.0, dtype=np.float32),
            'band': np.zeros(4, dtype=int)
        }
        
        env.step(action)
        
        # Positions should change by velocity * dt
        expected = old_positions + 5.0 * env.dt
        assert np.allclose(env.jammer_positions, expected)
    
    def test_step_updates_enemy_positions(self):
        """Test step makes enemies move."""
        env = JammerEnv()
        env.reset()
        
        old_enemy = env.enemy_swarm.positions.copy()
        
        action = {
            'velocity': np.zeros((env.M, 2), dtype=np.float32),
            'band': np.zeros(env.M, dtype=int)
        }
        env.step(action)
        
        # Enemy positions should change
        assert not np.allclose(env.enemy_swarm.positions, old_enemy)
    
    def test_step_boundaries_respected(self):
        """Test positions stay within arena."""
        env = JammerEnv(arena_size=200.0, v_max=10.0)
        env.reset()
        
        # Place jammers at edge
        env.jammer_positions = np.full((env.M, 2), 199.0)
        
        # Push toward boundary
        action = {
            'velocity': np.full((env.M, 2), 10.0, dtype=np.float32),
            'band': np.zeros(env.M, dtype=int)
        }
        
        env.step(action)
        
        assert np.all(env.jammer_positions <= 200.0)
        assert np.all(env.jammer_positions >= 0.0)
    
    def test_step_band_assignment(self):
        """Test step uses band from action."""
        env = JammerEnv(M=4)
        env.reset()
        
        action = {
            'velocity': np.zeros((4, 2), dtype=np.float32),
            'band': np.array([0, 1, 2, 3])
        }
        
        env.step(action)
        
        expected_bands = np.array([0, 1, 2, 3])
        assert np.array_equal(env.jammer_bands, expected_bands)
    
    def test_step_info_contains_positions(self):
        """Test info contains jammer and enemy positions."""
        env = JammerEnv()
        env.reset()
        
        action = {
            'velocity': np.zeros((env.M, 2), dtype=np.float32),
            'band': np.zeros(env.M, dtype=int)
        }
        _, _, _, _, info = env.step(action)
        
        assert "jammer_positions" in info
        assert "enemy_positions" in info
        assert info["jammer_positions"].shape == (env.M, 2)
        assert info["enemy_positions"].shape == (env.N, 2)


class TestEpisodeDynamics:
    """Tests for episode dynamics and termination."""
    
    def test_episode_length(self):
        """Test episode runs for max steps or terminates early."""
        env = JammerEnv(max_steps=10)
        env.reset()
        
        steps = 0
        truncated = False
        terminated = False
        
        while not (terminated or truncated):
            action = {
                'velocity': np.zeros((env.M, 2), dtype=np.float32),
                'band': np.full(env.M, env.enemy_band)
            }
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
        
        # Episode should terminate via truncation at max_steps or
        # early termination when swarm is disconnected (lambda2 = 0)
        assert steps <= 10
        assert terminated or truncated
    
    def test_truncated_at_max_steps(self):
        """Test truncated flag set at max steps."""
        env = JammerEnv(max_steps=5)
        env.reset()
        
        for i in range(5):
            action = {
                'velocity': np.zeros((env.M, 2), dtype=np.float32),
                'band': np.zeros(env.M, dtype=int)
            }
            _, _, terminated, truncated, _ = env.step(action)
        
        assert truncated == True
    
    def test_info_lambda2_tracked(self):
        """Test lambda2 is tracked in info across steps."""
        env = JammerEnv()
        env.reset()
        
        lambda2_values = []
        for _ in range(5):
            action = {
                'velocity': np.zeros((env.M, 2), dtype=np.float32),
                'band': np.full(env.M, env.enemy_band)
            }
            _, _, _, _, info = env.step(action)
            lambda2_values.append(info["lambda2_current"])
        
        # Should have 5 valid values
        assert len(lambda2_values) == 5
        assert all(isinstance(v, (int, float)) for v in lambda2_values)


class TestLambda2Computation:
    """Tests for lambda2 (Fiedler value) computation."""
    
    def test_lambda2_non_negative(self):
        """Test lambda2 is always non-negative."""
        env = JammerEnv()
        
        for _ in range(5):
            env.reset()
            action = {
                'velocity': np.zeros((env.M, 2), dtype=np.float32),
                'band': np.full(env.M, env.enemy_band)
            }
            _, _, _, _, info = env.step(action)
            
            assert info["lambda2_current"] >= 0
    
    def test_lambda2_finite(self):
        """Test lambda2 is always finite."""
        env = JammerEnv()
        
        for _ in range(5):
            env.reset()
            action = {
                'velocity': np.zeros((env.M, 2), dtype=np.float32),
                'band': np.zeros(env.M, dtype=int)
            }
            _, _, _, _, info = env.step(action)
            
            assert np.isfinite(info["lambda2_current"])
    
    def test_lambda2_unjammed_vs_jammed(self):
        """Test we can compute both jammed and unjammed lambda2."""
        env = JammerEnv(N=10, M=4, seed=42)
        env.reset()
        
        # Get unjammed lambda2
        unjammed_l2 = env._compute_lambda2(disrupted=False)
        
        # Get jammed lambda2
        jammed_l2 = env._compute_lambda2(disrupted=True)
        
        # Both should be valid
        assert np.isfinite(unjammed_l2)
        assert np.isfinite(jammed_l2)


class TestRender:
    """Tests for environment rendering."""
    
    def test_render_returns_array(self):
        """Test render returns RGB array."""
        env = JammerEnv(render_mode="rgb_array")
        env.reset()
        
        frame = env.render()
        
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3  # Height x Width x Channels
        assert frame.shape[2] == 3  # RGB


class TestGlobalObservation:
    """Tests for global observation (for centralized critic)."""
    
    def test_global_observation_shape(self):
        """Test global observation has correct shape."""
        env = JammerEnv(M=4)
        env.reset()
        
        global_obs = env.get_global_observation()
        
        # Should be mean-pooled to (5,)
        assert global_obs.shape == (5,)
    
    def test_global_observation_normalized(self):
        """Test global observation is normalized."""
        env = JammerEnv(M=4)
        env.reset()
        
        global_obs = env.get_global_observation()
        
        assert np.all(global_obs >= 0.0)
        assert np.all(global_obs <= 1.0)


class TestArrayStyleAction:
    """Tests for array-style action (M, 3) format."""
    
    def test_array_action_accepted(self):
        """Test environment accepts (M, 3) array action."""
        env = JammerEnv(M=4)
        env.reset()
        
        # Action as (M, 3) array: [vx, vy, band]
        action = np.zeros((4, 3), dtype=np.float32)
        action[:, 2] = [0, 1, 2, 3]  # Bands
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (4, 5)
        assert np.array_equal(env.jammer_bands, [0, 1, 2, 3])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
