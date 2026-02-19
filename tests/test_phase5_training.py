"""
Phase 5 Tests: Training Infrastructure
=======================================

Tests for configuration, metrics logging, and trainer.

Test Categories:
    - Configuration: Creation, serialization, presets
    - Metrics Logger: Logging, rolling stats, CSV output
    - Trainer: Initialization, rollouts, updates

Author: MARL Jammer Team
"""

import pytest
import numpy as np
import sys
import os
import json
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestTrainingConfig:
    """Test TrainingConfig class."""
    
    def test_default_creation(self):
        """Test creating config with defaults."""
        from training.config import TrainingConfig
        
        config = TrainingConfig()
        
        assert config.env.N == 10
        assert config.env.M == 4
        assert config.ppo.gamma == 0.99
        assert config.ppo.clip_eps == 0.2
        assert config.total_timesteps == 2_000_000
    
    def test_sub_configs(self):
        """Test sub-configuration objects."""
        from training.config import TrainingConfig
        
        config = TrainingConfig()
        
        # Environment config
        assert hasattr(config, 'env')
        assert config.env.arena_size == 100.0
        assert config.env.v_max == 5.0
        
        # Network config
        assert hasattr(config, 'network')
        assert config.network.obs_dim == 5
        assert config.network.hidden_dim == 128
        
        # PPO config
        assert hasattr(config, 'ppo')
        assert config.ppo.n_epochs == 10
        assert config.ppo.batch_size == 256
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        from training.config import TrainingConfig
        
        config = TrainingConfig()
        d = config.to_dict()
        
        assert isinstance(d, dict)
        assert "env" in d
        assert "network" in d
        assert "ppo" in d
        assert d["env"]["N"] == 10
        assert d["ppo"]["gamma"] == 0.99
    
    def test_save_and_load(self):
        """Test saving and loading config."""
        from training.config import TrainingConfig
        
        config = TrainingConfig()
        config.env.N = 20
        config.ppo.gamma = 0.95
        config.experiment_name = "test_experiment"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save(f.name)
            
            loaded = TrainingConfig.load(f.name)
            
            assert loaded.env.N == 20
            assert loaded.ppo.gamma == 0.95
            assert loaded.experiment_name == "test_experiment"
    
    def test_reward_weights(self):
        """Test reward weights configuration."""
        from training.config import TrainingConfig
        
        config = TrainingConfig()
        
        weights = config.env.reward_weights
        assert "lambda2_reduction" in weights
        assert "band_match" in weights
        assert "overlap" in weights
        assert weights["lambda2_reduction"] == 1.0


class TestConfigPresets:
    """Test configuration presets."""
    
    def test_debug_config(self):
        """Test debug configuration."""
        from training.config import get_debug_config
        
        config = get_debug_config()
        
        assert config.env.N == 5
        assert config.env.M == 2
        assert config.total_timesteps == 1000
        assert config.ppo.rollout_length == 128
    
    def test_fast_config(self):
        """Test fast configuration."""
        from training.config import get_fast_config
        
        config = get_fast_config()
        
        assert config.total_timesteps == 100_000
        assert config.ppo.rollout_length == 512
    
    def test_full_config(self):
        """Test full configuration."""
        from training.config import get_full_config
        
        config = get_full_config()
        
        assert config.total_timesteps == 2_000_000
        assert config.ppo.rollout_length == 2048
    
    def test_large_scale_config(self):
        """Test large scale configuration."""
        from training.config import get_large_scale_config
        
        config = get_large_scale_config()
        
        assert config.env.N == 100
        assert config.env.M == 40
        assert config.network.hidden_dim == 256


# =============================================================================
# METRICS TESTS
# =============================================================================

class TestRollingStats:
    """Test RollingStats class."""
    
    def test_creation(self):
        """Test creation with window size."""
        from training.metrics import RollingStats
        
        stats = RollingStats(window_size=10)
        
        assert len(stats) == 0
        assert stats.mean == 0.0
    
    def test_add_values(self):
        """Test adding values."""
        from training.metrics import RollingStats
        
        stats = RollingStats(window_size=5)
        
        for i in range(5):
            stats.add(float(i))
        
        assert len(stats) == 5
        assert stats.mean == 2.0  # mean of [0,1,2,3,4]
    
    def test_window_overflow(self):
        """Test window overflow behavior."""
        from training.metrics import RollingStats
        
        stats = RollingStats(window_size=5)
        
        for i in range(10):
            stats.add(float(i))
        
        assert len(stats) == 5
        # Should only have [5,6,7,8,9]
        assert abs(stats.mean - 7.0) < 0.01
    
    def test_statistics(self):
        """Test statistical properties."""
        from training.metrics import RollingStats
        
        stats = RollingStats(window_size=100)
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            stats.add(v)
        
        assert abs(stats.mean - 3.0) < 0.01
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.std > 0


class TestMetricsLogger:
    """Test MetricsLogger class."""
    
    def test_creation(self):
        """Test logger creation."""
        from training.metrics import MetricsLogger
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir, "test_exp")
            
            assert logger.output_dir.exists()
            assert logger.total_timesteps == 0
            assert logger.total_episodes == 0
            
            logger.close()
    
    def test_log_episode(self):
        """Test logging episodes."""
        from training.metrics import MetricsLogger
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir, "test_exp")
            
            logger.log_episode(
                reward=10.5,
                lambda2_reduction=75.0,
                length=150,
                timesteps=150
            )
            
            assert logger.total_episodes == 1
            assert logger.episode_rewards.mean == 10.5
            assert logger.lambda2_reductions.mean == 75.0
            
            logger.close()
    
    def test_log_update(self):
        """Test logging updates."""
        from training.metrics import MetricsLogger
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir, "test_exp")
            
            logger.log_update(
                actor_loss=0.05,
                critic_loss=0.3,
                entropy=1.5,
                clip_fraction=0.1,
                approx_kl=0.01
            )
            
            assert logger.total_updates == 1
            assert logger.actor_losses.mean == 0.05
            assert logger.entropies.mean == 1.5
            
            logger.close()
    
    def test_log_rollout(self):
        """Test logging complete rollout."""
        from training.metrics import MetricsLogger
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir, "test_exp")
            
            logger.log_rollout(
                timesteps=2048,
                reward=15.0,
                lambda2_reduction=80.0,
                actor_loss=0.02,
                critic_loss=0.25,
                entropy=1.2
            )
            
            assert len(logger.history["timestep"]) == 1
            assert logger.history["timestep"][0] == 2048
            
            logger.close()
    
    def test_get_stats(self):
        """Test getting statistics."""
        from training.metrics import MetricsLogger
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir, "test_exp")
            
            for i in range(5):
                logger.log_episode(
                    reward=10.0 + i,
                    lambda2_reduction=70.0 + i,
                    length=100 + i,
                    timesteps=100
                )
            
            stats = logger.get_stats()
            
            assert "reward_mean" in stats
            assert "lambda2_reduction_mean" in stats
            assert abs(stats["reward_mean"] - 12.0) < 0.01  # mean of [10,11,12,13,14]
            
            logger.close()
    
    def test_csv_logging(self):
        """Test CSV file creation."""
        from training.metrics import MetricsLogger
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir, "test_exp")
            
            logger.log_rollout(timesteps=100, reward=10.0)
            logger.log_rollout(timesteps=200, reward=15.0)
            
            logger.close()
            
            csv_path = logger.output_dir / "training_log.csv"
            assert csv_path.exists()
            
            # Read and verify
            import csv
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 2
            assert rows[0]["timestep"] == "100"
    
    def test_save_history(self):
        """Test saving history."""
        from training.metrics import MetricsLogger
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir, "test_exp")
            
            logger.log_rollout(timesteps=100, reward=10.0)
            logger.save()
            
            history_path = logger.output_dir / "history.json"
            assert history_path.exists()
            
            with open(history_path) as f:
                history = json.load(f)
            
            assert "timestep" in history
            assert history["timestep"][0] == 100
            
            logger.close()


class TestEvaluationResult:
    """Test EvaluationResult class."""
    
    def test_creation(self):
        """Test creation."""
        from training.metrics import EvaluationResult
        
        result = EvaluationResult()
        
        assert len(result.rewards) == 0
        assert len(result.lambda2_reductions) == 0
    
    def test_add_episodes(self):
        """Test adding episodes."""
        from training.metrics import EvaluationResult
        
        result = EvaluationResult()
        
        for i in range(5):
            result.add_episode(
                reward=10.0 + i,
                lambda2_reduction=70.0 + i,
                length=100 + i * 10
            )
        
        assert len(result.rewards) == 5
        assert result.rewards[0] == 10.0
    
    def test_summary(self):
        """Test summary statistics."""
        from training.metrics import EvaluationResult
        
        result = EvaluationResult()
        
        for i in range(10):
            result.add_episode(
                reward=float(i),
                lambda2_reduction=50.0 + float(i),
                length=100
            )
        
        summary = result.summary()
        
        assert abs(summary["reward_mean"] - 4.5) < 0.01
        assert summary["reward_std"] > 0
        assert summary["episode_length_mean"] == 100


# =============================================================================
# TRAINER TESTS (Mocked where needed)
# =============================================================================

class TestTrainerBasics:
    """Test Trainer basic functionality."""
    
    def test_config_verification(self):
        """Test config verification function."""
        from training.config import verify_config
        
        results = verify_config()
        
        for test_name, result in results.items():
            assert result.get("pass", False), f"{test_name} failed"
    
    def test_metrics_verification(self):
        """Test metrics verification function."""
        from training.metrics import verify_metrics_logger
        
        results = verify_metrics_logger()
        
        for test_name, result in results.items():
            assert result.get("pass", False), f"{test_name} failed"


# Skip integration tests if PyTorch not available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestTrainerIntegration:
    """Integration tests for trainer (require full system)."""
    
    def test_trainer_creation(self):
        """Test creating trainer with debug config."""
        from training.config import get_debug_config
        from training.trainer import Trainer
        
        config = get_debug_config()
        config.total_timesteps = 100  # Very short run
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = tmpdir
            config.experiment_name = "test_create"
            
            trainer = Trainer(config)
            
            assert trainer.config is not None
            assert trainer.agent is not None
            assert trainer.env is not None
            assert trainer.total_timesteps == 0
            
            trainer.logger.close()
    
    def test_collect_single_rollout(self):
        """Test collecting a single rollout."""
        from training.config import get_debug_config
        from training.trainer import Trainer
        
        config = get_debug_config()
        config.ppo.rollout_length = 32
        config.total_timesteps = 100
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = tmpdir
            config.experiment_name = "test_rollout"
            
            trainer = Trainer(config)
            
            # Initialize episode
            trainer._obs, info = trainer.env.reset()
            trainer._done = False
            trainer._episode_reward = 0.0
            trainer._episode_length = 0
            trainer._episode_lambda2_start = info.get('lambda2', 1.0)
            
            # Collect rollout
            stats = trainer.collect_rollout()
            
            assert stats["steps"] == 32
            assert trainer.total_timesteps == 32
            
            trainer.logger.close()
    
    def test_ppo_update(self):
        """Test PPO update after rollout."""
        from training.config import get_debug_config
        from training.trainer import Trainer
        
        config = get_debug_config()
        config.ppo.rollout_length = 32
        config.ppo.batch_size = 8
        config.ppo.n_epochs = 2
        config.total_timesteps = 100
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = tmpdir
            config.experiment_name = "test_update"
            
            trainer = Trainer(config)
            
            # Initialize
            trainer._obs, info = trainer.env.reset()
            trainer._done = False
            trainer._episode_reward = 0.0
            trainer._episode_length = 0
            trainer._episode_lambda2_start = info.get('lambda2', 1.0)
            
            # Collect rollout
            trainer.collect_rollout()
            
            # Update
            update_stats = trainer.update()
            
            assert "actor_loss" in update_stats
            assert "critic_loss" in update_stats
            assert "entropy" in update_stats
            assert np.isfinite(update_stats["actor_loss"])
            
            trainer.logger.close()
    
    def test_evaluation(self):
        """Test policy evaluation."""
        from training.config import get_debug_config
        from training.trainer import Trainer
        
        config = get_debug_config()
        config.env.max_steps = 20
        config.total_timesteps = 100
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = tmpdir
            config.experiment_name = "test_eval"
            
            trainer = Trainer(config)
            
            result = trainer.evaluate(n_episodes=2)
            
            assert len(result.rewards) == 2
            summary = result.summary()
            assert "reward_mean" in summary
            assert "lambda2_reduction_mean" in summary
            
            trainer.logger.close()
    
    def test_save_and_load_checkpoint(self):
        """Test checkpoint saving and loading."""
        from training.config import get_debug_config
        from training.trainer import Trainer
        
        config = get_debug_config()
        config.total_timesteps = 100
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = tmpdir
            config.experiment_name = "test_checkpoint"
            
            trainer = Trainer(config)
            trainer.total_timesteps = 500
            trainer.total_episodes = 10
            
            # Save
            trainer.save_checkpoint("test_ckpt")
            
            ckpt_path = Path(tmpdir) / "test_checkpoint" / "checkpoints" / "test_ckpt"
            assert (ckpt_path / "ppo_agent.pt").exists()
            assert (ckpt_path / "trainer_state.json").exists()
            
            # Create new trainer and load
            trainer2 = Trainer(config)
            trainer2.load_checkpoint(ckpt_path)
            
            assert trainer2.total_timesteps == 500
            assert trainer2.total_episodes == 10
            
            trainer.logger.close()
            trainer2.logger.close()
    
    def test_mini_training_loop(self):
        """Test a minimal training loop."""
        from training.config import get_debug_config
        from training.trainer import Trainer
        
        config = get_debug_config()
        config.ppo.rollout_length = 32
        config.ppo.batch_size = 8
        config.ppo.n_epochs = 1
        config.total_timesteps = 64  # 2 rollouts
        config.log_interval = 1
        config.save_interval = 10
        config.eval_interval = 10
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = tmpdir
            config.experiment_name = "test_loop"
            
            trainer = Trainer(config)
            
            # Run training
            results = trainer.train()
            
            assert results["total_timesteps"] >= 64
            assert results["total_rollouts"] >= 2
            assert "best_lambda2_reduction" in results
            
            # Check outputs
            output_dir = Path(tmpdir) / "test_loop"
            assert (output_dir / "config.json").exists()
            assert (output_dir / "history.json").exists()


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
