"""
Metrics Logger Module
=====================

Logging and metrics tracking for training.

Features:
    - Rolling statistics
    - CSV logging
    - TensorBoard support (optional)
    - Training curves

Author: MARL Jammer Team
"""

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import deque
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RollingStats:
    """Rolling statistics tracker."""
    
    window_size: int = 100
    _values: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def __post_init__(self):
        self._values = deque(maxlen=self.window_size)
    
    def add(self, value: float):
        """Add a value."""
        self._values.append(value)
    
    @property
    def mean(self) -> float:
        """Get rolling mean."""
        if len(self._values) == 0:
            return 0.0
        return float(np.mean(self._values))
    
    @property
    def std(self) -> float:
        """Get rolling std."""
        if len(self._values) < 2:
            return 0.0
        return float(np.std(self._values))
    
    @property
    def min(self) -> float:
        """Get rolling min."""
        if len(self._values) == 0:
            return 0.0
        return float(np.min(self._values))
    
    @property
    def max(self) -> float:
        """Get rolling max."""
        if len(self._values) == 0:
            return 0.0
        return float(np.max(self._values))
    
    def __len__(self) -> int:
        return len(self._values)


class MetricsLogger:
    """
    Comprehensive metrics logging for training.
    
    Tracks:
        - Episode rewards and returns
        - Lambda-2 reduction (primary metric)
        - Training losses (actor, critic)
        - Policy entropy
        - Band match rate
        - Episode lengths
    
    Example:
        >>> logger = MetricsLogger("outputs/run1")
        >>> logger.log_episode(reward=10.5, lambda2_reduction=0.75, length=150)
        >>> logger.log_update(actor_loss=0.01, critic_loss=0.5, entropy=1.2)
        >>> logger.save()
    """
    
    def __init__(
        self,
        output_dir: str,
        experiment_name: str = "experiment",
        window_size: int = 100
    ):
        """
        Initialize logger.
        
        Args:
            output_dir: Directory for log files
            experiment_name: Name of experiment
            window_size: Size of rolling statistics window
        """
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.window_size = window_size
        
        # Rolling statistics
        self.episode_rewards = RollingStats(window_size)
        self.episode_lengths = RollingStats(window_size)
        self.lambda2_reductions = RollingStats(window_size)
        self.band_match_rates = RollingStats(window_size)
        self.avg_jamming_powers = RollingStats(window_size)
        
        self.actor_losses = RollingStats(window_size)
        self.critic_losses = RollingStats(window_size)
        self.entropies = RollingStats(window_size)
        self.clip_fractions = RollingStats(window_size)
        self.approx_kls = RollingStats(window_size)
        
        # Full history for plotting
        self.history: Dict[str, List[float]] = {
            "timestep": [],
            "episode": [],
            "reward": [],
            "lambda2_reduction": [],
            "episode_length": [],
            "band_match_rate": [],
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "clip_fraction": [],
            "approx_kl": [],
            "time_elapsed": [],
            "avg_jamming_power_dbm": []
        }
        
        # Counters
        self.total_timesteps = 0
        self.total_episodes = 0
        self.total_updates = 0
        self.start_time = time.time()
        
        # CSV writer
        self._csv_file = None
        self._csv_writer = None
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV logging."""
        csv_path = self.output_dir / "training_log.csv"
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=list(self.history.keys())
        )
        self._csv_writer.writeheader()
    
    def log_episode(
        self,
        reward: float,
        lambda2_reduction: float,
        length: int,
        band_match_rate: float = 0.0,
        timesteps: int = 0
    ):
        """
        Log episode completion.
        
        Args:
            reward: Total episode reward
            lambda2_reduction: Final λ₂ reduction percentage
            length: Episode length (steps)
            band_match_rate: Fraction of correct band matches
            timesteps: Number of timesteps this episode
        """
        self.episode_rewards.add(reward)
        self.lambda2_reductions.add(lambda2_reduction)
        self.episode_lengths.add(length)
        self.band_match_rates.add(band_match_rate)
        
        self.total_episodes += 1
        self.total_timesteps += timesteps
    
    def log_update(
        self,
        actor_loss: float,
        critic_loss: float,
        entropy: float,
        clip_fraction: float = 0.0,
        approx_kl: float = 0.0
    ):
        """
        Log PPO update.
        
        Args:
            actor_loss: Actor policy loss
            critic_loss: Critic value loss
            entropy: Policy entropy
            clip_fraction: Fraction of clipped ratios
            approx_kl: Approximate KL divergence
        """
        self.actor_losses.add(actor_loss)
        self.critic_losses.add(critic_loss)
        self.entropies.add(entropy)
        self.clip_fractions.add(clip_fraction)
        self.approx_kls.add(approx_kl)
        
        self.total_updates += 1
    
    def log_rollout(
        self,
        timesteps: int,
        reward: float = None,
        lambda2_reduction: float = None,
        episode_length: float = None,
        band_match_rate: float = None,
        actor_loss: float = None,
        critic_loss: float = None,
        entropy: float = None,
        clip_fraction: float = None,
        approx_kl: float = None,
        avg_jamming_power_dbm: float = None
    ):
        """
        Log complete rollout with all metrics.
        
        Args:
            timesteps: Total timesteps so far
            **kwargs: Metric values
        """
        self.total_timesteps = timesteps
        time_elapsed = time.time() - self.start_time
        
        # Track avg jamming power
        if avg_jamming_power_dbm is not None:
            self.avg_jamming_powers.add(avg_jamming_power_dbm)
        
        # Build record
        record = {
            "timestep": timesteps,
            "episode": self.total_episodes,
            "reward": reward if reward is not None else self.episode_rewards.mean,
            "lambda2_reduction": lambda2_reduction if lambda2_reduction is not None else self.lambda2_reductions.mean,
            "episode_length": episode_length if episode_length is not None else self.episode_lengths.mean,
            "band_match_rate": band_match_rate if band_match_rate is not None else self.band_match_rates.mean,
            "actor_loss": actor_loss if actor_loss is not None else self.actor_losses.mean,
            "critic_loss": critic_loss if critic_loss is not None else self.critic_losses.mean,
            "entropy": entropy if entropy is not None else self.entropies.mean,
            "clip_fraction": clip_fraction if clip_fraction is not None else self.clip_fractions.mean,
            "approx_kl": approx_kl if approx_kl is not None else self.approx_kls.mean,
            "time_elapsed": time_elapsed,
            "avg_jamming_power_dbm": avg_jamming_power_dbm if avg_jamming_power_dbm is not None else self.avg_jamming_powers.mean
        }
        
        # Add to history
        for key, value in record.items():
            self.history[key].append(value)
        
        # Write to CSV
        if self._csv_writer:
            self._csv_writer.writerow(record)
            self._csv_file.flush()
    
    def get_stats(self) -> Dict[str, float]:
        """Get current rolling statistics."""
        return {
            "reward_mean": self.episode_rewards.mean,
            "reward_std": self.episode_rewards.std,
            "lambda2_reduction_mean": self.lambda2_reductions.mean,
            "lambda2_reduction_std": self.lambda2_reductions.std,
            "episode_length_mean": self.episode_lengths.mean,
            "band_match_rate_mean": self.band_match_rates.mean,
            "avg_jamming_power_dbm_mean": self.avg_jamming_powers.mean,
            "actor_loss_mean": self.actor_losses.mean,
            "critic_loss_mean": self.critic_losses.mean,
            "entropy_mean": self.entropies.mean,
            "clip_fraction_mean": self.clip_fractions.mean,
            "approx_kl_mean": self.approx_kls.mean,
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "total_updates": self.total_updates,
            "time_elapsed": time.time() - self.start_time
        }
    
    def print_stats(self, prefix: str = ""):
        """Print current statistics."""
        stats = self.get_stats()
        fps = self.total_timesteps / max(stats["time_elapsed"], 1e-6)
        
        print(f"{prefix}Steps: {self.total_timesteps:,} | "
              f"Episodes: {self.total_episodes} | "
              f"Reward: {stats['reward_mean']:.2f}+/-{stats['reward_std']:.2f} | "
              f"L2_red: {stats['lambda2_reduction_mean']:.1f}% | "
              f"Entropy: {stats['entropy_mean']:.3f} | "
              f"FPS: {fps:.0f}")
    
    def save(self):
        """Save all logs and history."""
        # Save history as JSON
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        # Save final stats
        with open(self.output_dir / "final_stats.json", "w") as f:
            json.dump(self.get_stats(), f, indent=2)
    
    def close(self):
        """Close file handles."""
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
    
    def __del__(self):
        self.close()


class EvaluationResult:
    """Container for evaluation results."""
    
    def __init__(self):
        self.rewards: List[float] = []
        self.lambda2_reductions: List[float] = []
        self.episode_lengths: List[int] = []
        self.band_match_rates: List[float] = []
        self.fragmentation_times: List[Optional[int]] = []
    
    def add_episode(
        self,
        reward: float,
        lambda2_reduction: float,
        length: int,
        band_match_rate: float = 0.0,
        fragmentation_time: Optional[int] = None
    ):
        """Add episode result."""
        self.rewards.append(reward)
        self.lambda2_reductions.append(lambda2_reduction)
        self.episode_lengths.append(length)
        self.band_match_rates.append(band_match_rate)
        self.fragmentation_times.append(fragmentation_time)
    
    def summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "reward_mean": float(np.mean(self.rewards)),
            "reward_std": float(np.std(self.rewards)),
            "lambda2_reduction_mean": float(np.mean(self.lambda2_reductions)),
            "lambda2_reduction_std": float(np.std(self.lambda2_reductions)),
            "episode_length_mean": float(np.mean(self.episode_lengths)),
            "band_match_rate_mean": float(np.mean(self.band_match_rates)),
            "fragmentation_rate": sum(1 for t in self.fragmentation_times if t is not None) / len(self.fragmentation_times) if self.fragmentation_times else 0.0
        }


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_metrics_logger() -> dict:
    """Verify metrics logger."""
    import tempfile
    results = {}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Creation
        logger = MetricsLogger(tmpdir, "test_exp")
        results["test_creation"] = {
            "output_dir_exists": logger.output_dir.exists(),
            "pass": logger.output_dir.exists()
        }
        
        # Test 2: Log episodes
        for i in range(10):
            logger.log_episode(
                reward=float(i),
                lambda2_reduction=50.0 + i * 2,
                length=100 + i * 10,
                timesteps=100
            )
        
        results["test_log_episodes"] = {
            "total_episodes": logger.total_episodes,
            "reward_mean": logger.episode_rewards.mean,
            "pass": logger.total_episodes == 10
        }
        
        # Test 3: Log updates
        for i in range(5):
            logger.log_update(
                actor_loss=0.1 - i * 0.01,
                critic_loss=0.5 - i * 0.05,
                entropy=1.0 - i * 0.1
            )
        
        results["test_log_updates"] = {
            "total_updates": logger.total_updates,
            "actor_loss_mean": logger.actor_losses.mean,
            "pass": logger.total_updates == 5
        }
        
        # Test 4: Save
        logger.save()
        history_path = logger.output_dir / "history.json"
        results["test_save"] = {
            "history_exists": history_path.exists(),
            "pass": history_path.exists()
        }
        
        logger.close()
    
    # Test 5: Rolling stats
    stats = RollingStats(window_size=5)
    for i in range(10):
        stats.add(float(i))
    
    results["test_rolling_stats"] = {
        "length": len(stats),
        "mean": stats.mean,  # Should be mean of [5,6,7,8,9] = 7.0
        "pass": len(stats) == 5 and abs(stats.mean - 7.0) < 0.01
    }
    
    # Test 6: Evaluation result
    eval_result = EvaluationResult()
    for i in range(5):
        eval_result.add_episode(
            reward=10.0 + i,
            lambda2_reduction=70.0 + i,
            length=150 + i * 10
        )
    
    summary = eval_result.summary()
    results["test_evaluation_result"] = {
        "reward_mean": summary["reward_mean"],
        "pass": abs(summary["reward_mean"] - 12.0) < 0.01
    }
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Metrics Logger Verification")
    print("=" * 60)
    
    results = verify_metrics_logger()
    
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
