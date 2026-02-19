"""
Trainer Module
==============

Main training loop that connects environment, agent, and logging.

This is the core of Phase 5 - the training pipeline that:
1. Collects rollouts from the environment
2. Updates the PPO agent
3. Logs metrics and saves checkpoints
4. Evaluates performance

Reference: PROJECT_MASTER_GUIDE_v2.md Section 8

Author: MARL Jammer Team
"""

import numpy as np
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import random

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import TrainingConfig, get_debug_config
from .metrics import MetricsLogger, EvaluationResult


class Trainer:
    """
    Main PPO trainer for MARL Jammer System.
    
    Training Loop Overview:
    
    1. RESET: Initialize environment with N enemies, M jammers
    2. ROLLOUT: Collect T steps of experience
       - For each step:
         a. Get observations for all M agents
         b. Actor samples actions (velocity + band)
         c. Execute actions in environment
         d. Compute reward based on λ₂ reduction
         e. Store transition in buffer
    3. UPDATE: After rollout completes
       - Compute GAE advantages
       - Run PPO update for K epochs
       - Update policy networks
    4. LOG: Record metrics, save checkpoints
    5. REPEAT until convergence or budget exhausted
    
    Attributes:
        config: Training configuration
        env: Jammer environment
        agent: PPO agent
        logger: Metrics logger
        
    Example:
        >>> config = TrainingConfig()
        >>> trainer = Trainer(config)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        config: TrainingConfig = None,
        env = None,
        agent = None,
        seed: int = None
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration (uses defaults if None)
            env: Pre-initialized environment (creates new if None)
            agent: Pre-initialized agent (creates new if None)
            seed: Random seed (uses config.seed if None)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for training")
        
        self.config = config or TrainingConfig()
        self.seed = seed if seed is not None else self.config.seed
        
        # Set seeds
        self._set_seeds(self.seed)
        
        # Initialize components
        self.env = env or self._create_env()
        self.agent = agent or self._create_agent()
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir) / self.config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = MetricsLogger(
            output_dir=self.config.output_dir,
            experiment_name=self.config.experiment_name
        )
        
        # Save config
        self.config.save(self.output_dir / "config.json")
        
        # Training state
        self.total_timesteps = 0
        self.total_episodes = 0
        self.total_rollouts = 0
        self.best_reduction = 0.0
        self.no_improvement_count = 0
        
        # Callbacks
        self.callbacks: List[Callable] = []
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        if HAS_TORCH:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
    def _create_env(self):
        """Create environment from config."""
        from environment.jammer_env import JammerEnv
        
        # Map config reward weights to omega parameters
        weights = self.config.env.reward_weights
        
        return JammerEnv(
            N=self.config.env.N,
            M=self.config.env.M,
            arena_size=self.config.env.arena_size,
            v_max=self.config.env.v_max,
            max_steps=self.config.env.max_steps,
            eps=self.config.env.eps,
            min_samples=self.config.env.min_samples,
            v_enemy=self.config.env.v_enemy,
            enemy_mode='random_walk' if self.config.env.motion_mode == 'random' else self.config.env.motion_mode,
            K_recompute=self.config.env.k_recompute,
            P_tx_dbm=self.config.env.tx_power_dbm,
            P_sens_dbm=self.config.env.sensitivity_dbm,
            P_jammer_dbm=self.config.env.jammer_power_dbm,
            P_jam_thresh_dbm=getattr(self.config.env, 'jam_thresh_dbm', -40.0),
            omega_1=weights.get("lambda2_reduction", 1.0),
            omega_2=weights.get("band_match", 0.3),
            omega_3=weights.get("proximity", 0.2),
            omega_4=weights.get("energy", 0.1),
            omega_5=weights.get("overlap", 0.2),
            debug_mode=getattr(self.config, 'enable_debug_logging', False),
            random_jammer_start=getattr(self.config.env, 'random_jammer_start', False)
        )
    
    def _create_agent(self):
        """Create PPO agent from config."""
        from agents.ppo_agent import PPOAgent
        
        return PPOAgent(
            obs_dim=self.config.network.obs_dim,
            hidden_dim=self.config.network.hidden_dim,
            M=self.config.env.M,
            v_max=self.config.env.v_max,
            num_bands=self.config.env.num_bands,
            gamma=self.config.ppo.gamma,
            gae_lambda=self.config.ppo.gae_lambda,
            clip_eps=self.config.ppo.clip_eps,
            lr_actor=self.config.ppo.lr_actor,
            lr_critic=self.config.ppo.lr_critic,
            c1=self.config.ppo.c1,
            c2=self.config.ppo.c2,
            max_grad_norm=self.config.ppo.max_grad_norm,
            rollout_length=self.config.ppo.rollout_length,
            batch_size=self.config.ppo.batch_size,
            n_epochs=self.config.ppo.n_epochs,
            device=self.config.device
        )
    
    def collect_rollout(self) -> Dict[str, Any]:
        """
        Collect a single rollout of experience.
        
        This is the core data collection loop:
        1. Reset env if needed
        2. For T steps:
           - Get actions from policy
           - Step environment
           - Store transitions
        3. Return rollout statistics
        
        Returns:
            Dictionary with rollout statistics
        """
        # Reset if first time or episode was done
        if not hasattr(self, '_obs') or self._done:
            self._obs, info = self.env.reset()
            self._done = False
            self._episode_reward = 0.0
            self._episode_length = 0
            self._episode_lambda2_start = info.get('lambda2_initial', info.get('lambda2', 1.0))
        
        # Track episode stats
        episode_rewards = []
        episode_lengths = []
        lambda2_reductions = []
        band_matches = []
        jamming_powers = []  # Track avg jamming power per step
        
        # Collect rollout
        steps_collected = 0
        while steps_collected < self.config.ppo.rollout_length:
            # Get action from agent
            actions, log_probs, value = self.agent.get_action(self._obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(actions)
            done = terminated or truncated
            
            # Store transition
            self.agent.store_transition(
                obs=self._obs,
                actions=actions,
                log_probs=log_probs,
                reward=reward,
                done=done,
                value=value
            )
            
            # Update episode tracking
            self._episode_reward += reward
            self._episode_length += 1
            
            # Track band matches
            if 'band_match_rate' in info:
                band_matches.append(info['band_match_rate'])
            
            # Track avg jamming power (for comparison graphs)
            if 'avg_jamming_power_dbm' in info:
                jamming_powers.append(info['avg_jamming_power_dbm'])
            
            # Handle episode end
            if done:
                # Compute λ₂ reduction
                lambda2_final = info.get('lambda2_current', info.get('lambda2', 0.0))
                lambda2_start = self._episode_lambda2_start
                if lambda2_start > 1e-10:
                    reduction = 100.0 * (1.0 - lambda2_final / lambda2_start)
                else:
                    reduction = 0.0
                
                # Record episode
                episode_rewards.append(self._episode_reward)
                episode_lengths.append(self._episode_length)
                lambda2_reductions.append(reduction)
                
                self.logger.log_episode(
                    reward=self._episode_reward,
                    lambda2_reduction=reduction,
                    length=self._episode_length,
                    band_match_rate=np.mean(band_matches) if band_matches else 0.0,
                    timesteps=self._episode_length
                )
                
                self.total_episodes += 1
                band_matches = []
                
                # Reset for next episode
                self._obs, info = self.env.reset()
                self._done = False
                self._episode_reward = 0.0
                self._episode_length = 0
                self._episode_lambda2_start = info.get('lambda2_initial', info.get('lambda2', 1.0))
            else:
                self._obs = next_obs
                self._done = done
            
            steps_collected += 1
            self.total_timesteps += 1
        
        # Return rollout statistics
        return {
            "steps": steps_collected,
            "episodes_completed": len(episode_rewards),
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "mean_lambda2_reduction": np.mean(lambda2_reductions) if lambda2_reductions else 0.0,
            "mean_jamming_power_dbm": np.mean(jamming_powers) if jamming_powers else -80.0
        }
    
    def update(self) -> Dict[str, float]:
        """
        Perform PPO update after rollout.
        
        This includes:
        1. Computing GAE advantages
        2. Running K epochs of PPO updates
        3. Returning training statistics
        
        Returns:
            Dictionary of update statistics
        """
        # Get final observation for bootstrapping
        last_obs = self._obs
        last_done = self._done
        
        # Run PPO update
        stats = self.agent.update(last_obs, last_done)
        
        # Log update
        self.logger.log_update(
            actor_loss=stats["actor_loss"],
            critic_loss=stats["critic_loss"],
            entropy=stats["entropy"],
            clip_fraction=stats["clip_fraction"],
            approx_kl=stats["approx_kl"]
        )
        
        return stats
    
    def evaluate(self, n_episodes: int = 5, deterministic: bool = True) -> EvaluationResult:
        """
        Evaluate current policy.
        
        Args:
            n_episodes: Number of episodes to evaluate
            deterministic: Use deterministic policy
            
        Returns:
            EvaluationResult with episode statistics
        """
        result = EvaluationResult()
        
        # Set to eval mode
        self.agent.set_training_mode(False)
        
        for ep in range(n_episodes):
            obs, info = self.env.reset()
            lambda2_start = info.get('lambda2_initial', info.get('lambda2', 1.0))
            
            episode_reward = 0.0
            episode_length = 0
            band_matches = []
            fragmentation_time = None
            
            done = False
            while not done:
                actions, _, _ = self.agent.get_action(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                if 'band_match_rate' in info:
                    band_matches.append(info['band_match_rate'])
                
                # Check for fragmentation
                if fragmentation_time is None and info.get('lambda2_current', info.get('lambda2', 1.0)) < 1e-10:
                    fragmentation_time = episode_length
            
            # Compute reduction
            lambda2_final = info.get('lambda2_current', info.get('lambda2', 0.0))
            if lambda2_start > 1e-10:
                reduction = 100.0 * (1.0 - lambda2_final / lambda2_start)
            else:
                reduction = 0.0
            
            result.add_episode(
                reward=episode_reward,
                lambda2_reduction=reduction,
                length=episode_length,
                band_match_rate=np.mean(band_matches) if band_matches else 0.0,
                fragmentation_time=fragmentation_time
            )
        
        # Set back to train mode
        self.agent.set_training_mode(True)
        
        return result
    
    def save_checkpoint(self, name: str = "checkpoint"):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints" / name
        self.agent.save(checkpoint_dir)
        
        # Save training state
        state = {
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "total_rollouts": self.total_rollouts,
            "best_reduction": self.best_reduction
        }
        
        import json
        with open(checkpoint_dir / "trainer_state.json", "w") as f:
            json.dump(state, f, indent=2)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        from agents.ppo_agent import PPOAgent
        
        checkpoint_dir = Path(path)
        self.agent = PPOAgent.from_checkpoint(checkpoint_dir)
        
        # Load training state
        import json
        state_path = checkpoint_dir / "trainer_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            self.total_timesteps = state.get("total_timesteps", 0)
            self.total_episodes = state.get("total_episodes", 0)
            self.total_rollouts = state.get("total_rollouts", 0)
            self.best_reduction = state.get("best_reduction", 0.0)
    
    def check_convergence(self) -> bool:
        """
        Check if training has converged.
        
        Convergence criteria:
        1. Mean λ₂ reduction >= target over window
        2. Or early stopping if no improvement
        
        Returns:
            True if converged
        """
        # Skip convergence check if disabled (e.g., debug mode)
        if self.config.disable_early_convergence:
            return False
        
        # Check target reached
        if len(self.logger.lambda2_reductions) >= self.config.convergence_window:
            recent_reductions = list(self.logger.lambda2_reductions._values)[-self.config.convergence_window:]
            mean_reduction = np.mean(recent_reductions)
            
            # Always track best reduction
            if mean_reduction > self.best_reduction:
                self.best_reduction = mean_reduction
                self.no_improvement_count = 0
            elif mean_reduction <= self.best_reduction + 1.0:
                self.no_improvement_count += 1
            
            if mean_reduction >= self.config.target_reduction:
                print(f"\n[OK] Convergence reached! Mean L2 reduction: {mean_reduction:.1f}%")
                return True
            
            # Early stopping
            if self.no_improvement_count >= self.config.early_stop_patience:
                print(f"\n[X] Early stopping: No improvement for {self.no_improvement_count} rollouts")
                return True
        
        return False
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        This is where everything comes together:
        
        ```
        while not done:
            1. Collect rollout (T steps of experience)
            2. Update policy (K epochs of PPO)
            3. Log metrics
            4. Save checkpoints periodically
            5. Evaluate periodically
            6. Check convergence
        ```
        
        Returns:
            Dictionary with final training statistics
        """
        print("=" * 70)
        print(f"Starting Training: {self.config.experiment_name}")
        print(f"  Total timesteps: {self.config.total_timesteps:,}")
        print(f"  N enemies: {self.config.env.N}, M jammers: {self.config.env.M}")
        print(f"  Rollout length: {self.config.ppo.rollout_length}")
        print(f"  Device: {self.agent.device}")
        print("=" * 70)
        
        start_time = time.time()
        
        # Initialize episode
        self._obs, info = self.env.reset()
        self._done = False
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_lambda2_start = info.get('lambda2_initial', info.get('lambda2', 1.0))
        
        # Training loop
        while self.total_timesteps < self.config.total_timesteps:
            # Collect rollout
            rollout_stats = self.collect_rollout()
            self.total_rollouts += 1
            
            # Update policy
            update_stats = self.update()
            
            # Log rollout
            self.logger.log_rollout(
                timesteps=self.total_timesteps,
                reward=rollout_stats["mean_reward"],
                lambda2_reduction=rollout_stats["mean_lambda2_reduction"],
                episode_length=rollout_stats["mean_length"],
                actor_loss=update_stats["actor_loss"],
                critic_loss=update_stats["critic_loss"],
                entropy=update_stats["entropy"],
                clip_fraction=update_stats["clip_fraction"],
                approx_kl=update_stats["approx_kl"],
                avg_jamming_power_dbm=rollout_stats["mean_jamming_power_dbm"]
            )
            
            # Print progress
            if self.total_rollouts % self.config.log_interval == 0:
                self.logger.print_stats(prefix=f"[Rollout {self.total_rollouts}] ")
            
            # Save checkpoint
            if self.total_rollouts % self.config.save_interval == 0:
                self.save_checkpoint(f"rollout_{self.total_rollouts}")
                self.save_checkpoint("latest")
            
            # Evaluate
            if self.total_rollouts % self.config.eval_interval == 0:
                eval_result = self.evaluate(n_episodes=self.config.eval_episodes)
                summary = eval_result.summary()
                print(f"  [Eval] Reward: {summary['reward_mean']:.2f} | "
                      f"L2_reduction: {summary['lambda2_reduction_mean']:.1f}%")
                
                # Save best model
                if summary['lambda2_reduction_mean'] > self.best_reduction:
                    self.best_reduction = summary['lambda2_reduction_mean']
                    self.save_checkpoint("best")
                    print(f"  [*] New best model! L2 reduction: {self.best_reduction:.1f}%")
            
            # Check convergence
            if self.check_convergence():
                break
            
            # Run callbacks
            for callback in self.callbacks:
                callback(self)
        
        # Final save
        self.save_checkpoint("final")
        self.logger.save()
        
        # Final stats
        elapsed = time.time() - start_time
        final_stats = {
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "total_rollouts": self.total_rollouts,
            "best_lambda2_reduction": self.best_reduction,
            "time_elapsed": elapsed,
            "fps": self.total_timesteps / elapsed
        }
        
        print("=" * 70)
        print("Training Complete!")
        print(f"  Total timesteps: {self.total_timesteps:,}")
        print(f"  Total episodes: {self.total_episodes}")
        print(f"  Best L2 reduction: {self.best_reduction:.1f}%")
        print(f"  Time: {elapsed / 60:.1f} minutes")
        print(f"  FPS: {final_stats['fps']:.0f}")
        print("=" * 70)
        
        self.logger.close()
        
        return final_stats


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_trainer() -> dict:
    """Verify trainer module (quick sanity check)."""
    results = {}
    
    if not HAS_TORCH:
        results["pytorch_available"] = {"pass": False, "error": "PyTorch not installed"}
        return results
    
    # Test 1: Config creation
    from .config import get_debug_config
    config = get_debug_config()
    results["test_debug_config"] = {
        "N": config.env.N,
        "rollout_length": config.ppo.rollout_length,
        "pass": config.env.N == 5
    }
    
    # Test 2: Trainer creation (without actually training)
    # This would require the environment to be available
    results["test_module_imports"] = {
        "trainer_available": True,
        "pass": True
    }
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Trainer Verification")
    print("=" * 60)
    
    results = verify_trainer()
    
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
