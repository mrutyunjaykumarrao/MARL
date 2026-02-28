"""
PPO Agent Module
================

Implements the Proximal Policy Optimization (PPO) agent for MARL training.

Reference: PROJECT_MASTER_GUIDE_v2.md Sections 3.10, 8.4-8.5

Key Features:
    - Clipped surrogate objective
    - GAE for advantage estimation
    - Entropy bonus for exploration
    - Parameter sharing across agents
    - Support for hybrid continuous+discrete actions

Author: MARL Jammer Team
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.utils import clip_grad_norm_
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .actor import Actor
from .critic import Critic, ActorCritic
from .rollout_buffer import RolloutBuffer, RolloutBufferSamples


class PPOAgent:
    """
    PPO Agent for Multi-Agent Jammer Drone System.
    
    Implements the PPO algorithm with:
        - Centralized Training, Decentralized Execution (CTDE)
        - Parameter sharing across all M agents
        - Hybrid continuous (velocity) + discrete (band) actions
        - GAE advantage estimation
        - Clipped surrogate loss
        - Entropy bonus
    
    Attributes:
        actor: Actor network (shared)
        critic: Critic network (centralized)
        actor_optimizer: Optimizer for actor
        critic_optimizer: Optimizer for critic
        buffer: Rollout buffer for experience collection
        
    Example:
        >>> agent = PPOAgent(obs_dim=5, M=4)
        >>> obs = np.random.randn(4, 5).astype(np.float32)
        >>> actions, log_probs, value = agent.get_action(obs)
        >>> # ... interact with environment ...
        >>> agent.store_transition(obs, actions, log_probs, reward, done, value)
        >>> # ... after rollout ...
        >>> losses = agent.update()
    """
    
    def __init__(
        self,
        obs_dim: int = 5,
        hidden_dim: int = 128,
        M: int = 4,
        v_max: float = 5.0,
        num_bands: int = 4,
        # Actor variance bounds
        log_std_min: float = -1.0,
        log_std_max: float = 0.5,
        # PPO hyperparameters
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        c1: float = 0.5,  # Value loss coefficient
        c2: float = 0.01,  # Entropy bonus coefficient
        max_grad_norm: float = 0.5,
        # Training parameters
        rollout_length: int = 2048,
        batch_size: int = 256,
        n_epochs: int = 10,
        # Device
        device: str = "auto"
    ):
        """
        Initialize PPO agent.
        
        Args:
            obs_dim: Observation dimensionality per agent
            hidden_dim: Hidden layer size for networks
            M: Number of jammer agents
            v_max: Maximum velocity
            num_bands: Number of frequency bands
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_eps: PPO clip epsilon
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            c1: Value loss coefficient
            c2: Entropy bonus coefficient
            max_grad_norm: Gradient clipping threshold
            rollout_length: Length of each rollout
            batch_size: Mini-batch size
            n_epochs: Number of PPO epochs per update
            device: Device to use ("auto", "cpu", or "cuda")
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for PPO agent")
        
        # Store hyperparameters
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.M = M
        self.v_max = v_max
        self.num_bands = num_bands
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.c1 = c1
        self.c2 = c2
        self.c2_initial = c2  # Store for decay
        self.max_grad_norm = max_grad_norm
        
        # Store initial learning rates for decay
        self.lr_actor_initial = lr_actor
        self.lr_critic_initial = lr_critic
        
        self.rollout_length = rollout_length
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize networks
        self.actor = Actor(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            v_max=v_max,
            num_bands=num_bands,
            log_std_min=log_std_min,
            log_std_max=log_std_max
        ).to(self.device)
        
        self.critic = Critic(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=lr_actor
        )
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=lr_critic
        )
        
        # Initialize rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            obs_dim=obs_dim,
            action_dim=3,  # vx, vy, band
            M=M,
            gamma=gamma,
            gae_lambda=gae_lambda
        )
        
        # Training statistics
        self.total_timesteps = 0
        self.total_updates = 0
        self.training_stats: Dict[str, List[float]] = {
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "clip_fraction": [],
            "approx_kl": []
        }
    
    def get_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get action from current policy.
        
        Args:
            obs: Per-agent observations, shape (M, obs_dim)
            deterministic: Use deterministic policy (for evaluation)
            
        Returns:
            Tuple of:
                - actions: Shape (M, 3)
                - log_probs: Shape (M,)
                - value: Scalar value estimate
        """
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            
            # Get actions from actor
            actions, log_probs, _ = self.actor.sample(obs_tensor, deterministic)
            
            # Get value from critic (mean-pooled)
            obs_pooled = obs_tensor.mean(dim=0, keepdim=True)
            value = self.critic.forward_pooled(obs_pooled)
        
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            float(value.cpu().numpy().squeeze())
        )
    
    def store_transition(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        reward: float,
        done: bool,
        value: float
    ):
        """
        Store a transition in the buffer.
        
        Args:
            obs: Per-agent observations, shape (M, obs_dim)
            actions: Per-agent actions, shape (M, 3)
            log_probs: Per-agent log probs, shape (M,)
            reward: Global reward
            done: Episode termination flag
            value: Value estimate
        """
        self.buffer.add(obs, actions, log_probs, reward, done, value)
        self.total_timesteps += 1
    
    def update(
        self,
        last_obs: np.ndarray,
        last_done: bool
    ) -> Dict[str, float]:
        """
        Perform PPO update after collecting a rollout.
        
        Args:
            last_obs: Final observation for bootstrapping
            last_done: Whether final state is terminal
            
        Returns:
            Dictionary of training statistics
        """
        # Compute last value for bootstrapping
        with torch.no_grad():
            last_obs_tensor = torch.from_numpy(last_obs).float().to(self.device)
            last_obs_pooled = last_obs_tensor.mean(dim=0, keepdim=True)
            last_value = float(
                self.critic.forward_pooled(last_obs_pooled).cpu().numpy().squeeze()
            )
        
        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(last_value, last_done)
        
        # Training statistics
        actor_losses = []
        critic_losses = []
        entropies = []
        clip_fractions = []
        approx_kls = []
        
        # Target KL for early stopping
        target_kl = 0.03
        
        # PPO epochs
        for epoch in range(self.n_epochs):
            epoch_kl = 0.0
            n_batches = 0
            
            for batch in self.buffer.get_minibatches(self.batch_size):
                # Convert to tensors
                obs = torch.from_numpy(batch.observations).float().to(self.device)
                actions = torch.from_numpy(batch.actions).float().to(self.device)
                old_log_probs = torch.from_numpy(batch.old_log_probs).float().to(self.device)
                advantages = torch.from_numpy(batch.advantages).float().to(self.device)
                returns = torch.from_numpy(batch.returns).float().to(self.device)
                
                # Evaluate actions
                log_probs, entropy = self.actor.evaluate_actions(obs, actions)
                values = self.critic.forward_pooled(obs).squeeze()
                
                # Compute ratio
                ratio = torch.exp(log_probs - old_log_probs)
                
                # Clipped surrogate loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                critic_loss = self.c1 * ((values - returns) ** 2).mean()
                
                # Entropy bonus
                entropy_loss = -self.c2 * entropy.mean()
                
                # Total loss
                total_loss = actor_loss + critic_loss + entropy_loss
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss_backward = actor_loss + entropy_loss
                actor_loss_backward.backward(retain_graph=True)
                clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # Record statistics
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())
                
                # Clip fraction
                with torch.no_grad():
                    clip_fraction = (
                        (torch.abs(ratio - 1) > self.clip_eps).float().mean().item()
                    )
                    clip_fractions.append(clip_fraction)
                    
                    # Approximate KL divergence
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    approx_kls.append(approx_kl)
                    epoch_kl += approx_kl
                    n_batches += 1
            
            # Early stopping based on KL divergence
            avg_epoch_kl = epoch_kl / max(n_batches, 1)
            if avg_epoch_kl > target_kl:
                break  # Stop training epochs early
        
        # Update counter and reset buffer
        self.total_updates += 1
        self.buffer.reset()
        
        # Compute mean statistics
        stats = {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy": np.mean(entropies),
            "clip_fraction": np.mean(clip_fractions),
            "approx_kl": np.mean(approx_kls)
        }
        
        # Store for logging
        for key, value in stats.items():
            self.training_stats[key].append(value)
        
        return stats
    
    def set_learning_rate(self, lr_actor: float, lr_critic: float):
        """Update learning rates for decay during training."""
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = lr_actor
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = lr_critic
    
    def set_entropy_coef(self, c2: float):
        """Update entropy coefficient for decay during training."""
        self.c2 = c2
    
    def save(self, path: str):
        """
        Save agent to disk.
        
        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "total_timesteps": self.total_timesteps,
            "total_updates": self.total_updates,
            "training_stats": self.training_stats,
            "config": {
                "obs_dim": self.obs_dim,
                "hidden_dim": self.hidden_dim,
                "M": self.M,
                "v_max": self.v_max,
                "num_bands": self.num_bands,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_eps": self.clip_eps,
                "c1": self.c1,
                "c2": self.c2,
                "max_grad_norm": self.max_grad_norm,
                "rollout_length": self.rollout_length,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs
            }
        }, path / "ppo_agent.pt")
    
    def load(self, path: str):
        """
        Load agent from disk.
        
        Args:
            path: Directory path to load from
        """
        path = Path(path)
        checkpoint = torch.load(path / "ppo_agent.pt", map_location=self.device, weights_only=False)
        
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.total_timesteps = checkpoint["total_timesteps"]
        self.total_updates = checkpoint["total_updates"]
        self.training_stats = checkpoint["training_stats"]
    
    @classmethod
    def from_checkpoint(cls, path: str, device: str = "auto") -> "PPOAgent":
        """
        Load agent from checkpoint.
        
        Args:
            path: Directory path to load from
            device: Device to use
            
        Returns:
            Loaded PPO agent
        """
        path = Path(path)
        checkpoint = torch.load(path / "ppo_agent.pt", map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        
        agent = cls(
            obs_dim=config["obs_dim"],
            hidden_dim=config["hidden_dim"],
            M=config["M"],
            v_max=config["v_max"],
            num_bands=config["num_bands"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_eps=config["clip_eps"],
            c1=config["c1"],
            c2=config["c2"],
            max_grad_norm=config["max_grad_norm"],
            rollout_length=config["rollout_length"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            device=device
        )
        
        agent.load(path)
        return agent
    
    def set_training_mode(self, training: bool = True):
        """Set networks to training or evaluation mode."""
        if training:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_ppo_agent() -> dict:
    """Run verification tests."""
    results = {}
    
    if not HAS_TORCH:
        results["pytorch_available"] = {"pass": False, "error": "PyTorch not installed"}
        return results
    
    # Test 1: Creation
    agent = PPOAgent(obs_dim=5, hidden_dim=64, M=4, rollout_length=64, batch_size=16)
    results["test_creation"] = {
        "obs_dim": agent.obs_dim,
        "M": agent.M,
        "device": str(agent.device),
        "pass": True
    }
    
    # Test 2: Get action
    obs = np.random.randn(4, 5).astype(np.float32)
    actions, log_probs, value = agent.get_action(obs)
    results["test_get_action"] = {
        "actions_shape": actions.shape,
        "log_probs_shape": log_probs.shape,
        "value_type": type(value).__name__,
        "pass": actions.shape == (4, 3) and log_probs.shape == (4,)
    }
    
    # Test 3: Store transitions
    for i in range(64):
        obs = np.random.randn(4, 5).astype(np.float32)
        actions, log_probs, value = agent.get_action(obs)
        reward = float(np.random.randn())
        done = i == 63
        agent.store_transition(obs, actions, log_probs, reward, done, value)
    
    results["test_store_transitions"] = {
        "buffer_size": len(agent.buffer),
        "total_timesteps": agent.total_timesteps,
        "pass": len(agent.buffer) == 64
    }
    
    # Test 4: Update
    last_obs = np.random.randn(4, 5).astype(np.float32)
    stats = agent.update(last_obs, last_done=True)
    results["test_update"] = {
        "actor_loss": stats["actor_loss"],
        "critic_loss": stats["critic_loss"],
        "entropy": stats["entropy"],
        "total_updates": agent.total_updates,
        "pass": agent.total_updates == 1 and np.isfinite(stats["actor_loss"])
    }
    
    # Test 5: Deterministic action
    obs = np.random.randn(4, 5).astype(np.float32)
    actions1, _, _ = agent.get_action(obs, deterministic=True)
    actions2, _, _ = agent.get_action(obs, deterministic=True)
    results["test_deterministic"] = {
        "actions_match": np.allclose(actions1, actions2),
        "pass": np.allclose(actions1, actions2)
    }
    
    # Test 6: Save and load
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        agent.save(tmpdir)
        agent2 = PPOAgent.from_checkpoint(tmpdir)
        
        results["test_save_load"] = {
            "total_timesteps_match": agent.total_timesteps == agent2.total_timesteps,
            "total_updates_match": agent.total_updates == agent2.total_updates,
            "pass": agent.total_timesteps == agent2.total_timesteps
        }
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("PPO Agent Verification")
    print("=" * 60)
    
    results = verify_ppo_agent()
    
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
