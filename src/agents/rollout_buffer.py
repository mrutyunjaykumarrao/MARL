"""
Rollout Buffer Module
=====================

Implements the experience buffer for PPO training.

Reference: PROJECT_MASTER_GUIDE_v2.md Section 8.3-8.4

Stores:
    - Observations
    - Actions
    - Log probabilities
    - Rewards
    - Values
    - Dones

Computes:
    - GAE advantages
    - Returns

Author: MARL Jammer Team
"""

import numpy as np
from typing import Tuple, Optional, Dict, Generator, NamedTuple


class RolloutBufferSamples(NamedTuple):
    """Container for mini-batch samples."""
    observations: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    old_values: np.ndarray


class RolloutBuffer:
    """
    Rollout buffer for PPO training.
    
    Collects experience from environment interactions and computes
    GAE (Generalized Advantage Estimation) for policy updates.
    
    Designed for multi-agent settings where each step contains
    M agent observations and actions.
    
    Attributes:
        buffer_size: Maximum number of timesteps to store
        obs_dim: Observation dimensionality per agent
        action_dim: Action dimensionality per agent
        M: Number of agents
        gamma: Discount factor
        gae_lambda: GAE smoothing parameter
        
    Example:
        >>> buffer = RolloutBuffer(buffer_size=2048, obs_dim=5, M=4)
        >>> buffer.add(obs, actions, log_probs, reward, done, values)
        >>> # ... collect more samples ...
        >>> buffer.compute_returns_and_advantages(last_values, dones)
        >>> for batch in buffer.get_minibatches(batch_size=256):
        ...     # Train on batch
    """
    
    def __init__(
        self,
        buffer_size: int = 2048,
        obs_dim: int = 5,
        action_dim: int = 3,
        M: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        Initialize rollout buffer.
        
        Args:
            buffer_size: Maximum number of timesteps (T)
            obs_dim: Observation dimensionality per agent
            action_dim: Action dimensionality per agent  
            M: Number of agents
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.M = M
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.pos = 0
        self.full = False
        
        # Allocate buffers
        # Shape: (T, M, dim) for per-agent data
        # Shape: (T,) for global data (rewards, dones)
        self.observations = np.zeros((buffer_size, M, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, M, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, M), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        # Values: one per timestep (critic uses mean-pooled obs)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        
        # Computed during finalization
        self.advantages = np.zeros((buffer_size, M), dtype=np.float32)
        self.returns = np.zeros((buffer_size, M), dtype=np.float32)
        
        # Generation counter to detect stale data
        self.generator = 0
    
    def reset(self):
        """Reset the buffer."""
        self.pos = 0
        self.full = False
        self.generator += 1
    
    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        reward: float,
        done: bool,
        value: float
    ):
        """
        Add a transition to the buffer.
        
        Args:
            obs: Per-agent observations, shape (M, obs_dim)
            actions: Per-agent actions, shape (M, action_dim)
            log_probs: Per-agent log probabilities, shape (M,)
            reward: Global reward (shared by all agents)
            done: Episode termination flag
            value: Critic value estimate (scalar, from pooled obs)
        """
        self.observations[self.pos] = obs
        self.actions[self.pos] = actions
        self.log_probs[self.pos] = log_probs
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.values[self.pos] = value
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(
        self,
        last_values: float,
        last_dones: bool
    ):
        """
        Compute GAE advantages and returns.
        
        Must be called after collecting a rollout and before sampling.
        
        Args:
            last_values: Value estimate for final state (scalar)
            last_dones: Whether final state is terminal
            
        Reference:
            A_t = δ_t + (γλ)A_{t+1}(1-done_t)
            δ_t = r_t + γV(s_{t+1})(1-done_t) - V(s_t)
            R_t = A_t + V(s_t)
        """
        size = self.pos if not self.full else self.buffer_size
        
        # Extend arrays for bootstrapping
        values_extended = np.zeros(size + 1, dtype=np.float32)
        values_extended[:size] = self.values[:size]
        values_extended[size] = last_values * (1 - float(last_dones))
        
        dones_extended = np.zeros(size + 1, dtype=np.float32)
        dones_extended[:size] = self.dones[:size]
        dones_extended[size] = float(last_dones)
        
        # GAE computation (backwards)
        advantages = np.zeros(size, dtype=np.float32)
        last_gae = 0
        
        for t in reversed(range(size)):
            next_non_terminal = 1.0 - dones_extended[t + 1]
            next_values = values_extended[t + 1]
            
            delta = (
                self.rewards[t] 
                + self.gamma * next_values * next_non_terminal 
                - values_extended[t]
            )
            
            advantages[t] = last_gae = (
                delta 
                + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            )
        
        # Broadcast to all agents (shared global advantage)
        self.advantages[:size] = advantages[:, np.newaxis]
        
        # Returns = advantages + values
        self.returns[:size] = self.advantages[:size] + self.values[:size, np.newaxis]
        
        # Normalize advantages
        adv_mean = self.advantages[:size].mean()
        adv_std = self.advantages[:size].std() + 1e-8
        self.advantages[:size] = (self.advantages[:size] - adv_mean) / adv_std
    
    def get_samples(self) -> Tuple[np.ndarray, ...]:
        """
        Get all samples flattened for training.
        
        Returns:
            Tuple of (observations, actions, log_probs, advantages, returns, values)
            All flattened to (size * M, dim) for batched training.
        """
        size = self.pos if not self.full else self.buffer_size
        
        # Flatten: (T, M, dim) -> (T*M, dim)
        obs = self.observations[:size].reshape(-1, self.obs_dim)
        actions = self.actions[:size].reshape(-1, self.action_dim)
        log_probs = self.log_probs[:size].reshape(-1)
        advantages = self.advantages[:size].reshape(-1)
        returns = self.returns[:size].reshape(-1)
        values = np.repeat(self.values[:size], self.M)
        
        return obs, actions, log_probs, advantages, returns, values
    
    def get_minibatches(
        self,
        batch_size: int = 256,
        shuffle: bool = True
    ) -> Generator[RolloutBufferSamples, None, None]:
        """
        Generate mini-batches for PPO training.
        
        Args:
            batch_size: Mini-batch size
            shuffle: Whether to shuffle samples
            
        Yields:
            RolloutBufferSamples named tuples
        """
        obs, actions, log_probs, advantages, returns, values = self.get_samples()
        
        n_samples = len(obs)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        # Generate batches
        start_idx = 0
        while start_idx < n_samples:
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield RolloutBufferSamples(
                observations=obs[batch_indices],
                actions=actions[batch_indices],
                old_log_probs=log_probs[batch_indices],
                advantages=advantages[batch_indices],
                returns=returns[batch_indices],
                old_values=values[batch_indices]
            )
            
            start_idx = end_idx
    
    def __len__(self) -> int:
        """Return number of stored transitions."""
        return self.buffer_size if self.full else self.pos
    
    @property
    def size(self) -> int:
        """Return actual number of transitions stored."""
        return len(self)


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_rollout_buffer() -> dict:
    """Run verification tests."""
    results = {}
    
    # Test 1: Creation
    buffer = RolloutBuffer(buffer_size=100, obs_dim=5, M=4)
    results["test_creation"] = {
        "buffer_size": buffer.buffer_size,
        "M": buffer.M,
        "pass": buffer.buffer_size == 100 and buffer.M == 4
    }
    
    # Test 2: Add transitions
    for i in range(50):
        obs = np.random.randn(4, 5).astype(np.float32)
        actions = np.random.randn(4, 3).astype(np.float32)
        log_probs = np.random.randn(4).astype(np.float32)
        reward = float(np.random.randn())
        done = False
        value = float(np.random.randn())
        
        buffer.add(obs, actions, log_probs, reward, done, value)
    
    results["test_add"] = {
        "pos": buffer.pos,
        "size": len(buffer),
        "pass": buffer.pos == 50
    }
    
    # Test 3: Compute returns and advantages
    buffer.compute_returns_and_advantages(last_values=0.0, last_dones=True)
    
    results["test_gae"] = {
        "advantages_shape": buffer.advantages[:50].shape,
        "returns_shape": buffer.returns[:50].shape,
        "advantages_normalized": abs(buffer.advantages[:50].mean()) < 0.1,
        "pass": buffer.advantages[:50].shape == (50, 4)
    }
    
    # Test 4: Get samples
    obs, actions, log_probs, advantages, returns, values = buffer.get_samples()
    
    results["test_get_samples"] = {
        "obs_shape": obs.shape,
        "actions_shape": actions.shape,
        "expected_size": 50 * 4,
        "pass": obs.shape == (200, 5) and actions.shape == (200, 3)
    }
    
    # Test 5: Mini-batches
    batch_count = 0
    total_samples = 0
    for batch in buffer.get_minibatches(batch_size=64):
        batch_count += 1
        total_samples += len(batch.observations)
    
    results["test_minibatches"] = {
        "batch_count": batch_count,
        "total_samples": total_samples,
        "expected_total": 200,
        "pass": total_samples == 200
    }
    
    # Test 6: Reset
    buffer.reset()
    results["test_reset"] = {
        "pos_after_reset": buffer.pos,
        "full_after_reset": buffer.full,
        "pass": buffer.pos == 0 and not buffer.full
    }
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Rollout Buffer Verification")
    print("=" * 60)
    
    results = verify_rollout_buffer()
    
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
