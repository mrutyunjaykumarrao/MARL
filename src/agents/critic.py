"""
Critic Network Module
=====================

Implements the Critic (value function) network for PPO.

Reference: PROJECT_MASTER_GUIDE_v2.md Section 3.9

Architecture:
    Input: s_pooled = (1/M) * sum_{j=1}^{M} s_j  [mean-pooled global state]
    FC(5→128) → ReLU → FC(128→128) → ReLU → FC(128→1) = V_φ(s)

Mean-pooled input keeps critic input size fixed regardless of M.
Enables scaling to M=40 without architecture change.

Author: MARL Jammer Team
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    
    class Critic(nn.Module):
        """
        Critic network estimating state value V(s).
        
        Takes mean-pooled observations across all agents to provide
        a global value estimate. This enables CTDE (Centralized Training,
        Decentralized Execution) where the critic sees global state during
        training but agents act on local observations.
        
        Attributes:
            obs_dim: Observation dimensionality (5)
            hidden_dim: Hidden layer size (128)
            
        Example:
            >>> critic = Critic(obs_dim=5, hidden_dim=128)
            >>> obs_all = torch.randn(4, 5)  # 4 agents, 5-dim obs each
            >>> value = critic(obs_all)
            >>> print(value.shape)  # (1,) - single global value
        """
        
        def __init__(
            self,
            obs_dim: int = 5,
            hidden_dim: int = 128
        ):
            """
            Initialize Critic network.
            
            Args:
                obs_dim: Observation dimensionality
                hidden_dim: Hidden layer size
            """
            super().__init__()
            
            self.obs_dim = obs_dim
            self.hidden_dim = hidden_dim
            
            # Value function network
            self.network = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            
            # Initialize weights
            self._init_weights()
        
        def _init_weights(self):
            """Initialize network weights using orthogonal initialization."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    nn.init.constant_(module.bias, 0.0)
            
            # Smaller initialization for output layer
            output_layer = self.network[-1]
            nn.init.orthogonal_(output_layer.weight, gain=1.0)
        
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            """
            Compute state value from observations.
            
            If input is (M, obs_dim), performs mean-pooling first.
            If input is already (batch, obs_dim), assumes pre-pooled.
            
            Args:
                obs: Observations
                    - Per-agent: shape (M, obs_dim) -> mean-pooled to (obs_dim,)
                    - Batched: shape (batch, obs_dim) -> direct forward
                    - Batched per-agent: shape (batch, M, obs_dim) -> mean over M
                    
            Returns:
                Value estimate(s)
            """
            # Handle different input shapes
            if obs.dim() == 3:
                # (batch, M, obs_dim) -> mean pool over M
                pooled = obs.mean(dim=1)  # (batch, obs_dim)
            elif obs.dim() == 2:
                # Could be (M, obs_dim) for single step or (batch, obs_dim) for batched
                # We assume if it came from a single step with M agents, 
                # caller should pool first. Here we assume (batch, obs_dim).
                pooled = obs
            else:
                # (obs_dim,) - single observation
                pooled = obs.unsqueeze(0)
            
            return self.network(pooled)
        
        def forward_pooled(self, obs_pooled: torch.Tensor) -> torch.Tensor:
            """
            Forward pass with pre-pooled observations.
            
            Args:
                obs_pooled: Mean-pooled observations, shape (batch, obs_dim)
                
            Returns:
                Value estimates, shape (batch, 1)
            """
            return self.network(obs_pooled)
        
        def get_value_numpy(self, obs: np.ndarray) -> np.ndarray:
            """
            Get value from numpy observations.
            
            Args:
                obs: Observations
                    - (M, obs_dim): Per-agent obs, will be mean-pooled
                    - (obs_dim,): Already pooled
                    
            Returns:
                Value as numpy scalar or array
            """
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float()
                
                # Mean-pool if per-agent
                if obs.ndim == 2:
                    obs_tensor = obs_tensor.mean(dim=0, keepdim=True)
                elif obs.ndim == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                
                value = self.forward(obs_tensor)
                
            return value.numpy().squeeze()


# NumPy fallback for systems without PyTorch
class CriticNumpy:
    """
    NumPy-based Critic for environments without PyTorch.
    
    Primarily for testing and lightweight inference.
    """
    
    def __init__(
        self,
        obs_dim: int = 5,
        hidden_dim: int = 128
    ):
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights
        self.w1 = np.random.randn(obs_dim, hidden_dim) * np.sqrt(2.0 / obs_dim)
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.w3 = np.random.randn(hidden_dim, 1) * 1.0
        self.b3 = np.zeros(1)
    
    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Forward pass with mean-pooling."""
        # Mean-pool if multiple agents
        if obs.ndim == 2:
            obs_pooled = obs.mean(axis=0, keepdims=True)
        else:
            obs_pooled = obs.reshape(1, -1)
        
        # Layer 1
        h1 = obs_pooled @ self.w1 + self.b1
        h1 = np.maximum(0, h1)  # ReLU
        
        # Layer 2
        h2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(0, h2)  # ReLU
        
        # Output
        value = h2 @ self.w3 + self.b3
        
        return value.squeeze()


# =============================================================================
# COMBINED ACTOR-CRITIC
# =============================================================================

if HAS_TORCH:
    
    class ActorCritic(nn.Module):
        """
        Combined Actor-Critic network with shared observations.
        
        Wraps separate Actor and Critic for convenience.
        Both networks share the same observation input.
        
        Attributes:
            actor: Actor network
            critic: Critic network
            
        Example:
            >>> ac = ActorCritic(obs_dim=5)
            >>> obs = torch.randn(4, 5)
            >>> actions, log_probs, entropy, values = ac.act(obs)
        """
        
        def __init__(
            self,
            obs_dim: int = 5,
            hidden_dim: int = 128,
            v_max: float = 5.0,
            num_bands: int = 4
        ):
            super().__init__()
            
            # Import Actor here to avoid circular imports
            from .actor import Actor
            
            self.actor = Actor(
                obs_dim=obs_dim,
                hidden_dim=hidden_dim,
                v_max=v_max,
                num_bands=num_bands
            )
            
            self.critic = Critic(
                obs_dim=obs_dim,
                hidden_dim=hidden_dim
            )
        
        def act(
            self,
            obs: torch.Tensor,
            deterministic: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Get actions and values for observations.
            
            Args:
                obs: Per-agent observations, shape (M, obs_dim)
                deterministic: Use deterministic policy
                
            Returns:
                Tuple of:
                    - actions: Shape (M, 3)
                    - log_probs: Shape (M,)
                    - entropy: Shape (M,)
                    - value: Shape (1,) - global value
            """
            # Get actions from actor
            actions, log_probs, entropy = self.actor.sample(obs, deterministic)
            
            # Get value from critic (mean-pooled)
            obs_pooled = obs.mean(dim=0, keepdim=True)
            value = self.critic.forward_pooled(obs_pooled)
            
            return actions, log_probs, entropy, value
        
        def evaluate(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Evaluate actions for PPO update.
            
            Args:
                obs: Observations, shape (batch, obs_dim)
                actions: Actions, shape (batch, 3)
                
            Returns:
                Tuple of:
                    - log_probs: Shape (batch,)
                    - entropy: Shape (batch,)
                    - values: Shape (batch, 1)
            """
            log_probs, entropy = self.actor.evaluate_actions(obs, actions)
            values = self.critic.forward_pooled(obs)
            
            return log_probs, entropy, values
        
        def get_value(self, obs: torch.Tensor) -> torch.Tensor:
            """
            Get value estimate for observations.
            
            Args:
                obs: Observations, shape (batch, obs_dim) or (M, obs_dim)
                
            Returns:
                Value estimate
            """
            if obs.dim() == 2 and obs.shape[0] > 1:
                # Multiple agents - mean pool
                obs_pooled = obs.mean(dim=0, keepdim=True)
            else:
                obs_pooled = obs
            
            return self.critic.forward_pooled(obs_pooled)


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_critic() -> dict:
    """Run verification tests."""
    results = {}
    
    if HAS_TORCH:
        # Test 1: Creation
        critic = Critic(obs_dim=5, hidden_dim=128)
        results["test_creation"] = {
            "obs_dim": critic.obs_dim,
            "hidden_dim": critic.hidden_dim,
            "pass": True
        }
        
        # Test 2: Forward with batched input
        obs = torch.randn(4, 5)  # 4 samples, already pooled
        value = critic(obs)
        results["test_forward_batched"] = {
            "value_shape": list(value.shape),
            "pass": value.shape == (4, 1)
        }
        
        # Test 3: Forward with pooled input
        obs_pooled = torch.randn(1, 5)
        value_single = critic.forward_pooled(obs_pooled)
        results["test_forward_pooled"] = {
            "value_shape": list(value_single.shape),
            "pass": value_single.shape == (1, 1)
        }
        
        # Test 4: NumPy interface
        obs_np = np.random.randn(4, 5).astype(np.float32)
        value_np = critic.get_value_numpy(obs_np)
        results["test_numpy_interface"] = {
            "value_type": type(value_np).__name__,
            "value_scalar": float(value_np),
            "pass": np.isfinite(value_np)
        }
        
        # Test 5: ActorCritic combined
        ac = ActorCritic(obs_dim=5)
        obs = torch.randn(4, 5)
        actions, log_probs, entropy, value = ac.act(obs)
        results["test_actor_critic"] = {
            "actions_shape": list(actions.shape),
            "log_probs_shape": list(log_probs.shape),
            "value_shape": list(value.shape),
            "pass": actions.shape == (4, 3) and value.shape == (1, 1)
        }
        
        # Test 6: Evaluate
        log_probs_eval, entropy_eval, values_eval = ac.evaluate(obs, actions)
        results["test_evaluate"] = {
            "log_probs_shape": list(log_probs_eval.shape),
            "values_shape": list(values_eval.shape),
            "pass": log_probs_eval.shape == (4,)
        }
        
    else:
        # NumPy fallback tests
        critic = CriticNumpy(obs_dim=5, hidden_dim=128)
        obs = np.random.randn(4, 5).astype(np.float32)
        value = critic.forward(obs)
        results["test_numpy_fallback"] = {
            "value_scalar": float(value),
            "pass": np.isfinite(value)
        }
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Critic Network Verification")
    print("=" * 60)
    
    results = verify_critic()
    
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
