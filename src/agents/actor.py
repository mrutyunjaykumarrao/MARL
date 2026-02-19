"""
Actor Network Module
====================

Implements the Actor network for PPO with hybrid continuous+discrete action space.

Reference: PROJECT_MASTER_GUIDE_v2.md Section 3.8

Architecture:
    Shared trunk: FC(5→128) → LayerNorm → ReLU → FC(128→128) → LayerNorm → ReLU
    Continuous head: μ = FC(128→2), log_σ = clamp(FC(128→2), -2, 2)
    Discrete head: logits = FC(128→4), π_band = softmax(logits)

Author: MARL Jammer Team
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal, Categorical
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    
    class Actor(nn.Module):
        """
        Actor network for hybrid continuous+discrete action space.
        
        Outputs:
            - Continuous: velocity (Vx, Vy) via Gaussian policy
            - Discrete: frequency band via Categorical policy
        
        Uses LayerNorm for stability with variable batch sizes and 
        shared-parameter multi-agent training.
        
        Attributes:
            obs_dim: Observation dimensionality (5)
            hidden_dim: Hidden layer size (128)
            v_max: Maximum velocity for action scaling
            num_bands: Number of frequency bands (4)
            
        Example:
            >>> actor = Actor(obs_dim=5, hidden_dim=128)
            >>> obs = torch.randn(4, 5)  # 4 agents, 5-dim obs
            >>> action, log_prob, entropy = actor.sample(obs)
            >>> print(action.shape)  # (4, 3): [vx, vy, band]
        """
        
        def __init__(
            self,
            obs_dim: int = 5,
            hidden_dim: int = 128,
            v_max: float = 5.0,
            num_bands: int = 4,
            log_std_min: float = -2.0,
            log_std_max: float = 2.0
        ):
            """
            Initialize Actor network.
            
            Args:
                obs_dim: Observation dimensionality
                hidden_dim: Hidden layer size
                v_max: Maximum velocity for action scaling
                num_bands: Number of frequency bands
                log_std_min: Minimum log standard deviation
                log_std_max: Maximum log standard deviation
            """
            super().__init__()
            
            self.obs_dim = obs_dim
            self.hidden_dim = hidden_dim
            self.v_max = v_max
            self.num_bands = num_bands
            self.log_std_min = log_std_min
            self.log_std_max = log_std_max
            
            # Shared trunk
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
            
            # Continuous head: velocity mean and log_std
            self.mu_head = nn.Linear(hidden_dim, 2)
            self.log_std_head = nn.Linear(hidden_dim, 2)
            
            # Discrete head: band logits
            self.band_head = nn.Linear(hidden_dim, num_bands)
            
            # Initialize weights
            self._init_weights()
        
        def _init_weights(self):
            """Initialize network weights using orthogonal initialization."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    nn.init.constant_(module.bias, 0.0)
            
            # Smaller initialization for output layers
            nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
            nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
            nn.init.orthogonal_(self.band_head.weight, gain=0.01)
        
        def forward(
            self,
            obs: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Forward pass returning distribution parameters.
            
            Args:
                obs: Observations, shape (batch, obs_dim)
                
            Returns:
                Tuple of:
                    - mu: Velocity means, shape (batch, 2)
                    - log_std: Velocity log stds, shape (batch, 2)
                    - band_logits: Band logits, shape (batch, num_bands)
            """
            features = self.trunk(obs)
            
            # Continuous: velocity distribution parameters
            mu = self.mu_head(features)
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            
            # Discrete: band logits
            band_logits = self.band_head(features)
            
            return mu, log_std, band_logits
        
        def sample(
            self,
            obs: torch.Tensor,
            deterministic: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Sample actions from the policy.
            
            Args:
                obs: Observations, shape (batch, obs_dim)
                deterministic: If True, use mean velocity and argmax band
                
            Returns:
                Tuple of:
                    - actions: Combined actions, shape (batch, 3)
                    - log_probs: Total log probabilities, shape (batch,)
                    - entropy: Policy entropy, shape (batch,)
            """
            mu, log_std, band_logits = self.forward(obs)
            std = torch.exp(log_std)
            
            # Create distributions
            velocity_dist = Normal(mu, std)
            band_dist = Categorical(logits=band_logits)
            
            if deterministic:
                # Use mode of distributions
                velocity = mu
                band = torch.argmax(band_logits, dim=-1)
            else:
                # Sample from distributions
                velocity = velocity_dist.rsample()  # Reparameterized sample
                band = band_dist.sample()
            
            # Clamp velocity to [-v_max, v_max] (simpler than tanh, more stable)
            velocity_clamped = torch.clamp(velocity, -self.v_max, self.v_max)
            
            # Compute log probabilities (simple Gaussian, no squashing correction needed)
            log_prob_velocity = velocity_dist.log_prob(velocity).sum(dim=-1)
            
            log_prob_band = band_dist.log_prob(band)
            
            # Total log probability
            log_prob = log_prob_velocity + log_prob_band
            
            # Entropy
            entropy_velocity = velocity_dist.entropy().sum(dim=-1)
            entropy_band = band_dist.entropy()
            entropy = entropy_velocity + entropy_band
            
            # Combine actions (use clamped velocity for environment)
            actions = torch.cat([
                velocity_clamped,
                band.unsqueeze(-1).float()
            ], dim=-1)
            
            return actions, log_prob, entropy
        
        def evaluate_actions(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Evaluate log probability and entropy for given actions.
            
            Used during PPO update to compute ratio.
            
            Args:
                obs: Observations, shape (batch, obs_dim)
                actions: Actions taken, shape (batch, 3)
                
            Returns:
                Tuple of:
                    - log_probs: Log probabilities, shape (batch,)
                    - entropy: Entropy, shape (batch,)
            """
            mu, log_std, band_logits = self.forward(obs)
            std = torch.exp(log_std)
            
            velocity = actions[:, :2]
            band = actions[:, 2].long()
            
            # Create distributions
            velocity_dist = Normal(mu, std)
            band_dist = Categorical(logits=band_logits)
            
            # Compute log probabilities (simple Gaussian - velocity was sampled from this dist)
            # Note: we evaluate at the raw velocity, not clamped, since log_prob was computed pre-clamp
            velocity_for_logprob = torch.clamp(velocity, -self.v_max * 2, self.v_max * 2)  # Allow slight extrapolation
            log_prob_velocity = velocity_dist.log_prob(velocity_for_logprob).sum(dim=-1)
            
            log_prob_band = band_dist.log_prob(band)
            
            log_prob = log_prob_velocity + log_prob_band
            
            # Entropy
            entropy_velocity = velocity_dist.entropy().sum(dim=-1)
            entropy_band = band_dist.entropy()
            entropy = entropy_velocity + entropy_band
            
            return log_prob, entropy
        
        def get_action_numpy(
            self,
            obs: np.ndarray,
            deterministic: bool = False
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Get actions from numpy observations.
            
            Convenience method for environment interaction.
            
            Args:
                obs: Observations, shape (M, obs_dim)
                deterministic: Use deterministic actions
                
            Returns:
                Tuple of (actions, log_probs, entropy) as numpy arrays
            """
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float()
                actions, log_probs, entropy = self.sample(obs_tensor, deterministic)
                
            return (
                actions.numpy(),
                log_probs.numpy(),
                entropy.numpy()
            )


# NumPy fallback for systems without PyTorch
class ActorNumpy:
    """
    NumPy-based Actor for environments without PyTorch.
    
    Uses simple linear layers with manual forward pass.
    Primarily for testing and lightweight inference.
    """
    
    def __init__(
        self,
        obs_dim: int = 5,
        hidden_dim: int = 128,
        v_max: float = 5.0,
        num_bands: int = 4
    ):
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.v_max = v_max
        self.num_bands = num_bands
        
        # Initialize weights
        self.w1 = np.random.randn(obs_dim, hidden_dim) * np.sqrt(2.0 / obs_dim)
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        
        self.w_mu = np.random.randn(hidden_dim, 2) * 0.01
        self.b_mu = np.zeros(2)
        self.w_std = np.random.randn(hidden_dim, 2) * 0.01
        self.b_std = np.zeros(2)
        
        self.w_band = np.random.randn(hidden_dim, num_bands) * 0.01
        self.b_band = np.zeros(num_bands)
    
    def forward(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass."""
        # Layer 1
        h1 = obs @ self.w1 + self.b1
        h1 = self._layer_norm(h1)
        h1 = np.maximum(0, h1)  # ReLU
        
        # Layer 2
        h2 = h1 @ self.w2 + self.b2
        h2 = self._layer_norm(h2)
        h2 = np.maximum(0, h2)  # ReLU
        
        # Heads
        mu = h2 @ self.w_mu + self.b_mu
        log_std = np.clip(h2 @ self.w_std + self.b_std, -2, 2)
        band_logits = h2 @ self.w_band + self.b_band
        
        return mu, log_std, band_logits
    
    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Simple layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + eps)
    
    def sample(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample actions."""
        mu, log_std, band_logits = self.forward(obs)
        std = np.exp(log_std)
        
        if deterministic:
            velocity = np.tanh(mu) * self.v_max
            band = np.argmax(band_logits, axis=-1)
        else:
            velocity_raw = mu + std * np.random.randn(*mu.shape)
            velocity = np.tanh(velocity_raw) * self.v_max
            
            # Softmax and sample
            exp_logits = np.exp(band_logits - np.max(band_logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            band = np.array([np.random.choice(self.num_bands, p=p) for p in probs])
        
        # Combine actions
        actions = np.column_stack([velocity, band])
        
        # Simplified log prob and entropy (placeholder)
        log_probs = np.zeros(obs.shape[0])
        entropy = np.ones(obs.shape[0])
        
        return actions, log_probs, entropy


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_actor() -> dict:
    """Run verification tests."""
    results = {}
    
    if HAS_TORCH:
        # Test 1: Creation
        actor = Actor(obs_dim=5, hidden_dim=128)
        results["test_creation"] = {
            "obs_dim": actor.obs_dim,
            "hidden_dim": actor.hidden_dim,
            "pass": True
        }
        
        # Test 2: Forward pass
        obs = torch.randn(4, 5)
        mu, log_std, logits = actor.forward(obs)
        results["test_forward"] = {
            "mu_shape": list(mu.shape),
            "log_std_shape": list(log_std.shape),
            "logits_shape": list(logits.shape),
            "pass": mu.shape == (4, 2) and logits.shape == (4, 4)
        }
        
        # Test 3: Sample
        actions, log_probs, entropy = actor.sample(obs)
        results["test_sample"] = {
            "actions_shape": list(actions.shape),
            "log_probs_shape": list(log_probs.shape),
            "entropy_shape": list(entropy.shape),
            "pass": actions.shape == (4, 3)
        }
        
        # Test 4: Velocity in range
        results["test_velocity_range"] = {
            "max_velocity": float(actions[:, :2].abs().max()),
            "v_max": actor.v_max,
            "pass": actions[:, :2].abs().max() <= actor.v_max
        }
        
        # Test 5: Evaluate actions
        log_probs_eval, entropy_eval = actor.evaluate_actions(obs, actions)
        results["test_evaluate"] = {
            "log_probs_shape": list(log_probs_eval.shape),
            "finite": bool(torch.isfinite(log_probs_eval).all()),
            "pass": log_probs_eval.shape == (4,)
        }
        
        # Test 6: NumPy interface
        obs_np = np.random.randn(4, 5).astype(np.float32)
        actions_np, _, _ = actor.get_action_numpy(obs_np)
        results["test_numpy_interface"] = {
            "actions_shape": list(actions_np.shape),
            "pass": actions_np.shape == (4, 3)
        }
    else:
        # NumPy fallback tests
        actor = ActorNumpy(obs_dim=5, hidden_dim=128)
        obs = np.random.randn(4, 5).astype(np.float32)
        actions, _, _ = actor.sample(obs)
        results["test_numpy_fallback"] = {
            "actions_shape": list(actions.shape),
            "pass": actions.shape == (4, 3)
        }
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Actor Network Verification")
    print("=" * 60)
    
    results = verify_actor()
    
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
