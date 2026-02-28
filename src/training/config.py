"""
Training Configuration Module
=============================

Centralized configuration for all training hyperparameters.

Reference: PROJECT_MASTER_GUIDE_v2.md Section 8.5

Author: MARL Jammer Team
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    
    # Enemy swarm
    N: int = 10                     # Number of enemy drones
    arena_size: float = 100.0       # Arena side length (meters)
    
    # Jammer agents
    M: int = 4                      # Number of jammer drones
    v_max: float = 5.0              # Max jammer velocity (m/s)
    
    # Enemy dynamics
    v_enemy: float = 2.0            # Enemy random walk step (m/s)
    motion_mode: str = "random"     # "static", "random", or "coordinated"
    
    # Frequency bands
    num_bands: int = 4              # {433MHz, 915MHz, 2.4GHz, 5.8GHz}
    
    # DBSCAN clustering
    eps: float = 25.0               # DBSCAN neighborhood radius
    min_samples: int = 2            # DBSCAN min cluster size
    k_recompute: int = 10           # Re-cluster every K steps
    
    # Episode parameters
    max_steps: int = 200            # Max steps per episode
    
    # RF parameters
    tx_power_dbm: float = 20.0      # Enemy transmit power
    sensitivity_dbm: float = -90.0  # Receiver sensitivity
    jammer_power_dbm: float = 30.0  # Jammer power (1W)
    jam_thresh_dbm: float = -30.0   # Jamming threshold: -30dBm gives ~10m range at 2.4GHz
                                    # Random placement achieves ~6% disruption
                                    # Trained agent should reach 60-80%
    random_jammer_start: bool = False  # Start jammers at random positions
    
    # Reward weights (ω₁ to ω₅)
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "lambda2_reduction": 1.0,
        "band_match": 0.3,
        "proximity": 0.2,
        "energy": 0.1,
        "overlap": 0.2
    })


@dataclass
class NetworkConfig:
    """Neural network configuration."""
    
    # Architecture
    obs_dim: int = 5                # Observation dimensionality
    hidden_dim: int = 128           # Hidden layer size
    
    # Actor specifics - TIGHTER BOUNDS to prevent entropy explosion
    log_std_min: float = -1.0       # Min log std (std=0.37) - not too deterministic
    log_std_max: float = 0.5        # Max log std (std=1.65) - prevents random policy


@dataclass
class PPOConfig:
    """PPO algorithm configuration."""
    
    # Core hyperparameters
    gamma: float = 0.99             # Discount factor
    gae_lambda: float = 0.95        # GAE smoothing parameter
    clip_eps: float = 0.2           # PPO clipping epsilon
    
    # Learning rates
    lr_actor: float = 3e-4          # Actor learning rate
    lr_critic: float = 1e-3         # Critic learning rate
    
    # Loss coefficients
    c1: float = 0.5                 # Value loss coefficient
    c2: float = 0.001               # Entropy bonus coefficient (very low for multi-agent)
    
    # Training parameters
    n_epochs: int = 10              # PPO epochs per rollout
    batch_size: int = 256           # Mini-batch size
    max_grad_norm: float = 0.5      # Gradient clipping threshold
    
    # Rollout
    rollout_length: int = 2048      # Steps per rollout


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # Sub-configs
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    
    # Training budget
    total_timesteps: int = 2_000_000  # Total training steps
    
    # Logging
    log_interval: int = 1           # Log every N rollouts
    save_interval: int = 10         # Save checkpoint every N rollouts
    eval_interval: int = 10         # Evaluate every N rollouts
    eval_episodes: int = 5          # Episodes per evaluation
    
    # Paths
    experiment_name: str = "marl_jammer"
    output_dir: str = "outputs"
    
    # Reproducibility
    seed: int = 42
    
    # Device
    device: str = "auto"            # "auto", "cpu", or "cuda"
    
    # Convergence
    target_reduction: float = 70.0  # Target lambda2 reduction (%)
    convergence_window: int = 50    # Episodes to check convergence
    early_stop_patience: int = 100  # Rollouts without improvement
    disable_early_convergence: bool = False  # If True, run full timesteps (for debug)
    enable_debug_logging: bool = False  # If True, print verbose step-by-step logs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "env": {
                "N": self.env.N,
                "M": self.env.M,
                "arena_size": self.env.arena_size,
                "v_max": self.env.v_max,
                "v_enemy": self.env.v_enemy,
                "motion_mode": self.env.motion_mode,
                "num_bands": self.env.num_bands,
                "eps": self.env.eps,
                "min_samples": self.env.min_samples,
                "k_recompute": self.env.k_recompute,
                "max_steps": self.env.max_steps,
                "tx_power_dbm": self.env.tx_power_dbm,
                "sensitivity_dbm": self.env.sensitivity_dbm,
                "jammer_power_dbm": self.env.jammer_power_dbm,
                "reward_weights": self.env.reward_weights
            },
            "network": {
                "obs_dim": self.network.obs_dim,
                "hidden_dim": self.network.hidden_dim,
                "log_std_min": self.network.log_std_min,
                "log_std_max": self.network.log_std_max
            },
            "ppo": {
                "gamma": self.ppo.gamma,
                "gae_lambda": self.ppo.gae_lambda,
                "clip_eps": self.ppo.clip_eps,
                "lr_actor": self.ppo.lr_actor,
                "lr_critic": self.ppo.lr_critic,
                "c1": self.ppo.c1,
                "c2": self.ppo.c2,
                "n_epochs": self.ppo.n_epochs,
                "batch_size": self.ppo.batch_size,
                "max_grad_norm": self.ppo.max_grad_norm,
                "rollout_length": self.ppo.rollout_length
            },
            "total_timesteps": self.total_timesteps,
            "log_interval": self.log_interval,
            "save_interval": self.save_interval,
            "eval_interval": self.eval_interval,
            "eval_episodes": self.eval_episodes,
            "experiment_name": self.experiment_name,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "device": self.device,
            "target_reduction": self.target_reduction,
            "convergence_window": self.convergence_window,
            "early_stop_patience": self.early_stop_patience
        }
    
    def save(self, path: str):
        """Save configuration to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load configuration from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        
        config = cls()
        
        # Environment
        if "env" in data:
            for k, v in data["env"].items():
                if hasattr(config.env, k):
                    setattr(config.env, k, v)
        
        # Network
        if "network" in data:
            for k, v in data["network"].items():
                if hasattr(config.network, k):
                    setattr(config.network, k, v)
        
        # PPO
        if "ppo" in data:
            for k, v in data["ppo"].items():
                if hasattr(config.ppo, k):
                    setattr(config.ppo, k, v)
        
        # Top-level
        for k in ["total_timesteps", "log_interval", "save_interval", 
                  "eval_interval", "eval_episodes", "experiment_name",
                  "output_dir", "seed", "device", "target_reduction",
                  "convergence_window", "early_stop_patience"]:
            if k in data:
                setattr(config, k, data[k])
        
        return config


# =============================================================================
# PRESETS
# =============================================================================

def get_debug_config() -> TrainingConfig:
    """Fast config for debugging."""
    config = TrainingConfig()
    config.env.N = 5
    config.env.M = 2
    config.env.max_steps = 50
    config.ppo.rollout_length = 128
    config.ppo.batch_size = 32
    config.ppo.n_epochs = 2
    config.total_timesteps = 1000
    config.log_interval = 1
    config.save_interval = 2
    config.eval_interval = 2
    config.eval_episodes = 2
    config.disable_early_convergence = True  # Run full timesteps in debug
    config.enable_debug_logging = True  # Verbose step-by-step logging
    return config


def get_fast_config() -> TrainingConfig:
    """Faster training for initial experiments."""
    config = TrainingConfig()
    config.env.arena_size = 200.0  # Larger arena - harder
    config.env.jammer_power_dbm = 20.0  # Match enemy TX power
    config.env.jam_thresh_dbm = -40.0  # Realistic threshold: R_jam ~ 30-50m
    config.env.random_jammer_start = True  # Start at random positions
    config.ppo.rollout_length = 2048  # Longer rollouts for better learning
    config.ppo.n_epochs = 10  # More epochs as per guide
    config.ppo.lr_actor = 1e-4  # Lower LR for stability
    config.ppo.lr_critic = 3e-4  # Lower LR for stability
    config.ppo.batch_size = 128  # Smaller batches, more updates
    config.total_timesteps = 100_000
    config.disable_early_convergence = True  # Run full timesteps
    return config


def get_full_config() -> TrainingConfig:
    """Full training configuration."""
    config = TrainingConfig()
    config.env.arena_size = 200.0  # Larger arena
    config.env.jammer_power_dbm = 20.0  # Realistic jammer power
    config.env.jam_thresh_dbm = -40.0  # Realistic threshold: R_jam ~ 30-50m
    config.env.random_jammer_start = True  # Start at random positions
    config.ppo.rollout_length = 2048  # As per guide
    config.ppo.n_epochs = 10  # As per guide
    config.ppo.lr_actor = 1e-4  # Lower LR for stability
    config.ppo.lr_critic = 3e-4  # Lower LR for stability
    config.ppo.batch_size = 256  # As per guide
    config.disable_early_convergence = True  # Run full timesteps
    return config


def get_large_scale_config() -> TrainingConfig:
    """DEMO CONFIG - Designed for clear learning curve.
    
    This configuration shows visible learning progression:
    - Random placement: ~30-40% L2 reduction
    - Trained policy: ~85-95% L2 reduction
    - Clear upward trend in L2 reduction over training
    
    Physics: jam_thresh=-35dBm at 2.4GHz with 30dBm jammer → ~18m range
    With only 6 jammers, random placement covers ~30% of arena
    Trained policy must coordinate to cover ~90%
    """
    config = TrainingConfig()
    
    # HARD task - few jammers, must coordinate precisely
    config.env.N = 30               # 30 enemies (moderate)
    config.env.M = 6                # ONLY 6 jammers - must be very strategic
    config.env.arena_size = 150.0   # Compact arena
    
    # TIGHT jamming threshold - ~18m range requires precise positioning
    config.env.jam_thresh_dbm = -35.0
    
    # Random start - MUST learn navigation
    config.env.random_jammer_start = True
    
    # PURE L2 reward scaled to fit [-10, +10] after clipping
    # omega_1=10 means L2_reduction of 1.0 gives reward=10.0 (max clipped)
    config.env.reward_weights = {
        "lambda2_reduction": 10.0,  # Scales [0,1] -> [0,10]
        "band_match": 0.0,
        "proximity": 0.0,
        "energy": 0.0,
        "overlap": 0.0
    }
    
    # DBSCAN for compact scenario
    config.env.eps = 25.0           # Clustering radius
    config.env.min_samples = 2
    
    # Episode length
    config.env.max_steps = 150      # Shorter episodes, faster learning
    
    # Compact network
    config.network.hidden_dim = 64
    
    # AGGRESSIVE training for fast convergence
    config.total_timesteps = 2_000_000
    config.ppo.rollout_length = 1024    # Smaller rollouts, more updates
    config.ppo.batch_size = 128
    config.ppo.lr_actor = 3e-4          # Higher LR for faster learning
    config.ppo.lr_critic = 1e-3
    config.ppo.n_epochs = 15            # More epochs per update
    config.ppo.c2 = 0.01                # Some exploration
    config.ppo.clip_eps = 0.2
    config.ppo.max_grad_norm = 0.5
    
    # Wider action variance for exploration
    config.network.log_std_min = -2.0
    config.network.log_std_max = 0.5
    
    config.disable_early_convergence = True
    
    return config


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_config() -> dict:
    """Verify configuration module."""
    results = {}
    
    # Test 1: Default creation
    config = TrainingConfig()
    results["test_default_creation"] = {
        "N": config.env.N,
        "M": config.env.M,
        "gamma": config.ppo.gamma,
        "pass": config.env.N == 10 and config.ppo.gamma == 0.99
    }
    
    # Test 2: To dict
    d = config.to_dict()
    results["test_to_dict"] = {
        "has_env": "env" in d,
        "has_ppo": "ppo" in d,
        "pass": "env" in d and "ppo" in d
    }
    
    # Test 3: Save and load
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config.save(f.name)
        loaded = TrainingConfig.load(f.name)
        results["test_save_load"] = {
            "N_match": config.env.N == loaded.env.N,
            "gamma_match": config.ppo.gamma == loaded.ppo.gamma,
            "pass": config.env.N == loaded.env.N
        }
    
    # Test 4: Presets
    debug = get_debug_config()
    fast = get_fast_config()
    full = get_full_config()
    results["test_presets"] = {
        "debug_timesteps": debug.total_timesteps,
        "fast_timesteps": fast.total_timesteps,
        "full_timesteps": full.total_timesteps,
        "pass": debug.total_timesteps < fast.total_timesteps < full.total_timesteps
    }
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Training Config Verification")
    print("=" * 60)
    
    results = verify_config()
    
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
