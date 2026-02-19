#!/usr/bin/env python
"""
MARL Jammer Deployment Module
=============================

This module provides the minimal code required for deploying the trained
MARL jammer system. It contains only what's needed for inference - no training code.

Usage:
    # Minimal inference
    from deploy import JammerController
    controller = JammerController.load("outputs/deployment/actor_state_dict.pt")
    
    # Get action from sensor data
    actions = controller.get_actions(observations)

Author: MARL Jammer Team
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import json

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MinimalActor(nn.Module):
    """
    Minimal actor network for deployment.
    
    This is a stripped-down version containing only the inference code.
    No training utilities, distributions, or logging.
    
    Architecture (matches training):
        Input:  [5]  - [Δx, Δy, vx, vy, enemy_band]
        Hidden: [128, 128] with LayerNorm + ReLU
        Output: [2, 4] - velocity (vx, vy) + band logits
    """
    
    def __init__(
        self,
        obs_dim: int = 5,
        hidden_dim: int = 128,
        v_max: float = 5.0,
        num_bands: int = 4
    ):
        super().__init__()
        
        self.v_max = v_max
        
        # Shared trunk (must match training architecture)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Output heads
        self.mu_head = nn.Linear(hidden_dim, 2)      # Velocity mean
        self.band_head = nn.Linear(hidden_dim, num_bands)  # Band logits
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for deployment (deterministic).
        
        Args:
            obs: Observation tensor, shape (batch, 5)
            
        Returns:
            velocity: Velocity actions, shape (batch, 2)
            band: Selected band indices, shape (batch,)
        """
        x = self.trunk(obs)
        
        # Get velocity (deterministic - use mean directly)
        velocity = self.mu_head(x)
        velocity = torch.clamp(velocity, -self.v_max, self.v_max)
        
        # Get band (argmax for deterministic selection)
        band_logits = self.band_head(x)
        band = torch.argmax(band_logits, dim=-1)
        
        return velocity, band


class JammerController:
    """
    High-level controller for jammer drones in deployment.
    
    This class wraps the neural network and provides a clean interface
    for integration with drone control systems.
    
    Example:
        >>> controller = JammerController.load("actor_state_dict.pt")
        >>> 
        >>> # Each jammer gets an observation from sensors
        >>> obs = np.array([
        ...     [0.2, -0.1, 0.5, 0.0, 0.66],  # Jammer 1
        ...     [-.3,  0.2, 0.0, 0.5, 0.66],  # Jammer 2
        ...     # ... etc
        ... ])
        >>> 
        >>> velocities, bands = controller.get_actions(obs)
        >>> 
        >>> # Send to drone controllers
        >>> for i, (vel, band) in enumerate(zip(velocities, bands)):
        ...     drone[i].set_velocity(vel[0], vel[1])
        ...     drone[i].set_frequency_band(band)
    
    Attributes:
        model: The neural network model
        device: Torch device (cpu/cuda)
        config: Deployment configuration
    """
    
    def __init__(
        self,
        model: MinimalActor,
        device: str = "cpu",
        config: Dict[str, Any] = None
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        self.config = config or {}
        
    @classmethod
    def load(
        cls,
        weights_path: str,
        config_path: str = None,
        device: str = "cpu"
    ) -> "JammerController":
        """
        Load controller from trained weights.
        
        Args:
            weights_path: Path to actor_state_dict.pt
            config_path: Optional path to config.json
            device: Device to use ("cpu" or "cuda")
            
        Returns:
            Loaded JammerController ready for inference
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for deployment")
        
        weights_path = Path(weights_path)
        
        # Load config if available
        config = {}
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif (weights_path.parent / "deployment_config.json").exists():
            with open(weights_path.parent / "deployment_config.json", 'r') as f:
                config = json.load(f)
        
        # Get architecture from config or use defaults
        obs_dim = config.get('obs_dim', 5)
        hidden_dim = config.get('hidden_dim', 128)
        v_max = config.get('v_max', 5.0)
        num_bands = config.get('num_bands', 4)
        
        # Create model
        model = MinimalActor(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            v_max=v_max,
            num_bands=num_bands
        )
        
        # Load weights
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        
        return cls(model, device, config)
    
    def get_actions(
        self,
        observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get actions for all jammers given their observations.
        
        Args:
            observations: Array of observations, shape (M, 5)
                Each row: [delta_x, delta_y, velocity_x, velocity_y, enemy_band]
                
        Returns:
            velocities: Velocity commands, shape (M, 2)
            bands: Selected frequency bands, shape (M,)
        """
        # Convert to tensor
        obs_tensor = torch.tensor(
            observations, 
            dtype=torch.float32,
            device=self.device
        )
        
        # Get actions (no gradient computation needed)
        with torch.no_grad():
            velocities, bands = self.model(obs_tensor)
        
        # Convert back to numpy
        return velocities.cpu().numpy(), bands.cpu().numpy()
    
    def get_single_action(
        self,
        observation: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        Get action for a single jammer.
        
        Args:
            observation: Single observation, shape (5,)
            
        Returns:
            velocity: (vx, vy) velocity command
            band: Selected frequency band
        """
        obs = observation.reshape(1, -1)
        velocities, bands = self.get_actions(obs)
        return velocities[0], int(bands[0])


def prepare_deployment_artifacts(
    experiment_dir: str,
    output_dir: str = "outputs/deployment"
) -> Dict[str, str]:
    """
    Prepare deployment artifacts from a trained experiment.
    
    This function extracts only the essential files needed for deployment
    from a full training output directory.
    
    Args:
        experiment_dir: Path to experiment output (e.g., "outputs/my_experiment")
        output_dir: Path to deployment output directory
        
    Returns:
        Dict mapping artifact names to their paths
    """
    import shutil
    
    exp_dir = Path(experiment_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    artifacts = {}
    
    # Copy actor weights (REQUIRED)
    actor_src = exp_dir / "actor_state_dict.pt"
    if actor_src.exists():
        actor_dst = out_dir / "actor_state_dict.pt"
        shutil.copy2(actor_src, actor_dst)
        artifacts["actor_weights"] = str(actor_dst)
        print(f"[+] Copied actor weights: {actor_dst}")
    else:
        print(f"[!] WARNING: Actor weights not found at {actor_src}")
    
    # Create deployment config (minimal)
    config_src = exp_dir / "config.json"
    if config_src.exists():
        with open(config_src, 'r') as f:
            full_config = json.load(f)
        
        # Extract only deployment-relevant config
        deploy_config = {
            "obs_dim": full_config.get("network", {}).get("obs_dim", 5),
            "hidden_dim": full_config.get("network", {}).get("hidden_dim", 128),
            "v_max": full_config.get("env", {}).get("v_max", 5.0),
            "num_bands": full_config.get("env", {}).get("num_bands", 4),
            "M": full_config.get("env", {}).get("M", 4),
            "arena_size": full_config.get("env", {}).get("arena_size", 200.0),
        }
        
        config_dst = out_dir / "deployment_config.json"
        with open(config_dst, 'w') as f:
            json.dump(deploy_config, f, indent=2)
        artifacts["config"] = str(config_dst)
        print(f"[+] Created deployment config: {config_dst}")
    
    # Copy training summary if available
    stats_src = exp_dir / "final_stats.json"
    if stats_src.exists():
        stats_dst = out_dir / "training_summary.json"
        shutil.copy2(stats_src, stats_dst)
        artifacts["summary"] = str(stats_dst)
        print(f"[+] Copied training summary: {stats_dst}")
    
    # Create README
    readme_content = f"""# MARL Jammer Deployment Package

## Contents
- `actor_state_dict.pt` - Trained neural network weights (~100KB)
- `deployment_config.json` - Model configuration
- `training_summary.json` - Training statistics

## Quick Start

```python
from deploy import JammerController

# Load the trained model
controller = JammerController.load("actor_state_dict.pt")

# Get actions for M jammers
observations = get_sensor_data()  # Shape: (M, 5)
velocities, bands = controller.get_actions(observations)

# Send to drone controllers
for i in range(M):
    drone[i].set_velocity(velocities[i])
    drone[i].set_frequency(bands[i])
```

## Observation Format
Each row: `[delta_x, delta_y, velocity_x, velocity_y, enemy_band]`
- delta_x, delta_y: Relative position to target centroid (normalized)
- velocity_x, velocity_y: Current velocity (normalized)
- enemy_band: Enemy frequency band (0-1 normalized)

## Training Info
- Experiment: {exp_dir.name}
- Generated: {np.datetime64('now')}

"""
    readme_path = out_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    artifacts["readme"] = str(readme_path)
    print(f"[+] Created README: {readme_path}")
    
    print(f"\n[SUCCESS] Deployment artifacts prepared in: {out_dir}")
    print(f"[INFO] Total size: {sum(Path(p).stat().st_size for p in artifacts.values() if Path(p).exists()) / 1024:.1f} KB")
    
    return artifacts


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """CLI for deployment utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MARL Jammer Deployment Utilities"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Prepare command
    prep_parser = subparsers.add_parser('prepare', help='Prepare deployment artifacts')
    prep_parser.add_argument('--experiment', '-e', required=True,
                            help='Path to experiment directory')
    prep_parser.add_argument('--output', '-o', default='outputs/deployment',
                            help='Output directory for deployment artifacts')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test deployment')
    test_parser.add_argument('--weights', '-w', required=True,
                            help='Path to actor weights')
    test_parser.add_argument('--config', '-c', default=None,
                            help='Path to deployment config')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        prepare_deployment_artifacts(args.experiment, args.output)
        
    elif args.command == 'test':
        print("Testing deployment...")
        controller = JammerController.load(args.weights, args.config)
        
        # Create dummy observations (4 agents)
        obs = np.random.randn(4, 5).astype(np.float32)
        obs[:, 4] = 0.66  # Normalize band
        
        velocities, bands = controller.get_actions(obs)
        
        print(f"\nTest Results:")
        print(f"  Observations: {obs.shape}")
        print(f"  Velocities: {velocities}")
        print(f"  Bands: {bands}")
        print(f"\n[SUCCESS] Deployment test passed!")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
