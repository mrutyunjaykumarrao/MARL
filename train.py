#!/usr/bin/env python
"""
MARL Jammer Training Script
============================

Main entry point for training the cooperative jammer drone system.

Usage:
    # Quick debug run
    python train.py --mode debug
    
    # Fast experiment
    python train.py --mode fast --name my_experiment
    
    # Full training
    python train.py --mode full --name production_run
    
    # Custom config
    python train.py --config my_config.json
    
    # Large scale (N=100, M=40)
    python train.py --mode large --name scale_test

Author: MARL Jammer Team
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MARL Jammer Drone System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --mode debug              # Quick test (1000 steps)
  python train.py --mode fast               # Fast training (100K steps)  
  python train.py --mode full               # Full training (2M steps)
  python train.py --config my_config.json   # Custom configuration
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["debug", "fast", "full", "large"],
        default="fast",
        help="Training mode preset (default: fast)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config JSON file"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: from config)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use (default: auto)"
    )
    
    # Environment overrides
    parser.add_argument("--N", type=int, default=None, help="Number of enemies")
    parser.add_argument("--M", type=int, default=None, help="Number of jammers")
    parser.add_argument("--steps", type=int, default=None, help="Total timesteps")
    
    return parser.parse_args()


def get_config(args):
    """Get training configuration from args."""
    from training.config import (
        TrainingConfig,
        get_debug_config,
        get_fast_config,
        get_full_config,
        get_large_scale_config
    )
    
    # Load from file or preset
    if args.config:
        config = TrainingConfig.load(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        config_map = {
            "debug": get_debug_config,
            "fast": get_fast_config,
            "full": get_full_config,
            "large": get_large_scale_config
        }
        config = config_map[args.mode]()
        print(f"Using preset: {args.mode}")
    
    # Apply overrides
    if args.name:
        config.experiment_name = args.name
    elif not args.config:
        # Auto-generate name
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment_name = f"{args.mode}_{timestamp}"
    
    if args.output:
        config.output_dir = args.output
    
    if args.seed is not None:
        config.seed = args.seed
    
    if args.device:
        config.device = args.device
    
    if args.N is not None:
        config.env.N = args.N
    
    if args.M is not None:
        config.env.M = args.M
    
    if args.steps is not None:
        config.total_timesteps = args.steps
    
    return config


def main():
    """Main training entry point."""
    args = parse_args()
    
    print("=" * 70)
    print("MARL JAMMER DRONE SYSTEM")
    print("Multi-Agent Reinforcement Learning for Cooperative Jamming")
    print("=" * 70)
    print()
    
    # Check PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA: Not available, using CPU")
    except ImportError:
        print("ERROR: PyTorch not installed!")
        print("Please install with: pip install torch")
        return 1
    
    print()
    
    # Get configuration
    config = get_config(args)
    
    # Import trainer
    from training.trainer import Trainer
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    try:
        results = trainer.train()
        
        print("\nFinal Results:")
        print(f"  Best λ₂ reduction: {results['best_lambda2_reduction']:.1f}%")
        print(f"  Total timesteps: {results['total_timesteps']:,}")
        print(f"  Training time: {results['time_elapsed'] / 60:.1f} minutes")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        trainer.save_checkpoint("interrupted")
        trainer.logger.save()
        return 1
    
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
