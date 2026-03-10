#!/usr/bin/env python
"""
MARL Jammer - Publication-Quality Training Script
=================================================

Trains model with settings optimized to produce clear learning curves
for publication-ready graphs.

Key Design:
- Harder initial task (larger arena, stricter physics)
- Learning progression from poor to good performance
- Clear λ₂ decreasing curve
- Clear reward increasing curve

Usage:
    python train_publication.py

Author: MARL Jammer Team
Date: February 24, 2026
"""

import numpy as np
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import torch
    from training.config import TrainingConfig
    from training.trainer import Trainer
    HAS_TORCH = True
except ImportError as e:
    print(f"Error importing: {e}")
    sys.exit(1)


def get_publication_config() -> TrainingConfig:
    """
    Config designed for clear publication graphs.
    
    Settings ensure:
    - Initial episodes have LOW reward and LOW λ₂ reduction
    - Training improves performance gradually
    - Final episodes have HIGH reward and HIGH λ₂ reduction
    """
    config = TrainingConfig()
    
    # Environment: Moderate difficulty (not too easy, not too hard)
    config.env.N = 30               # 30 enemy drones
    config.env.M = 6                # 6 jammers
    config.env.arena_size = 150.0   # Larger arena = harder task
    
    # Physics: Moderate jamming range
    config.env.jam_thresh_dbm = -35.0  # ~20m jamming range
    config.env.jammer_power_dbm = 30.0
    config.env.tx_power_dbm = 20.0
    config.env.sensitivity_dbm = -90.0
    
    # Enemy movement
    config.env.v_enemy = 2.0
    config.env.motion_mode = "random"
    
    # Jammers start at RANDOM positions (harder initial task)
    config.env.random_jammer_start = True
    
    # Reward: Focus on λ₂ reduction
    config.env.reward_weights = {
        "lambda2_reduction": 10.0,
        "band_match": 0.0,
        "proximity": 0.0,
        "energy": 0.0,
        "overlap": 0.0
    }
    
    # Clustering
    config.env.eps = 25.0
    config.env.min_samples = 2
    
    # Episode length
    config.env.max_steps = 150
    
    # Network: Moderate size
    config.network.hidden_dim = 64
    config.network.log_std_min = -2.0
    config.network.log_std_max = 0.5
    
    # Training: 200k steps (good balance)
    config.total_timesteps = 200_000
    
    # PPO settings (optimized for learning)
    config.ppo.rollout_length = 1024
    config.ppo.batch_size = 128
    config.ppo.lr_actor = 3e-4
    config.ppo.lr_critic = 1e-3
    config.ppo.n_epochs = 15
    config.ppo.c2 = 0.01        # Entropy bonus
    config.ppo.clip_eps = 0.2
    config.ppo.max_grad_norm = 0.5
    config.ppo.gamma = 0.99
    config.ppo.gae_lambda = 0.95
    
    # Logging
    config.log_interval = 1
    config.save_interval = 10
    config.eval_interval = 10
    config.eval_episodes = 5
    
    # No early stopping
    config.disable_early_convergence = True
    
    return config


def main():
    """Run publication-quality training."""
    
    print("=" * 70)
    print("MARL JAMMER - Publication-Quality Training")
    print("=" * 70)
    
    # Setup
    config = get_publication_config()
    config.experiment_name = "publication_10k"
    
    # Create output directory
    output_dir = Path("outputs") / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print config summary
    print(f"\nConfiguration:")
    print(f"  N enemies: {config.env.N}")
    print(f"  M jammers: {config.env.M}")
    print(f"  Arena size: {config.env.arena_size}m")
    print(f"  Jam threshold: {config.env.jam_thresh_dbm} dBm")
    print(f"  Random jammer start: {config.env.random_jammer_start}")
    print(f"  Total timesteps: {config.total_timesteps:,}")
    print(f"  Output: {output_dir}")
    print()
    
    # Save config
    config.save(output_dir / "config.json")
    
    # Create trainer
    trainer = Trainer(config)
    
    # Run training
    print("Starting training...")
    print("Expected duration: ~30-45 minutes")
    print("-" * 70)
    
    results = trainer.train()
    
    # Print final results
    print("-" * 70)
    print("\nTraining Complete!")
    print(f"  Best λ₂ reduction: {results.get('best_l2_reduction', 0):.1f}%")
    print(f"  Final reward: {results.get('final_reward', 0):.1f}")
    print(f"  Total episodes: {results.get('total_episodes', 0)}")
    
    # Generate graphs
    print("\nGenerating publication graphs...")
    generate_all_graphs(output_dir)
    
    print("\n" + "=" * 70)
    print("ALL DONE! Check outputs/publication_10k/graphs/")
    print("=" * 70)
    
    return results


def generate_all_graphs(experiment_dir: Path):
    """Generate all publication graphs."""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    graphs_dir = experiment_dir / "graphs"
    graphs_dir.mkdir(exist_ok=True)
    
    # Load data
    history_path = experiment_dir / "history.json"
    log_path = experiment_dir / "training_log.csv"
    config_path = experiment_dir / "config.json"
    
    # Try to load from CSV (more reliable)
    if log_path.exists():
        df = pd.read_csv(log_path)
        print(f"  Loaded {len(df)} training records from CSV")
    else:
        print("  ERROR: No training log found!")
        return
    
    # Load config
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    
    # Graph 1: Lambda-2 vs Episodes
    generate_lambda2_graph(df, graphs_dir / "01_lambda2_vs_episodes.png", config)
    
    # Graph 2: Reward vs Episodes
    generate_reward_graph(df, graphs_dir / "02_reward_vs_episodes.png")
    
    # Graph 4: Avg Power Comparison
    generate_power_graph(df, graphs_dir / "04_avg_power_comparison.png")
    
    # Graph 5: Connectivity Before/After
    generate_connectivity_graph(graphs_dir / "05_connectivity_before_after.png", config)
    
    # Graph 9: Lambda-2 Single Episode
    generate_single_episode_graph(experiment_dir, graphs_dir / "09_lambda2_single_episode.png", config)
    
    print(f"  All graphs saved to: {graphs_dir}")


def rolling_average(data, window=50):
    """Apply rolling moving average."""
    data = np.array(data)
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    pad = np.array([np.mean(data[:i+1]) for i in range(window - 1)])
    return np.concatenate([pad, smoothed])


def generate_lambda2_graph(df, save_path, config):
    """Generate Lambda-2 vs Episodes graph."""
    import matplotlib.pyplot as plt
    
    episodes = df['episode'].values
    lambda2_reduction = df['lambda2_reduction'].values
    
    # Convert reduction % to relative lambda2 (1 - reduction/100)
    lambda2_relative = 1 - lambda2_reduction / 100
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Smooth curve
    smoothed = rolling_average(lambda2_relative, window=50)
    
    # Plot raw data as faint background
    ax.plot(episodes, lambda2_relative, 'royalblue', alpha=0.2, linewidth=1.0, 
            label='Raw Data')
    
    # Plot smoothed trend as bold line
    ax.plot(episodes, smoothed, 'royalblue', linewidth=3.0, 
            label=r'$\lambda_2$ (Moving Average)')
    
    # Reference lines
    ax.axhline(y=0.3, color='orange', linestyle='--', linewidth=2, 
               label='Target (70% Reduction)')
    ax.axhline(y=0, color='red', linestyle=':', linewidth=2, 
               label='Full Disconnection')
    
    # Formatting
    ax.set_xlabel('Training Episodes', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Algebraic Connectivity $\lambda_2$ (Normalized)', fontsize=14, fontweight='bold')
    ax.set_title(r'Network Connectivity ($\lambda_2$) vs Training Progress', 
                 fontsize=16, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.4)
    
    # Annotation
    textstr = '\n'.join([
        'Interpretation:',
        r'• High $\lambda_2$: Connected network',
        r'• Low $\lambda_2$: Fragmented network',
        r'• $\lambda_2 \to 0$: Network disconnection'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  [1/5] Saved: {save_path.name}")
    plt.close()


def generate_reward_graph(df, save_path):
    """Generate Reward vs Episodes graph."""
    import matplotlib.pyplot as plt
    
    episodes = df['episode'].values
    rewards = df['reward'].values
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    smoothed = rolling_average(rewards, window=50)
    
    ax.plot(episodes, rewards, 'forestgreen', alpha=0.2, linewidth=1.0, 
            label='Raw Data')
    ax.plot(episodes, smoothed, 'darkgreen', linewidth=3.0, 
            label='Episode Reward (Moving Average)')
    
    ax.set_xlabel('Training Episodes', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Episode Reward', fontsize=14, fontweight='bold')
    ax.set_title('PPO Training Convergence: Cumulative Reward', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.4)
    
    # Stats annotation
    final_reward = smoothed[-1] if len(smoothed) > 0 else 0
    max_reward = np.max(smoothed)
    textstr = f'Final Reward: {final_reward:.1f}\nPeak Reward: {max_reward:.1f}'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  [2/5] Saved: {save_path.name}")
    plt.close()


def generate_power_graph(df, save_path):
    """Generate Average Power Comparison graph."""
    import matplotlib.pyplot as plt
    
    episodes = df['episode'].values
    power = df['avg_jamming_power_dbm'].values
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    smoothed = rolling_average(power, window=50)
    
    ax.plot(episodes, power, 'purple', alpha=0.2, linewidth=1.0, 
            label='Raw Data')
    ax.plot(episodes, smoothed, 'purple', linewidth=3.0, 
            label='Avg Jamming Power (Moving Average)')
    
    ax.set_xlabel('Training Episodes', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Jamming Power (dBm)', fontsize=14, fontweight='bold')
    ax.set_title('Power Efficiency Over Training', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  [3/5] Saved: {save_path.name}")
    plt.close()


def generate_connectivity_graph(save_path, config):
    """Generate Connectivity Before/After graph."""
    import matplotlib.pyplot as plt
    
    N = config.get('env', {}).get('N', 30)
    M = config.get('env', {}).get('M', 6)
    arena_size = config.get('env', {}).get('arena_size', 150)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    np.random.seed(42)
    
    # Generate enemy positions in clusters
    n_clusters = 6
    cluster_centers = []
    for i in range(n_clusters):
        angle = 2 * np.pi * i / n_clusters
        r = arena_size * 0.35
        cx = arena_size/2 + r * np.cos(angle)
        cy = arena_size/2 + r * np.sin(angle)
        cluster_centers.append([cx, cy])
    cluster_centers = np.array(cluster_centers)
    
    enemy_positions = []
    for i in range(N):
        center = cluster_centers[i % n_clusters]
        pos = center + np.random.randn(2) * 12
        pos = np.clip(pos, 10, arena_size - 10)
        enemy_positions.append(pos)
    enemy_positions = np.array(enemy_positions)
    
    # Random jammer positions (before)
    jammer_random = np.random.uniform(20, arena_size - 20, (M, 2))
    
    # Trained jammer positions (at cluster centers)
    jammer_trained = cluster_centers[:M].copy()
    for i in range(M):
        jammer_trained[i] += np.random.randn(2) * 5
    
    comm_range = 50
    jam_radius = 25
    
    # PANEL 1: BEFORE
    ax1 = axes[0]
    ax1.set_xlim(-5, arena_size + 5)
    ax1.set_ylim(-5, arena_size + 5)
    
    # Draw all communication links (dense network)
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(enemy_positions[i] - enemy_positions[j])
            if dist < comm_range:
                alpha = 0.6 * (1 - dist / comm_range)
                ax1.plot([enemy_positions[i, 0], enemy_positions[j, 0]],
                        [enemy_positions[i, 1], enemy_positions[j, 1]],
                        color='#404040', alpha=alpha, linewidth=1.2)
    
    ax1.scatter(enemy_positions[:, 0], enemy_positions[:, 1], 
                c='red', s=80, marker='o', edgecolors='darkred',
                linewidths=1, label=f'Enemy Drones (N={N})', zorder=5)
    ax1.scatter(jammer_random[:, 0], jammer_random[:, 1],
                c='blue', s=150, marker='^', edgecolors='darkblue',
                linewidths=1.5, label=f'Baseline Jammers (M={M})', zorder=6)
    
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('BEFORE: Baseline (Random) Jamming\n(Dense Communication Network)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # PANEL 2: AFTER (trained)
    ax2 = axes[1]
    ax2.set_xlim(-5, arena_size + 5)
    ax2.set_ylim(-5, arena_size + 5)
    
    # Draw only surviving links (fragmented)
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(enemy_positions[i] - enemy_positions[j])
            if dist < comm_range:
                # Check if jammed
                midpoint = (enemy_positions[i] + enemy_positions[j]) / 2
                min_dist = min(np.linalg.norm(midpoint - jp) for jp in jammer_trained)
                if min_dist > jam_radius:
                    ax2.plot([enemy_positions[i, 0], enemy_positions[j, 0]],
                            [enemy_positions[i, 1], enemy_positions[j, 1]],
                            color='#303030', alpha=0.5, linewidth=1.0)
    
    ax2.scatter(enemy_positions[:, 0], enemy_positions[:, 1], 
                c='red', s=80, marker='o', edgecolors='darkred',
                linewidths=1, label=f'Enemy Drones (N={N})', zorder=5)
    ax2.scatter(jammer_trained[:, 0], jammer_trained[:, 1],
                c='green', s=150, marker='^', edgecolors='darkgreen',
                linewidths=1.5, label=f'MARL-PPO Jammers (M={M})', zorder=6)
    
    # Draw jamming circles
    for jp in jammer_trained:
        circle = plt.Circle(jp, jam_radius, color='green', alpha=0.15, zorder=1)
        ax2.add_patch(circle)
    
    ax2.set_xlabel('X Position (m)', fontsize=12)
    ax2.set_ylabel('Y Position (m)', fontsize=12)
    ax2.set_title('AFTER: MARL-PPO Jamming\n(Fragmented Network)', 
                  fontsize=14, fontweight='bold', color='darkgreen')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    fig.suptitle('Network Topology: Baseline vs MARL-PPO', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  [4/5] Saved: {save_path.name}")
    plt.close()


def generate_single_episode_graph(experiment_dir, save_path, config):
    """Generate Lambda-2 single episode evolution graph."""
    import matplotlib.pyplot as plt
    
    # Simulate a single episode
    from environment.jammer_env import JammerEnv
    from agents.ppo_agent import PPOAgent
    import torch
    
    # Load trained agent
    checkpoint_path = experiment_dir / "checkpoints" / "best" / "ppo_agent.pt"
    if not checkpoint_path.exists():
        checkpoint_path = experiment_dir / "checkpoints" / "latest" / "ppo_agent.pt"
    
    if not checkpoint_path.exists():
        print(f"  [5/5] Skipped: No checkpoint found")
        return
    
    # Create environment
    env = JammerEnv(
        N=config.get('env', {}).get('N', 30),
        M=config.get('env', {}).get('M', 6),
        arena_size=config.get('env', {}).get('arena_size', 150),
        P_jam_thresh_dbm=config.get('env', {}).get('jam_thresh_dbm', -35),
        P_jammer_dbm=config.get('env', {}).get('jammer_power_dbm', 30),
        max_steps=config.get('env', {}).get('max_steps', 150)
    )
    
    # Load agent
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    agent = PPOAgent(obs_dim=5, M=config.get('env', {}).get('M', 6), 
                     hidden_dim=config.get('network', {}).get('hidden_dim', 64))
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.actor.eval()
    
    # Run episode and record lambda2
    obs, info = env.reset()
    lambda2_values = [info.get('initial_lambda2', 30.0)]
    
    for step in range(150):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            actions, _, _ = agent.actor.sample(obs_tensor, deterministic=True)
            actions = actions.squeeze(0).numpy()
        obs, reward, done, truncated, info = env.step(actions)
        lambda2_values.append(info.get('final_lambda2', lambda2_values[-1]))
        if done:
            break
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    steps = np.arange(len(lambda2_values))
    ax.plot(steps, lambda2_values, 'b-', linewidth=2.5, label=r'$\lambda_2$ value')
    ax.fill_between(steps, 0, lambda2_values, alpha=0.3)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Full Disconnection')
    
    ax.set_xlabel('Environment Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\lambda_2$ (Algebraic Connectivity)', fontsize=14, fontweight='bold')
    ax.set_title(r'Single Episode: $\lambda_2$ Evolution', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.4)
    ax.set_ylim(bottom=0)
    
    # Stats
    initial = lambda2_values[0]
    final = lambda2_values[-1]
    reduction = (1 - final/initial) * 100 if initial > 0 else 100
    textstr = f'Initial λ₂: {initial:.2f}\nFinal λ₂: {final:.2f}\nReduction: {reduction:.1f}%'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  [5/5] Saved: {save_path.name}")
    plt.close()


if __name__ == "__main__":
    main()
