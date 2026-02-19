"""
MARL Jammer Drone System
========================

Multi-Agent Reinforcement Learning based Cooperative Jammer Drone System
for disrupting enemy drone swarm communication using Graph Laplacian reward.

Modules:
    - physics: FSPL calculations, communication graph, jamming disruption
    - clustering: DBSCAN spatial clustering
    - environment: Gym environment with dynamic enemy swarm
    - agents: Actor-Critic networks with PPO
    - training: PPO training pipeline with GAE
    - evaluation: Metrics, baselines, ablation studies
    - visualization: Plots, heatmaps, animations
"""

__version__ = "2.0.0"
__author__ = "MARL Jammer Team"
