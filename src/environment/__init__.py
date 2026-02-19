"""
Environment Module
==================

Gymnasium-compatible environment for MARL jammer drone simulation.
    - enemy_swarm: Dynamic enemy motion models (random walk, coordinated turn)
    - observation: 5-dimensional observation builder
    - action_space: Hybrid continuous + discrete action handling
    - reward: 5-term reward function with lambda-2
    - jammer_env: Main Gym environment
"""

from .enemy_swarm import (
    EnemySwarm,
    create_clustered_swarm,
    create_formation_swarm,
)
from .observation import ObservationBuilder, build_global_observation
from .action_space import ActionHandler
from .reward import RewardCalculator, RewardComponents
from .jammer_env import JammerEnv

__all__ = [
    "EnemySwarm",
    "create_clustered_swarm",
    "create_formation_swarm",
    "ObservationBuilder",
    "build_global_observation",
    "ActionHandler",
    "RewardCalculator",
    "RewardComponents",
    "JammerEnv",
]
