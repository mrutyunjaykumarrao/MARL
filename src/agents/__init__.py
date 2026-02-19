"""
Agents Module
=============

Neural network architectures for MARL-PPO.
    - actor: Actor network with shared trunk and dual heads
    - critic: Critic network with mean-pooled input
    - rollout_buffer: Experience buffer with GAE computation
    - ppo_agent: Combined PPO agent with parameter sharing
"""

from .actor import Actor, ActorNumpy, verify_actor
from .critic import Critic, CriticNumpy, ActorCritic, verify_critic
from .rollout_buffer import RolloutBuffer, RolloutBufferSamples, verify_rollout_buffer
from .ppo_agent import PPOAgent, verify_ppo_agent

__all__ = [
    # Actor
    "Actor",
    "ActorNumpy",
    "verify_actor",
    # Critic
    "Critic",
    "CriticNumpy",
    "ActorCritic",
    "verify_critic",
    # Buffer
    "RolloutBuffer",
    "RolloutBufferSamples",
    "verify_rollout_buffer",
    # PPO
    "PPOAgent",
    "verify_ppo_agent",
]
