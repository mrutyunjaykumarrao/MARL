"""
Phase 4 Tests: Actor-Critic Networks + PPO Agent
=================================================

Comprehensive tests for neural network architectures and PPO training.

Test Categories:
    - Actor Network: Architecture, forward pass, action sampling
    - Critic Network: Architecture, value estimation, mean pooling
    - RolloutBuffer: Storage, GAE computation, mini-batches
    - PPO Agent: Training loop, losses, checkpointing

Author: MARL Jammer Team
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Skip all tests if PyTorch not available
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


# =============================================================================
# ACTOR NETWORK TESTS
# =============================================================================

class TestActorArchitecture:
    """Test Actor network architecture."""
    
    def test_actor_creation_default(self):
        """Test Actor creation with default parameters."""
        from agents.actor import Actor
        
        actor = Actor()
        
        assert actor.obs_dim == 5
        assert actor.hidden_dim == 128
        assert actor.v_max == 5.0
        assert actor.num_bands == 4
    
    def test_actor_creation_custom(self):
        """Test Actor creation with custom parameters."""
        from agents.actor import Actor
        
        actor = Actor(obs_dim=7, hidden_dim=256, v_max=10.0, num_bands=6)
        
        assert actor.obs_dim == 7
        assert actor.hidden_dim == 256
        assert actor.v_max == 10.0
        assert actor.num_bands == 6
    
    def test_actor_layers_structure(self):
        """Test Actor has correct layer structure."""
        from agents.actor import Actor
        
        actor = Actor(obs_dim=5, hidden_dim=128)
        
        # Check shared trunk (Sequential module)
        assert hasattr(actor, 'trunk')
        
        # Check continuous head
        assert hasattr(actor, 'mu_head')
        assert hasattr(actor, 'log_std_head')
        
        # Check discrete head
        assert hasattr(actor, 'band_head')
    
    def test_actor_layer_sizes(self):
        """Test Actor layers have correct sizes."""
        from agents.actor import Actor
        
        actor = Actor(obs_dim=5, hidden_dim=128)
        
        # Trunk first layer: 5 -> 128
        assert actor.trunk[0].in_features == 5
        assert actor.trunk[0].out_features == 128
        
        # Trunk third layer: 128 -> 128
        assert actor.trunk[3].in_features == 128
        assert actor.trunk[3].out_features == 128
        
        # Continuous head: 128 -> 2
        assert actor.mu_head.in_features == 128
        assert actor.mu_head.out_features == 2
        
        # Discrete head: 128 -> 4
        assert actor.band_head.in_features == 128
        assert actor.band_head.out_features == 4


class TestActorForwardPass:
    """Test Actor forward pass and outputs."""
    
    def test_forward_single_observation(self):
        """Test forward pass with single observation."""
        from agents.actor import Actor
        
        actor = Actor(obs_dim=5, hidden_dim=64)
        obs = torch.randn(5)
        
        mu, log_std, logits = actor(obs)
        
        assert mu.shape == (2,)
        assert log_std.shape == (2,)
        assert logits.shape == (4,)
    
    def test_forward_batch_observations(self):
        """Test forward pass with batch of observations."""
        from agents.actor import Actor
        
        actor = Actor(obs_dim=5, hidden_dim=64)
        obs = torch.randn(32, 5)
        
        mu, log_std, logits = actor(obs)
        
        assert mu.shape == (32, 2)
        assert log_std.shape == (32, 2)
        assert logits.shape == (32, 4)
    
    def test_log_std_clamping(self):
        """Test log_std is properly clamped."""
        from agents.actor import Actor
        
        actor = Actor(obs_dim=5, hidden_dim=64)
        
        # Forward pass many times
        for _ in range(10):
            obs = torch.randn(16, 5) * 10  # Large values
            _, log_std, _ = actor(obs)
            
            assert (log_std >= -2.0).all()
            assert (log_std <= 2.0).all()
    
    def test_mu_tanh_bounded(self):
        """Test mu is tanh-bounded to [-1, 1]."""
        from agents.actor import Actor
        
        actor = Actor(obs_dim=5, hidden_dim=64, v_max=5.0)
        
        for _ in range(10):
            obs = torch.randn(16, 5) * 10
            mu, _, _ = actor(obs)
            
            # mu should be in [-1, 1] after tanh
            assert (mu >= -1.0).all()
            assert (mu <= 1.0).all()


class TestActorSampling:
    """Test Actor action sampling."""
    
    def test_sample_shape(self):
        """Test sample returns correct shapes."""
        from agents.actor import Actor
        
        actor = Actor(obs_dim=5, hidden_dim=64, v_max=5.0, num_bands=4)
        obs = torch.randn(4, 5)  # 4 agents
        
        actions, log_probs, entropy = actor.sample(obs)
        
        assert actions.shape == (4, 3)  # vx, vy, band
        assert log_probs.shape == (4,)
        assert entropy.shape == (4,)
    
    def test_sample_velocity_bounds(self):
        """Test sampled velocities are within bounds."""
        from agents.actor import Actor
        
        actor = Actor(obs_dim=5, hidden_dim=64, v_max=5.0)
        obs = torch.randn(100, 5)
        
        for _ in range(10):
            actions, _, _ = actor.sample(obs)
            
            # Velocities in columns 0, 1
            velocities = actions[:, :2]
            assert (velocities >= -5.0).all()
            assert (velocities <= 5.0).all()
    
    def test_sample_band_valid(self):
        """Test sampled bands are valid integers."""
        from agents.actor import Actor
        
        actor = Actor(obs_dim=5, hidden_dim=64, num_bands=4)
        obs = torch.randn(100, 5)
        
        for _ in range(10):
            actions, _, _ = actor.sample(obs)
            
            # Band in column 2
            bands = actions[:, 2]
            assert (bands >= 0).all()
            assert (bands < 4).all()
    
    def test_sample_deterministic(self):
        """Test deterministic sampling is reproducible."""
        from agents.actor import Actor
        
        actor = Actor(obs_dim=5, hidden_dim=64)
        obs = torch.randn(4, 5)
        
        actions1, _, _ = actor.sample(obs, deterministic=True)
        actions2, _, _ = actor.sample(obs, deterministic=True)
        
        assert torch.allclose(actions1, actions2)
    
    def test_sample_stochastic_differs(self):
        """Test stochastic sampling produces different results."""
        from agents.actor import Actor
        
        actor = Actor(obs_dim=5, hidden_dim=64)
        obs = torch.randn(4, 5)
        
        # Multiple samples should differ
        samples = [actor.sample(obs, deterministic=False)[0] for _ in range(5)]
        
        # At least one pair should be different
        all_same = all(torch.allclose(samples[0], s) for s in samples[1:])
        assert not all_same


class TestActorEvaluate:
    """Test Actor evaluate_actions method."""
    
    def test_evaluate_returns_correct_shapes(self):
        """Test evaluate_actions returns correct shapes."""
        from agents.actor import Actor
        
        actor = Actor(obs_dim=5, hidden_dim=64)
        obs = torch.randn(32, 5)
        actions = torch.randn(32, 3)
        actions[:, 2] = torch.randint(0, 4, (32,)).float()
        
        log_probs, entropy = actor.evaluate_actions(obs, actions)
        
        assert log_probs.shape == (32,)
        assert entropy.shape == (32,)
    
    def test_evaluate_log_probs_finite(self):
        """Test log probs are finite."""
        from agents.actor import Actor
        
        actor = Actor(obs_dim=5, hidden_dim=64, v_max=5.0)
        obs = torch.randn(32, 5)
        
        # Get valid actions
        actions, _, _ = actor.sample(obs)
        
        log_probs, entropy = actor.evaluate_actions(obs, actions)
        
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(entropy).all()


# =============================================================================
# CRITIC NETWORK TESTS
# =============================================================================

class TestCriticArchitecture:
    """Test Critic network architecture."""
    
    def test_critic_creation_default(self):
        """Test Critic creation with default parameters."""
        from agents.critic import Critic
        
        critic = Critic()
        
        assert critic.obs_dim == 5
        assert critic.hidden_dim == 128
    
    def test_critic_creation_custom(self):
        """Test Critic creation with custom parameters."""
        from agents.critic import Critic
        
        critic = Critic(obs_dim=7, hidden_dim=256)
        
        assert critic.obs_dim == 7
        assert critic.hidden_dim == 256
    
    def test_critic_layers_structure(self):
        """Test Critic has correct layer structure."""
        from agents.critic import Critic
        
        critic = Critic(obs_dim=5, hidden_dim=128)
        
        # Critic uses a Sequential network module
        assert hasattr(critic, 'network')
    
    def test_critic_layer_sizes(self):
        """Test Critic layers have correct sizes."""
        from agents.critic import Critic
        
        critic = Critic(obs_dim=5, hidden_dim=128)
        
        # Network[0]: 5 -> 128
        assert critic.network[0].in_features == 5
        assert critic.network[0].out_features == 128
        
        # Network[2]: 128 -> 128
        assert critic.network[2].in_features == 128
        assert critic.network[2].out_features == 128
        
        # Value head (Network[4]): 128 -> 1
        assert critic.network[4].in_features == 128
        assert critic.network[4].out_features == 1


class TestCriticForwardPass:
    """Test Critic forward pass."""
    
    def test_forward_single_observation(self):
        """Test forward pass with single observation."""
        from agents.critic import Critic
        
        critic = Critic(obs_dim=5, hidden_dim=64)
        obs = torch.randn(1, 5)  # Batch size 1
        
        value = critic(obs)
        
        assert value.shape == (1, 1)
    
    def test_forward_batch_observations(self):
        """Test forward pass with batch of observations."""
        from agents.critic import Critic
        
        critic = Critic(obs_dim=5, hidden_dim=64)
        obs = torch.randn(32, 5)
        
        value = critic(obs)
        
        assert value.shape == (32, 1)
    
    def test_forward_pooled(self):
        """Test forward_pooled with pre-pooled observations."""
        from agents.critic import Critic
        
        critic = Critic(obs_dim=5, hidden_dim=64)
        
        # Simulate mean-pooled observations for 4 agents
        obs = torch.randn(4, 5)
        pooled = obs.mean(dim=0, keepdim=True)
        
        value = critic.forward_pooled(pooled)
        
        assert value.shape == (1, 1)
    
    def test_value_is_scalar(self):
        """Test value output is effectively scalar."""
        from agents.critic import Critic
        
        critic = Critic(obs_dim=5, hidden_dim=64)
        obs = torch.randn(1, 5)
        
        value = critic(obs)
        
        # Can convert to float
        scalar = float(value.squeeze())
        assert isinstance(scalar, float)


class TestActorCriticCombined:
    """Test ActorCritic combined class."""
    
    def test_actor_critic_creation(self):
        """Test ActorCritic creation."""
        from agents.critic import ActorCritic
        
        ac = ActorCritic(obs_dim=5, hidden_dim=64, v_max=5.0, num_bands=4)
        
        assert hasattr(ac, 'actor')
        assert hasattr(ac, 'critic')
    
    def test_actor_critic_forward(self):
        """Test ActorCritic forward pass."""
        from agents.critic import ActorCritic
        
        ac = ActorCritic(obs_dim=5, hidden_dim=64)
        obs = torch.randn(4, 5)  # 4 agents
        
        # Test actor component
        actions, log_probs, entropy = ac.actor.sample(obs)
        assert actions.shape == (4, 3)
        assert log_probs.shape == (4,)
        assert entropy.shape == (4,)
        
        # Test critic component with pooled obs
        obs_pooled = obs.mean(dim=0, keepdim=True)
        value = ac.critic.forward_pooled(obs_pooled)
        assert value.shape == (1, 1)


# =============================================================================
# ROLLOUT BUFFER TESTS
# =============================================================================

class TestRolloutBufferBasics:
    """Test RolloutBuffer basic functionality."""
    
    def test_buffer_creation(self):
        """Test buffer creation."""
        from agents.rollout_buffer import RolloutBuffer
        
        buffer = RolloutBuffer(
            buffer_size=256,
            obs_dim=5,
            action_dim=3,
            M=4
        )
        
        assert buffer.buffer_size == 256
        assert buffer.obs_dim == 5
        assert buffer.action_dim == 3
        assert buffer.M == 4
        assert len(buffer) == 0
    
    def test_buffer_add_single(self):
        """Test adding single transition."""
        from agents.rollout_buffer import RolloutBuffer
        
        buffer = RolloutBuffer(buffer_size=256, obs_dim=5, action_dim=3, M=4)
        
        obs = np.random.randn(4, 5).astype(np.float32)
        actions = np.random.randn(4, 3).astype(np.float32)
        log_probs = np.random.randn(4).astype(np.float32)
        reward = 1.0
        done = False
        value = 0.5
        
        buffer.add(obs, actions, log_probs, reward, done, value)
        
        assert len(buffer) == 1
    
    def test_buffer_add_multiple(self):
        """Test adding multiple transitions."""
        from agents.rollout_buffer import RolloutBuffer
        
        buffer = RolloutBuffer(buffer_size=256, obs_dim=5, action_dim=3, M=4)
        
        for i in range(100):
            obs = np.random.randn(4, 5).astype(np.float32)
            actions = np.random.randn(4, 3).astype(np.float32)
            log_probs = np.random.randn(4).astype(np.float32)
            reward = float(i)
            done = i == 99
            value = float(np.random.randn())
            
            buffer.add(obs, actions, log_probs, reward, done, value)
        
        assert len(buffer) == 100
    
    def test_buffer_reset(self):
        """Test buffer reset."""
        from agents.rollout_buffer import RolloutBuffer
        
        buffer = RolloutBuffer(buffer_size=256, obs_dim=5, action_dim=3, M=4)
        
        # Add transitions
        for _ in range(50):
            buffer.add(
                np.random.randn(4, 5).astype(np.float32),
                np.random.randn(4, 3).astype(np.float32),
                np.random.randn(4).astype(np.float32),
                1.0, False, 0.5
            )
        
        assert len(buffer) == 50
        
        buffer.reset()
        
        assert len(buffer) == 0


class TestRolloutBufferGAE:
    """Test GAE computation."""
    
    def test_compute_returns_and_advantages(self):
        """Test GAE computation produces correct shapes."""
        from agents.rollout_buffer import RolloutBuffer
        
        buffer = RolloutBuffer(
            buffer_size=64,
            obs_dim=5,
            action_dim=3,
            M=4,
            gamma=0.99,
            gae_lambda=0.95
        )
        
        # Fill buffer
        for i in range(64):
            buffer.add(
                np.random.randn(4, 5).astype(np.float32),
                np.random.randn(4, 3).astype(np.float32),
                np.random.randn(4).astype(np.float32),
                reward=1.0,
                done=(i == 63),
                value=float(np.random.randn())
            )
        
        # Compute GAE
        buffer.compute_returns_and_advantages(0.0, True)
        
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert len(buffer.advantages) == 64
        assert len(buffer.returns) == 64
    
    def test_gae_advantages_finite(self):
        """Test GAE advantages are finite."""
        from agents.rollout_buffer import RolloutBuffer
        
        buffer = RolloutBuffer(buffer_size=64, obs_dim=5, action_dim=3, M=4)
        
        for i in range(64):
            buffer.add(
                np.random.randn(4, 5).astype(np.float32),
                np.random.randn(4, 3).astype(np.float32),
                np.random.randn(4).astype(np.float32),
                reward=1.0,
                done=False,
                value=0.5
            )
        
        buffer.compute_returns_and_advantages(0.5, False)
        
        assert np.all(np.isfinite(buffer.advantages))
        assert np.all(np.isfinite(buffer.returns))
    
    def test_gae_normalized(self):
        """Test advantages are normalized."""
        from agents.rollout_buffer import RolloutBuffer
        
        buffer = RolloutBuffer(buffer_size=100, obs_dim=5, action_dim=3, M=4)
        
        for _ in range(100):
            buffer.add(
                np.random.randn(4, 5).astype(np.float32),
                np.random.randn(4, 3).astype(np.float32),
                np.random.randn(4).astype(np.float32),
                reward=np.random.randn(),
                done=False,
                value=np.random.randn()
            )
        
        buffer.compute_returns_and_advantages(0.0, True)
        
        # After normalization, mean should be ~0, std should be ~1
        mean = np.mean(buffer.advantages)
        std = np.std(buffer.advantages)
        
        assert abs(mean) < 0.1
        assert abs(std - 1.0) < 0.1


class TestRolloutBufferMinibatches:
    """Test minibatch generation."""
    
    def test_minibatch_shapes(self):
        """Test minibatch samples have correct shapes."""
        from agents.rollout_buffer import RolloutBuffer
        
        buffer = RolloutBuffer(buffer_size=64, obs_dim=5, action_dim=3, M=4)
        
        for _ in range(64):
            buffer.add(
                np.random.randn(4, 5).astype(np.float32),
                np.random.randn(4, 3).astype(np.float32),
                np.random.randn(4).astype(np.float32),
                reward=1.0,
                done=False,
                value=0.5
            )
        
        buffer.compute_returns_and_advantages(0.0, True)
        
        batch_size = 16
        for batch in buffer.get_minibatches(batch_size):
            # Each sample should have M * batch_entries agents
            assert batch.observations.shape[0] <= batch_size * 4
            assert batch.observations.shape[1] == 5
            assert batch.actions.shape[1] == 3
            break
    
    def test_minibatch_coverage(self):
        """Test minibatches cover all data."""
        from agents.rollout_buffer import RolloutBuffer
        
        buffer = RolloutBuffer(buffer_size=64, obs_dim=5, action_dim=3, M=4)
        
        for _ in range(64):
            buffer.add(
                np.random.randn(4, 5).astype(np.float32),
                np.random.randn(4, 3).astype(np.float32),
                np.random.randn(4).astype(np.float32),
                reward=1.0,
                done=False,
                value=0.5
            )
        
        buffer.compute_returns_and_advantages(0.0, True)
        
        # Count total samples
        total_samples = sum(len(batch.returns) for batch in buffer.get_minibatches(16))
        
        # Should have 64 * 4 = 256 total agent samples
        assert total_samples == 64 * 4


# =============================================================================
# PPO AGENT TESTS
# =============================================================================

class TestPPOAgentCreation:
    """Test PPO agent creation."""
    
    def test_ppo_creation_default(self):
        """Test PPO creation with defaults."""
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent()
        
        assert agent.obs_dim == 5
        assert agent.M == 4
        assert agent.gamma == 0.99
        assert agent.clip_eps == 0.2
    
    def test_ppo_creation_custom(self):
        """Test PPO creation with custom parameters."""
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(
            obs_dim=7,
            hidden_dim=256,
            M=6,
            gamma=0.95,
            clip_eps=0.3
        )
        
        assert agent.obs_dim == 7
        assert agent.hidden_dim == 256
        assert agent.M == 6
        assert agent.gamma == 0.95
        assert agent.clip_eps == 0.3
    
    def test_ppo_has_networks(self):
        """Test PPO has required networks."""
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(obs_dim=5, hidden_dim=64, M=4)
        
        assert hasattr(agent, 'actor')
        assert hasattr(agent, 'critic')
        assert hasattr(agent, 'buffer')
    
    def test_ppo_device_selection(self):
        """Test PPO device selection."""
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(device="cpu")
        
        assert str(agent.device) == "cpu"


class TestPPOAgentGetAction:
    """Test PPO get_action method."""
    
    def test_get_action_shapes(self):
        """Test get_action returns correct shapes."""
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(obs_dim=5, hidden_dim=64, M=4)
        obs = np.random.randn(4, 5).astype(np.float32)
        
        actions, log_probs, value = agent.get_action(obs)
        
        assert actions.shape == (4, 3)
        assert log_probs.shape == (4,)
        assert isinstance(value, float)
    
    def test_get_action_velocity_bounds(self):
        """Test actions have bounded velocities."""
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(obs_dim=5, hidden_dim=64, M=4, v_max=5.0)
        
        for _ in range(10):
            obs = np.random.randn(4, 5).astype(np.float32)
            actions, _, _ = agent.get_action(obs)
            
            velocities = actions[:, :2]
            assert np.all(velocities >= -5.0)
            assert np.all(velocities <= 5.0)
    
    def test_get_action_band_valid(self):
        """Test bands are valid."""
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(obs_dim=5, hidden_dim=64, M=4, num_bands=4)
        
        for _ in range(10):
            obs = np.random.randn(4, 5).astype(np.float32)
            actions, _, _ = agent.get_action(obs)
            
            bands = actions[:, 2]
            assert np.all(bands >= 0)
            assert np.all(bands < 4)
    
    def test_get_action_deterministic(self):
        """Test deterministic action is reproducible."""
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(obs_dim=5, hidden_dim=64, M=4)
        obs = np.random.randn(4, 5).astype(np.float32)
        
        actions1, _, _ = agent.get_action(obs, deterministic=True)
        actions2, _, _ = agent.get_action(obs, deterministic=True)
        
        assert np.allclose(actions1, actions2)


class TestPPOAgentStoreTransition:
    """Test PPO store_transition method."""
    
    def test_store_single_transition(self):
        """Test storing single transition."""
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(obs_dim=5, hidden_dim=64, M=4, rollout_length=64)
        
        obs = np.random.randn(4, 5).astype(np.float32)
        actions, log_probs, value = agent.get_action(obs)
        
        agent.store_transition(obs, actions, log_probs, reward=1.0, done=False, value=value)
        
        assert len(agent.buffer) == 1
        assert agent.total_timesteps == 1
    
    def test_store_full_rollout(self):
        """Test storing full rollout."""
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(obs_dim=5, hidden_dim=64, M=4, rollout_length=64)
        
        for i in range(64):
            obs = np.random.randn(4, 5).astype(np.float32)
            actions, log_probs, value = agent.get_action(obs)
            agent.store_transition(obs, actions, log_probs, reward=1.0, done=(i==63), value=value)
        
        assert len(agent.buffer) == 64
        assert agent.total_timesteps == 64


class TestPPOAgentUpdate:
    """Test PPO update method."""
    
    def test_update_returns_stats(self):
        """Test update returns statistics."""
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(
            obs_dim=5, hidden_dim=64, M=4,
            rollout_length=64, batch_size=16, n_epochs=2
        )
        
        # Fill buffer
        for i in range(64):
            obs = np.random.randn(4, 5).astype(np.float32)
            actions, log_probs, value = agent.get_action(obs)
            agent.store_transition(obs, actions, log_probs, reward=1.0, done=(i==63), value=value)
        
        last_obs = np.random.randn(4, 5).astype(np.float32)
        stats = agent.update(last_obs, last_done=True)
        
        assert "actor_loss" in stats
        assert "critic_loss" in stats
        assert "entropy" in stats
        assert "clip_fraction" in stats
        assert "approx_kl" in stats
    
    def test_update_increments_counter(self):
        """Test update increments update counter."""
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(
            obs_dim=5, hidden_dim=64, M=4,
            rollout_length=32, batch_size=8, n_epochs=1
        )
        
        assert agent.total_updates == 0
        
        for i in range(32):
            obs = np.random.randn(4, 5).astype(np.float32)
            actions, log_probs, value = agent.get_action(obs)
            agent.store_transition(obs, actions, log_probs, reward=1.0, done=(i==31), value=value)
        
        last_obs = np.random.randn(4, 5).astype(np.float32)
        agent.update(last_obs, last_done=True)
        
        assert agent.total_updates == 1
    
    def test_update_clears_buffer(self):
        """Test update clears buffer."""
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(
            obs_dim=5, hidden_dim=64, M=4,
            rollout_length=32, batch_size=8, n_epochs=1
        )
        
        for i in range(32):
            obs = np.random.randn(4, 5).astype(np.float32)
            actions, log_probs, value = agent.get_action(obs)
            agent.store_transition(obs, actions, log_probs, reward=1.0, done=(i==31), value=value)
        
        assert len(agent.buffer) == 32
        
        last_obs = np.random.randn(4, 5).astype(np.float32)
        agent.update(last_obs, last_done=True)
        
        assert len(agent.buffer) == 0
    
    def test_update_losses_finite(self):
        """Test losses are finite."""
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(
            obs_dim=5, hidden_dim=64, M=4,
            rollout_length=64, batch_size=16, n_epochs=3
        )
        
        for i in range(64):
            obs = np.random.randn(4, 5).astype(np.float32)
            actions, log_probs, value = agent.get_action(obs)
            reward = np.random.randn()
            agent.store_transition(obs, actions, log_probs, reward=reward, done=(i==63), value=value)
        
        last_obs = np.random.randn(4, 5).astype(np.float32)
        stats = agent.update(last_obs, last_done=True)
        
        assert np.isfinite(stats["actor_loss"])
        assert np.isfinite(stats["critic_loss"])
        assert np.isfinite(stats["entropy"])


class TestPPOAgentSaveLoad:
    """Test PPO save/load functionality."""
    
    def test_save_creates_file(self):
        """Test save creates checkpoint file."""
        import tempfile
        import os
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(obs_dim=5, hidden_dim=64, M=4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save(tmpdir)
            
            assert os.path.exists(os.path.join(tmpdir, "ppo_agent.pt"))
    
    def test_load_restores_state(self):
        """Test load restores agent state."""
        import tempfile
        from agents.ppo_agent import PPOAgent
        
        # Create and train agent
        agent = PPOAgent(
            obs_dim=5, hidden_dim=64, M=4,
            rollout_length=32, batch_size=8, n_epochs=1
        )
        
        for i in range(32):
            obs = np.random.randn(4, 5).astype(np.float32)
            actions, log_probs, value = agent.get_action(obs)
            agent.store_transition(obs, actions, log_probs, reward=1.0, done=(i==31), value=value)
        
        last_obs = np.random.randn(4, 5).astype(np.float32)
        agent.update(last_obs, last_done=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save(tmpdir)
            
            agent2 = PPOAgent.from_checkpoint(tmpdir)
            
            assert agent2.total_updates == agent.total_updates
            assert agent2.total_timesteps == agent.total_timesteps
            assert agent2.obs_dim == agent.obs_dim
    
    def test_loaded_agent_produces_same_actions(self):
        """Test loaded agent produces identical deterministic actions."""
        import tempfile
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(obs_dim=5, hidden_dim=64, M=4)
        
        obs = np.random.randn(4, 5).astype(np.float32)
        actions1, _, _ = agent.get_action(obs, deterministic=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save(tmpdir)
            agent2 = PPOAgent.from_checkpoint(tmpdir)
            
            actions2, _, _ = agent2.get_action(obs, deterministic=True)
            
            assert np.allclose(actions1, actions2)


class TestPPOAgentTrainingMode:
    """Test PPO training mode setting."""
    
    def test_set_training_mode_train(self):
        """Test setting training mode."""
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(obs_dim=5, hidden_dim=64, M=4)
        agent.set_training_mode(True)
        
        assert agent.actor.training
        assert agent.critic.training
    
    def test_set_training_mode_eval(self):
        """Test setting evaluation mode."""
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(obs_dim=5, hidden_dim=64, M=4)
        agent.set_training_mode(False)
        
        assert not agent.actor.training
        assert not agent.critic.training


# =============================================================================
# VERIFICATION FUNCTION TESTS
# =============================================================================

class TestVerificationFunctions:
    """Test built-in verification functions."""
    
    def test_verify_actor(self):
        """Test verify_actor function."""
        from agents.actor import verify_actor
        
        results = verify_actor()
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # All tests should pass
        for test_name, result in results.items():
            assert result.get("pass", False), f"{test_name} failed"
    
    def test_verify_critic(self):
        """Test verify_critic function."""
        from agents.critic import verify_critic
        
        results = verify_critic()
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for test_name, result in results.items():
            assert result.get("pass", False), f"{test_name} failed"
    
    def test_verify_rollout_buffer(self):
        """Test verify_rollout_buffer function."""
        from agents.rollout_buffer import verify_rollout_buffer
        
        results = verify_rollout_buffer()
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for test_name, result in results.items():
            assert result.get("pass", False), f"{test_name} failed"
    
    def test_verify_ppo_agent(self):
        """Test verify_ppo_agent function."""
        from agents.ppo_agent import verify_ppo_agent
        
        results = verify_ppo_agent()
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for test_name, result in results.items():
            assert result.get("pass", False), f"{test_name} failed"


# =============================================================================
# NUMPY FALLBACK TESTS
# =============================================================================

class TestNumpyFallbacks:
    """Test NumPy fallback implementations."""
    
    def test_actor_numpy_creation(self):
        """Test ActorNumpy creation."""
        from agents.actor import ActorNumpy
        
        actor = ActorNumpy(obs_dim=5, hidden_dim=64, v_max=5.0, num_bands=4)
        
        assert actor.obs_dim == 5
        assert actor.hidden_dim == 64
    
    def test_actor_numpy_get_action(self):
        """Test ActorNumpy action generation."""
        from agents.actor import ActorNumpy
        
        actor = ActorNumpy(obs_dim=5, hidden_dim=64, v_max=5.0, num_bands=4)
        obs = np.random.randn(4, 5).astype(np.float32)
        
        actions, _, _ = actor.sample(obs)
        
        assert actions.shape == (4, 3)
        assert np.all(actions[:, :2] >= -5.0)
        assert np.all(actions[:, :2] <= 5.0)
        assert np.all(actions[:, 2] >= 0)
        assert np.all(actions[:, 2] < 4)
    
    def test_critic_numpy_creation(self):
        """Test CriticNumpy creation."""
        from agents.critic import CriticNumpy
        
        critic = CriticNumpy(obs_dim=5, hidden_dim=64)
        
        assert critic.obs_dim == 5
        assert critic.hidden_dim == 64
    
    def test_critic_numpy_get_value(self):
        """Test CriticNumpy value estimation."""
        from agents.critic import CriticNumpy
        
        critic = CriticNumpy(obs_dim=5, hidden_dim=64)
        obs = np.random.randn(4, 5).astype(np.float32)
        
        value = critic.forward(obs)
        
        assert isinstance(value, (float, np.floating, np.ndarray))


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for full training loop."""
    
    def test_single_rollout_and_update(self):
        """Test complete single rollout and update cycle."""
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(
            obs_dim=5, hidden_dim=64, M=4,
            rollout_length=32, batch_size=8, n_epochs=2
        )
        
        # Simulate environment interaction
        obs = np.random.randn(4, 5).astype(np.float32)
        
        for step in range(32):
            actions, log_probs, value = agent.get_action(obs)
            
            # Simulate environment step
            next_obs = np.random.randn(4, 5).astype(np.float32)
            reward = np.random.randn()
            done = step == 31
            
            agent.store_transition(obs, actions, log_probs, reward, done, value)
            obs = next_obs
        
        # Update
        stats = agent.update(obs, last_done=True)
        
        assert agent.total_updates == 1
        assert np.isfinite(stats["actor_loss"])
        assert np.isfinite(stats["critic_loss"])
    
    def test_multiple_rollouts(self):
        """Test multiple rollouts and updates."""
        from agents.ppo_agent import PPOAgent
        
        agent = PPOAgent(
            obs_dim=5, hidden_dim=64, M=4,
            rollout_length=16, batch_size=8, n_epochs=1
        )
        
        for rollout in range(3):
            obs = np.random.randn(4, 5).astype(np.float32)
            
            for step in range(16):
                actions, log_probs, value = agent.get_action(obs)
                next_obs = np.random.randn(4, 5).astype(np.float32)
                reward = np.random.randn()
                done = step == 15
                
                agent.store_transition(obs, actions, log_probs, reward, done, value)
                obs = next_obs
            
            agent.update(obs, last_done=True)
        
        assert agent.total_updates == 3
        assert agent.total_timesteps == 48


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
