"""
Action Space Handler Module
===========================

Handles hybrid continuous + discrete action space for jammer agents.

Reference: PROJECT_MASTER_GUIDE_v2.md Section 6

Action Space:
    Continuous: (Vx, Vy) velocity in [-v_max, v_max]^2
    Discrete: band selection in {0, 1, 2, 3}
    
Combined action shape: (M, 3) where [:, :2] is velocity, [:, 2] is band

Author: MARL Jammer Team
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        HAS_GYMNASIUM = True
    except ImportError:
        HAS_GYMNASIUM = False


class ActionHandler:
    """
    Handles action space construction and action processing.
    
    Supports both training (stochastic) and inference (deterministic) modes.
    
    Attributes:
        v_max: Maximum velocity per axis (m/s)
        num_bands: Number of frequency bands
        M: Number of jammer agents
        
    Example:
        >>> handler = ActionHandler(v_max=5.0, num_bands=4)
        >>> action = handler.sample(M=4)  # Shape: (4, 3)
        >>> velocity, bands = handler.split_action(action)
    """
    
    # Frequency bands in Hz
    BANDS = {
        0: 433e6,   # 433 MHz
        1: 915e6,   # 915 MHz
        2: 2.4e9,   # 2.4 GHz
        3: 5.8e9    # 5.8 GHz
    }
    
    BAND_NAMES = {
        0: "433 MHz",
        1: "915 MHz",
        2: "2.4 GHz",
        3: "5.8 GHz"
    }
    
    def __init__(
        self,
        v_max: float = 5.0,
        num_bands: int = 4,
        arena_size: float = 200.0
    ):
        """
        Initialize action handler.
        
        Args:
            v_max: Maximum velocity per axis (m/s)
            num_bands: Number of frequency bands (default: 4)
            arena_size: Arena size for position clipping
        """
        self.v_max = v_max
        self.num_bands = num_bands
        self.arena_size = arena_size
    
    def get_action_space(self, M: int = 1) -> "spaces.Dict":
        """
        Get Gymnasium action space for M agents.
        
        Returns:
            Dict space with 'velocity' (Box) and 'band' (MultiDiscrete)
        """
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium or gym required for action space")
        
        return spaces.Dict({
            'velocity': spaces.Box(
                low=-self.v_max,
                high=self.v_max,
                shape=(M, 2),
                dtype=np.float32
            ),
            'band': spaces.MultiDiscrete([self.num_bands] * M)
        })
    
    def get_single_action_space(self) -> "spaces.Tuple":
        """
        Get action space for a single agent.
        
        Returns:
            Tuple of (Box for velocity, Discrete for band)
        """
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium or gym required for action space")
        
        return spaces.Tuple((
            spaces.Box(
                low=-self.v_max,
                high=self.v_max,
                shape=(2,),
                dtype=np.float32
            ),
            spaces.Discrete(self.num_bands)
        ))
    
    def sample(self, M: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """
        Sample random actions for M agents.
        
        Args:
            M: Number of agents
            seed: Random seed
            
        Returns:
            Actions array, shape (M, 3)
            [:, 0:2] = velocity (Vx, Vy)
            [:, 2] = band (integer 0-3)
        """
        rng = np.random.RandomState(seed)
        
        actions = np.zeros((M, 3), dtype=np.float32)
        
        # Sample velocities
        actions[:, :2] = rng.uniform(-self.v_max, self.v_max, size=(M, 2))
        
        # Sample bands
        actions[:, 2] = rng.randint(0, self.num_bands, size=M)
        
        return actions
    
    def split_action(
        self,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split combined action array into velocity and band components.
        
        Args:
            actions: Combined actions, shape (M, 3)
            
        Returns:
            Tuple of:
                - velocity: Shape (M, 2)
                - bands: Shape (M,), dtype int
        """
        velocity = actions[:, :2].astype(np.float32)
        bands = actions[:, 2].astype(np.int32)
        
        return velocity, bands
    
    def combine_action(
        self,
        velocity: np.ndarray,
        bands: np.ndarray
    ) -> np.ndarray:
        """
        Combine velocity and band into single action array.
        
        Args:
            velocity: Shape (M, 2)
            bands: Shape (M,)
            
        Returns:
            Combined actions, shape (M, 3)
        """
        M = velocity.shape[0]
        actions = np.zeros((M, 3), dtype=np.float32)
        actions[:, :2] = velocity
        actions[:, 2] = bands
        return actions
    
    def clip_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """
        Clip velocity to valid range.
        
        Args:
            velocity: Shape (M, 2) or (2,)
            
        Returns:
            Clipped velocity
        """
        return np.clip(velocity, -self.v_max, self.v_max)
    
    def clip_bands(self, bands: np.ndarray) -> np.ndarray:
        """
        Clip bands to valid range.
        
        Args:
            bands: Shape (M,)
            
        Returns:
            Clipped bands (integers in [0, num_bands-1])
        """
        return np.clip(bands.astype(np.int32), 0, self.num_bands - 1)
    
    def apply_action(
        self,
        positions: np.ndarray,
        velocity: np.ndarray,
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Apply velocity action to update positions.
        
        Args:
            positions: Current positions, shape (M, 2)
            velocity: Velocity actions, shape (M, 2)
            dt: Time step (seconds)
            
        Returns:
            New positions, clipped to arena bounds
        """
        # Clip velocity
        velocity = self.clip_velocity(velocity)
        
        # Update positions
        new_positions = positions + velocity * dt
        
        # Clip to arena bounds
        new_positions = np.clip(new_positions, 0, self.arena_size)
        
        return new_positions
    
    def get_band_frequency(self, band: int) -> float:
        """
        Get frequency in Hz for a band index.
        
        Args:
            band: Band index (0-3)
            
        Returns:
            Frequency in Hz
        """
        return self.BANDS.get(band, 2.4e9)  # Default to 2.4 GHz
    
    def get_band_name(self, band: int) -> str:
        """
        Get human-readable band name.
        
        Args:
            band: Band index (0-3)
            
        Returns:
            Band name string
        """
        return self.BAND_NAMES.get(band, "Unknown")
    
    def from_network_output(
        self,
        mu: np.ndarray,
        log_sigma: np.ndarray,
        logits: np.ndarray,
        deterministic: bool = False,
        rng: Optional[np.random.RandomState] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert network outputs to actions.
        
        Args:
            mu: Velocity mean from actor, shape (M, 2)
            log_sigma: Log std dev, shape (M, 2)
            logits: Band logits, shape (M, num_bands)
            deterministic: If True, use mean velocity and argmax band
            rng: Random state for sampling
            
        Returns:
            Tuple of:
                - actions: Combined actions, shape (M, 3)
                - log_probs: Log probabilities, shape (M,)
                - entropy: Entropy of distributions, shape (M,)
        """
        if rng is None:
            rng = np.random.RandomState()
        
        M = mu.shape[0]
        
        if deterministic:
            # Deterministic: use mean velocity and argmax band
            velocity = mu.copy()
            bands = np.argmax(logits, axis=1)
            
            # Log prob is 0 for deterministic (or can compute at mean)
            log_probs = np.zeros(M, dtype=np.float32)
            entropy = np.zeros(M, dtype=np.float32)
        else:
            # Stochastic: sample from distributions
            sigma = np.exp(log_sigma)
            
            # Sample velocity from Gaussian
            velocity = mu + sigma * rng.randn(M, 2)
            velocity = self.clip_velocity(velocity)
            
            # Sample band from categorical
            probs = self._softmax(logits)
            bands = np.array([
                rng.choice(self.num_bands, p=probs[i])
                for i in range(M)
            ], dtype=np.int32)
            
            # Compute log probabilities
            log_probs = self._compute_log_probs(velocity, mu, sigma, bands, probs)
            
            # Compute entropy
            entropy = self._compute_entropy(sigma, probs)
        
        actions = self.combine_action(velocity, bands)
        
        return actions, log_probs, entropy
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        # Subtract max for numerical stability
        logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    def _compute_log_probs(
        self,
        velocity: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        bands: np.ndarray,
        probs: np.ndarray
    ) -> np.ndarray:
        """Compute log probabilities for actions."""
        M = velocity.shape[0]
        log_probs = np.zeros(M, dtype=np.float32)
        
        # Gaussian log prob for velocity
        log_2pi = np.log(2 * np.pi)
        for i in range(M):
            # Log prob of Gaussian
            diff = velocity[i] - mu[i]
            var = sigma[i] ** 2
            log_p_vel = -0.5 * np.sum(
                log_2pi + np.log(var) + diff**2 / var
            )
            
            # Log prob of categorical for band
            log_p_band = np.log(probs[i, bands[i]] + 1e-8)
            
            log_probs[i] = log_p_vel + log_p_band
        
        return log_probs
    
    def _compute_entropy(
        self,
        sigma: np.ndarray,
        probs: np.ndarray
    ) -> np.ndarray:
        """Compute entropy of action distributions."""
        M = sigma.shape[0]
        entropy = np.zeros(M, dtype=np.float32)
        
        for i in range(M):
            # Gaussian entropy
            # H = 0.5 * log(2 * pi * e * sigma^2) for each dimension
            h_gaussian = 0.5 * np.sum(np.log(2 * np.pi * np.e * sigma[i]**2))
            
            # Categorical entropy
            # H = -sum(p * log(p))
            h_categorical = -np.sum(probs[i] * np.log(probs[i] + 1e-8))
            
            entropy[i] = h_gaussian + h_categorical
        
        return entropy


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_action_handler() -> dict:
    """Run verification tests."""
    results = {}
    
    handler = ActionHandler(v_max=5.0, num_bands=4, arena_size=200.0)
    
    # Test 1: Sample actions
    actions = handler.sample(M=4, seed=42)
    
    results["test_sample"] = {
        "shape": actions.shape,
        "expected_shape": (4, 3),
        "velocity_in_range": np.all(np.abs(actions[:, :2]) <= 5.0),
        "bands_valid": np.all((actions[:, 2] >= 0) & (actions[:, 2] < 4)),
        "pass": actions.shape == (4, 3)
    }
    
    # Test 2: Split action
    velocity, bands = handler.split_action(actions)
    
    results["test_split"] = {
        "velocity_shape": velocity.shape,
        "bands_shape": bands.shape,
        "pass": velocity.shape == (4, 2) and bands.shape == (4,)
    }
    
    # Test 3: Apply action
    positions = np.array([[100, 100], [50, 50]])
    velocity = np.array([[5, 0], [-5, 0]])
    
    new_pos = handler.apply_action(positions, velocity, dt=1.0)
    
    results["test_apply"] = {
        "expected_x": [105, 45],
        "actual_x": new_pos[:, 0].tolist(),
        "pass": np.allclose(new_pos[:, 0], [105, 45])
    }
    
    # Test 4: Clip at boundaries
    positions = np.array([[198, 198]])
    velocity = np.array([[10, 10]])
    
    new_pos = handler.apply_action(positions, velocity, dt=1.0)
    
    results["test_boundary_clip"] = {
        "new_position": new_pos[0].tolist(),
        "pass": np.allclose(new_pos[0], [200, 200])
    }
    
    # Test 5: Network output conversion
    mu = np.array([[0, 0], [1, 1]])
    log_sigma = np.array([[-1, -1], [-1, -1]])
    logits = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])  # Band 0 and 2
    
    actions, log_probs, entropy = handler.from_network_output(
        mu, log_sigma, logits, deterministic=True
    )
    
    results["test_network_output_det"] = {
        "actions_shape": actions.shape,
        "bands": actions[:, 2].tolist(),
        "expected_bands": [0, 2],
        "pass": np.allclose(actions[:, 2], [0, 2])
    }
    
    # Test 6: Stochastic sampling
    actions_stoch, log_probs_stoch, entropy_stoch = handler.from_network_output(
        mu, log_sigma, logits, deterministic=False, rng=np.random.RandomState(42)
    )
    
    results["test_network_output_stoch"] = {
        "actions_shape": actions_stoch.shape,
        "log_probs_finite": np.all(np.isfinite(log_probs_stoch)),
        "entropy_positive": np.all(entropy_stoch > 0),
        "pass": actions_stoch.shape == (2, 3)
    }
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Action Handler Verification")
    print("=" * 60)
    
    results = verify_action_handler()
    
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
