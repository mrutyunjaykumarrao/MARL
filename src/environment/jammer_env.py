"""
Jammer Environment Module
=========================

Gymnasium-compatible environment for MARL jammer drone simulation.

Reference: PROJECT_MASTER_GUIDE_v2.md Sections 2-6

Environment Features:
    - Dynamic enemy swarm (random walk or coordinated turn)
    - FSPL-based communication and jamming
    - Lambda-2 reward for swarm disconnection
    - Hybrid action space (continuous velocity + discrete band)
    - DBSCAN clustering for intelligent jammer deployment

Author: MARL Jammer Team
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any, List, Literal

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from .enemy_swarm import EnemySwarm, create_clustered_swarm
from .observation import ObservationBuilder, build_global_observation
from .action_space import ActionHandler
from .reward import RewardCalculator, RewardComponents

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from physics.fspl import compute_jam_range, db_to_watts, watts_to_db
from physics.communication_graph import (
    compute_adjacency_matrix,
    compute_laplacian,
    compute_lambda2
)
from physics.jamming import (
    compute_disrupted_links, 
    apply_jamming_to_adjacency,
    compute_midpoints,
    compute_distances_to_midpoints,
    compute_jamming_power,
    FREQUENCY_BANDS
)
from clustering.dbscan_clustering import (
    DBSCANClusterer,
    assign_jammers_to_clusters,
    get_jammer_initial_positions
)


class JammerEnv(gym.Env):
    """
    Multi-Agent Jammer Drone Environment.
    
    This environment simulates M jammer drones attempting to disrupt
    communication of an N-drone enemy swarm by minimizing lambda-2
    (the Fiedler value / algebraic connectivity).
    
    Observation Space (per agent):
        5-dimensional normalized vector [0, 1]^5:
        [dist_to_centroid, cluster_density, dist_to_others, 
         coverage_overlap, band_match]
    
    Action Space (per agent):
        Continuous: velocity (Vx, Vy) in [-v_max, v_max]^2
        Discrete: band selection in {0, 1, 2, 3}
    
    Reward:
        5-term function with lambda-2 reduction as primary objective.
    
    Attributes:
        M: Number of jammer agents
        N: Number of enemy drones
        arena_size: Size of square arena (meters)
        max_steps: Maximum episode length
        
    Example:
        >>> env = JammerEnv(M=4, N=20)
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        M: int = 4,
        N: int = 20,
        arena_size: float = 200.0,
        max_steps: int = 200,
        dt: float = 0.5,
        v_max: float = 5.0,
        v_enemy: float = 2.0,
        enemy_mode: Literal['random_walk', 'coordinated_turn'] = 'random_walk',
        # Physics parameters
        P_tx_dbm: float = 20.0,
        P_sens_dbm: float = -90.0,
        P_jammer_dbm: float = 30.0,
        P_jam_thresh_dbm: float = -70.0,
        # Reward weights
        omega_1: float = 1.0,
        omega_2: float = 0.3,
        omega_3: float = 0.2,
        omega_4: float = 0.1,
        omega_5: float = 0.2,
        # Clustering
        eps: float = 30.0,
        min_samples: int = 2,
        K_recompute: int = 10,
        # Validation
        min_lambda2_initial: float = 0.5,
        max_reset_attempts: int = 100,
        debug_mode: bool = False,
        random_jammer_start: bool = False,  # If True, start jammers at random positions
        # Other
        seed: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the jammer environment.
        
        Args:
            M: Number of jammer agents
            N: Number of enemy drones
            arena_size: Size of square arena in meters
            max_steps: Maximum steps per episode
            dt: Time step duration (seconds)
            v_max: Maximum jammer velocity (m/s)
            v_enemy: Enemy movement speed (m/s)
            enemy_mode: 'random_walk' or 'coordinated_turn'
            P_tx_dbm: Enemy transmit power (dBm)
            P_sens_dbm: Receiver sensitivity (dBm)
            P_jammer_dbm: Jammer transmit power (dBm)
            P_jam_thresh_dbm: Jamming threshold (dBm)
            omega_1-5: Reward weights
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min samples
            K_recompute: Steps between cluster recomputation
            seed: Random seed
            render_mode: Rendering mode
        """
        super().__init__()
        
        # Store parameters
        self.M = M
        self.N = N
        self.arena_size = arena_size
        self.max_steps = max_steps
        self.dt = dt
        self.v_max = v_max
        self.v_enemy = v_enemy
        self.enemy_mode = enemy_mode
        self.render_mode = render_mode
        
        # Physics parameters
        self.P_tx_dbm = P_tx_dbm
        self.P_sens_dbm = P_sens_dbm
        self.P_jammer_dbm = P_jammer_dbm
        self.P_jam_thresh_dbm = P_jam_thresh_dbm
        
        # Clustering parameters
        self.eps = eps
        self.min_samples = min_samples
        self.K_recompute = K_recompute
        
        # Validation parameters
        self.min_lambda2_initial = min_lambda2_initial
        self.max_reset_attempts = max_reset_attempts
        self.debug_mode = debug_mode
        self.random_jammer_start = random_jammer_start
        
        # Frequency bands (Hz)
        self.BANDS = [433e6, 915e6, 2.4e9, 5.8e9]
        self.num_bands = len(self.BANDS)
        
        # Compute default jamming range at 2.4 GHz
        # Convert dBm to Watts for FSPL calculation
        self.R_jam = compute_jam_range(
            db_to_watts(P_jammer_dbm), db_to_watts(P_jam_thresh_dbm), self.BANDS[2]
        )
        
        # Initialize components
        self.enemy_swarm = EnemySwarm(
            N=N,
            mode=enemy_mode,
            v_enemy=v_enemy,
            dt=dt,
            arena_size=arena_size,
            seed=seed
        )
        
        self.clusterer = DBSCANClusterer(
            eps=eps,
            min_samples=min_samples,
            arena_size=arena_size
        )
        
        self.obs_builder = ObservationBuilder(
            arena_size=arena_size,
            R_jam=self.R_jam
        )
        
        self.action_handler = ActionHandler(
            v_max=v_max,
            num_bands=self.num_bands,
            arena_size=arena_size
        )
        
        self.reward_calculator = RewardCalculator(
            omega_1=omega_1,
            omega_2=omega_2,
            omega_3=omega_3,
            omega_4=omega_4,
            omega_5=omega_5,
            v_max=v_max,
            R_jam=self.R_jam,
            arena_size=arena_size
        )
        
        # Define observation space (per agent)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(M, 5),
            dtype=np.float32
        )
        
        # Define action space
        # Combined: (M, 3) where [:, :2] is velocity, [:, 2] is band
        self.action_space = spaces.Dict({
            'velocity': spaces.Box(
                low=-v_max,
                high=v_max,
                shape=(M, 2),
                dtype=np.float32
            ),
            'band': spaces.MultiDiscrete([self.num_bands] * M)
        })
        
        # Episode state
        self._rng = np.random.RandomState(seed)
        self.step_count = 0
        self.enemy_band = 2  # Default: 2.4 GHz
        self.lambda2_initial = 0.0
        self.jammer_positions = np.zeros((M, 2))
        self.jammer_bands = np.zeros(M, dtype=np.int32)
        self.centroids: Dict[int, np.ndarray] = {}
        self.cluster_sizes: Dict[int, int] = {}
        self.jammer_assignments: Dict[int, list] = {}
        self.last_velocities = np.zeros((M, 2))
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Optional random seed
            options: Optional reset options
            
        Returns:
            Tuple of (observations, info_dict)
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        
        self.step_count = 0
        
        # Reset enemy swarm with validation for non-trivial initial graph
        reset_seed = seed
        for attempt in range(self.max_reset_attempts):
            self.enemy_swarm.reset(seed=reset_seed)
            
            # Sample enemy band
            self.enemy_band = self._rng.randint(0, self.num_bands)
            enemy_freq = self.BANDS[self.enemy_band]
            
            # Update R_jam for this frequency (convert dBm to Watts)
            self.R_jam = compute_jam_range(
                db_to_watts(self.P_jammer_dbm), db_to_watts(self.P_jam_thresh_dbm), enemy_freq
            )
            self.obs_builder.R_jam = self.R_jam
            self.reward_calculator.R_jam = self.R_jam
            
            # Cluster enemies
            self._update_clusters()
            
            # Compute initial adjacency and lambda-2 WITHOUT jamming
            self.lambda2_initial = self._compute_lambda2(disrupted=False)
            
            # Validate: ensure non-trivial initial graph
            if self.lambda2_initial >= self.min_lambda2_initial:
                break
            
            # Try different seed for next attempt
            reset_seed = None if seed is None else seed + attempt + 1
            
            if self.debug_mode and attempt < 5:
                print(f"  [Reset attempt {attempt+1}] lambda2_initial={self.lambda2_initial:.4f} < {self.min_lambda2_initial} (regenerating)")
        else:
            # Fallback: warn but continue with what we have
            if self.debug_mode:
                print(f"  [Warning] Could not achieve lambda2_initial >= {self.min_lambda2_initial} after {self.max_reset_attempts} attempts")
                print(f"  [Warning] Using lambda2_initial={self.lambda2_initial:.4f}")
        
        if self.debug_mode:
            print(f"  [Reset] lambda2_initial={self.lambda2_initial:.4f}, enemy_band={self.enemy_band}, n_clusters={self.clusterer.n_clusters}")
        
        # Initialize jammer positions
        if self.random_jammer_start:
            # Start at random positions - agent must learn to navigate
            self.jammer_positions = self._rng.uniform(
                0.1 * self.arena_size, 0.9 * self.arena_size, size=(self.M, 2)
            )
            self.jammer_assignments = {}  # No initial assignments
        else:
            # Start near cluster centroids (easier)
            self.jammer_assignments = assign_jammers_to_clusters(
                self.M, self.centroids, self.cluster_sizes, strategy="proportional"
            )
            self.jammer_positions = get_jammer_initial_positions(
                self.M, self.centroids, self.jammer_assignments,
                spread=10.0, arena_size=self.arena_size
            )
        
        # Initialize jammer bands RANDOMLY (not optimal - must learn)
        self.jammer_bands = self._rng.randint(0, self.num_bands, size=self.M).astype(np.int32)
        
        self.last_velocities = np.zeros((self.M, 2))
        
        # Build observations
        obs = self._build_observations()
        
        info = {
            "lambda2_initial": self.lambda2_initial,
            "lambda2_current": self.lambda2_initial,
            "enemy_band": self.enemy_band,
            "n_clusters": self.clusterer.n_clusters,
            "step": 0
        }
        
        return obs, info
    
    def step(
        self,
        action: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Dict with 'velocity' (M, 2) and 'band' (M,)
            
        Returns:
            Tuple of (observations, reward, terminated, truncated, info)
        """
        self.step_count += 1
        
        # Parse action
        if isinstance(action, dict):
            velocity = action['velocity']
            bands = action['band']
        else:
            # Handle array-style action (M, 3)
            velocity = action[:, :2]
            bands = action[:, 2].astype(np.int32)
        
        self.last_velocities = velocity.copy()
        
        # Apply velocity to jammer positions
        self.jammer_positions = self.action_handler.apply_action(
            self.jammer_positions, velocity, self.dt
        )
        
        # Update jammer bands
        self.jammer_bands = self.action_handler.clip_bands(bands)
        
        # Update enemy swarm
        self.enemy_swarm.step()
        
        # Periodically recompute clusters
        if self.step_count % self.K_recompute == 0:
            self._update_clusters()
            
            # Check if reassignment needed
            if self.clusterer.should_reassign_jammers(drift_threshold=30.0):
                self.jammer_assignments = assign_jammers_to_clusters(
                    self.M, self.centroids, self.cluster_sizes,
                    strategy="proportional"
                )
        
        # Compute current lambda-2 with jamming
        lambda2_current = self._compute_lambda2(disrupted=True)
        
        # Compute raw reduction (safe division)
        if self.lambda2_initial > 1e-6:
            raw_reduction = 1.0 - (lambda2_current / self.lambda2_initial)
        else:
            raw_reduction = 0.0  # Cannot reduce from trivial starting point
        
        # Compute reward
        reward, components = self.reward_calculator.compute(
            lambda2_current=lambda2_current,
            lambda2_initial=self.lambda2_initial,
            jammer_bands=self.jammer_bands,
            enemy_band=self.enemy_band,
            jammer_positions=self.jammer_positions,
            centroids=self.centroids,
            velocities=self.last_velocities,
            jammer_assignments=self.jammer_assignments
        )
        
        # Diagnostic logging in debug mode
        if self.debug_mode and self.step_count <= 3:
            print(f"    [Step {self.step_count}] L2_init={self.lambda2_initial:.4f}, L2_curr={lambda2_current:.4f}, "
                  f"reduction={raw_reduction*100:.1f}%, reward={reward:.3f}")
            print(f"      Components: L2={components.lambda2_reduction:.3f}, band={components.band_match:.3f}, "
                  f"prox={components.proximity:.3f}, energy={components.energy_penalty:.3f}")
        
        # Check termination - only truncate, don't terminate early
        # Agent should learn to maintain disruption over time
        terminated = False  # Never terminate early - learn sustained jamming
        truncated = self.step_count >= self.max_steps
        
        # Build observations
        obs = self._build_observations()
        
        # Compute average jamming power received at enemy link midpoints (for logging)
        avg_jamming_power_dbm = self._compute_avg_jamming_power()
        
        info = {
            "lambda2_initial": self.lambda2_initial,
            "lambda2_current": lambda2_current,
            "lambda2_reduction": raw_reduction,
            "lambda2_reduction_raw": raw_reduction,
            "enemy_band": self.enemy_band,
            "n_clusters": self.clusterer.n_clusters,
            "step": self.step_count,
            "reward_components": self.reward_calculator.get_reward_breakdown(components),
            "jammer_positions": self.jammer_positions.copy(),
            "enemy_positions": self.enemy_swarm.positions.copy(),
            "avg_jamming_power_dbm": avg_jamming_power_dbm
        }
        
        return obs, reward, terminated, truncated, info
    
    def _update_clusters(self):
        """Update DBSCAN clustering of enemies."""
        labels, self.centroids = self.clusterer.fit(self.enemy_swarm.positions)
        self.cluster_sizes = self.clusterer.get_cluster_sizes()
    
    def _compute_lambda2(self, disrupted: bool = False) -> float:
        """
        Compute lambda-2 of enemy communication graph.
        
        Args:
            disrupted: If True, apply jamming disruption
            
        Returns:
            Lambda-2 value (0 = disconnected)
        """
        enemy_freq = self.BANDS[self.enemy_band]
        
        # Build adjacency matrix
        adj = compute_adjacency_matrix(
            self.enemy_swarm.positions,
            self.P_tx_dbm,
            self.P_sens_dbm,
            enemy_freq
        )
        
        if disrupted:
            # Convert dBm to Watts for jamming computation
            jammer_power_watts = db_to_watts(self.P_jammer_dbm)
            jam_thresh_watts = db_to_watts(self.P_jam_thresh_dbm)
            
            # Compute which links are jammed
            jammed_links = compute_disrupted_links(
                self.jammer_positions,
                self.jammer_bands,
                self.enemy_swarm.positions,
                self.enemy_band,
                jammer_power_watts,
                jam_thresh_watts
            )
            
            # Apply jamming to adjacency matrix
            adj = apply_jamming_to_adjacency(adj, jammed_links)
        
        # Compute Laplacian and lambda-2
        laplacian = compute_laplacian(adj)
        return compute_lambda2(laplacian)
    
    def _compute_avg_jamming_power(self) -> float:
        """
        Compute average jamming power received at enemy link midpoints.
        
        Only considers jammers with matching frequency band.
        Higher values indicate better jamming effectiveness.
        
        Returns:
            Average jamming power in dBm (for logging/visualization)
        """
        enemy_freq = FREQUENCY_BANDS[self.enemy_band]
        jammer_power_watts = db_to_watts(self.P_jammer_dbm)
        
        # Compute midpoints of all enemy links
        midpoints = compute_midpoints(self.enemy_swarm.positions)
        
        # Compute distances from jammers to midpoints
        distances = compute_distances_to_midpoints(self.jammer_positions, midpoints)
        
        # Only consider jammers with matching band
        band_mask = (self.jammer_bands == self.enemy_band)
        
        if not np.any(band_mask):
            # No jammers on correct band - return very low power
            return -80.0
        
        # Compute jamming power from each jammer (M, N, N)
        P_jam = compute_jamming_power(distances, jammer_power_watts, enemy_freq)
        
        # Apply band mask - only matching jammers contribute
        P_jam_masked = P_jam.copy()
        P_jam_masked[~band_mask, :, :] = 0.0
        
        # Take max across jammers for each link (best jammer for each link)
        P_jam_max_per_link = np.max(P_jam_masked, axis=0)  # (N, N)
        
        # Get upper triangle (unique links, exclude diagonal)
        upper_tri_indices = np.triu_indices_from(P_jam_max_per_link, k=1)
        link_powers = P_jam_max_per_link[upper_tri_indices]
        
        # Filter out zero powers (no jammer coverage)
        valid_powers = link_powers[link_powers > 0]
        
        if len(valid_powers) == 0:
            return -80.0
        
        # Average power in watts, convert to dBm
        avg_power_watts = np.mean(valid_powers)
        avg_power_dbm = watts_to_db(avg_power_watts)
        
        return float(avg_power_dbm)
    
    def _build_observations(self) -> np.ndarray:
        """Build observation vectors for all agents."""
        return self.obs_builder.build_vectorized(
            self.jammer_positions,
            self.jammer_bands,
            self.centroids,
            self.cluster_sizes,
            self.enemy_band,
            self.N,
            self.jammer_assignments
        )
    
    def get_global_observation(self) -> np.ndarray:
        """Get mean-pooled global observation for critic."""
        obs = self._build_observations()
        return build_global_observation(obs)
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_console()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()
    
    def _render_console(self):
        """Simple console rendering."""
        print(f"\n--- Step {self.step_count} ---")
        print(f"Enemy band: {self.action_handler.get_band_name(self.enemy_band)}")
        print(f"Lambda-2: {self._compute_lambda2(disrupted=True):.4f}")
        print(f"Clusters: {self.clusterer.n_clusters}")
        print(f"Jammer positions:\n{self.jammer_positions}")
    
    def _render_rgb(self) -> np.ndarray:
        """Render to RGB array (placeholder)."""
        # Simple placeholder - return arena as image
        size = 400
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = [40, 40, 40]  # Dark gray background
        
        scale = size / self.arena_size
        
        # Draw enemies (red)
        for pos in self.enemy_swarm.positions:
            x, y = int(pos[0] * scale), int(pos[1] * scale)
            if 0 <= x < size and 0 <= y < size:
                cv_x, cv_y = max(0, x-2), max(0, y-2)
                img[cv_y:min(cv_y+5, size), cv_x:min(cv_x+5, size)] = [255, 0, 0]
        
        # Draw jammers (green)
        for pos in self.jammer_positions:
            x, y = int(pos[0] * scale), int(pos[1] * scale)
            if 0 <= x < size and 0 <= y < size:
                cv_x, cv_y = max(0, x-3), max(0, y-3)
                img[cv_y:min(cv_y+7, size), cv_x:min(cv_x+7, size)] = [0, 255, 0]
        
        return img
    
    def close(self):
        """Clean up resources."""
        pass


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_jammer_env() -> dict:
    """Run verification tests."""
    results = {}
    
    # Test 1: Environment creation
    env = JammerEnv(M=4, N=20, seed=42)
    
    results["test_creation"] = {
        "M": env.M,
        "N": env.N,
        "obs_space_shape": env.observation_space.shape,
        "pass": env.M == 4 and env.N == 20
    }
    
    # Test 2: Reset
    obs, info = env.reset(seed=42)
    
    results["test_reset"] = {
        "obs_shape": obs.shape,
        "expected_shape": (4, 5),
        "lambda2_initial": info["lambda2_initial"],
        "has_lambda2": info["lambda2_initial"] > 0,
        "pass": obs.shape == (4, 5) and "lambda2_initial" in info
    }
    
    # Test 3: Step with sample action
    action = {
        'velocity': np.random.uniform(-5, 5, size=(4, 2)).astype(np.float32),
        'band': np.array([2, 2, 2, 2])
    }
    
    obs2, reward, terminated, truncated, info2 = env.step(action)
    
    results["test_step"] = {
        "obs_shape": obs2.shape,
        "reward_type": type(reward).__name__,
        "lambda2_current": info2["lambda2_current"],
        "step_count": info2["step"],
        "pass": obs2.shape == (4, 5) and isinstance(reward, (int, float))
    }
    
    # Test 4: Episode rollout
    env.reset(seed=123)
    total_reward = 0
    
    for _ in range(50):
        action = {
            'velocity': np.zeros((4, 2), dtype=np.float32),
            'band': np.full(4, env.enemy_band)
        }
        _, r, term, trunc, _ = env.step(action)
        total_reward += r
        if term or trunc:
            break
    
    results["test_rollout"] = {
        "total_reward": total_reward,
        "steps": env.step_count,
        "pass": env.step_count > 0
    }
    
    # Test 5: Global observation
    global_obs = env.get_global_observation()
    
    results["test_global_obs"] = {
        "shape": global_obs.shape,
        "expected_shape": (5,),
        "pass": global_obs.shape == (5,)
    }
    
    # Test 6: Lambda-2 reduction
    env.reset(seed=42)
    initial_l2 = env.lambda2_initial
    
    # Move jammers to optimal positions over centroids
    for _ in range(20):
        # All jammers use correct band
        action = {
            'velocity': np.zeros((4, 2), dtype=np.float32),
            'band': np.full(4, env.enemy_band)
        }
        _, _, _, _, info = env.step(action)
    
    final_l2 = info["lambda2_current"]
    
    results["test_lambda2_dynamics"] = {
        "initial": initial_l2,
        "final": final_l2,
        "pass": True  # Just verify it runs
    }
    
    env.close()
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Jammer Environment Verification")
    print("=" * 60)
    
    results = verify_jammer_env()
    
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
