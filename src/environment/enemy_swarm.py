"""
Enemy Swarm Dynamics Module
===========================

Implements enemy drone movement models for realistic swarm simulation.

Reference: PROJECT_MASTER_GUIDE_v2.md Section 4.3

Movement Models:
    Mode A - Random Walk: Independent Brownian motion with reflective boundaries
    Mode B - Coordinated Turn: State-space model matching IEEE TMC 2024 baseline

State Vector (Coordinated Turn):
    s = [x, x_dot, y, y_dot, omega]
    - x, y: position
    - x_dot, y_dot: velocity  
    - omega: turn rate

Transition Matrix (Eq. 3-5 from baseline):
    Uses CT (Coordinated Turn) model with constant turn rate assumption

Author: MARL Jammer Team
"""

import numpy as np
from typing import Optional, Tuple, Literal


class EnemySwarm:
    """
    Enemy swarm dynamics simulation.
    
    Supports two movement modes:
        1. Random Walk: Each drone moves independently with random velocity
        2. Coordinated Turn: State-space model for realistic maneuvering
    
    Attributes:
        N: Number of enemy drones
        positions: Current positions, shape (N, 2)
        velocities: Current velocities, shape (N, 2)
        mode: Movement mode ('random_walk' or 'coordinated_turn')
        v_enemy: Base enemy speed (m/s)
        dt: Time step (seconds)
        
    Example:
        >>> swarm = EnemySwarm(N=20, mode='random_walk', arena_size=200.0)
        >>> for _ in range(100):
        ...     swarm.step()
        >>> final_positions = swarm.positions
    """
    
    def __init__(
        self,
        N: int = 20,
        positions: Optional[np.ndarray] = None,
        mode: Literal['random_walk', 'coordinated_turn'] = 'random_walk',
        v_enemy: float = 2.0,
        dt: float = 0.5,
        arena_size: float = 200.0,
        omega_max: float = 0.5,
        process_noise_std: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize enemy swarm.
        
        Args:
            N: Number of enemy drones
            positions: Initial positions, shape (N, 2). If None, randomly initialized.
            mode: 'random_walk' or 'coordinated_turn'
            v_enemy: Base enemy speed in m/s (default: 2.0)
            dt: Time step in seconds (default: 0.5)
            arena_size: Size of square arena in meters (default: 200)
            omega_max: Maximum turn rate for coordinated turn (rad/s, default: 0.5)
            process_noise_std: Process noise standard deviation (default: 0.1)
            seed: Random seed for reproducibility
        """
        self.N = N
        self.mode = mode
        self.v_enemy = v_enemy
        self.dt = dt
        self.arena_size = arena_size
        self.omega_max = omega_max
        self.process_noise_std = process_noise_std
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            self._rng = np.random.RandomState(seed)
        else:
            self._rng = np.random.RandomState()
        
        # Initialize positions
        if positions is not None:
            assert positions.shape == (N, 2), f"Expected shape ({N}, 2), got {positions.shape}"
            self.positions = positions.astype(np.float64).copy()
        else:
            # Random initial positions in arena
            margin = arena_size * 0.1  # 10% margin from edges
            self.positions = self._rng.uniform(
                margin, arena_size - margin,
                size=(N, 2)
            )
        
        # Initialize velocities (random directions, constant speed)
        angles = self._rng.uniform(0, 2 * np.pi, size=N)
        self.velocities = np.column_stack([
            v_enemy * np.cos(angles),
            v_enemy * np.sin(angles)
        ]).astype(np.float64)
        
        # Coordinated turn state: [x, x_dot, y, y_dot, omega]
        if mode == 'coordinated_turn':
            self._init_coordinated_turn_state()
        
        # Step counter
        self.step_count = 0
    
    def _init_coordinated_turn_state(self):
        """Initialize state vector for coordinated turn model."""
        # State: [x, x_dot, y, y_dot, omega] for each drone
        self.state = np.zeros((self.N, 5))
        
        # Position
        self.state[:, 0] = self.positions[:, 0]  # x
        self.state[:, 2] = self.positions[:, 1]  # y
        
        # Velocity
        self.state[:, 1] = self.velocities[:, 0]  # x_dot
        self.state[:, 3] = self.velocities[:, 1]  # y_dot
        
        # Turn rate (initially small random values)
        self.state[:, 4] = self._rng.uniform(
            -self.omega_max * 0.5,
            self.omega_max * 0.5,
            size=self.N
        )
    
    def step(self) -> np.ndarray:
        """
        Advance swarm by one time step.
        
        Uses the selected movement model to update positions and velocities.
        Applies boundary conditions (reflective for random walk, wrapping for CT).
        
        Returns:
            Updated positions, shape (N, 2)
        """
        if self.mode == 'random_walk':
            self._step_random_walk()
        else:
            self._step_coordinated_turn()
        
        self.step_count += 1
        return self.positions.copy()
    
    def _step_random_walk(self):
        """
        Random walk with reflective boundary conditions.
        
        Each drone moves in its current direction with small random perturbations.
        When hitting a boundary, velocity is reflected.
        """
        # Add noise to velocity direction
        noise = self._rng.normal(0, self.process_noise_std, size=(self.N, 2))
        self.velocities += noise
        
        # Normalize to maintain constant speed
        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        speeds = np.maximum(speeds, 1e-6)  # Avoid division by zero
        self.velocities = self.velocities / speeds * self.v_enemy
        
        # Update positions
        new_positions = self.positions + self.velocities * self.dt
        
        # Reflective boundary conditions
        for dim in range(2):
            # Below zero
            below = new_positions[:, dim] < 0
            new_positions[below, dim] = -new_positions[below, dim]
            self.velocities[below, dim] = -self.velocities[below, dim]
            
            # Above arena_size
            above = new_positions[:, dim] > self.arena_size
            new_positions[above, dim] = 2 * self.arena_size - new_positions[above, dim]
            self.velocities[above, dim] = -self.velocities[above, dim]
        
        # Clip to ensure bounds (in case of numerical issues)
        self.positions = np.clip(new_positions, 0, self.arena_size)
    
    def _step_coordinated_turn(self):
        """
        Coordinated Turn (CT) model step.
        
        State transition based on IEEE TMC 2024 baseline equations.
        State: s = [x, x_dot, y, y_dot, omega]
        
        Transition equations:
            x(k+1) = x(k) + sin(omega*dt)/omega * x_dot - (1-cos(omega*dt))/omega * y_dot
            x_dot(k+1) = cos(omega*dt) * x_dot - sin(omega*dt) * y_dot
            y(k+1) = y(k) + (1-cos(omega*dt))/omega * x_dot + sin(omega*dt)/omega * y_dot  
            y_dot(k+1) = sin(omega*dt) * x_dot + cos(omega*dt) * y_dot
            omega(k+1) = omega(k) + noise
        """
        dt = self.dt
        
        for i in range(self.N):
            x, x_dot, y, y_dot, omega = self.state[i]
            
            # Handle near-zero omega to avoid division by zero
            if abs(omega) < 1e-6:
                # Linear motion approximation
                new_x = x + x_dot * dt
                new_y = y + y_dot * dt
                new_x_dot = x_dot
                new_y_dot = y_dot
            else:
                # Full coordinated turn model
                sin_wt = np.sin(omega * dt)
                cos_wt = np.cos(omega * dt)
                
                new_x = x + (sin_wt / omega) * x_dot - ((1 - cos_wt) / omega) * y_dot
                new_x_dot = cos_wt * x_dot - sin_wt * y_dot
                new_y = y + ((1 - cos_wt) / omega) * x_dot + (sin_wt / omega) * y_dot
                new_y_dot = sin_wt * x_dot + cos_wt * y_dot
            
            # Add process noise to omega
            omega_noise = self._rng.normal(0, self.process_noise_std * 0.5)
            new_omega = np.clip(omega + omega_noise, -self.omega_max, self.omega_max)
            
            # Update state
            self.state[i] = [new_x, new_x_dot, new_y, new_y_dot, new_omega]
        
        # Apply boundary conditions (wrapping)
        self.state[:, 0] = np.mod(self.state[:, 0], self.arena_size)  # x
        self.state[:, 2] = np.mod(self.state[:, 2], self.arena_size)  # y
        
        # Update positions and velocities from state
        self.positions[:, 0] = self.state[:, 0]
        self.positions[:, 1] = self.state[:, 2]
        self.velocities[:, 0] = self.state[:, 1]
        self.velocities[:, 1] = self.state[:, 3]
    
    def reset(
        self,
        positions: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Reset swarm to initial state.
        
        Args:
            positions: New initial positions. If None, random initialization.
            seed: New random seed. If None, continues with current RNG.
            
        Returns:
            Initial positions after reset
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        
        if positions is not None:
            assert positions.shape == (self.N, 2)
            self.positions = positions.copy()
        else:
            margin = self.arena_size * 0.1
            self.positions = self._rng.uniform(
                margin, self.arena_size - margin,
                size=(self.N, 2)
            )
        
        # Reinitialize velocities
        angles = self._rng.uniform(0, 2 * np.pi, size=self.N)
        self.velocities = np.column_stack([
            self.v_enemy * np.cos(angles),
            self.v_enemy * np.sin(angles)
        ])
        
        if self.mode == 'coordinated_turn':
            self._init_coordinated_turn_state()
        
        self.step_count = 0
        return self.positions.copy()
    
    def get_speeds(self) -> np.ndarray:
        """Get current speed of each drone."""
        return np.linalg.norm(self.velocities, axis=1)
    
    def get_headings(self) -> np.ndarray:
        """Get current heading angle (radians) of each drone."""
        return np.arctan2(self.velocities[:, 1], self.velocities[:, 0])
    
    def set_positions(self, positions: np.ndarray):
        """Manually set positions (e.g., for testing)."""
        assert positions.shape == (self.N, 2)
        self.positions = positions.copy()
        if self.mode == 'coordinated_turn':
            self.state[:, 0] = positions[:, 0]
            self.state[:, 2] = positions[:, 1]


def create_clustered_swarm(
    n_drones: int,
    n_clusters: int,
    arena_size: float = 200.0,
    cluster_radius: float = 30.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate initial positions with drones grouped in clusters.
    
    Useful for testing clustering algorithms and jammer deployment strategies.
    
    Args:
        n_drones: Total number of drones
        n_clusters: Number of clusters to create
        arena_size: Size of arena
        cluster_radius: Radius of each cluster
        seed: Random seed
        
    Returns:
        Initial positions, shape (n_drones, 2)
    """
    rng = np.random.RandomState(seed)
    
    # Generate cluster centers
    margin = cluster_radius * 2
    centers = rng.uniform(margin, arena_size - margin, size=(n_clusters, 2))
    
    # Assign drones to clusters
    positions = np.zeros((n_drones, 2))
    drones_per_cluster = n_drones // n_clusters
    remainder = n_drones % n_clusters
    
    drone_idx = 0
    for c in range(n_clusters):
        n_in_cluster = drones_per_cluster + (1 if c < remainder else 0)
        
        # Random positions around cluster center
        angles = rng.uniform(0, 2 * np.pi, size=n_in_cluster)
        radii = rng.uniform(0, cluster_radius, size=n_in_cluster)
        
        positions[drone_idx:drone_idx + n_in_cluster, 0] = (
            centers[c, 0] + radii * np.cos(angles)
        )
        positions[drone_idx:drone_idx + n_in_cluster, 1] = (
            centers[c, 1] + radii * np.sin(angles)
        )
        
        drone_idx += n_in_cluster
    
    # Clip to arena bounds
    positions = np.clip(positions, 0, arena_size)
    
    return positions


def create_formation_swarm(
    n_drones: int,
    formation: Literal['grid', 'line', 'circle', 'v_shape'] = 'grid',
    center: Tuple[float, float] = (100.0, 100.0),
    spacing: float = 20.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate initial positions in a specific formation.
    
    Args:
        n_drones: Number of drones
        formation: Formation type
        center: Center of formation
        spacing: Distance between drones
        seed: Random seed (for adding noise)
        
    Returns:
        Initial positions, shape (n_drones, 2)
    """
    rng = np.random.RandomState(seed)
    positions = np.zeros((n_drones, 2))
    cx, cy = center
    
    if formation == 'grid':
        # Square grid formation
        side = int(np.ceil(np.sqrt(n_drones)))
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= n_drones:
                    break
                positions[idx] = [
                    cx + (i - side/2) * spacing,
                    cy + (j - side/2) * spacing
                ]
                idx += 1
                
    elif formation == 'line':
        # Line formation
        for i in range(n_drones):
            positions[i] = [
                cx + (i - n_drones/2) * spacing,
                cy
            ]
            
    elif formation == 'circle':
        # Circular formation
        angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
        radius = spacing * n_drones / (2 * np.pi)
        positions[:, 0] = cx + radius * np.cos(angles)
        positions[:, 1] = cy + radius * np.sin(angles)
        
    elif formation == 'v_shape':
        # V formation
        half = n_drones // 2
        for i, idx in enumerate(range(half)):
            positions[idx] = [
                cx - (i + 1) * spacing * 0.7,
                cy + (i + 1) * spacing
            ]
        for i, idx in enumerate(range(half, n_drones)):
            positions[idx] = [
                cx + (i + 1) * spacing * 0.7,
                cy + (i + 1) * spacing
            ]
    
    # Add small noise
    positions += rng.normal(0, spacing * 0.05, size=positions.shape)
    
    return positions


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def verify_enemy_swarm() -> dict:
    """
    Run verification tests on enemy swarm dynamics.
    
    Returns:
        Dictionary with test results
    """
    results = {}
    
    # Test 1: Random walk stays in bounds
    swarm = EnemySwarm(N=20, mode='random_walk', arena_size=200.0, seed=42)
    
    for _ in range(200):
        swarm.step()
    
    in_bounds = np.all((swarm.positions >= 0) & (swarm.positions <= 200))
    results["test_random_walk_bounds"] = {
        "final_positions_min": float(swarm.positions.min()),
        "final_positions_max": float(swarm.positions.max()),
        "in_bounds": in_bounds,
        "pass": in_bounds
    }
    
    # Test 2: Speed consistency (random walk)
    speeds = swarm.get_speeds()
    speed_consistent = np.allclose(speeds, 2.0, atol=0.5)  # Within 0.5 m/s of target
    
    results["test_random_walk_speed"] = {
        "mean_speed": float(speeds.mean()),
        "expected": 2.0,
        "pass": speed_consistent
    }
    
    # Test 3: Coordinated turn model
    swarm_ct = EnemySwarm(N=10, mode='coordinated_turn', arena_size=200.0, seed=42)
    initial_pos = swarm_ct.positions.copy()
    
    for _ in range(10):
        swarm_ct.step()
    
    # Position should change
    position_changed = not np.allclose(initial_pos, swarm_ct.positions)
    
    results["test_coordinated_turn_motion"] = {
        "initial_center": initial_pos.mean(axis=0).tolist(),
        "final_center": swarm_ct.positions.mean(axis=0).tolist(),
        "position_changed": position_changed,
        "pass": position_changed
    }
    
    # Test 4: Reset functionality
    swarm.reset(seed=42)
    reset_step_count = swarm.step_count
    
    results["test_reset"] = {
        "step_count_after_reset": reset_step_count,
        "pass": reset_step_count == 0
    }
    
    # Test 5: Clustered swarm generation
    clustered = create_clustered_swarm(20, 3, arena_size=200.0, seed=42)
    
    results["test_clustered_generation"] = {
        "shape": clustered.shape,
        "expected_shape": (20, 2),
        "pass": clustered.shape == (20, 2)
    }
    
    # Test 6: Formation swarm generation
    circle = create_formation_swarm(10, 'circle', center=(100, 100), spacing=20)
    
    # Check points are roughly equidistant from center
    distances = np.linalg.norm(circle - np.array([100, 100]), axis=1)
    distance_variance = np.var(distances)
    
    results["test_circle_formation"] = {
        "mean_distance_from_center": float(distances.mean()),
        "distance_variance": float(distance_variance),
        "pass": distance_variance < 10.0  # Should be nearly constant
    }
    
    return results


# =============================================================================
# MAIN (for standalone testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Enemy Swarm Dynamics Module Verification")
    print("=" * 60)
    
    results = verify_enemy_swarm()
    
    all_passed = True
    for test_name, test_result in results.items():
        print(f"\n{test_name}:")
        if isinstance(test_result, dict):
            for key, val in test_result.items():
                print(f"  {key}: {val}")
            if "pass" in test_result:
                status = "PASS" if test_result["pass"] else "FAIL"
                print(f"  STATUS: {status}")
                if not test_result["pass"]:
                    all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All enemy swarm tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
