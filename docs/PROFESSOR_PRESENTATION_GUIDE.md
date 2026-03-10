# MARL Jammer Project - Complete Technical Presentation Guide

**Date: February 23, 2026**  
**Project: Multi-Agent Reinforcement Learning for Enemy Drone Swarm Communication Disruption**

---

## Table of Contents

1. [Project Overview - The Story](#1-project-overview---the-story)
2. [Problem Statement](#2-problem-statement)
3. [Mathematical Foundation](#3-mathematical-foundation)
4. [Physics Layer - FSPL Model](#4-physics-layer---fspl-model)
5. [Communication Graph Theory](#5-communication-graph-theory)
6. [DBSCAN Clustering](#6-dbscan-clustering)
7. [Environment Design](#7-environment-design)
8. [Reward Function Design](#8-reward-function-design)
9. [PPO Agent Architecture](#9-ppo-agent-architecture)
10. [Training Pipeline](#10-training-pipeline)
11. [Results and Graphs Explanation](#11-results-and-graphs-explanation)
12. [Key Code Walkthrough](#12-key-code-walkthrough)
13. [Conclusion](#13-conclusion)

---

## 1. Project Overview - The Story

### Opening Statement

"Professor, this project addresses a critical military/defense challenge: How can we use autonomous jammer drones to **disconnect an enemy drone swarm's communication network** using Multi-Agent Reinforcement Learning?"

### The Core Idea

- **Enemy Swarm**: N drones (e.g., 30 drones) communicating wirelessly
- **Our Jammers**: M drones (e.g., 6 jammers) trying to disrupt their communication
- **Objective**: Minimize the swarm's **algebraic connectivity (λ₂)** to break their coordination

### Why This Matters

- Enemy drone swarms rely on inter-drone communication for coordination
- Breaking their communication fragments the swarm into isolated groups
- Each fragment loses collective intelligence and becomes easy to defeat

### Our Approach vs Previous Work

| Aspect      | Previous Paper (Q-table)        | Our Approach (MARL-PPO)                    |
| ----------- | ------------------------------- | ------------------------------------------ |
| Algorithm   | Q-learning with lookup tables   | PPO with neural networks                   |
| Scalability | Fails at N>20, M>4              | Scales to N=100, M=40                      |
| State Space | Discretized (explosion problem) | Continuous (neural function approximation) |
| Physics     | Simple distance threshold       | FSPL-based realistic RF propagation        |

---

## 2. Problem Statement

### Formal Definition

Given:

- N enemy drones at positions $\{p_1, p_2, ..., p_N\} \in \mathbb{R}^2$
- M jammer drones with controllable positions and frequency bands
- Communication graph $G = (V, E)$ where edges exist based on received power

**Objective**: Find jammer positions and band selections that minimize:
$$\lambda_2(L(G_{\text{jammed}})) \rightarrow 0$$

where $L$ is the Graph Laplacian and $\lambda_2$ is the Fiedler value (algebraic connectivity).

### Key Insight

**Proposition 1 (from our theory)**:

> If $\lambda_2 = 0$, the graph is disconnected. Therefore, driving $\lambda_2$ to zero guarantees swarm fragmentation.

---

## 3. Mathematical Foundation

### Graph Laplacian Theory

For a graph $G = (V, E)$ with adjacency matrix $A$:

**Degree Matrix**: $D_{ii} = \sum_j A_{ij}$

**Laplacian Matrix**: $L = D - A$

**Properties of L**:

```
L = | d₁  -a₁₂  -a₁₃ |
    |-a₂₁   d₂  -a₂₃ |
    |-a₃₁  -a₃₂   d₃ |
```

**Eigenvalues**: $0 = \lambda_1 \leq \lambda_2 \leq ... \leq \lambda_N$

**Key Theorem**:

- $\lambda_2 > 0$ ⟺ Graph is connected
- $\lambda_2 = 0$ ⟺ Graph has ≥2 components (disconnected!)

### Code Implementation

```python
# From src/physics/communication_graph.py

def compute_laplacian(A: np.ndarray) -> np.ndarray:
    """
    Compute the Graph Laplacian L = D - A.
    """
    D = np.diag(np.sum(A, axis=1))  # Degree matrix
    L = D - A
    return L

def compute_lambda2(L: np.ndarray) -> float:
    """
    Compute lambda-2 (Fiedler value / algebraic connectivity).
    """
    eigenvalues = np.linalg.eigvalsh(L)  # Hermitian eigenvalues
    eigenvalues = np.sort(eigenvalues)   # Sort ascending

    if len(eigenvalues) < 2:
        return 0.0

    return float(eigenvalues[1])  # Second smallest = λ₂
```

---

## 4. Physics Layer - FSPL Model

### Free Space Path Loss (FSPL)

The received power at distance $d$ is:

$$P_R = P_{tx} \cdot \left(\frac{c}{4\pi f d}\right)^2$$

In dB form:
$$\text{FSPL}(d, f) = 20\log_{10}(d) + 20\log_{10}(f) + 20\log_{10}\left(\frac{4\pi}{c}\right)$$

### Code Implementation

```python
# From src/physics/fspl.py

SPEED_OF_LIGHT = 3.0e8  # m/s

def received_power_watts(
    tx_power_watts: float,
    distance_m: float,
    frequency_hz: float,
    eps: float = 1e-10
) -> float:
    """
    Calculate received power using FSPL model.

    Formula: P_R = P_tx * (c / (4 * pi * f * d))^2
    """
    d = max(distance_m, eps)  # Prevent division by zero
    wavelength = SPEED_OF_LIGHT / frequency_hz
    path_loss_linear = (wavelength / (4 * np.pi * d)) ** 2
    return tx_power_watts * path_loss_linear

def db_to_watts(power_dbm: float) -> float:
    """Convert dBm to Watts: P_watts = 10^((P_dBm - 30) / 10)"""
    return 10 ** ((power_dbm - 30) / 10)

def watts_to_db(power_watts: float) -> float:
    """Convert Watts to dBm: P_dBm = 10*log10(P_watts) + 30"""
    return 10 * np.log10(power_watts + 1e-30) + 30
```

### Jamming Range Calculation

```python
def compute_jam_range(
    jammer_power_dbm: float,
    jam_threshold_dbm: float,
    frequency_hz: float
) -> float:
    """
    Compute effective jamming radius.

    At this range, received jamming power = threshold

    R_jam = (c / (4 * pi * f)) * sqrt(P_tx / P_thresh)
    """
    P_tx = db_to_watts(jammer_power_dbm)   # 30 dBm = 1W
    P_thresh = db_to_watts(jam_threshold_dbm)  # -35 dBm

    wavelength = SPEED_OF_LIGHT / frequency_hz
    R_jam = (wavelength / (4 * np.pi)) * np.sqrt(P_tx / P_thresh)
    return R_jam  # ~18 meters at 2.4GHz with our parameters
```

### Physical Parameters

| Parameter       | Value       | Explanation                             |
| --------------- | ----------- | --------------------------------------- |
| Jammer Power    | 30 dBm (1W) | Standard drone transmit power           |
| Jam Threshold   | -35 dBm     | Power level that disrupts communication |
| Frequency       | 2.4 GHz     | Common drone communication band         |
| Effective Range | ~18m        | Radius where jamming is effective       |

---

## 5. Communication Graph Theory

### Adjacency Matrix Construction

An edge (i,j) exists in the communication graph if:

1. Received power $P_R(i,j) \geq P_{\text{sensitivity}}$ (-90 dBm)
2. Link is NOT jammed

```python
# From src/physics/communication_graph.py

def compute_adjacency_matrix(
    positions: np.ndarray,      # (N, 2) enemy positions
    tx_power_watts: float,      # 0.1W = 20 dBm
    sensitivity_watts: float,   # 1e-12W = -90 dBm
    frequency: float,           # 2.4e9 Hz
    jammed_links: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute adjacency matrix of enemy communication graph.
    """
    N = positions.shape[0]

    # Compute received power matrix (N, N)
    P_R = compute_pairwise_received_power(positions, tx_power_watts, frequency)

    # Links exist where received power >= sensitivity
    A = (P_R >= sensitivity_watts).astype(np.float64)

    # Remove self-loops
    np.fill_diagonal(A, 0)

    # Apply jamming mask
    if jammed_links is not None:
        A = A * (~jammed_links).astype(np.float64)

    return A
```

### Jamming Effect on Links

```python
# From src/physics/jamming.py

def compute_disrupted_links(
    enemy_positions: np.ndarray,     # (N, 2)
    jammer_positions: np.ndarray,    # (M, 2)
    jammer_bands: np.ndarray,        # (M,) band indices
    enemy_band: int,                 # Enemy's operating band
    jammer_power_dbm: float = 30.0,  # 1W
    jam_threshold_dbm: float = -35.0
) -> np.ndarray:
    """
    Determine which enemy links are disrupted by jammers.

    A link (i,j) is jammed if:
    1. Any jammer is within range of link midpoint
    2. Jammer is on same frequency band as enemy

    Returns: Boolean matrix (N, N), True where link is jammed
    """
    N = enemy_positions.shape[0]
    M = jammer_positions.shape[0]

    # Compute link midpoints (N, N, 2)
    midpoints = compute_midpoints(enemy_positions)

    # For each jammer, check if it jams each link
    jammed = np.zeros((N, N), dtype=bool)

    for k in range(M):
        # Check band match
        if jammer_bands[k] != enemy_band:
            continue  # Wrong band, no jamming effect

        # Compute distances to all midpoints
        distances = np.linalg.norm(midpoints - jammer_positions[k], axis=2)

        # Check if within jamming range
        jam_power_received = compute_jamming_power(distances, jammer_power_dbm)
        jammed |= (jam_power_received >= jam_threshold_dbm)

    return jammed
```

---

## 6. DBSCAN Clustering

### Why Clustering?

- Enemy drones often form **clusters** (groups)
- Deploying jammers near cluster **centroids** is more effective than random placement
- DBSCAN finds these clusters automatically without knowing the number beforehand

### DBSCAN Algorithm

```python
# From src/clustering/dbscan_clustering.py

class DBSCANClusterer:
    def __init__(
        self,
        eps: float = 30.0,      # Neighborhood radius (meters)
        min_samples: int = 2,   # Min drones to form cluster
        arena_size: float = 150.0
    ):
        self._dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.centroids = {}
        self.n_clusters = 0

    def fit(self, positions: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Run DBSCAN on enemy positions.

        Returns cluster labels and centroids.
        Label -1 = noise (not in any cluster)
        """
        self.labels = self._dbscan.fit_predict(positions)

        # Compute centroids for each cluster
        unique_labels = set(self.labels) - {-1}  # Exclude noise
        self.n_clusters = len(unique_labels)

        self.centroids = {}
        for label in unique_labels:
            mask = self.labels == label
            self.centroids[label] = positions[mask].mean(axis=0)

        return self.labels, self.centroids
```

### Jammer Assignment to Clusters

```python
def assign_jammers_to_clusters(
    centroids: Dict[int, np.ndarray],
    M: int,
    method: str = "round_robin"
) -> Dict[int, List[int]]:
    """
    Assign M jammers to K clusters.

    Strategy:
    - If M >= K: Distribute evenly, extra jammers go to largest clusters
    - If M < K: Each jammer covers nearest cluster(s)

    Returns: {cluster_id: [jammer_indices]}
    """
    K = len(centroids)
    assignments = {k: [] for k in centroids.keys()}

    if K == 0:
        return assignments

    cluster_ids = list(centroids.keys())

    # Round-robin assignment
    for j in range(M):
        cluster = cluster_ids[j % K]
        assignments[cluster].append(j)

    return assignments
```

---

## 7. Environment Design

### JammerEnv - Gymnasium Compatible

```python
# From src/environment/jammer_env.py

class JammerEnv(gym.Env):
    """
    Observation Space (per agent): 5D vector [0, 1]^5
        [0] dist_to_centroid - Distance to assigned cluster centroid
        [1] cluster_density - Density of nearby enemies
        [2] dist_to_others - Average distance to other jammers
        [3] coverage_overlap - How much coverage overlaps with others
        [4] band_match - Whether on same band as enemy (0 or 1)

    Action Space (per agent):
        Continuous: velocity (Vx, Vy) ∈ [-5, 5]^2 m/s
        Discrete: band_selection ∈ {0, 1, 2, 3}
    """

    def __init__(
        self,
        M: int = 6,                    # Number of jammers
        N: int = 30,                   # Number of enemy drones
        arena_size: float = 150.0,     # Arena size (meters)
        max_steps: int = 200,          # Episode length
        dt: float = 0.5,               # Time step (seconds)
        # Physics parameters
        jammer_power_dbm: float = 30.0,
        jam_threshold_dbm: float = -35.0,
        frequency_hz: float = 2.4e9,
        # Motion parameters
        v_max: float = 5.0,            # Max velocity (m/s)
        num_bands: int = 4,
        # DBSCAN parameters
        dbscan_eps: float = 30.0,
        dbscan_min_samples: int = 2
    ):
        super().__init__()

        # Store config
        self.M = M
        self.N = N
        self.arena_size = arena_size
        # ...

        # Define spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(M, 5), dtype=np.float32
        )

        # Hybrid action space: continuous velocity + discrete band
        self.action_space = spaces.Dict({
            'velocity': spaces.Box(low=-v_max, high=v_max, shape=(M, 2)),
            'band': spaces.MultiDiscrete([num_bands] * M)
        })
```

### Episode Flow

```python
def reset(self, seed=None):
    """Initialize new episode."""
    # 1. Create enemy swarm
    self.enemy = EnemySwarm(N=self.N, arena_size=self.arena_size)

    # 2. Run DBSCAN clustering
    self.clusterer.fit(self.enemy.positions)

    # 3. Place jammers near cluster centroids
    self.jammer_positions = get_jammer_initial_positions(
        self.clusterer.centroids, self.M
    )

    # 4. Compute initial λ₂
    A = compute_adjacency_matrix(self.enemy.positions, ...)
    L = compute_laplacian(A)
    self.lambda2_initial = compute_lambda2(L)

    # 5. Build observations
    obs = self._build_observations()

    return obs, info

def step(self, action):
    """Execute one timestep."""
    # 1. Parse action
    velocities = action['velocity']  # (M, 2)
    bands = action['band']           # (M,)

    # 2. Update jammer positions
    self.jammer_positions += velocities * self.dt
    self.jammer_positions = np.clip(self.jammer_positions, 0, self.arena_size)

    # 3. Update enemy positions (random walk)
    self.enemy.step()

    # 4. Re-cluster enemies
    self.clusterer.fit(self.enemy.positions)

    # 5. Compute jammed links
    jammed = compute_disrupted_links(
        self.enemy.positions,
        self.jammer_positions,
        bands,
        self.enemy.band
    )

    # 6. Compute new λ₂
    A = compute_adjacency_matrix(self.enemy.positions, jammed_links=jammed)
    L = compute_laplacian(A)
    lambda2_current = compute_lambda2(L)

    # 7. Compute reward
    reward, components = self.reward_calculator.compute(
        lambda2_current=lambda2_current,
        lambda2_initial=self.lambda2_initial,
        jammer_bands=bands,
        enemy_band=self.enemy.band,
        jammer_positions=self.jammer_positions,
        centroids=self.clusterer.centroids,
        velocities=velocities
    )

    # 8. Check termination
    terminated = lambda2_current < 0.01  # Swarm disconnected!
    truncated = self.step_count >= self.max_steps

    return obs, reward, terminated, truncated, info
```

---

## 8. Reward Function Design

### 5-Term Reward Function

$$R(t) = \omega_1 \cdot R_{\lambda_2} + \omega_2 \cdot R_{\text{band}} + \omega_3 \cdot R_{\text{prox}} - \omega_4 \cdot R_{\text{energy}} - \omega_5 \cdot R_{\text{overlap}}$$

### Reward Components

```python
# From src/environment/reward.py

class RewardCalculator:
    def __init__(
        self,
        omega_1: float = 10.0,  # λ₂ reduction (PRIMARY)
        omega_2: float = 0.0,   # Band match (disabled)
        omega_3: float = 0.0,   # Proximity (disabled)
        omega_4: float = 0.0,   # Energy penalty (disabled)
        omega_5: float = 0.0    # Overlap penalty (disabled)
    ):
        pass

    def _compute_lambda2_reward(self, lambda2_current, lambda2_initial):
        """
        Primary reward: How much λ₂ was reduced.

        R_λ₂ = 1 - (λ₂_current / λ₂_initial)

        If λ₂ reduced by 80%: R_λ₂ = 1 - 0.2 = 0.8
        """
        if lambda2_initial < 1e-10:
            return 0.0
        reduction = 1.0 - (lambda2_current / lambda2_initial)
        return np.clip(reduction, -1.0, 1.0)
```

### Reward Clipping

```python
# In trainer.py - Critical for stable training!

reward = np.clip(reward, -10.0, 10.0)
```

**Why clipping?** Without it, extreme rewards cause:

- Gradient explosion
- Entropy collapse (policy becomes deterministic too fast)
- Unstable learning

---

## 9. PPO Agent Architecture

### Actor Network (Policy)

```python
# From src/agents/actor.py

class Actor(nn.Module):
    """
    Actor network for hybrid action space.

    Outputs:
        - Velocity mean μ and std σ for Gaussian distribution
        - Band logits for Categorical distribution
    """

    def __init__(self, obs_dim=5, hidden_dim=128, v_max=5.0, num_bands=4):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Velocity head (continuous)
        self.velocity_mean = nn.Linear(hidden_dim, 2)
        self.velocity_log_std = nn.Parameter(torch.zeros(2))

        # Band head (discrete)
        self.band_logits = nn.Linear(hidden_dim, num_bands)

    def forward(self, obs):
        features = self.shared(obs)

        # Velocity: Gaussian distribution
        v_mean = self.velocity_mean(features) * v_max
        v_std = torch.exp(self.velocity_log_std)

        # Band: Categorical distribution
        band_logits = self.band_logits(features)

        return v_mean, v_std, band_logits
```

### Critic Network (Value Function)

```python
# From src/agents/critic.py

class Critic(nn.Module):
    """
    Centralized critic for CTDE paradigm.

    Takes global state and outputs value estimate V(s).
    """

    def __init__(self, obs_dim=5, M=6, hidden_dim=128):
        super().__init__()

        # Input: concatenated observations of all M agents
        self.network = nn.Sequential(
            nn.Linear(obs_dim * M, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, global_obs):
        return self.network(global_obs)
```

### PPO Algorithm

```python
# From src/agents/ppo_agent.py

class PPOAgent:
    def update(self):
        """
        PPO update with clipped surrogate objective.
        """
        # Compute advantages using GAE
        advantages = self._compute_gae(rewards, values, dones)

        for epoch in range(n_epochs):
            for batch in self.buffer.sample(batch_size):
                # Compute new log probabilities
                new_log_probs = self.actor.evaluate(batch.obs, batch.actions)

                # Importance ratio
                ratio = torch.exp(new_log_probs - batch.old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch.advantages
                surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * batch.advantages

                # PPO Loss
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                critic_loss = F.mse_loss(self.critic(batch.obs), batch.returns)

                # Entropy bonus (encourages exploration)
                entropy = self.actor.entropy(batch.obs)

                # Total loss
                loss = actor_loss + c1 * critic_loss - c2 * entropy

                # Gradient descent
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.parameters(), max_grad_norm)
                self.optimizer.step()
```

### Hyperparameters

```python
# Final training config (config.json)

{
    "gamma": 0.99,          # Discount factor
    "gae_lambda": 0.95,     # GAE lambda
    "clip_eps": 0.2,        # PPO clip epsilon
    "lr_actor": 3e-4,       # Actor learning rate
    "lr_critic": 1e-3,      # Critic learning rate
    "c1": 0.5,              # Value loss coefficient
    "c2": 0.02,             # Entropy coefficient (increased for exploration)
    "rollout_length": 1024, # Steps per rollout
    "batch_size": 128,      # Mini-batch size
    "n_epochs": 15          # PPO epochs per update
}
```

---

## 10. Training Pipeline

### Training Flow

```python
# From train.py

def train(config):
    # 1. Create environment
    env = JammerEnv(
        M=config.M,
        N=config.N,
        arena_size=config.arena_size,
        jam_threshold_dbm=config.jam_threshold_dbm
    )

    # 2. Create PPO agent
    agent = PPOAgent(
        obs_dim=5,
        M=config.M,
        lr_actor=config.lr_actor,
        rollout_length=config.rollout_length,
        batch_size=config.batch_size
    )

    # 3. Training loop
    for rollout in range(total_rollouts):
        # Collect rollout
        obs = env.reset()
        for step in range(rollout_length):
            action, log_prob, value = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)

            # Clip reward for stability
            reward = np.clip(reward, -10.0, 10.0)

            agent.store_transition(obs, action, log_prob, reward, done, value)
            obs = next_obs

        # Update policy
        losses = agent.update()

        # Log metrics
        lambda2_reduction = info['lambda2_reduction']
        print(f"Rollout {rollout}: λ₂ reduction = {lambda2_reduction:.1f}%")
```

### Training Results

```
Training Configuration:
- N = 30 enemies
- M = 6 jammers
- Arena = 150m × 150m
- Total Steps = 200,000
- Training Time = 19.7 minutes

Final Results:
- Best λ₂ Reduction: 78%
- Average Reward: 6.2
- Policy Entropy: 2.1 (healthy exploration)
```

---

## 11. Results and Graphs Explanation

### Graph 1: Lambda-2 vs Episodes (01_lambda2_vs_episodes.png)

**What it shows**: Training progress over time

- X-axis: Training episodes
- Y-axis: λ₂ reduction percentage
- Blue line: Raw values
- Orange line: Smoothed trend

**Key insight**: Agent learns to reduce connectivity from ~30% initially to ~78% by end of training.

---

### Graph 2: Training Curves 4-Panel (07_training_curves_4panel.png)

**What it shows**: Complete training diagnostics

| Panel        | Metric         | What to Look For                         |
| ------------ | -------------- | ---------------------------------------- |
| Top-Left     | Episode Reward | Should increase over time                |
| Top-Right    | λ₂ Reduction   | Should increase (our main objective)     |
| Bottom-Left  | Policy Entropy | Should stay moderate (not collapse to 0) |
| Bottom-Right | Value Loss     | Should decrease (critic learning)        |

---

### Graph 3: Baseline Comparison Bar (08_baseline_comparison_bar.png)

**What it shows**: Our method vs 5 baselines

| Method       | Description              | Expected Performance |
| ------------ | ------------------------ | -------------------- |
| MARL-PPO     | Our method               | ~78% (best)          |
| Random       | Random actions           | ~10-15%              |
| Static       | Fixed positions          | ~20-30%              |
| Center-Based | Naive centroid targeting | ~40-50%              |
| Greedy       | Heuristic                | ~45-55%              |

---

### Graph 4: Avg Power Comparison (04_avg_power_comparison.png)

**What it shows**: MARL-PPO vs Q-table baseline

- **Purple line (MARL-PPO)**: Starts at -55 dBm, improves to -43 dBm
  - Higher power = better jamming = agents learned optimal positioning
- **Orange line (Q-table)**: Stays flat at -65 dBm
  - Cannot learn effectively due to state-space explosion
- **Black dashed (Threshold)**: -67 dBm minimum effective jamming

**Key insight**: Our neural network-based approach achieves **+22 dB better** jamming power than Q-table approach.

---

### Graph 5: Connectivity Before/After (05_connectivity_before_after.png)

**What it shows**: Network topology visualization

| Left Panel (Before) | Right Panel (After)     |
| ------------------- | ----------------------- |
| Full connectivity   | Fragmented network      |
| Many edges (thick)  | Few edges (thin/broken) |
| High λ₂             | Low/zero λ₂             |

**Visual proof**: Jammers successfully broke the enemy communication graph!

---

### Graph 6: Scalability - Enemy Count (10_scalability_enemy_count.png)

**What it shows**: Performance as N increases (5 → 100 enemies)

**Key insight**: Performance stays >70% even with N=100 enemies, proving scalability.

---

### Graph 7: Scalability - Jammer Count (11_scalability_jammer_count.png)

**What it shows**: Performance vs number of jammers (M=2 to M=8)

**Key insight**: More jammers = better performance, with diminishing returns after M≈6.

---

### Graph 8: GAE vs MC Ablation (16_ablation_gae_vs_mc.png)

**What it shows**: Advantage estimation comparison

- **Green (GAE λ=0.95)**: Faster convergence, lower variance
- **Red (Monte Carlo)**: Slower convergence, higher variance

**Key insight**: GAE reaches 70% target **2.5× faster** than vanilla MC returns.

---

### Graph 9: Dynamic Enemy Tracking (17_dynamic_enemy_tracking.png)

**What it shows**: Adaptation to moving enemies

- **Left panel**: Tracking error over time
  - MARL-PPO: Low error (~5m)
  - Static baseline: High error (~25m)
- **Right panel**: Trajectory visualization
  - Blue: Enemy swarm centroid path
  - Green: MARL-PPO jammer tracking path
  - Red X: Static baseline (fixed position)

**Key insight**: Our method adapts to dynamic enemy motion, unlike static baselines.

---

## 12. Key Code Walkthrough

### Start-to-End Flow

```
1. Config Loading (config.json)
   └── Hyperparameters, physics params

2. Environment Creation (jammer_env.py)
   ├── Enemy swarm initialization
   ├── DBSCAN clustering
   └── Jammer placement

3. Agent Creation (ppo_agent.py)
   ├── Actor network (policy)
   └── Critic network (value)

4. Training Loop (trainer.py)
   ├── Rollout collection
   │   ├── env.step(action)
   │   ├── Physics computation (FSPL, jamming)
   │   └── Reward computation
   │
   └── PPO Update
       ├── Compute advantages (GAE)
       ├── Clipped surrogate loss
       └── Gradient descent

5. Evaluation & Visualization (generate_graphs.py)
   └── 17 publication-quality graphs
```

### Critical Files Summary

| File                                  | Purpose          | Key Functions                                   |
| ------------------------------------- | ---------------- | ----------------------------------------------- |
| `src/physics/fspl.py`                 | RF propagation   | `received_power_watts()`, `compute_jam_range()` |
| `src/physics/communication_graph.py`  | Graph theory     | `compute_laplacian()`, `compute_lambda2()`      |
| `src/physics/jamming.py`              | Link disruption  | `compute_disrupted_links()`                     |
| `src/clustering/dbscan_clustering.py` | Enemy clustering | `DBSCANClusterer.fit()`                         |
| `src/environment/jammer_env.py`       | RL environment   | `reset()`, `step()`                             |
| `src/environment/reward.py`           | Reward function  | `RewardCalculator.compute()`                    |
| `src/agents/ppo_agent.py`             | PPO algorithm    | `get_action()`, `update()`                      |
| `train.py`                            | Entry point      | `train()`                                       |

---

## 13. Conclusion

### Summary

1. **Problem**: Disrupt enemy drone swarm communication
2. **Method**: MARL with PPO, FSPL-based physics, Graph Laplacian theory
3. **Results**: 78% λ₂ reduction, scalable to N=100, M=40

### Key Contributions

1. **Physically-grounded model**: FSPL instead of crude distance thresholds
2. **Scalable MARL**: PPO with parameter sharing beats Q-table approaches
3. **Theoretically sound**: Proposition 1 guarantees disconnection when λ₂ → 0
4. **Comprehensive evaluation**: 17 graphs covering all aspects

### Future Work

- 3D environment extension
- Real-time deployment on physical drones
- Multi-swarm adversarial scenarios

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    MARL JAMMER - Quick Reference            │
├─────────────────────────────────────────────────────────────┤
│ Objective:  Minimize λ₂ (algebraic connectivity)            │
│ Method:     MARL + PPO with parameter sharing              │
│ Physics:    FSPL-based RF propagation                      │
│ Clustering: DBSCAN (eps=30m, min_samples=2)                │
│                                                             │
│ Key Parameters:                                             │
│   N = 30 enemies    │  M = 6 jammers                       │
│   Arena = 150m      │  Jam range ≈ 18m                     │
│   lr = 3e-4         │  clip_eps = 0.2                      │
│   rollout = 1024    │  batch = 128                         │
│                                                             │
│ Result: 78% λ₂ reduction in 200k steps (19.7 min)          │
└─────────────────────────────────────────────────────────────┘
```

---

_Document prepared for professor presentation - February 23, 2026_
