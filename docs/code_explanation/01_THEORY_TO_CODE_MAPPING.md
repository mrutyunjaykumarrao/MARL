# Theory to Code Mapping: Complete Guide

## How Every Equation from PROJECT_MASTER_GUIDE_v2.md is Implemented

This document maps each theoretical equation to its exact implementation in the codebase.

---

## 📚 Table of Contents

1. [FSPL Communication Model](#1-fspl-communication-model)
2. [Adjacency Matrix Construction](#2-adjacency-matrix-construction)
3. [Graph Laplacian & Lambda-2](#3-graph-laplacian--lambda-2)
4. [Jamming Disruption Model](#4-jamming-disruption-model)
5. [5-Term Reward Function](#5-5-term-reward-function)
6. [Actor Network Architecture](#6-actor-network-architecture)
7. [Critic Network Architecture](#7-critic-network-architecture)
8. [PPO Algorithm](#8-ppo-algorithm)
9. [GAE Advantage Estimation](#9-gae-advantage-estimation)

---

## 1. FSPL Communication Model

### Theory (Section 3.2)

**Free-Space Path Loss Formula:**

```
FSPL(i,j) = 20*log₁₀(d_ij) + 20*log₁₀(f) + 20*log₁₀(4π/c)
```

**Received Power:**

```
P_R(i,j) = P_tx * (c / (4πfd_ij))²
```

### Code Implementation

**File:** `src/physics/fspl.py`

```python
# Lines 30-37: Physical Constants
SPEED_OF_LIGHT = 3.0e8  # c = 3×10⁸ m/s
FSPL_CONSTANT_DB = 20 * np.log10(4 * np.pi / SPEED_OF_LIGHT)  # ≈ -147.55 dB

# Lines 98-128: FSPL in dB
def fspl_db(distance, frequency, eps=1e-10):
    """
    FSPL = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)
    """
    d = np.maximum(distance, eps)  # Prevent log(0)
    fspl = 20 * np.log10(d) + 20 * np.log10(frequency) + FSPL_CONSTANT_DB
    return fspl

# Lines 165-200: Received Power in Watts
def received_power_watts(tx_power_watts, distance, frequency, eps=1e-10):
    """
    P_R = P_tx * (c / (4πfd))²

    This is THE core formula from Section 3.2
    """
    d = np.maximum(distance, eps)
    wavelength = SPEED_OF_LIGHT / frequency  # λ = c/f

    # P_R = P_tx * (λ / (4πd))² = P_tx * (c / (4πfd))²
    received = tx_power_watts * (wavelength / (4 * np.pi * d)) ** 2
    return received
```

### Line-by-Line Explanation

| Line                                  | Code                | Theory Mapping           |
| ------------------------------------- | ------------------- | ------------------------ |
| `SPEED_OF_LIGHT = 3.0e8`              | Physical constant c | Speed of light in vacuum |
| `np.log10(d)`                         | `log₁₀(d_ij)`       | Distance component       |
| `np.log10(frequency)`                 | `log₁₀(f)`          | Frequency component      |
| `(wavelength / (4 * np.pi * d)) ** 2` | `(c / (4πfd))²`     | FSPL linear factor       |

---

## 2. Adjacency Matrix Construction

### Theory (Section 3.2)

**Link Exists if:**

```
A[i,j] = 1  iff  P_R(i,j) ≥ P_sens  AND  link(i,j) is not jammed
```

Where `P_sens` is receiver sensitivity (default: -90 dBm = 10⁻¹² W)

### Code Implementation

**File:** `src/physics/communication_graph.py`

```python
# Lines 45-95: Adjacency Matrix
def compute_adjacency_matrix(positions, tx_power_watts, sensitivity_watts,
                              frequency, jammed_links=None, eps=1e-10):
    """
    A[i,j] = 1 if:
        1. P_R(i,j) >= P_sens (received power above sensitivity)
        2. Link is not jammed (jammed_links[i,j] == False)
    """
    N = positions.shape[0]

    # Step 1: Compute received power for all pairs
    P_R = compute_pairwise_received_power(positions, tx_power_watts, frequency, eps)

    # Step 2: Links exist where received power >= sensitivity
    A = (P_R >= sensitivity_watts).astype(np.float64)

    # Step 3: Remove self-loops (diagonal = 0)
    np.fill_diagonal(A, 0)

    # Step 4: Apply jamming mask
    if jammed_links is not None:
        A = A * (~jammed_links).astype(np.float64)  # Where jammed, set A=0

    return A
```

### Visual Explanation

```
Enemy Positions:        Distance Matrix:        Received Power:         Adjacency:
 D1(10,10)             D1   D2   D3            D1    D2    D3          D1  D2  D3
 D2(50,10)         D1 [ 0   40   90]       D1 [inf  high  low ]    D1 [ 0   1   0 ]
 D3(100,10)        D2 [40    0   50]       D2 [high inf  med ]    D2 [ 1   0   1 ]
                   D3 [90   50    0]       D3 [low  med  inf ]    D3 [ 0   1   0 ]

                   d_12 = 40m               P_R(1,2) > P_sens       Connected!
                   d_13 = 90m               P_R(1,3) < P_sens       Not connected
```

---

## 3. Graph Laplacian & Lambda-2

### Theory (Section 3.4-3.5)

**Degree Matrix:**

```
D[i,i] = Σⱼ A[i,j]  (sum of row i)
```

**Laplacian:**

```
L = D - A
```

**Lambda-2 (Fiedler Value):**

```
Eigenvalues of L: 0 = λ₁ ≤ λ₂ ≤ ... ≤ λₙ
λ₂ > 0  ⟺  Graph is connected
λ₂ = 0  ⟺  Graph is disconnected
```

### Code Implementation

**File:** `src/physics/communication_graph.py`

```python
# Lines 160-175: Degree Matrix
def compute_degree_matrix(A):
    """D[i,i] = sum of row i of A"""
    degrees = np.sum(A, axis=1)  # Sum each row
    D = np.diag(degrees)         # Put on diagonal
    return D

# Lines 177-195: Laplacian
def compute_laplacian(A):
    """L = D - A"""
    D = compute_degree_matrix(A)
    L = D - A
    return L

# Lines 220-280: Lambda-2 (THE KEY METRIC)
def compute_lambda2(L, use_sparse=False):
    """
    Lambda-2 = second smallest eigenvalue of L

    Properties:
    - λ₂ > 0  ⟹  graph connected (swarm can coordinate)
    - λ₂ = 0  ⟹  graph disconnected (swarm fragmented!)
    """
    N = L.shape[0]

    if N <= 1:
        return 0.0

    # Compute only 2 smallest eigenvalues (efficient!)
    # This is O(N²) instead of O(N³) for full decomposition
    eigenvalues = scipy.linalg.eigh(L, subset_by_index=[0, 1], eigvals_only=True)

    # eigenvalues[0] ≈ 0 (always), eigenvalues[1] = λ₂
    return float(eigenvalues[1])
```

### Numerical Example

```python
# Triangle Graph (fully connected, 3 nodes)
A = [[0, 1, 1],
     [1, 0, 1],
     [1, 1, 0]]

# Degree Matrix
D = [[2, 0, 0],
     [0, 2, 0],
     [0, 0, 2]]

# Laplacian L = D - A
L = [[ 2, -1, -1],
     [-1,  2, -1],
     [-1, -1,  2]]

# Eigenvalues: [0, 3, 3]
# λ₂ = 3 > 0 → Graph is connected!

# After jamming (remove edge 1-2):
A_jammed = [[0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]]

L_jammed = [[ 1, -1,  0],
            [-1,  2, -1],
            [ 0, -1,  1]]

# Eigenvalues: [0, 1, 3]
# λ₂ = 1 > 0 → Still connected (but weaker!)

# If ALL edges removed: λ₂ = 0 → Disconnected!
```

---

## 4. Jamming Disruption Model

### Theory (Section 3.3)

**Jammer k disrupts link (i,j) if:**

```
P_jam(k, m_ij) ≥ P_jam_thresh  AND  band_k = band_enemy
```

Where:

- `m_ij = (x_i + x_j) / 2` (midpoint of link)
- `P_jam = P_jammer * (c / (4πf_jam * d_km))²`

### Code Implementation

**File:** `src/physics/jamming.py`

```python
# Lines 40-65: Compute Midpoints
def compute_midpoints(positions):
    """
    m_ij = (positions[i] + positions[j]) / 2

    Uses broadcasting for O(N²) vectorized computation
    """
    # positions[:, None, :] shape: (N, 1, 2)
    # positions[None, :, :] shape: (1, N, 2)
    midpoints = (positions[:, None, :] + positions[None, :, :]) / 2
    return midpoints  # Shape: (N, N, 2)

# Lines 70-100: Distance from jammer to midpoints
def compute_distances_to_midpoints(jammer_positions, midpoints):
    """
    d_km = ||p_k - m_ij||

    Vectorized: No Python loops!
    """
    # jammer_positions shape: (M, 2) → (M, 1, 1, 2)
    # midpoints shape: (N, N, 2) → (1, N, N, 2)
    diff = jammer_positions[:, None, None, :] - midpoints[None, :, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=3))
    return distances  # Shape: (M, N, N)

# Lines 105-130: Jamming Power
def compute_jamming_power(distances, jammer_power_watts, frequency, eps=1e-10):
    """
    P_jam = P_jammer * (c / (4πfd))²
    """
    P_jam = received_power_watts(jammer_power_watts, distances, frequency, eps)
    return P_jam  # Shape: (M, N, N)

# Lines 160-220: Complete Disruption Logic
def compute_disrupted_links(jammer_positions, jammer_bands, enemy_positions,
                            enemy_band, jammer_power_watts, jam_threshold_watts):
    """
    CRITICAL: Band matching required!
    Wrong band = ZERO disruption even at close range
    """
    M = len(jammer_positions)
    N = len(enemy_positions)

    # Step 1: Compute midpoints
    midpoints = compute_midpoints(enemy_positions)

    # Step 2: Distances from each jammer to each midpoint
    distances = compute_distances_to_midpoints(jammer_positions, midpoints)

    # Step 3: Jamming power at each midpoint
    frequency = FREQUENCY_BANDS[enemy_band]
    P_jam = compute_jamming_power(distances, jammer_power_watts, frequency)

    # Step 4: Band matching mask (CRITICAL!)
    band_match = (jammer_bands == enemy_band)  # Shape: (M,)
    band_mask = band_match[:, None, None]       # Shape: (M, 1, 1) for broadcast

    # Step 5: Jammed if power sufficient AND band matches
    jammed_by_each = (P_jam >= jam_threshold_watts) & band_mask

    # Step 6: Link is jammed if ANY jammer disrupts it
    jammed = np.any(jammed_by_each, axis=0)  # Shape: (N, N)

    return jammed
```

### Critical Insight: Band Matching

```
Scenario 1: Wrong Band
  Enemy Band: 2 (2.4 GHz)
  Jammer Band: 0 (433 MHz)
  Distance: 1 meter (very close!)
  Result: NO DISRUPTION (band_k ≠ band_enemy)

Scenario 2: Right Band
  Enemy Band: 2 (2.4 GHz)
  Jammer Band: 2 (2.4 GHz)
  Distance: 30 meters
  Result: LINK DISRUPTED (if P_jam ≥ threshold)

This forces the agent to LEARN frequency selection!
```

---

## 5. 5-Term Reward Function

### Theory (Section 3.7)

```
R(t) = ω₁ * [1 - λ₂(t)/λ₂(0)]           # Lambda-2 reduction
     + ω₂ * (1/M) * Σ 1[band_k = band_e]  # Band match
     + ω₃ * (1/M) * Σ exp(-κ*d_centroid)  # Proximity
     - ω₄ * (1/M) * Σ ||v_k||²/v_max²     # Energy penalty
     - ω₅ * overlap_penalty               # Overlap penalty
```

### Code Implementation

**File:** `src/environment/reward.py`

```python
# Lines 70-110: Main Reward Computation
def compute(self, lambda2_current, lambda2_initial, jammer_bands, enemy_band,
            jammer_positions, centroids, velocities, jammer_assignments=None):
    """
    Compute the complete 5-term reward function.
    """
    M = jammer_positions.shape[0]

    # Term 1: λ₂ reduction (PRIMARY OBJECTIVE)
    r_lambda2 = self._compute_lambda2_reward(lambda2_current, lambda2_initial)

    # Term 2: Band match
    r_band = self._compute_band_match_reward(jammer_bands, enemy_band)

    # Term 3: Proximity to centroids
    r_proximity = self._compute_proximity_reward(jammer_positions, centroids)

    # Term 4: Energy penalty
    r_energy = self._compute_energy_penalty(velocities)

    # Term 5: Overlap penalty
    r_overlap = self._compute_overlap_penalty(jammer_positions)

    # Weighted sum (note: energy and overlap are SUBTRACTED)
    total = (
        self.omega_1 * r_lambda2      # +1.0
        + self.omega_2 * r_band       # +0.3
        + self.omega_3 * r_proximity  # +0.2
        - self.omega_4 * r_energy     # -0.1 (penalty)
        - self.omega_5 * r_overlap    # -0.2 (penalty)
    )

    return total, components

# Lines 175-190: Lambda-2 Reward
def _compute_lambda2_reward(self, lambda2_current, lambda2_initial):
    """
    [1 - λ₂(t)/λ₂(0)] ∈ [0, 1]

    0 = no reduction (λ₂ unchanged)
    1 = complete disconnection (λ₂ = 0)
    """
    if lambda2_initial <= 0:
        return 0.0  # Already disconnected

    reduction = 1.0 - (lambda2_current / lambda2_initial)
    return np.clip(reduction, 0.0, 1.0)

# Lines 195-210: Band Match Reward
def _compute_band_match_reward(self, jammer_bands, enemy_band):
    """
    (1/M) * Σ 1[band_k = band_enemy]

    Returns fraction of jammers with correct frequency.
    Random baseline: 1/4 = 0.25 (4 bands)
    """
    matches = np.sum(jammer_bands == enemy_band)
    return float(matches) / len(jammer_bands)

# Lines 240-260: Energy Penalty
def _compute_energy_penalty(self, velocities):
    """
    (1/M) * Σ ||v_k||²/v_max²

    High speed = more energy = penalty
    Encourages efficient hovering over targets
    """
    speed_sq = np.sum(velocities ** 2, axis=1)  # ||v||² for each jammer
    normalized = speed_sq / (self.v_max ** 2)
    return float(np.mean(normalized))
```

### Reward Weights (Default Values)

| Weight | Value | Purpose                       |
| ------ | ----- | ----------------------------- |
| ω₁     | 1.0   | λ₂ reduction (most important) |
| ω₂     | 0.3   | Band match reward             |
| ω₃     | 0.2   | Centroid proximity            |
| ω₄     | 0.1   | Energy penalty (negative)     |
| ω₅     | 0.2   | Overlap penalty (negative)    |

---

## 6. Actor Network Architecture

### Theory (Section 3.8)

```
Input: s_j ∈ ℝ⁵

Shared trunk: FC(5→128) → LayerNorm → ReLU → FC(128→128) → LayerNorm → ReLU

Continuous head: μ = FC(128→2), log_σ = clamp(FC(128→2), -2, 2)
Discrete head:   logits = FC(128→4), π_band = softmax(logits)

log π(a|s) = log N(Vx,Vy | μ, σ) + log Categorical(band | π_band)
```

### Code Implementation

**File:** `src/agents/actor.py`

```python
# Lines 45-100: Actor Network Definition
class Actor(nn.Module):
    def __init__(self, obs_dim=5, hidden_dim=128, v_max=5.0, num_bands=4,
                 log_std_min=-2.0, log_std_max=2.0):
        super().__init__()

        # Shared trunk (as per theory)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),     # FC(5→128)
            nn.LayerNorm(hidden_dim),           # LayerNorm
            nn.ReLU(),                          # ReLU
            nn.Linear(hidden_dim, hidden_dim),  # FC(128→128)
            nn.LayerNorm(hidden_dim),           # LayerNorm
            nn.ReLU()                           # ReLU
        )

        # Continuous head: velocity (Vx, Vy)
        self.mu_head = nn.Linear(hidden_dim, 2)      # FC(128→2) for mean
        self.log_std_head = nn.Linear(hidden_dim, 2) # FC(128→2) for log_std

        # Discrete head: band selection
        self.band_head = nn.Linear(hidden_dim, num_bands)  # FC(128→4)

    def forward(self, obs):
        """
        Forward pass through the network.

        Input:  obs ∈ ℝ^(batch × 5)
        Output: μ, log_σ, band_logits
        """
        features = self.trunk(obs)  # Shared features

        # Continuous: velocity
        mu = self.mu_head(features)            # Shape: (batch, 2)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # Discrete: band
        band_logits = self.band_head(features)  # Shape: (batch, 4)

        return mu, log_std, band_logits

    def sample(self, obs, deterministic=False):
        """
        Sample actions from policy.

        Returns: actions, log_probs, entropy
        """
        mu, log_std, band_logits = self.forward(obs)
        std = torch.exp(log_std)

        # Create distributions
        velocity_dist = Normal(mu, std)          # Gaussian for velocity
        band_dist = Categorical(logits=band_logits)  # Categorical for band

        if deterministic:
            velocity = mu                         # Use mean
            band = torch.argmax(band_logits, dim=-1)  # Use mode
        else:
            velocity = velocity_dist.rsample()   # Sample from Gaussian
            band = band_dist.sample()            # Sample from Categorical

        # Clamp velocity to [-v_max, v_max]
        velocity_clamped = torch.clamp(velocity, -self.v_max, self.v_max)

        # Log probabilities
        log_prob_velocity = velocity_dist.log_prob(velocity).sum(dim=-1)
        log_prob_band = band_dist.log_prob(band)
        log_prob = log_prob_velocity + log_prob_band  # Total log π(a|s)

        return actions, log_prob, entropy
```

---

## 7. Critic Network Architecture

### Theory (Section 3.9)

```
Input: s_pooled = (1/M) * Σ s_j  (mean-pooled global state)

FC(5→128) → ReLU → FC(128→128) → ReLU → FC(128→1) = V_φ(s)
```

### Code Implementation

**File:** `src/agents/critic.py`

```python
# Lines 35-80: Critic Network
class Critic(nn.Module):
    def __init__(self, obs_dim=5, hidden_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),    # FC(5→128)
            nn.ReLU(),                         # ReLU
            nn.Linear(hidden_dim, hidden_dim), # FC(128→128)
            nn.ReLU(),                         # ReLU
            nn.Linear(hidden_dim, 1)           # FC(128→1) = V(s)
        )

    def forward_pooled(self, obs):
        """
        Input: obs of shape (M, 5) - all agent observations
        Output: V(s) scalar

        Uses mean-pooling for CTDE:
        s_pooled = (1/M) * Σⱼ sⱼ
        """
        if obs.dim() == 2 and obs.shape[0] > 1:
            # Mean pool across agents
            obs_pooled = obs.mean(dim=0, keepdim=True)  # (1, 5)
        else:
            obs_pooled = obs

        return self.network(obs_pooled)  # Returns V(s)
```

### Why Mean-Pooling?

```
Without pooling: Critic input size = M × 5 (varies with M)
With pooling:    Critic input size = 5 (fixed!)

This enables:
- Same model for M=4 jammers and M=40 jammers
- Scalability without architecture change
- CTDE: Centralized Training (critic sees all), Decentralized Execution (actor sees local)
```

---

## 8. PPO Algorithm

### Theory (Section 3.10)

**Clipped Surrogate Loss:**

```
r_t(θ) = exp(log π_θ(a|s) - log π_θ_old(a|s))

L_CLIP = E[min(r_t × A_t, clip(r_t, 1-ε, 1+ε) × A_t)]

L_total = -L_CLIP + c₁ × (V_φ(s) - R_t)² - c₂ × H(π)
```

### Code Implementation

**File:** `src/agents/ppo_agent.py`

```python
# Lines 270-340: PPO Update
def update(self, last_obs, last_done):
    """
    Perform PPO update after collecting a rollout.
    """
    # Step 1: Compute GAE advantages
    self.buffer.compute_returns_and_advantages(last_value, last_done)

    # Step 2: PPO epochs (K=10 by default)
    for epoch in range(self.n_epochs):
        for batch in self.buffer.get_minibatches(self.batch_size):

            # Get current policy probabilities
            log_probs, entropy = self.actor.evaluate_actions(obs, actions)
            values = self.critic.forward_pooled(obs)

            # Compute probability ratio
            # r_t = π_θ(a|s) / π_θ_old(a|s) = exp(log π_θ - log π_old)
            ratio = torch.exp(log_probs - old_log_probs)

            # Clipped surrogate loss
            surr1 = ratio * advantages            # r_t × A_t
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()  # Negative for gradient ascent

            # Value (critic) loss
            critic_loss = self.c1 * ((values - returns) ** 2).mean()

            # Entropy bonus (encourages exploration)
            entropy_loss = -self.c2 * entropy.mean()

            # Backward pass and optimization
            self.actor_optimizer.zero_grad()
            (actor_loss + entropy_loss).backward()
            clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
```

---

## 9. GAE Advantage Estimation

### Theory (Section 3.10)

```
δ_t = r_t + γ × V(s_{t+1}) × (1-done_t) - V(s_t)

A_t = δ_t + (γλ) × A_{t+1} × (1-done_t)   [computed backwards]

R_t = A_t + V(s_t)

A_t ← (A_t - mean(A)) / (std(A) + 1e-8)   [normalize]
```

### Code Implementation

**File:** `src/agents/rollout_buffer.py`

```python
# Lines 150-200: GAE Computation
def compute_returns_and_advantages(self, last_values, last_dones):
    """
    Compute GAE advantages and returns.

    GAE reduces variance while maintaining low bias.
    """
    size = self.pos

    # Extend arrays for bootstrapping
    values_extended = np.zeros(size + 1)
    values_extended[:size] = self.values[:size]
    values_extended[size] = last_values * (1 - float(last_dones))

    # GAE computation (BACKWARDS loop - critical!)
    advantages = np.zeros(size)
    last_gae = 0

    for t in reversed(range(size)):  # t = T-1, T-2, ..., 0
        next_non_terminal = 1.0 - self.dones[t + 1]
        next_values = values_extended[t + 1]

        # TD error: δ_t = r_t + γV(s_{t+1})(1-done) - V(s_t)
        delta = (
            self.rewards[t]
            + self.gamma * next_values * next_non_terminal
            - values_extended[t]
        )

        # GAE: A_t = δ_t + γλ × A_{t+1} × (1-done)
        advantages[t] = last_gae = (
            delta
            + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        )

    # Returns: R_t = A_t + V(s_t)
    self.returns[:size] = advantages + self.values[:size]

    # Normalize advantages (crucial for stable training!)
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    self.advantages[:size] = (advantages - adv_mean) / adv_std
```

---

## 🗣️ Hinglish Technical Summary

**Theory to Code Mapping ka Summary:**

"Dekho, humne PROJECT_MASTER_GUIDE mein jo bhi equations likhi hain - FSPL formula, adjacency matrix, Laplacian, lambda-2, reward function - sab kuch code mein exactly waise hi implement kiya hai. Jaise theory mein likha `P_R = P_tx * (c / (4πfd))²`, wahi exactly `received_power_watts()` function mein hai. Lambda-2 jo second smallest eigenvalue hai Laplacian ka, wohi `compute_lambda2()` mein scipy.linalg.eigh se compute hota hai with `subset_by_index=[0,1]` for efficiency.

Reward function ke 5 terms - lambda2 reduction, band match, proximity, energy penalty, overlap penalty - sab separately compute hote hain aur weighted sum kiya jaata hai exactly as per theory. Actor network mein shared trunk hai with LayerNorm (BatchNorm nahi because multi-agent scenario hai), continuous head velocity ke liye (Gaussian distribution), discrete head band selection ke liye (Categorical distribution). Critic mean-pooled observations leta hai taaki input size fixed rahe irrespective of M agents - yahi scalability ka secret hai.

PPO algorithm mein clipped surrogate loss use hota hai with ratio r = exp(log*π_new - log*π_old), aur GAE backwards loop mein compute hota hai. Yeh sab industry standard implementation hai, Stable-Baselines3 jaisa. Professor ko bolo: 'Every equation from theory has a 1-to-1 mapping to code, physically grounded and mathematically correct.'"

---

**Next:** See `02_STEP_BY_STEP_DEMO.md` for what to run in order while presenting.
