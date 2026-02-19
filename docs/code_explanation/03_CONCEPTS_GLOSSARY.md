# Concepts Glossary: RL and MARL Terms Explained

## Complete Reference for Understanding the Codebase

This document explains every concept, variable, and term used in the MARL Jammer project.

---

## 📚 Table of Contents

1. [Reinforcement Learning Basics](#1-reinforcement-learning-basics)
2. [PPO-Specific Terms](#2-ppo-specific-terms)
3. [Multi-Agent Concepts](#3-multi-agent-concepts)
4. [Physics & Graph Theory](#4-physics--graph-theory)
5. [Training Variables](#5-training-variables)
6. [Neural Network Terms](#6-neural-network-terms)
7. [Code Variables Reference](#7-code-variables-reference)

---

## 1. Reinforcement Learning Basics

### Step (Timestep)

**Definition:** A single interaction between agent and environment.

```
Time t → t+1:
  1. Agent observes state s_t
  2. Agent takes action a_t
  3. Environment returns reward r_t and next state s_{t+1}
```

**In our code:**

```python
# One step in training loop
obs, reward, done, info = env.step(action)
```

**Scale:**

- 1 step = 0.1 seconds in simulation
- 200,000 steps = 20,000 simulated seconds

---

### Episode

**Definition:** A complete run from start to termination.

```
Episode = step_0 → step_1 → ... → step_T (done=True)
```

**In our project:**

- Episode length: 100 steps (fixed)
- Episode ends when: step counter reaches max OR early termination
- 200,000 training steps ÷ 100 steps/episode ≈ 2,000 episodes

**Code:**

```python
for step in range(max_episode_steps):
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break  # Episode ends
```

---

### Rollout

**Definition:** A collection of steps gathered BEFORE updating the neural network.

```
Rollout = {(s_1, a_1, r_1), (s_2, a_2, r_2), ..., (s_n, a_n, r_n)}
```

**In our config:**

```python
rollout_size = 2048  # Collect 2048 steps before updating
```

**Why rollout matters:**

- Too short (512): High variance, unstable updates
- Too long (8192): Outdated samples, slow learning
- Sweet spot (2048): Enough variance reduction, still relevant data

**Important:** Rollout is stored in a **buffer** during training but NOT needed for deployment.

---

### Reward

**Definition:** Scalar feedback signal that defines the objective.

**Our 5-term reward:**

```python
R = ω₁*(λ₂ reduction)     # +1.0 weight (most important)
  + ω₂*(band match)       # +0.3 weight
  + ω₃*(proximity)        # +0.2 weight
  - ω₄*(energy use)       # -0.1 penalty
  - ω₅*(overlap)          # -0.2 penalty
```

**Typical values:**

- Episode start reward: ~13 (random behavior)
- Trained policy reward: ~82 (learned behavior)

---

### Return (G)

**Definition:** Cumulative discounted future reward.

```
G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γⁿ*r_{t+n}
```

**Why discount (γ < 1)?**

- Future is uncertain
- Prefer immediate reward over distant reward
- Our γ = 0.99 (slight discounting)

---

### Observation / State

**Definition:** What the agent "sees" at each timestep.

**Our observation vector (5 dimensions per agent):**

```python
obs = [
    Δx,    # Relative x to assigned centroid (normalized)
    Δy,    # Relative y to assigned centroid (normalized)
    v_x,   # Current x velocity (normalized)
    v_y,   # Current y velocity (normalized)
    band   # Enemy band (one-hot or integer encoding)
]
```

**Shape:** `(M, 5)` where M = number of jammers

---

### Action

**Definition:** What the agent does in response to observation.

**Our action space (hybrid: continuous + discrete):**

```python
action = {
    'velocity': [v_x, v_y],  # Continuous ∈ [-5, 5]²
    'band': k                # Discrete ∈ {0, 1, 2, 3}
}
```

**Action shape:** `(M, 3)` - 2 continuous + 1 discrete per agent

---

### Policy (π)

**Definition:** Mapping from states to actions.

```
π(a|s) = probability of taking action a in state s
```

**Types:**

- **Deterministic:** π(s) = a (one action per state)
- **Stochastic:** π(a|s) = P(a|s) (distribution over actions)

**Our policy (stochastic):**

```python
# Continuous: Gaussian distribution
velocity ~ N(μ, σ²)  where μ, σ = Actor(obs)

# Discrete: Categorical distribution
band ~ Cat(logits)   where logits = Actor(obs)
```

---

### Value Function (V)

**Definition:** Expected return from a state under current policy.

```
V(s) = E[G_t | s_t = s, policy π]
```

**Our implementation:**

- Critic network outputs V(s)
- Uses mean-pooled observations for CTDE

---

## 2. PPO-Specific Terms

### Clip Range (ε)

**Definition:** How much the policy ratio is allowed to change.

```python
clip_eps = 0.2  # Allow ratio ∈ [0.8, 1.2]
```

**Formula:**

```
ratio = π_new(a|s) / π_old(a|s)
clipped_ratio = clip(ratio, 1-ε, 1+ε)
```

**Purpose:** Prevents catastrophic policy updates.

---

### KL Divergence

**Definition:** Measures "distance" between old and new policy distributions.

```
KL(π_old || π_new) = E[log(π_old/π_new)]
```

**Our implementation:**

```python
# Approximate KL using log probability difference
approx_kl = (old_log_probs - new_log_probs).mean()

# Early stopping if KL too high
if approx_kl > target_kl:  # target_kl = 0.03
    break  # Stop PPO epochs
```

**Why:**

- KL > 0.03: Policy changing too fast, risk of collapse
- KL < 0.001: Policy barely changing, possibly stuck

---

### Generalized Advantage Estimation (GAE)

**Definition:** Method to compute advantage with bias-variance tradeoff.

```
δ_t = r_t + γV(s_{t+1}) - V(s_t)   # TD error
A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}     # GAE
```

**Parameters:**

```python
gamma = 0.99       # Discount factor
gae_lambda = 0.95  # GAE λ (higher = less bias, more variance)
```

**λ = 0:** Just TD error (high bias, low variance)
**λ = 1:** Monte Carlo estimate (low bias, high variance)
**λ = 0.95:** Sweet spot

---

### Entropy Bonus

**Definition:** Reward for policy randomness (exploration).

```
Entropy H(π) = -Σ π(a|s) log π(a|s)
```

**In loss:**

```python
entropy_coef = 0.01
loss = -actor_loss - entropy_coef * entropy  # Negative = maximize entropy
```

**Effect:**

- High entropy: Agent tries many actions
- Low entropy: Agent deterministic (exploitation)

---

### PPO Epochs (n_epochs)

**Definition:** How many times to reuse rollout data.

```python
n_epochs = 10  # Train on same rollout 10 times
```

**Why multiple epochs?**

- More sample efficiency
- But too many epochs → overfitting to old data

---

### Mini-batch

**Definition:** Subset of rollout used for one gradient update.

```python
rollout_size = 2048
batch_size = 256
num_batches = 2048 / 256 = 8  # 8 batches per epoch
```

**Total gradient updates per rollout:**

```
updates = n_epochs × num_batches = 10 × 8 = 80
```

---

## 3. Multi-Agent Concepts

### Parameter Sharing

**Definition:** All agents use the same neural network weights.

```python
# One actor network, shared by all M agents
actor = Actor()  # θ

# Each agent j observes s_j, takes action a_j = π_θ(s_j)
for j in range(M):
    action_j = actor(obs[j])
```

**Benefits:**

- 4 agents = 4× samples for same network
- Automatic generalization
- Scalability (M=4 → M=40 without retraining)

---

### CTDE (Centralized Training, Decentralized Execution)

**Definition:** Critic sees everything during training, actors see only local.

```
Training:
  Critic input = mean(all observations) → V(s) = global value

Execution:
  Actor_j input = obs_j only → a_j = local action
```

**Why:**

- Training: Use all information for better value estimates
- Execution: No communication needed between agents

---

### Homogeneous Agents

**Definition:** All agents are identical (same type, same action space).

Our project: All 4 jammers are identical drones.

**Alternative (heterogeneous):** Different agent types with different abilities.

---

### Emergent Coordination

**Definition:** Coordination that arises without explicit programming.

```
We did NOT program: "Jammer 1 go left, Jammer 2 go right"
But they LEARNED: To spread out and cover different clusters
```

This happens because:

1. Overlap penalty discourages agents from same location
2. λ₂ reduction requires covering multiple links
3. Shared experience through parameter sharing

---

## 4. Physics & Graph Theory

### Free-Space Path Loss (FSPL)

**Definition:** Signal power loss over distance in free space.

```
FSPL(dB) = 20log₁₀(d) + 20log₁₀(f) + 20log₁₀(4π/c)
```

**Variables:**

- d: Distance in meters
- f: Frequency in Hz
- c: Speed of light (3×10⁸ m/s)

---

### Frequency Bands

**Our 4 bands:**

```python
FREQUENCY_BANDS = {
    0: 433e6,    # 433 MHz (ISM band)
    1: 915e6,    # 915 MHz (ISM band)
    2: 2.4e9,    # 2.4 GHz (WiFi/Bluetooth)
    3: 5.8e9     # 5.8 GHz (WiFi 5)
}
```

**Jamming rule:** Jammer must be on SAME band as enemy to disrupt.

---

### Graph Laplacian (L)

**Definition:** Matrix that encodes graph structure.

```
L = D - A

Where:
  A = adjacency matrix (1 if connected, 0 otherwise)
  D = degree matrix (diagonal, D[i,i] = sum of row i in A)
```

**Properties:**

- Symmetric (L = Lᵀ)
- All eigenvalues ≥ 0
- Smallest eigenvalue = 0 (always)

---

### Lambda-2 (Fiedler Value)

**Definition:** Second smallest eigenvalue of Laplacian.

```
Eigenvalues of L: 0 = λ₁ ≤ λ₂ ≤ λ₃ ≤ ... ≤ λₙ

λ₂ > 0  →  Graph is connected
λ₂ = 0  →  Graph is disconnected
```

**Our objective:** Minimize λ₂ (disrupt connectivity)

**Our metric:** Report λ₂/λ₂₀ × 100 (percent of initial)

---

### Receiver Sensitivity

**Definition:** Minimum signal power needed to maintain connection.

```python
sensitivity_dbm = -90  # -90 dBm
sensitivity_watts = db_to_watts(sensitivity_dbm)  # ≈ 10⁻¹² W
```

**If P_R < sensitivity:** Link fails (no communication)

---

### Jamming Threshold

**Definition:** Minimum jamming power to disrupt a link.

```python
jam_threshold_dbm = -70  # -70 dBm
jam_threshold_watts = db_to_watts(jam_threshold_dbm)  # ≈ 10⁻¹⁰ W
```

**Link is jammed if:** P_jam ≥ threshold AND bands match

---

## 5. Training Variables

### total_timesteps

**Definition:** Total number of environment steps during training.

```python
total_timesteps = 200000  # 200K steps
```

**Mapping:**

- 200K steps ÷ 100 steps/episode ≈ 2000 episodes
- 200K steps ÷ 2048 rollout ≈ 97 policy updates

---

### learning_rate (lr)

**Definition:** Step size for gradient descent optimization.

```python
learning_rate = 1e-4  # 0.0001
```

**Effect:**

- Too high (3e-4): Unstable, policy collapse
- Too low (1e-5): Very slow learning
- Sweet spot (1e-4): Stable and reasonably fast

---

### max_grad_norm

**Definition:** Maximum gradient magnitude allowed.

```python
max_grad_norm = 0.5

# Applied during training:
torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
```

**Purpose:** Prevents exploding gradients from destabilizing training.

---

## 6. Neural Network Terms

### LayerNorm vs BatchNorm

**BatchNorm:** Normalizes across batch dimension

- Problem: Needs large batches, inconsistent with multi-agent

**LayerNorm:** Normalizes across feature dimension

- Works with any batch size, consistent between training/inference

**Our choice:** LayerNorm (more stable for MARL)

---

### Softmax

**Definition:** Converts logits to probabilities.

```python
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

**Used for:** Band selection (4 classes → 4 probabilities)

---

### Log-Probability

**Definition:** Logarithm of probability (used for numerical stability).

```python
log_prob = log π(a|s)

# Instead of computing π directly, we compute log π
# This avoids numerical underflow for small probabilities
```

---

### Entropy

**Definition:** Measure of randomness in distribution.

```python
# For discrete (band):
H_discrete = -Σ p_i log(p_i)

# For continuous (velocity):
H_gaussian = 0.5 * log(2πeσ²)
```

---

## 7. Code Variables Reference

### config.py Key Variables

| Variable          | Default | Description                 |
| ----------------- | ------- | --------------------------- |
| `num_jammers`     | 4       | M: Number of jammer agents  |
| `num_enemies`     | 10      | N: Number of enemy nodes    |
| `total_timesteps` | 200000  | Training steps              |
| `rollout_size`    | 2048    | Steps per rollout           |
| `learning_rate`   | 1e-4    | Adam learning rate          |
| `gamma`           | 0.99    | Discount factor             |
| `gae_lambda`      | 0.95    | GAE lambda                  |
| `clip_eps`        | 0.2     | PPO clip range              |
| `n_epochs`        | 10      | PPO epochs per update       |
| `batch_size`      | 256     | Mini-batch size             |
| `target_kl`       | 0.03    | KL early stopping threshold |

### reward.py Weight Variables

| Variable  | Default | Term            |
| --------- | ------- | --------------- |
| `omega_1` | 1.0     | λ₂ reduction    |
| `omega_2` | 0.3     | Band match      |
| `omega_3` | 0.2     | Proximity       |
| `omega_4` | 0.1     | Energy penalty  |
| `omega_5` | 0.2     | Overlap penalty |

### physics Constants

| Variable          | Value | Description             |
| ----------------- | ----- | ----------------------- |
| `SPEED_OF_LIGHT`  | 3e8   | c in m/s                |
| `P_TX_DBM`        | 20    | Transmit power (100 mW) |
| `SENSITIVITY_DBM` | -90   | Receiver sensitivity    |
| `JAM_THRESH_DBM`  | -70   | Jamming threshold       |
| `AREA_SIZE`       | 100   | Environment in meters   |

---

## 🗣️ Hinglish Glossary Summary

**Concepts ka Summary:**

"RL mein **step** matlab ek interaction hai - observe karo, action lo, reward milo. **Episode** matlab poori game - start se end tak, humare case mein 100 steps. **Rollout** matlab kitne steps collect karte ho neural network update karne se pehle - humare case mein 2048 steps.

**Policy** wo function hai jo state dekhke action deta hai. Humare paas stochastic policy hai - velocity ke liye Gaussian distribution, band ke liye Categorical distribution. **Value function** predict karta hai expected total reward ek state se.

PPO mein **clip** karte hain ratio ko taaki policy bahut zyada change na ho ek hi update mein. **KL divergence** se measure karte hain kitna change hua - 0.03 se zyada ho toh early stop karte hain.

**Lambda-2** graph ka algebraic connectivity hai - agar zero ho gaya toh graph disconnect ho gaya. Yahi humara main objective hai - enemy ka λ₂ kam karo.

**Parameter sharing** matlab ek hi neural network 4 agents share karte hain. **CTDE** matlab training mein global information use karo (critic), execution mein local only (actor).

**FSPL** real physics hai - signal power distance aur frequency ke saath decrease hoti hai. **Frequency bands** 4 hain, jamming sirf matching band pe kaam karti hai."

---

**Next:** See `04_DEPLOYMENT_GUIDE.md` for understanding where weights are stored and how to deploy.
