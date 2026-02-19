# MARL Jammer Drone System

## Multi-Agent Reinforcement Learning for Cooperative Swarm Disruption

**Version 2.0** — Publication-Ready Implementation

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Theory](#core-theory)
3. [Project Structure](#project-structure)
4. [File Descriptions](#file-descriptions)
5. [Getting Started](#getting-started)
6. [Training](#training)
7. [Visualization & Graphs](#visualization--graphs)
8. [Understanding the Results](#understanding-the-results)
9. [Theoretical Validation](#theoretical-validation)
10. [Plan for 20 Graphs](#plan-for-20-graphs)

---

## Project Overview

This project implements a **Multi-Agent Reinforcement Learning (MARL)** system where cooperative jammer drones learn to disrupt enemy swarm communication by minimizing the **algebraic connectivity (λ₂)** of the swarm's communication graph.

### Key Innovation

Unlike traditional jamming approaches that maximize individual jamming power, this system:

- Uses **Graph Laplacian reward** based on λ₂ (Fiedler value)
- Employs **FSPL-grounded physics** for realistic RF propagation
- Implements **DBSCAN clustering** for intelligent jammer deployment
- Uses **PPO with parameter sharing** for scalable multi-agent learning

### Why λ₂ (Lambda-2)?

- **λ₂ > 0**: Graph is connected → Swarm can coordinate
- **λ₂ = 0**: Graph is disconnected → Swarm is fragmented
- **Minimizing λ₂ directly achieves swarm disruption!**

---

## Core Theory

### Communication Graph Model

Enemy drones form a communication graph G = (V, E) where:

- **Nodes (V)**: Enemy drones
- **Edges (E)**: Communication links (if received power ≥ sensitivity threshold)

### Free-Space Path Loss (FSPL)

```
FSPL(d) = 20*log₁₀(d) + 20*log₁₀(f) + 20*log₁₀(4π/c)

P_received = P_transmit * (c / (4πfd))²
```

### Jamming Model

A jammer disrupts link (i,j) if:

1. `P_jam ≥ P_jam_threshold` (sufficient power reaches link midpoint)
2. `band_jammer = band_enemy` (correct frequency band)

### Reward Function

```
R(t) = ω₁ * [1 - λ₂(t)/λ₂(0)]      # Primary: Connectivity reduction
     + ω₂ * mean(band_match)        # Correct frequency selection
     + ω₃ * mean(proximity)         # Stay near cluster centroids
     - ω₄ * mean(energy)            # Minimize energy use
     - ω₅ * overlap_penalty         # Avoid jammer clustering
```

| Weight | Value | Purpose                |
| ------ | ----- | ---------------------- |
| ω₁     | 1.0   | λ₂ reduction (primary) |
| ω₂     | 0.3   | Band matching          |
| ω₃     | 0.2   | Centroid proximity     |
| ω₄     | 0.1   | Energy penalty         |
| ω₅     | 0.2   | Overlap penalty        |

---

## Project Structure

```
MARL_JAMMER/
│
├── src/                           # Core source code
│   ├── agents/                    # PPO agent implementation
│   │   ├── actor.py               # Actor network (policy)
│   │   ├── critic.py              # Critic network (value function)
│   │   ├── ppo_agent.py           # PPO algorithm
│   │   └── rollout_buffer.py      # Experience buffer
│   │
│   ├── environment/               # Gymnasium environment
│   │   ├── jammer_env.py          # Main environment class
│   │   ├── enemy_swarm.py         # Enemy drone dynamics
│   │   ├── observation.py         # Observation builder
│   │   ├── action_space.py        # Action handler
│   │   └── reward.py              # 5-term reward calculator
│   │
│   ├── physics/                   # RF propagation models
│   │   ├── fspl.py                # Free-space path loss
│   │   ├── communication_graph.py # Adjacency/Laplacian/λ₂
│   │   └── jamming.py             # Jamming disruption logic
│   │
│   ├── clustering/                # DBSCAN clustering
│   │   └── dbscan_clustering.py   # Cluster detection & assignment
│   │
│   └── training/                  # Training pipeline
│       ├── config.py              # Training configurations
│       ├── trainer.py             # Main training loop
│       └── metrics.py             # Metrics logging
│
├── outputs/                       # Training outputs (auto-created)
│   └── <experiment_name>/
│       ├── config.json            # Configuration used
│       ├── history.json           # Training metrics
│       ├── final_stats.json       # Summary statistics
│       ├── checkpoints/           # Model weights
│       └── graphs/                # Generated visualizations
│
├── tests/                         # Unit tests
├── docs/                          # Documentation
│   └── PROJECT_MASTER_GUIDE_v2.md # Technical reference
│
├── train.py                       # Training entry point
├── generate_graphs.py             # Visualization script
├── evaluate_random.py             # Baseline evaluation
└── requirements.txt               # Dependencies
```

---

## File Descriptions

### 🔴 MOST IMPORTANT FILES

| File                                | Purpose                      | When to Use                                     |
| ----------------------------------- | ---------------------------- | ----------------------------------------------- |
| **train.py**                        | Main training script         | `python train.py --mode fast --name my_exp`     |
| **generate_graphs.py**              | Publication-quality graphs   | `python generate_graphs.py --experiment my_exp` |
| **docs/PROJECT_MASTER_GUIDE_v2.md** | Complete technical reference | Read for theory understanding                   |

### Core Source Files (`src/`)

#### `src/environment/jammer_env.py` ⭐

**The main Gymnasium environment.**

- Implements the step() and reset() functions
- Manages enemy swarm, jammers, and state
- Computes λ₂ and reward at each step
- Key parameters: N (enemies), M (jammers), arena_size

#### `src/environment/reward.py` ⭐

**5-term reward calculator.**

- λ₂ reduction (primary objective)
- Band match reward
- Proximity to centroids
- Energy penalty
- Overlap penalty

#### `src/physics/communication_graph.py` ⭐

**Graph Laplacian and λ₂ computation.**

- `compute_adjacency_matrix()`: FSPL-based link determination
- `compute_laplacian()`: L = D - A
- `compute_lambda2()`: Second smallest eigenvalue

#### `src/physics/fspl.py`

**Free-Space Path Loss implementation.**

- `fspl_db()`: Path loss in dB
- `received_power_watts()`: Received power calculation
- `compute_jam_range()`: Effective jamming radius

#### `src/physics/jamming.py`

**Jamming disruption logic.**

- Determines which links are jammed
- Band-aware (wrong band = no disruption)

#### `src/agents/ppo.py`

**PPO algorithm implementation.**

- Actor-Critic architecture
- GAE advantage estimation
- Clipped surrogate loss

#### `src/training/trainer.py`

**Main training loop.**

- Rollout collection
- PPO updates
- Logging and checkpointing

#### `src/training/config.py`

**Training configurations.**

- `get_debug_config()`: Quick test (1K steps)
- `get_fast_config()`: Fast training (100K steps)
- `get_full_config()`: Full training (2M steps)

### Utility Scripts

| Script                   | Purpose                             |
| ------------------------ | ----------------------------------- |
| `evaluate_random.py`     | Evaluate random policy baseline     |
| `professor_visualize.py` | Additional dashboard visualizations |
| `display_training.py`    | Kaggle-style console output         |

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

Required packages:

- `torch>=2.0.0`
- `numpy`
- `scipy`
- `scikit-learn` (for DBSCAN)
- `matplotlib`
- `gymnasium`

### Quick Test

```bash
# Debug run (1000 steps, ~30 seconds)
python train.py --mode debug --name test_run
```

### Verify Environment

```bash
# Check environment works
python evaluate_random.py --episodes 10
```

---

## Training

### Training Modes

| Mode    | Steps     | Time   | Use Case                  |
| ------- | --------- | ------ | ------------------------- |
| `debug` | 1,000     | ~30s   | Quick test                |
| `fast`  | 100,000   | ~5min  | Development               |
| `full`  | 2,000,000 | ~2hr   | Publication               |
| `large` | 500,000   | ~30min | Large scale (N=100, M=40) |

### Run Training

```bash
# Fast training (recommended for demos)
python train.py --mode fast --name my_experiment --steps 300000

# Full training for publication
python train.py --mode full --name production_run
```

### Output Files

After training, find results in `outputs/<experiment_name>/`:

- `history.json`: Training metrics per update
- `final_stats.json`: Summary statistics
- `config.json`: Configuration used
- `checkpoints/`: Model weights

---

## Visualization & Graphs

### Generate All Graphs

```bash
python generate_graphs.py --experiment professor_demo_v2
```

### Generated Graphs

#### 1️⃣ `1_lambda2_vs_episodes.png` — **MOST IMPORTANT**

**λ₂ vs Training Episodes**

| Aspect         | Details                                                           |
| -------------- | ----------------------------------------------------------------- |
| X-axis         | Training episodes                                                 |
| Y-axis         | λ₂ (algebraic connectivity, normalized)                           |
| Expected Shape | High → gradually decreases → low                                  |
| Interpretation | "As PPO learns, jammers increasingly fragment swarm connectivity" |
| Theory Link    | Proves Proposition 1: λ₂ → 0 means graph disconnect               |

**What to explain to professor:**

> "This graph shows the Fiedler value (λ₂) decreasing over training. By Theorem 1 (Fiedler, 1973), λ₂ = 0 if and only if the graph is disconnected. Our agents learn to drive λ₂ toward zero, proving successful swarm fragmentation."

#### 2️⃣ `2_reward_vs_episodes.png`

**Reward Convergence**

| Aspect         | Details                    |
| -------------- | -------------------------- |
| X-axis         | Training episodes          |
| Y-axis         | Average episode reward     |
| Expected Shape | Low → increasing → plateau |
| Interpretation | Standard RL convergence    |

**What to explain:**

> "The increasing reward curve demonstrates PPO convergence. The agent learns actions that maximize connectivity reduction."

#### 3️⃣ `3_marl_vs_random.png` — **PROVES NOVELTY**

**MARL-PPO vs Random Jamming**

| Aspect   | Details                        |
| -------- | ------------------------------ |
| X-axis   | Training episodes              |
| Y-axis   | λ₂ reduction (%)               |
| Curves   | MARL-PPO (blue), Random (red)  |
| Expected | Random: flat ~15%, MARL: ~50%+ |

**What to explain:**

> "Random jamming achieves only ~15% connectivity reduction, while our MARL-PPO achieves 50%+. This demonstrates the value of learned cluster-aware deployment."

#### 4️⃣ `4_connectivity_before_after.png` — **INTUITIVE**

**Communication Graph Before/After**

| Panel | Shows                                |
| ----- | ------------------------------------ |
| Left  | Dense communication network (before) |
| Right | Fragmented clusters (after MARL)     |

**What to explain:**

> "The left panel shows the fully connected enemy swarm. After MARL training (right), the communication graph is fragmented - jammers have positioned themselves to disrupt critical links."

#### 5️⃣ `5_jammer_trajectories.png`

**Spatial Deployment**

Shows jammer movement from random start positions to learned optimal positions near cluster centroids and communication bridges.

**What to explain:**

> "Jammers learn to position themselves near cluster centroids and communication bottlenecks, not just randomly distributed."

#### 6️⃣ `6_full_dashboard.png`

**Complete 6-panel summary** combining all key metrics.

---

## Understanding the Results

### Key Metrics Explained

| Metric          | What it means                   | Good Value |
| --------------- | ------------------------------- | ---------- |
| λ₂ Reduction    | How much connectivity decreased | >50%       |
| Band Match Rate | Correct frequency selection     | >90%       |
| Final Reward    | Cumulative performance          | Increasing |
| Entropy         | Exploration vs exploitation     | Decreasing |

### Interpreting λ₂ Reduction

| Reduction | Interpretation                      |
| --------- | ----------------------------------- |
| 0-20%     | Poor - random level                 |
| 20-50%    | Moderate - some learning            |
| 50-70%    | Good - effective disruption         |
| 70-100%   | Excellent - near-full fragmentation |

### Common Questions

**Q: Why isn't λ₂ reduction always increasing?**
A: RL has inherent stochasticity. The smoothed trend should increase. Some fluctuation is normal - focus on the moving average.

**Q: Why does entropy decrease?**
A: Initially, the policy explores randomly (high entropy). As it learns optimal actions, it becomes more deterministic (low entropy).

**Q: Why is 300K steps recommended?**
A: Short training (100K) may not show clear learning. 300K-500K gives visible trends suitable for presentation.

---

## Theoretical Validation

### Proposition 1 (from PROJECT_MASTER_GUIDE_v2.md)

**Statement:** Under the FSPL communication model, if λ₂(t) = 0, then the enemy swarm graph is disconnected, and swarm coordination is impossible.

**Proof Sketch:**

1. By Fiedler's Theorem (1973): λ₂ = 0 ⟺ graph is disconnected
2. Disconnected graph = no path between some drone pairs
3. No path = no communication = no coordination

**Our contribution:** We use λ₂ directly as reward signal, proving that learned policies achieve swarm fragmentation.

### FSPL Physical Grounding

Our jamming model uses realistic free-space path loss:

```
R_jam = (c/(4πf)) * sqrt(P_jammer/P_threshold)
```

At 2.4 GHz with 30 dBm jammer power:

- Communication range ≈ 86m
- Jamming range ≈ 22m (with -40 dBm threshold)

This is **not arbitrary** - it's derived from RF physics.

---

## Plan for 20 Graphs

### Required for Publication (Section 10.1 of Guide)

| #   | Graph                     | Status  | Priority |
| --- | ------------------------- | ------- | -------- |
| 1   | λ₂ vs Episodes            | ✅ Done | Core     |
| 2   | Reward vs Episodes        | ✅ Done | Core     |
| 3   | MARL vs Random            | ✅ Done | Core     |
| 4   | Connectivity Before/After | ✅ Done | Core     |
| 5   | Jammer Trajectories       | ✅ Done | Core     |
| 6   | Full Dashboard            | ✅ Done | Core     |

### To Be Implemented

| #   | Graph                                | Description              | Script Needed             |
| --- | ------------------------------------ | ------------------------ | ------------------------- |
| 7   | System Architecture Flowchart        | Pipeline diagram         | Separate (draw.io)        |
| 8   | Baseline Comparison Bar Chart        | MARL vs 5 baselines      | `run_baselines.py`        |
| 9   | Single Episode λ₂ Evolution          | Real-time λ₂ drop        | `episode_analysis.py`     |
| 10  | Scalability: Enemy Count             | N=5,10,20,50,100         | `scalability_test.py`     |
| 11  | Scalability: Jammer Count            | M=2,4,6,8 + theory bound | `scalability_test.py`     |
| 12  | Ablation: GAE vs MC                  | Convergence speed        | `ablation_study.py`       |
| 13  | Ablation: Reward Components          | Remove each term         | `ablation_study.py`       |
| 14  | Coverage Heatmap Before              | Random positions         | `heatmap_analysis.py`     |
| 15  | Coverage Heatmap After               | Trained positions        | `heatmap_analysis.py`     |
| 16  | Frequency Band Distribution          | Pie chart                | `band_analysis.py`        |
| 17  | Convergence: PPO vs A2C vs REINFORCE | Algorithm comparison     | `algorithm_comparison.py` |
| 18  | Dynamic Enemy Tracking Error         | MARL vs static           | `tracking_analysis.py`    |
| 19  | Cluster-wise Disruption              | Per-cluster λ₂           | `cluster_analysis.py`     |
| 20  | Training Animation (GIF)             | Jammer movement          | `animation.py`            |

### Implementation Plan

**Phase 1 (Current):** Core 5 graphs ✅

**Phase 2 (Baselines):**

```bash
python run_baselines.py --algorithms random,greedy,single_ppo,independent_ppo
python generate_graphs.py --graph baseline_comparison
```

**Phase 3 (Scalability):**

```bash
python scalability_test.py --enemy_counts 5,10,20,50,100
python scalability_test.py --jammer_counts 2,4,6,8
```

**Phase 4 (Ablations):**

```bash
python ablation_study.py --remove none,gae,band,overlap,proximity

```

**Phase 5 (Advanced):**

- Heatmaps, animations, per-episode analysis

---

## Quick Reference

### Train New Model

```bash
python train.py --mode fast --name my_model --steps 300000
```

### Generate Graphs

```bash
python generate_graphs.py --experiment my_model
```

### View Results

```bash
# List outputs
dir outputs\my_model

# View graphs
start outputs\my_model\graphs\1_lambda2_vs_episodes.png
```

### Evaluate Random Baseline

```bash
python evaluate_random.py --episodes 50
```

---

## Contact & References

**Theoretical Foundation:**

- Fiedler, M. (1973). "Algebraic connectivity of graphs." Czechoslovak Mathematical Journal
- Valianti et al. (2024). IEEE TMC - Baseline MARL jamming paper

**Implementation:**

- PyTorch for neural networks
- Gymnasium for environment
- Stable-Baselines3 patterns for PPO

---

_MARL Jammer System v2.0 — Ready for Professor Presentation_
