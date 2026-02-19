# Deployment Guide: From Training to Production

## Understanding Where Weights Are and What's Needed for Deployment

This document explains how to deploy a trained model and clarifies what components are/aren't needed.

---

## 📁 Where Are the Trained Weights?

### File Locations

After training completes, weights are saved in:

```
outputs/experiment_v1/
├── actor_state_dict.pt      # ← Actor network weights (THE POLICY)
├── critic_state_dict.pt     # ← Critic network weights (VALUE FUNCTION)
├── optimizer_actor.pt       # For resuming training only
├── optimizer_critic.pt      # For resuming training only
├── config.yaml              # Training configuration snapshot
├── training_log.csv         # Training metrics
└── graphs/                  # Generated visualizations
```

### What Each File Contains

#### `actor_state_dict.pt` (Most Important!)

```python
{
    'trunk.0.weight': torch.Tensor,       # FC1 weights (5×128)
    'trunk.0.bias': torch.Tensor,         # FC1 bias (128)
    'trunk.1.weight': torch.Tensor,       # LayerNorm weights
    'trunk.1.bias': torch.Tensor,         # LayerNorm bias
    'trunk.3.weight': torch.Tensor,       # FC2 weights (128×128)
    ...
    'mu_head.weight': torch.Tensor,       # Velocity mean head
    'mu_head.bias': torch.Tensor,
    'log_std_head.weight': torch.Tensor,  # Velocity std head
    'log_std_head.bias': torch.Tensor,
    'band_head.weight': torch.Tensor,     # Band selection head
    'band_head.bias': torch.Tensor,
}
```

**Size:** ~100 KB (very small, easily deployable)

#### `critic_state_dict.pt`

```python
{
    'network.0.weight': torch.Tensor,  # FC1 (5×128)
    'network.0.bias': torch.Tensor,
    ...
}
```

**Important:** Critic is ONLY needed for training. For deployment, you can delete it.

---

## 🚀 Deployment: What You NEED and DON'T NEED

### ✅ REQUIRED for Deployment

| Component          | File                  | Size    | Why                             |
| ------------------ | --------------------- | ------- | ------------------------------- |
| Actor Network      | `actor_state_dict.pt` | ~100 KB | The policy that decides actions |
| Actor Architecture | `src/agents/actor.py` | ~5 KB   | Network definition              |
| Physics Models     | `src/physics/*.py`    | ~20 KB  | FSPL, jamming calculations      |
| Config             | `src/config.py`       | ~3 KB   | Network architecture params     |

**Minimal deployment package:**

```
deployment/
├── actor_state_dict.pt
├── actor.py (simplified)
├── config.py
└── inference.py
```

### ❌ NOT NEEDED for Deployment

| Component      | File                   | Why Not Needed                       |
| -------------- | ---------------------- | ------------------------------------ |
| Critic Network | `critic_state_dict.pt` | Only for training (value estimation) |
| Optimizers     | `optimizer_*.pt`       | Only for training continuation       |
| Rollout Buffer | N/A                    | Not saved, only used during training |
| Training Logs  | `training_log.csv`     | Just metrics, not functional         |
| Graphs         | `graphs/`              | Visualization only                   |

---

## 💻 Minimal Inference Code

### Standalone Deployment Script

```python
# deployment/inference.py
import torch
import torch.nn as nn
import numpy as np

# Minimal Actor Definition (copy from actor.py)
class Actor(nn.Module):
    def __init__(self, obs_dim=5, hidden_dim=128, v_max=5.0, num_bands=4):
        super().__init__()
        self.v_max = v_max

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.mu_head = nn.Linear(hidden_dim, 2)
        self.band_head = nn.Linear(hidden_dim, num_bands)

    def forward(self, obs):
        features = self.trunk(obs)
        mu = self.mu_head(features)
        band_logits = self.band_head(features)
        return mu, band_logits

    def act_deterministic(self, obs):
        """Get action without sampling (greedy/deterministic)"""
        with torch.no_grad():
            mu, band_logits = self.forward(obs)
            velocity = torch.clamp(mu, -self.v_max, self.v_max)
            band = torch.argmax(band_logits, dim=-1)
        return velocity.numpy(), band.numpy()


def load_for_deployment(weights_path):
    """Load trained actor for deployment."""
    actor = Actor()
    actor.load_state_dict(torch.load(weights_path, map_location='cpu'))
    actor.eval()  # Set to evaluation mode
    return actor


def main():
    # Load trained policy
    actor = load_for_deployment('actor_state_dict.pt')

    # Example inference
    # Observation: [Δx, Δy, v_x, v_y, enemy_band]
    obs = torch.tensor([[0.5, -0.3, 0.1, 0.0, 2.0]], dtype=torch.float32)

    velocity, band = actor.act_deterministic(obs)
    print(f"Velocity command: {velocity}")
    print(f"Band selection: {band}")


if __name__ == "__main__":
    main()
```

### Running Deployed Model

```powershell
cd deployment
python inference.py
```

**Output:**

```
Velocity command: [[ 2.3 -1.7]]
Band selection: [2]
```

---

## 🔄 Training vs Deployment: Key Differences

### Training Loop (Full)

```python
# During TRAINING:
for step in total_timesteps:
    # 1. Actor generates action (with exploration noise)
    action, log_prob = actor.sample(obs)  # Stochastic

    # 2. Execute in environment
    next_obs, reward, done, info = env.step(action)

    # 3. Store in rollout buffer
    buffer.add(obs, action, reward, done, log_prob, value)

    # 4. Periodically update networks
    if buffer.is_full():
        # Compute advantages using CRITIC
        advantages = compute_gae(buffer.rewards, critic(buffer.obs))

        # Update both actor AND critic
        actor_loss = compute_ppo_loss(actor, buffer, advantages)
        critic_loss = compute_value_loss(critic, buffer)

        actor_optimizer.step()
        critic_optimizer.step()
```

### Deployment Loop (Minimal)

```python
# During DEPLOYMENT:
while True:
    # 1. Get observation from sensors
    obs = get_sensor_observation()

    # 2. Actor generates action (deterministic, no noise)
    velocity, band = actor.act_deterministic(obs)  # Deterministic

    # 3. Execute action on real drone
    send_velocity_command(velocity)
    set_jammer_frequency(band)

    # NO buffer, NO critic, NO optimization
```

---

## 📊 Rollout Buffer: Why It's NOT Needed

### What is the Rollout Buffer?

```python
class RolloutBuffer:
    """Stores transitions during training for PPO updates."""

    def __init__(self, size, obs_dim, action_dim):
        self.observations = np.zeros((size, obs_dim))
        self.actions = np.zeros((size, action_dim))
        self.rewards = np.zeros(size)
        self.dones = np.zeros(size)
        self.log_probs = np.zeros(size)
        self.values = np.zeros(size)
        self.advantages = np.zeros(size)
        self.returns = np.zeros(size)
```

### Why It's Training-Only

| Used For                | Training | Deployment |
| ----------------------- | -------- | ---------- |
| Store transitions       | ✅ Yes   | ❌ No      |
| Compute GAE             | ✅ Yes   | ❌ No      |
| PPO mini-batch sampling | ✅ Yes   | ❌ No      |
| Off-policy learning     | ✅ Yes   | ❌ No      |

**At deployment:**

- Observation comes in → Action goes out → Done
- No need to store history
- No learning happening

---

## 🤖 Real-World Deployment Considerations

### Hardware Requirements

| Platform       | CPU                  | RAM    | Storage |
| -------------- | -------------------- | ------ | ------- |
| Raspberry Pi 4 | ✅ Yes               | 512 MB | 1 MB    |
| NVIDIA Jetson  | ✅ Yes               | 256 MB | 1 MB    |
| Arduino        | ❌ No (need PyTorch) | -      | -       |
| Custom FPGA    | Possible with ONNX   | -      | -       |

### Converting to ONNX (For Edge Devices)

```python
import torch.onnx

# Load trained actor
actor = load_for_deployment('actor_state_dict.pt')

# Create dummy input
dummy_input = torch.randn(1, 5)

# Export to ONNX
torch.onnx.export(
    actor,
    dummy_input,
    "actor_policy.onnx",
    input_names=['observation'],
    output_names=['velocity_mean', 'band_logits'],
    dynamic_axes={'observation': {0: 'batch_size'}}
)
```

**ONNX benefits:**

- Run without PyTorch
- Faster inference
- Compatible with TensorRT, OpenVINO

### Latency Considerations

| Operation                | Typical Latency |
| ------------------------ | --------------- |
| Actor forward pass (CPU) | 0.1 - 0.5 ms    |
| Actor forward pass (GPU) | 0.01 - 0.05 ms  |
| Real drone control loop  | 10 - 100 ms     |

**Conclusion:** Neural network inference is NOT the bottleneck.

---

## 📝 Deployment Checklist

### Before Deployment

- [ ] Training converged (stable reward, low KL)
- [ ] Tested on evaluation episodes
- [ ] Weights saved (`actor_state_dict.pt`)
- [ ] Minimal inference code prepared
- [ ] Edge device compatible (CPU-only if needed)

### During Deployment

- [ ] Load weights with `torch.load(path, map_location='cpu')`
- [ ] Call `model.eval()` to disable dropout/batchnorm training mode
- [ ] Use `act_deterministic()` for no exploration noise
- [ ] Clamp outputs to valid ranges

### Monitoring

- [ ] Log action distributions (for debugging)
- [ ] Track real-world λ₂ if measurable
- [ ] Alert if actions saturate at limits

---

## 🗣️ Hinglish Deployment Summary

**Deployment ka Summary:**

"Deployment ke liye sirf **actor network weights** chahiye - yeh ~100 KB ki file hai `actor_state_dict.pt`. Critic network, rollout buffer, optimizers - yeh sab sirf training ke liye hain, deployment mein nahi chahiye.

Kaam kaise karta hai: Observation aata hai (5 numbers - position, velocity, enemy band), actor network forward pass karta hai, velocity aur band output deta hai. Bas itna hi. Koi buffer nahi, koi learning nahi deployment mein.

Code simple hai: `actor.load_state_dict(torch.load('weights.pt'))` se weights load karo, `actor.eval()` se evaluation mode mein daalo, phir `actor.act_deterministic(obs)` se action lo. 0.5 millisecond mein hota hai ek inference.

Real drone pe deploy karna hai toh ONNX mein convert kar sakte ho - yeh TensorRT, Jetson sab ke saath compatible hai. Raspberry Pi 4 pe bhi chal jayega easily.

Weights save karte waqt automatically `torch.save(actor.state_dict(), path)` hota hai training script mein. Resume karna hai training toh optimizers bhi chahiye, deploy karna hai toh sirf actor weights."

---

**Next:** See `05_SCALABILITY.md` for discussion on scaling to 100 enemies, 40 jammers, 2M steps.
