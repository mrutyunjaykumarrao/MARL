# Real-World Deployment Guide

## From Trained Policy to Physical Drones

---

## 1. The Gap Between Simulation and Reality

### Current Simulation (What We Have)

```
Simulation Environment:
├── Enemy positions: Generated as (x, y) coordinates
├── Observations: Computed directly from coordinates
├── Actions: Velocity vectors (Vx, Vy) + band selection
└── Reward: Computed from λ₂ reduction
```

### Real-World Scenario (What Professor Wants)

```
Real Deployment:
├── Input: Camera/radar images of enemy drones
├── Need: Convert images → (x, y) positions
├── Need: Feed positions to trained policy
├── Output: Control commands for jammer drones
```

---

## 2. Complete Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    REAL-WORLD DEPLOYMENT PIPELINE               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐  │
│   │   PERCEPTION  │ => │    STATE      │ => │    POLICY     │  │
│   │    MODULE     │    │  ESTIMATOR    │    │   NETWORK     │  │
│   │               │    │               │    │   (Trained)   │  │
│   │ Camera/Radar  │    │ Positions     │    │ Actor Network │  │
│   │ → Detection   │    │ → Observation │    │ → Actions     │  │
│   └───────────────┘    └───────────────┘    └───────────────┘  │
│         │                     │                     │          │
│         ▼                     ▼                     ▼          │
│   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐  │
│   │ Detected      │    │ 5D Obs Vector │    │ Velocity +    │  │
│   │ Objects       │    │ [normalized]  │    │ Band Command  │  │
│   │ (x,y) per     │    │               │    │               │  │
│   │ drone         │    │               │    │               │  │
│   └───────────────┘    └───────────────┘    └───────────────┘  │
│                                                    │           │
│                                                    ▼           │
│                                            ┌───────────────┐   │
│                                            │ DRONE CONTROL │   │
│                                            │ (Execute)     │   │
│                                            └───────────────┘   │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Deployment

### Step 1: Perception Module (Image → Positions)

**What it does:** Detects enemy drones in camera/radar images

**Options:**

1. **YOLO/Faster RCNN** - Object detection for visual cameras
2. **Radar Signal Processing** - For radar-based detection
3. **LiDAR Point Cloud** - For 3D spatial detection

**Output:** List of (x, y, z) positions or (x, y) in 2D

```python
# Example: Using a pretrained drone detector
class PerceptionModule:
    def __init__(self, detector_weights="yolo_drone_v8.pt"):
        self.detector = load_yolo(detector_weights)

    def process_frame(self, image):
        """
        Input: Camera image (RGB, 640x480)
        Output: Enemy positions [(x1,y1), (x2,y2), ...]
        """
        detections = self.detector.predict(image)

        # Convert pixel coordinates to world coordinates
        # Requires camera calibration + depth estimation
        positions = []
        for det in detections:
            if det.class_name == "drone":
                world_pos = self.pixel_to_world(det.center, det.depth)
                positions.append(world_pos)

        return np.array(positions)
```

### Step 2: State Estimator (Positions → Observations)

**What it does:** Converts raw positions to normalized observations

```python
# This is already in our codebase: src/environment/observation.py

class StateEstimator:
    def __init__(self, arena_size=100.0):
        self.arena_size = arena_size
        self.clusterer = DBSCANClusterer(eps=20.0)

    def compute_observations(self, enemy_positions, jammer_positions):
        """
        Input:
            enemy_positions: (N, 2) array
            jammer_positions: (M, 2) array
        Output:
            observations: (M, 5) normalized array
        """
        # Run DBSCAN clustering
        labels, centroids = self.clusterer.fit(enemy_positions)

        # For each jammer, compute 5D observation
        obs = []
        for j in range(len(jammer_positions)):
            # [dist_to_centroid, cluster_density, dist_to_others,
            #  coverage_overlap, band_match]
            o = self._compute_single_obs(
                jammer_positions[j],
                enemy_positions,
                jammer_positions,
                centroids,
                j
            )
            obs.append(o)

        return np.array(obs, dtype=np.float32)
```

### Step 3: Policy Network (Observations → Actions)

**What it does:** Uses trained actor network to get actions

```python
# Deployment code using trained weights

class DeployedPolicy:
    def __init__(self, weights_path="outputs/optimal_10k/checkpoints/best/ppo_agent.pt"):
        # Load trained actor network
        self.agent = PPOAgent(obs_dim=5, M=6, hidden_dim=256)
        self.agent.load(weights_path)
        self.agent.actor.eval()  # Set to evaluation mode

    def get_action(self, observations):
        """
        Input: observations (M, 5) array
        Output: velocities (M, 2), bands (M,)
        """
        with torch.no_grad():
            # Get deterministic action (mean, not sampled)
            action = self.agent.get_deterministic_action(observations)

        return action['velocity'], action['band']
```

### Step 4: Drone Control (Actions → Commands)

```python
class DroneController:
    def __init__(self, drone_ids):
        self.drones = [connect_to_drone(did) for did in drone_ids]

    def execute_actions(self, velocities, bands):
        """
        Input:
            velocities: (M, 2) - [vx, vy] per drone
            bands: (M,) - frequency band index
        """
        for i, drone in enumerate(self.drones):
            # Send velocity command
            drone.set_velocity(velocities[i, 0], velocities[i, 1])

            # Switch to appropriate frequency band
            frequency = FREQUENCY_BANDS[bands[i]]
            drone.set_jamming_frequency(frequency)
```

---

## 4. Complete Deployment Loop

```python
def real_time_deployment():
    """Complete deployment loop."""

    # Initialize modules
    perception = PerceptionModule("yolo_drone_v8.pt")
    state_estimator = StateEstimator(arena_size=100.0)
    policy = DeployedPolicy("outputs/optimal_10k/checkpoints/best/ppo_agent.pt")
    controller = DroneController(["jammer_1", "jammer_2", ..., "jammer_6"])

    # Get initial jammer positions (from GPS/IMU)
    jammer_positions = get_current_jammer_positions()

    # Control loop
    while mission_active:
        # 1. Capture frame from camera
        frame = camera.capture()

        # 2. Detect enemy drones
        enemy_positions = perception.process_frame(frame)

        # 3. Estimate current jammer positions
        jammer_positions = get_current_jammer_positions()

        # 4. Compute observations
        observations = state_estimator.compute_observations(
            enemy_positions, jammer_positions
        )

        # 5. Get actions from trained policy
        velocities, bands = policy.get_action(observations)

        # 6. Execute on physical drones
        controller.execute_actions(velocities, bands)

        # 7. Sleep for dt (e.g., 0.1 seconds)
        time.sleep(0.1)
```

---

## 5. Available Datasets for Testing

### Drone Detection Datasets (for Perception Module)

| Dataset           | Description                      | Use Case                  | Size   |
| ----------------- | -------------------------------- | ------------------------- | ------ |
| **Anti-UAV**      | IR + RGB drone videos            | Train YOLO detector       | ~15GB  |
| **DUT-UAV**       | High-altitude drone images       | Detection benchmarks      | ~3GB   |
| **Drone-vs-Bird** | Distinguishing drones from birds | Robust detection          | ~2GB   |
| **UAV-BB**        | Bounding box annotations         | Object detection training | ~500MB |
| **USC Drone**     | Multi-scale UAV images           | Detection + tracking      | ~8GB   |

### Dataset Download Links

**1. Anti-UAV Dataset (RECOMMENDED)**

- Website: https://anti-uav.github.io/
- Paper: "Anti-UAV: A Large Multi-Modal Benchmark for UAV Tracking"
- Contains: 318 video sequences, IR + RGB
- Download: Apply via Google Form on website

**2. DUT-UAV Dataset**

- GitHub: https://github.com/wangxiao5791509/DUT-Anti-UAV
- Contains: 10,000+ images with bounding boxes
- Direct download link on repository

**3. Drone-vs-Bird Challenge**

- Website: https://wosdetc2020.wordpress.com/
- Annual challenge dataset
- Contains: Drone and bird videos for classification

**4. USC Drone Dataset**

- Website: https://usc-actlab.github.io/DroneDetect/
- Contains: Multi-scale drone images from ground cameras

**5. UAVDet Dataset**

- GitHub: https://github.com/VisDrone/VisDrone-Dataset
- Contains: Detection + tracking + counting
- Large-scale with diverse scenarios

### How to Download and Use

```bash
# Step 1: Create a datasets folder
mkdir datasets
cd datasets

# Step 2: Download Anti-UAV (after approval)
# 1. Go to https://anti-uav.github.io/
# 2. Fill the application form
# 3. Download links will be emailed

# Step 3: For quick testing, use DUT-UAV
git clone https://github.com/wangxiao5791509/DUT-Anti-UAV.git
# Follow README for download instructions

# Step 4: For VisDrone (large dataset)
# Download from: https://github.com/VisDrone/VisDrone-Dataset
```

### For Testing Our RL Policy

Our RL policy works on **abstract state** (positions, observations), NOT raw images. So:

1. **Use drone detection dataset** → Train/use object detector (YOLO v8)
2. **Extract positions** from detected bboxes + depth estimation
3. **Feed positions** to our State Estimator (cluster assignment + observation)
4. **Run our trained policy** to get actions (velocity + band)

---

## 6. Sim-to-Real Gap Considerations

### Challenges

```
Simulation:
✓ Perfect position knowledge
✓ No sensor noise
✓ Instant action execution
✓ Known enemy band

Reality:
✗ Position errors from detection
✗ Sensor noise & occlusions
✗ Action delays & drone dynamics
✗ Unknown enemy band (need detection)
```

### Solutions

1. **Domain Randomization**: Add noise during training
2. **Robust Observations**: Use relative positions, not absolute
3. **Delay Handling**: Add action delay in simulation
4. **Band Detection**: Add RF spectrum sensing module

---

## 7. Summary for Professor

**Answer to "How to use trained model with real data":**

```
Three-Stage Pipeline:

1. PERCEPTION: Image/Video → Drone Detection → (x,y) Positions
   - Use pretrained YOLO/Faster-RCNN
   - Existing datasets: Anti-UAV, DUT-UAV, Drone-vs-Bird

2. STATE ESTIMATION: Positions → Normalized Observations
   - Already implemented in our code
   - DBSCAN clustering + observation builder

3. POLICY INFERENCE: Observations → Actions
   - Load trained actor weights
   - Deterministic forward pass
   - No training needed at deployment

The trained RL policy is "blind" to image format - it only sees
5D observations. The perception module bridges the gap between
raw sensor data and abstract state representation.
```

---

## 8. Next Steps for Real Deployment

1. **Train Object Detector** on drone images (Anti-UAV dataset)
2. **Calibrate Camera** for pixel-to-world coordinate mapping
3. **Test Pipeline** in simulation with added noise
4. **Hardware Integration** with actual drone control APIs
5. **Field Testing** with real jammer drones

---

## 9. Dataset Integration Guide

### Directory Structure for Datasets

```
MARL JAMMER/
├── datasets/
│   ├── anti_uav/
│   │   ├── train/
│   │   │   ├── IR/
│   │   │   └── RGB/
│   │   ├── test/
│   │   └── annotations/
│   ├── dut_uav/
│   │   ├── images/
│   │   └── labels/
│   └── visdrone/
│       ├── VisDrone2019-DET-train/
│       └── VisDrone2019-DET-val/
```

### Basic Dataset Loader Script

Create `src/perception/dataset_loader.py`:

```python
import os
import cv2
import numpy as np
from pathlib import Path

class UAVDatasetLoader:
    """Load and process UAV detection datasets."""

    def __init__(self, dataset_path: str, dataset_type: str = "anti_uav"):
        self.path = Path(dataset_path)
        self.type = dataset_type

    def load_anti_uav(self):
        """Load Anti-UAV dataset sequences."""
        sequences = []
        for seq_dir in (self.path / "train").iterdir():
            if seq_dir.is_dir():
                frames = sorted(seq_dir.glob("*.jpg"))
                sequences.append({
                    "name": seq_dir.name,
                    "frames": frames,
                    "annotations": self._load_annotations(seq_dir.name)
                })
        return sequences

    def load_dut_uav(self):
        """Load DUT-UAV dataset."""
        images = []
        labels_dir = self.path / "labels"
        for img_path in (self.path / "images").glob("*.jpg"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            images.append({
                "image": img_path,
                "label": label_path if label_path.exists() else None
            })
        return images

    def process_frame(self, frame_path: str) -> tuple:
        """Process single frame and extract drone positions.

        Returns: (image, detected_positions)
        """
        image = cv2.imread(str(frame_path))
        # TODO: Add YOLO detection here
        # detected_positions = yolo_detector.predict(image)
        return image, None

# Usage example
if __name__ == "__main__":
    loader = UAVDatasetLoader("datasets/anti_uav", "anti_uav")
    sequences = loader.load_anti_uav()
    print(f"Loaded {len(sequences)} sequences")
```

### Testing with Dataset (After Perception Module Ready)

```python
from perception.dataset_loader import UAVDatasetLoader
from perception.yolo_detector import DroneDetector
from deployment.state_estimator import StateEstimator
from deployment.policy_runner import PolicyRunner

# Load components
dataset = UAVDatasetLoader("datasets/anti_uav")
detector = DroneDetector("weights/yolo_drone.pt")
state_estimator = StateEstimator(arena_size=100.0)
policy = PolicyRunner("outputs/optimal_95pct/checkpoints/best")

# Process video sequence
for frame_path in dataset.load_anti_uav()[0]["frames"]:
    # 1. Load frame
    image = cv2.imread(str(frame_path))

    # 2. Detect drones
    positions = detector.predict(image)

    # 3. Convert to observations
    obs = state_estimator.positions_to_observations(positions)

    # 4. Get jammer actions
    actions = policy.get_actions(obs)

    # 5. Visualize/execute
    print(f"Actions: {actions}")
```

---

_Document created: February 24, 2026_
