# grasp_lab — VLA Robot Learning Framework

A Vision-Language-Action (VLA) training framework for the i2rt YAM arm.
Two training backends share a common LeRobot v3 data pipeline:

| Backend | Framework | Status |
|---|---|---|
| **LeRobot** | PyTorch (ACT / Diffusion) | Primary — ready |
| **openpi / π0** | JAX | Conversion only — see `openpi_training/README.md` |

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r grasp_lab/requirements.txt
```

### 2. Collect demonstrations

```bash
# Real hardware
python grasp_lab/collection/collect_demos.py \
    --follower-can can0 \
    --leader-can can1 \
    --gripper crank_4310 \
    --task grasp_mug \
    --dataset-dir ./data/grasp_mug \
    --num-episodes 50

# Smoke-test without hardware
python grasp_lab/collection/collect_demos.py --dry-run
```

### 3. Inspect the dataset

```bash
python grasp_lab/visualization/visualize_dataset.py \
    --dataset-dir ./data/grasp_mug \
    --episode-index 0
```

### 4. Train a policy

```bash
# ACT
python grasp_lab/lerobot_training/train.py \
    --dataset-dir ./data/grasp_mug \
    --policy act \
    --output-dir ./runs/grasp_mug_act

# Diffusion Policy
python grasp_lab/lerobot_training/train.py \
    --dataset-dir ./data/grasp_mug \
    --policy diffusion \
    --output-dir ./runs/grasp_mug_diffusion

# Dry-run (config check, no training)
python grasp_lab/lerobot_training/train.py \
    --dataset-dir ./data/grasp_mug --policy act --dry-run
```

### 5. Evaluate on the real robot

```bash
python grasp_lab/lerobot_training/eval.py \
    --checkpoint ./runs/grasp_mug_act/checkpoints/last \
    --follower-can can0 \
    --gripper crank_4310 \
    --num-rollouts 5
```

### 6. Replay a stored episode

```bash
python grasp_lab/collection/replay_episode.py \
    --dataset-dir ./data/grasp_mug \
    --episode-index 0 \
    --follower-can can0
```

---

## Folder Structure

```
grasp_lab/
├── README.md
├── requirements.txt
├── data/                       # Shared LeRobot v3 utilities
│   ├── dataset.py              # DataLoader factory
│   ├── transforms.py           # Image & state augmentations
│   └── realsense.py            # RealSense camera (RGB + depth + point cloud)
├── collection/                 # Data collection pipeline
│   ├── collect_demos.py        # Record demos → LeRobot v3
│   └── replay_episode.py       # Replay a stored episode
├── lerobot_training/           # PyTorch VLA (ACT / Diffusion)
│   ├── configs/
│   │   ├── robot/i2rt_yam.yaml
│   │   ├── act.yaml
│   │   └── diffusion.yaml
│   ├── robot_env.py            # LeRobot Robot wrapper for i2rt
│   ├── train.py                # Training entry point
│   └── eval.py                 # Evaluation loop on real robot
├── openpi_training/            # JAX (π0 / openpi) — conversion only
│   ├── convert_dataset.py      # LeRobot v3 → RLDS/TFRecord
│   └── README.md
└── visualization/
    ├── visualize_dataset.py    # Episode viewer + action plots
    ├── realsense_3d.py         # Live 3D point cloud viewer
    └── rollout_viewer.py       # Policy rollout overlay on video
```

---

## Data Format: LeRobot v3

```
data/
  chunk-000/
    episode_000000.parquet   # per-step: state, action, timestamps
    ...
videos/
  chunk-000/
    observation.images.wrist_episode_000000.mp4
    observation.images.top_episode_000000.mp4   (optional)
meta/
  info.json       # dataset metadata and feature shapes
  episodes.jsonl  # per-episode metadata
  stats.json      # normalization statistics (mean/std/min/max)
```

### Feature names

| Key | Shape | Description |
|---|---|---|
| `observation.state` | (7,) | Follower joint positions (6 arm + 1 gripper) |
| `action` | (7,) | Leader joint positions (target) |
| `observation.images.wrist` | (H, W, 3) | Wrist RGB frame |
| `observation.images.top` | (H, W, 3) | Top-down RGB frame (optional) |

---

## Hardware Setup

| Component | Details |
|---|---|
| Robot arm | i2rt YAM (6 DOF + gripper) |
| Leader arm | YAM + teaching handle (zero-gravity mode) |
| Camera | Intel RealSense D-series (wrist-mounted) |
| CAN bus | follower → `can0`, leader → `can1` |

### Teaching-handle protocol

- **Press button** → start recording an episode
- **Move the arm** to demonstrate the task
- **Press button again** → stop and save the episode
- **Quick second press** (within 0.8 s) → discard the episode

---

## Verification

```bash
# Smoke-test collection pipeline
python grasp_lab/collection/collect_demos.py --dry-run

# Verify dataloader
python grasp_lab/data/dataset.py --dataset-dir ./test_data

# Verify training config
python grasp_lab/lerobot_training/train.py \
    --dataset-dir ./test_data --policy act --dry-run

# Browse dataset
python grasp_lab/visualization/visualize_dataset.py \
    --dataset-dir ./test_data
```
