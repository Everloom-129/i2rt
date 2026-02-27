# grasp_lab — Development Plan

_Last updated: 2026-02-26_

---

## Goal

Build a complete VLA (Vision-Language-Action) robot learning framework for the i2rt YAM arm,
hosted under `grasp_lab/` in the i2rt repo. Two training backends share one LeRobot v3 data pipeline:

| Backend | Framework | Priority |
|---|---|---|
| **LeRobot** | PyTorch — ACT / Diffusion Policy | Primary |
| **openpi / π0** | JAX | Secondary (conversion stub only for now) |

---

## Environment Setup

The project uses a `uv`-managed Python 3.11 virtualenv at the repo root.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.11
source .venv/bin/activate
sudo apt update && sudo apt install build-essential python3-dev linux-headers-$(uname -r)
uv pip install -e .
```

Install `grasp_lab` extras:
```bash
uv pip install -r grasp_lab/requirements.txt
```

> **Docker** (`ssh -p 6600 root@158.130.50.26`, password `rtx4090`) is the intended
> long-term dev/training environment but was unavailable during initial setup.
> Switch back to Docker once it is running.

---

## Completed Work

### Folder structure created
```
grasp_lab/
├── README.md
├── requirements.txt
├── docs/
│   └── dev_plan.md              ← this file
├── data/
│   ├── __init__.py
│   ├── dataset.py               LeRobot v3 DataLoader factory
│   ├── transforms.py            Image & state augmentations
│   └── realsense.py             RealSense D-series camera
├── collection/
│   ├── __init__.py
│   ├── collect_demos.py         Teleoperation → LeRobot v3 recording
│   └── replay_episode.py        Replay stored episode on real robot
├── lerobot_training/
│   ├── __init__.py
│   ├── configs/
│   │   ├── robot/i2rt_yam.yaml  YAM arm + RealSense config
│   │   ├── act.yaml             ACT policy hyperparams
│   │   └── diffusion.yaml       Diffusion policy hyperparams
│   ├── robot_env.py             LeRobot Robot wrapper for i2rt
│   ├── train.py                 Training entry point
│   └── eval.py                  Evaluation loop on real robot
├── openpi_training/
│   ├── __init__.py
│   ├── convert_dataset.py       LeRobot v3 → RLDS/TFRecord
│   └── README.md
└── visualization/
    ├── __init__.py
    ├── visualize_dataset.py     Episode viewer + action plots
    ├── realsense_3d.py          Live 3D point cloud
    └── rollout_viewer.py        Policy rollout overlay on video
```

### Key design decisions
- **Data format**: LeRobot v3 — Parquet shards for tabular data, MP4 shards for video, JSON for metadata.
- **State / action dim**: 7 DOF (6 arm joints + 1 gripper).
- **Control rate**: 30 Hz.
- **Episode trigger**: Teaching-handle button (`io_inputs[0] > 0.5`) starts/stops recording.
  Quick double-press (< 0.8 s) discards the episode.
- **Leader API**: `YAMLeaderRobot.get_info()` → `(joint_pos_7dof, io_inputs)` from `scripts/minimum_gello.py`.

---

## Next Steps

### 1. Environment smoke-test (immediate)
- [ ] Activate venv: `source .venv/bin/activate`
- [ ] Install deps: `uv pip install -e . && uv pip install -r grasp_lab/requirements.txt`
- [ ] Run: `python grasp_lab/collection/collect_demos.py --dry-run`
- [ ] Run: `python grasp_lab/lerobot_training/train.py --dataset-dir ./test_data --policy act --dry-run`
- [ ] Run: `python grasp_lab/lerobot_training/robot_env.py --dry-run`

### 2. Install LeRobot
```bash
uv pip install 'lerobot @ git+https://github.com/huggingface/lerobot.git'
```
- Verify `lerobot.common.datasets.lerobot_dataset.LeRobotDataset` loads a local dataset.
- Confirm `dataset.py` `make_dataloader()` works end-to-end.

### 3. Collect first real dataset
- Hardware: follower on `can0`, leader on `can1`, wrist RealSense.
- Task: simple pick-and-place or grasp.
- Target: ≥ 50 episodes for meaningful training.

### 4. Train first policy (ACT)
- Start with ACT (faster to converge than Diffusion).
- Log to wandb project `grasp_lab`.
- Checkpoint every 10 000 steps.

### 5. Evaluate on robot
- Use `eval.py` to run 10 rollouts and record success rate.

### 6. Diffusion Policy
- Switch `--policy diffusion` and compare against ACT baseline.
- May need to resize images to 96×96 (see `diffusion.yaml`).

### 7. openpi / π0 fine-tuning (future)
- Docker environment with JAX + CUDA.
- Convert dataset with `openpi_training/convert_dataset.py`.
- Fine-tune π0 checkpoint on grasp task.

### 8. Multi-camera support
- Add `--enable-top-camera` to `collect_demos.py`.
- Update `robot_env.py` `top_camera_serial` path.

---

## Known Issues / TODOs

| Item | File | Notes |
|---|---|---|
| `train.py` calls `lerobot.scripts.train` — exact API depends on lerobot version | `lerobot_training/train.py` | May need adjustment after installing lerobot |
| Depth frames not saved during collection | `collection/collect_demos.py` | Add depth shard writer if needed for 3D viz |
| `realsense_3d.py` replay uses dummy flat depth | `visualization/realsense_3d.py` | Will be fixed once depth recording is in place |
| Docker dev environment offline | — | Switch back once container is running again |
| `lerobot_training/eval.py` `load_policy` uses Hydra config — verify against installed version | `lerobot_training/eval.py` | Test after lerobot install |

---

## References

- [LeRobot repo](https://github.com/huggingface/lerobot)
- [ACT paper](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy paper](https://diffusion-policy.cs.columbia.edu/)
- [openpi / π0](https://github.com/Physical-Intelligence/openpi)
- [RLDS format](https://github.com/google-research/rlds)
