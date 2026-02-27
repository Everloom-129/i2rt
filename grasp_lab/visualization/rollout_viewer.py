"""Overlay policy rollout trajectory on episode video.

Plots the predicted action sequence alongside the recorded RGB frames,
providing a visual diff between what the robot did and what the policy predicts.

Usage::

    python grasp_lab/visualization/rollout_viewer.py \\
        --dataset-dir ./data/grasp_mug \\
        --checkpoint ./runs/grasp_mug_act/checkpoints/last \\
        --episode-index 0 \\
        --output ./rollout_overlay.mp4
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tyro

logger = logging.getLogger(__name__)


@dataclass
class Args:
    dataset_dir: str
    """Path to the LeRobot v3 dataset."""
    checkpoint: Optional[str] = None
    """Path to a trained checkpoint. If None, only shows recorded actions."""
    episode_index: int = 0
    device: str = "cuda"
    output: Optional[str] = None
    """Save the overlay video to this path (MP4). If None, display in window."""
    fps: int = 10


def _load_policy(checkpoint: str, device: str):
    from lerobot.common.policies.factory import make_policy
    from lerobot.common.utils.utils import init_hydra_config

    cfg = init_hydra_config(Path(checkpoint) / "config.yaml")
    policy = make_policy(cfg.policy, pretrained_policy_name_or_path=checkpoint)
    policy = policy.to(device).eval()
    return policy, cfg


def _run_policy_on_episode(policy, states, video_frames, device: str) -> np.ndarray:
    """Run the policy on every frame of an episode and collect predicted actions."""
    import torch

    predicted = []
    policy.reset()

    for t in range(len(states)):
        obs = {"observation.state": torch.from_numpy(states[t]).float().unsqueeze(0).to(device)}
        key = next(iter(video_frames), None)
        if key is not None and t < len(video_frames[key]):
            img = video_frames[key][t]
            t_img = torch.from_numpy(img).float().unsqueeze(0).permute(0, 3, 1, 2) / 255.0
            obs["observation.images.wrist"] = t_img.to(device)

        with torch.no_grad():
            action = policy.select_action(obs)
        predicted.append(action[0].cpu().numpy())

    return np.array(predicted, dtype=np.float32)


def _draw_overlay(frame: np.ndarray, recorded: np.ndarray, predicted: Optional[np.ndarray], step: int) -> np.ndarray:
    """Draw action values as text overlay on a frame."""
    import cv2

    out = frame.copy()
    h, w = out.shape[:2]

    def _put(text, y, color):
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    _put(f"Step {step:04d}", 20, (255, 255, 255))
    rec_str = "  ".join(f"{v:+.2f}" for v in recorded)
    _put(f"Rec:  {rec_str}", 40, (100, 255, 100))
    if predicted is not None:
        pred_str = "  ".join(f"{v:+.2f}" for v in predicted)
        _put(f"Pred: {pred_str}", 60, (100, 100, 255))

    return out


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    import cv2

    from grasp_lab.visualization.visualize_dataset import load_episode

    dataset_dir = Path(args.dataset_dir)
    df, states, actions, video_frames = load_episode(dataset_dir, args.episode_index)
    n = len(df)

    predicted = None
    if args.checkpoint is not None:
        logger.info(f"Loading policy from {args.checkpoint} …")
        policy, _ = _load_policy(args.checkpoint, args.device)
        predicted = _run_policy_on_episode(policy, states, video_frames, args.device)

    # Pick wrist camera preferentially
    video_key = next((k for k in video_frames if "wrist" in k), next(iter(video_frames), None))
    frames = video_frames.get(video_key) if video_key else None

    writer = None
    if args.output:
        h = frames[0].shape[0] if frames is not None else 480
        w = frames[0].shape[1] if frames is not None else 640
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

    dt_ms = int(1000 / args.fps)
    for t in range(n):
        if frames is not None and t < len(frames):
            frame = frames[t]
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        pred_t = predicted[t] if predicted is not None else None
        overlay = _draw_overlay(frame, actions[t], pred_t, t)
        bgr = overlay[:, :, ::-1]

        if writer is not None:
            writer.write(bgr)
        else:
            cv2.imshow("Rollout Viewer", bgr)
            if cv2.waitKey(dt_ms) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()
        print(f"Saved to {args.output}")
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(tyro.cli(Args))
