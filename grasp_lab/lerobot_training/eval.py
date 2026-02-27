"""Evaluate a trained LeRobot policy on the real i2rt robot.

Usage::

    python grasp_lab/lerobot_training/eval.py \\
        --checkpoint ./runs/grasp_mug_act/checkpoints/last \\
        --follower-can can0 \\
        --gripper crank_4310 \\
        --num-rollouts 5
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import tyro

logger = logging.getLogger(__name__)

CONTROL_HZ = 30


@dataclass
class Args:
    checkpoint: str
    """Path to a trained LeRobot checkpoint directory."""
    follower_can: str = "can0"
    gripper: str = "crank_4310"
    wrist_camera_serial: Optional[str] = None
    top_camera_serial: Optional[str] = None
    num_rollouts: int = 5
    """Number of evaluation rollouts to run."""
    max_steps: int = 300
    """Maximum steps per rollout (safety cutoff)."""
    device: str = "cuda"
    dry_run: bool = False
    """Run without real hardware: prints dummy actions."""
    record_video: bool = False
    """Save rollout videos to <checkpoint>/eval_videos/."""


def load_policy(checkpoint: str, device: str):
    """Load a serialised LeRobot policy from a checkpoint directory."""
    try:
        from lerobot.common.policies.factory import make_policy
        from lerobot.common.utils.utils import init_hydra_config
    except ImportError as e:
        raise ImportError("lerobot not installed.") from e

    cfg = init_hydra_config(Path(checkpoint) / "config.yaml")
    policy = make_policy(cfg.policy, pretrained_policy_name_or_path=checkpoint)
    policy = policy.to(device)
    policy.eval()
    return policy, cfg


def run_rollout(policy, env, max_steps: int, device: str, record_video: bool, video_path: Optional[Path]) -> dict:
    """Run a single rollout and return stats."""
    import torch

    video_writer = None
    if record_video and video_path is not None:
        from grasp_lab.collection.collect_demos import _VideoWriter
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = _VideoWriter(video_path, fps=CONTROL_HZ)

    obs = env.get_observation()
    policy.reset()

    dt = 1.0 / CONTROL_HZ
    steps = 0
    t0 = time.time()

    try:
        while steps < max_steps:
            t_step = time.time()

            # Convert obs to tensors
            obs_tensors = {}
            for k, v in obs.items():
                t = torch.from_numpy(v).float().unsqueeze(0).to(device)
                # Images: (1, H, W, C) → (1, C, H, W) and normalize
                if v.ndim == 3 and v.shape[-1] == 3:
                    t = t.permute(0, 3, 1, 2) / 255.0
                obs_tensors[k] = t

            with torch.no_grad():
                action_batch = policy.select_action(obs_tensors)

            action = action_batch[0].cpu().numpy()
            env.send_action(action)

            if video_writer is not None and "observation.images.wrist" in obs:
                video_writer.write(obs["observation.images.wrist"])

            obs = env.get_observation()
            steps += 1

            elapsed = time.time() - t_step
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        logger.info("Rollout interrupted.")
    finally:
        if video_writer is not None:
            video_writer.close()

    return {"steps": steps, "duration": time.time() - t0}


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dry_run:
        logger.info("DRY-RUN mode: no hardware or model loading.")
        print(f"Would load checkpoint: {args.checkpoint}")
        print(f"Would run {args.num_rollouts} rollouts × {args.max_steps} steps on {args.follower_can}")
        return

    # Load policy
    logger.info(f"Loading policy from {args.checkpoint} …")
    policy, cfg = load_policy(args.checkpoint, args.device)

    # Connect robot env
    from grasp_lab.lerobot_training.robot_env import I2RTRobotEnv

    env = I2RTRobotEnv(
        follower_can=args.follower_can,
        gripper=args.gripper,
        wrist_camera_serial=args.wrist_camera_serial,
        top_camera_serial=args.top_camera_serial,
    )
    env.connect()

    results = []
    try:
        for i in range(args.num_rollouts):
            logger.info(f"=== Rollout {i + 1}/{args.num_rollouts} ===")
            video_path = None
            if args.record_video:
                video_path = Path(args.checkpoint) / "eval_videos" / f"rollout_{i:03d}.mp4"

            stats = run_rollout(
                policy=policy,
                env=env,
                max_steps=args.max_steps,
                device=args.device,
                record_video=args.record_video,
                video_path=video_path,
            )
            results.append(stats)
            logger.info(f"  steps={stats['steps']}, duration={stats['duration']:.1f}s")
            time.sleep(1.0)  # brief pause between rollouts
    finally:
        env.disconnect()

    # Summary
    print("\n=== Eval Summary ===")
    avg_steps = np.mean([r["steps"] for r in results])
    print(f"  Rollouts     : {len(results)}")
    print(f"  Avg steps    : {avg_steps:.1f}")
    print(f"  Avg duration : {np.mean([r['duration'] for r in results]):.1f}s")


if __name__ == "__main__":
    main(tyro.cli(Args))
