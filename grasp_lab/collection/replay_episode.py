"""Replay a stored LeRobot v3 episode on the real robot.

Usage::

    python grasp_lab/collection/replay_episode.py \\
        --dataset-dir ./data/grasp_mug \\
        --episode-index 0 \\
        --follower-can can0 \\
        --gripper crank_4310
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tyro

logger = logging.getLogger(__name__)

CONTROL_HZ = 30


@dataclass
class Args:
    dataset_dir: str
    """Path to the LeRobot v3 dataset root."""
    episode_index: int = 0
    """Index of the episode to replay."""
    follower_can: str = "can0"
    gripper: str = "crank_4310"
    dry_run: bool = False
    """Print actions without sending to robot."""


def load_episode_actions(dataset_dir: Path, episode_index: int) -> np.ndarray:
    """Load the action sequence from a parquet shard."""
    import pandas as pd

    # Find the parquet file for this episode
    pattern = f"episode_{episode_index:06d}.parquet"
    matches = list(dataset_dir.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No parquet file found for episode {episode_index} in {dataset_dir}")

    df = pd.read_parquet(matches[0])
    actions = np.array(df["action"].tolist(), dtype=np.float32)
    return actions


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dataset_dir = Path(args.dataset_dir)

    logger.info(f"Loading episode {args.episode_index} from {dataset_dir}")
    actions = load_episode_actions(dataset_dir, args.episode_index)
    logger.info(f"Episode has {len(actions)} steps ({len(actions)/CONTROL_HZ:.1f}s)")

    if args.dry_run:
        logger.info("DRY-RUN: printing first 5 actions only.")
        for i, a in enumerate(actions[:5]):
            print(f"  step {i:03d}: {np.round(a, 3)}")
        return

    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    gripper_type = GripperType.from_string_name(args.gripper)
    robot = get_yam_robot(channel=args.follower_can, gripper_type=gripper_type)

    dt = 1.0 / CONTROL_HZ
    print(f"\nReplaying {len(actions)} steps. Press Ctrl-C to abort.\n")

    try:
        for step, action in enumerate(actions):
            t0 = time.time()
            robot.command_joint_pos(action)
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
            if step % CONTROL_HZ == 0:
                logger.info(f"  step {step}/{len(actions)}")
    except KeyboardInterrupt:
        logger.info("Replay aborted.")

    logger.info("Replay complete.")


if __name__ == "__main__":
    main(tyro.cli(Args))
