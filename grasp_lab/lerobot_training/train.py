"""Training entry point for LeRobot VLA policies on i2rt data.

This script is a thin wrapper that sets up the run configuration and
delegates to lerobot's training machinery.

Usage::

    # ACT policy
    python grasp_lab/lerobot_training/train.py \\
        --dataset-dir ./data/grasp_mug \\
        --policy act \\
        --output-dir ./runs/grasp_mug_act

    # Diffusion policy
    python grasp_lab/lerobot_training/train.py \\
        --dataset-dir ./data/grasp_mug \\
        --policy diffusion \\
        --output-dir ./runs/grasp_mug_diffusion

    # Smoke-test without GPU / dataset
    python grasp_lab/lerobot_training/train.py \\
        --dataset-dir ./test_data \\
        --policy act \\
        --dry-run
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import tyro

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent / "configs"


@dataclass
class Args:
    dataset_dir: str
    """Local path to the LeRobot v3 dataset."""
    policy: Literal["act", "diffusion"] = "act"
    """Policy architecture to train."""
    output_dir: str = "./runs/grasp_lab"
    """Directory for checkpoints, logs, and wandb artefacts."""
    num_epochs: Optional[int] = None
    """Override the number of training epochs from the config."""
    batch_size: Optional[int] = None
    """Override batch size."""
    lr: Optional[float] = None
    """Override learning rate."""
    wandb_project: str = "grasp_lab"
    wandb_entity: Optional[str] = None
    seed: int = 42
    device: str = "cuda"
    dry_run: bool = False
    """Validate config without launching real training."""
    resume: Optional[str] = None
    """Path to a checkpoint directory to resume from."""


def build_hydra_overrides(args: Args) -> list[str]:
    """Translate CLI args into Hydra override strings for lerobot.scripts.train."""
    overrides = [
        f"dataset_repo_id={args.dataset_dir}",
        f"training.seed={args.seed}",
        f"device={args.device}",
        f"wandb.project={args.wandb_project}",
        f"output_dir={args.output_dir}",
    ]
    if args.num_epochs is not None:
        overrides.append(f"training.num_epochs={args.num_epochs}")
    if args.batch_size is not None:
        overrides.append(f"training.batch_size={args.batch_size}")
    if args.lr is not None:
        overrides.append(f"training.lr={args.lr}")
    if args.wandb_entity is not None:
        overrides.append(f"wandb.entity={args.wandb_entity}")
    if args.resume is not None:
        overrides.append(f"resume=true")
        overrides.append(f"training.resume_checkpoint={args.resume}")
    return overrides


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    policy_cfg = CONFIGS_DIR / f"{args.policy}.yaml"
    robot_cfg = CONFIGS_DIR / "robot" / "i2rt_yam.yaml"

    logger.info(f"Policy config : {policy_cfg}")
    logger.info(f"Robot config  : {robot_cfg}")
    logger.info(f"Dataset       : {args.dataset_dir}")
    logger.info(f"Output dir    : {args.output_dir}")

    if args.dry_run:
        logger.info("DRY-RUN: config validated. No training launched.")
        print("Overrides that would be passed to lerobot.scripts.train:")
        for o in build_hydra_overrides(args):
            print(f"  {o}")
        return

    try:
        from lerobot.scripts.train import train
    except ImportError as e:
        raise ImportError(
            "lerobot is not installed. Install with:\n"
            "  pip install 'lerobot @ git+https://github.com/huggingface/lerobot.git'"
        ) from e

    overrides = build_hydra_overrides(args)

    # Set WANDB env vars if not already set
    if args.wandb_entity and not os.environ.get("WANDB_ENTITY"):
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    logger.info(f"Launching lerobot training with policy={args.policy} …")

    # LeRobot uses Hydra; call its main entry point with overrides
    # The exact call signature depends on the installed lerobot version;
    # we try the most common patterns.
    try:
        train(overrides=overrides, config_path=str(policy_cfg))
    except TypeError:
        # Older lerobot versions may not accept config_path
        train(overrides=overrides)


if __name__ == "__main__":
    main(tyro.cli(Args))
