"""Convert a LeRobot v3 dataset to the RLDS/TFRecord format expected by openpi.

Feature mapping
---------------
LeRobot v3              → openpi / RLDS
observation.state       → observation/state
action                  → action
observation.images.*    → observation/image  (wrist), observation/image_top (top)

Usage::

    python grasp_lab/openpi_training/convert_dataset.py \\
        --dataset-dir ./data/grasp_mug \\
        --output-dir ./data/grasp_mug_rlds

    # Dry run (print shapes, no writing)
    python grasp_lab/openpi_training/convert_dataset.py \\
        --dataset-dir ./data/grasp_mug \\
        --output-dir /tmp/test_rlds \\
        --dry-run
"""

from __future__ import annotations

import json
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
    """Path to the LeRobot v3 dataset root."""
    output_dir: str
    """Directory where TFRecord shards will be written."""
    dry_run: bool = False
    """Print shapes and exit without writing."""
    shard_size: int = 100
    """Number of episodes per TFRecord shard."""


# ---------------------------------------------------------------------------
# RLDS step structure
# ---------------------------------------------------------------------------

def _make_rlds_step(
    state: np.ndarray,
    action: np.ndarray,
    images: dict[str, np.ndarray],
    is_first: bool,
    is_last: bool,
    is_terminal: bool,
) -> dict:
    """Pack arrays into an RLDS step dict (to be serialised as tf.Example)."""
    obs = {"state": state}
    if "observation.images.wrist" in images:
        obs["image"] = images["observation.images.wrist"]
    if "observation.images.top" in images:
        obs["image_top"] = images["observation.images.top"]

    return {
        "observation": obs,
        "action": action,
        "is_first": bool(is_first),
        "is_last": bool(is_last),
        "is_terminal": bool(is_terminal),
        "reward": 0.0,
        "discount": 1.0,
    }


def _encode_tf_example(step: dict):
    """Encode a step dict as a tf.train.Example proto."""
    import tensorflow as tf

    def _bytes(arr: np.ndarray) -> bytes:
        return arr.tobytes()

    feature = {}
    # Observation state
    feature["observation/state"] = tf.train.Feature(
        float_list=tf.train.FloatList(value=step["observation"]["state"].tolist())
    )
    # Images
    for key in ("image", "image_top"):
        if key in step["observation"]:
            encoded = _encode_image_bytes(step["observation"][key])
            feature[f"observation/{key}"] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[encoded])
            )
    # Action
    feature["action"] = tf.train.Feature(
        float_list=tf.train.FloatList(value=step["action"].tolist())
    )
    # Scalar flags
    for flag in ("is_first", "is_last", "is_terminal"):
        feature[flag] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(step[flag])])
        )
    feature["reward"] = tf.train.Feature(
        float_list=tf.train.FloatList(value=[step["reward"]])
    )
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def _encode_image_bytes(rgb: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 RGB array as a PNG byte string."""
    import cv2

    bgr = rgb[:, :, ::-1]
    _, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Episode loading helpers
# ---------------------------------------------------------------------------

def _load_episode_parquet(dataset_dir: Path, episode_index: int):
    """Load a single episode's tabular data from parquet."""
    import pandas as pd

    pattern = f"episode_{episode_index:06d}.parquet"
    matches = list((dataset_dir / "data").rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No parquet found for episode {episode_index}")
    df = pd.read_parquet(matches[0])
    return df


def _load_episode_video_frames(dataset_dir: Path, episode_index: int, image_key: str) -> Optional[np.ndarray]:
    """Decode an MP4 shard into an array of shape (T, H, W, 3)."""
    import cv2

    pattern = f"{image_key}_episode_{episode_index:06d}.mp4"
    matches = list((dataset_dir / "videos").rglob(pattern))
    if not matches:
        return None

    cap = cv2.VideoCapture(str(matches[0]))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = frame[:, :, ::-1]
        frames.append(rgb)
    cap.release()
    return np.array(frames, dtype=np.uint8) if frames else None


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(args: Args) -> None:
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    info_path = dataset_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    num_episodes = info["total_episodes"]
    image_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    logger.info(f"Dataset: {num_episodes} episodes, image keys: {image_keys}")

    if args.dry_run:
        # Print shapes for the first episode and return
        df = _load_episode_parquet(dataset_dir, 0)
        states = np.array(df["observation.state"].tolist())
        actions = np.array(df["action"].tolist())
        print(f"Episode 0: {len(df)} steps")
        print(f"  state shape : {states.shape}")
        print(f"  action shape: {actions.shape}")
        for key in image_keys:
            frames = _load_episode_video_frames(dataset_dir, 0, key)
            if frames is not None:
                print(f"  {key} frames : {frames.shape}")
        print("Dry-run complete.")
        return

    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "tensorflow is required for RLDS conversion. "
            "Install with: pip install tensorflow"
        ) from e

    output_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    writer = None

    for ep_idx in range(num_episodes):
        if ep_idx % args.shard_size == 0:
            if writer is not None:
                writer.close()
            shard_path = output_dir / f"shard_{shard_idx:05d}.tfrecord"
            writer = tf.io.TFRecordWriter(str(shard_path))
            shard_idx += 1
            logger.info(f"Writing shard {shard_idx}: {shard_path.name}")

        df = _load_episode_parquet(dataset_dir, ep_idx)
        states = np.array(df["observation.state"].tolist(), dtype=np.float32)
        actions = np.array(df["action"].tolist(), dtype=np.float32)

        video_frames: dict[str, Optional[np.ndarray]] = {}
        for key in image_keys:
            video_frames[key] = _load_episode_video_frames(dataset_dir, ep_idx, key)

        n_steps = len(df)
        for t in range(n_steps):
            images = {}
            for key in image_keys:
                if video_frames[key] is not None and t < len(video_frames[key]):
                    images[key] = video_frames[key][t]

            step = _make_rlds_step(
                state=states[t],
                action=actions[t],
                images=images,
                is_first=(t == 0),
                is_last=(t == n_steps - 1),
                is_terminal=(t == n_steps - 1),
            )
            writer.write(_encode_tf_example(step))

        if ep_idx % 10 == 0:
            logger.info(f"  Converted episode {ep_idx}/{num_episodes}")

    if writer is not None:
        writer.close()

    # Write dataset metadata
    meta = {
        "num_episodes": num_episodes,
        "num_shards": shard_idx,
        "shard_size": args.shard_size,
        "source": str(dataset_dir),
        "image_keys": image_keys,
        "feature_map": {
            "observation.state": "observation/state",
            "action": "action",
            **{k: f"observation/{'image' if 'wrist' in k else 'image_top'}" for k in image_keys},
        },
    }
    with open(output_dir / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Conversion complete: {num_episodes} episodes → {shard_idx} shards in {output_dir}")


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    convert(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
