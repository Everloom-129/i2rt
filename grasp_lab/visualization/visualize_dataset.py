"""Dataset browser: episode viewer and action plots.

Usage::

    python grasp_lab/visualization/visualize_dataset.py \\
        --dataset-dir ./data/grasp_mug

    # Show a specific episode
    python grasp_lab/visualization/visualize_dataset.py \\
        --dataset-dir ./data/grasp_mug \\
        --episode-index 2

    # Export episode frames as a GIF
    python grasp_lab/visualization/visualize_dataset.py \\
        --dataset-dir ./data/grasp_mug \\
        --episode-index 0 \\
        --export-gif ./episode_0.gif
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
    episode_index: Optional[int] = None
    """Which episode to show. If None, shows a summary of all episodes."""
    export_gif: Optional[str] = None
    """Path to export the episode frames as a GIF."""
    fps: int = 10
    """Playback FPS for the episode viewer."""
    show_depth: bool = False
    """Show depth frames if available."""


def load_episode(dataset_dir: Path, episode_index: int):
    """Load parquet + video frames for one episode."""
    import pandas as pd

    pattern = f"episode_{episode_index:06d}.parquet"
    matches = list((dataset_dir / "data").rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Episode {episode_index} not found in {dataset_dir}")

    df = pd.read_parquet(matches[0])
    states = np.array(df["observation.state"].tolist(), dtype=np.float32)
    actions = np.array(df["action"].tolist(), dtype=np.float32)

    # Load video frames
    video_frames: dict[str, np.ndarray] = {}
    for video_path in (dataset_dir / "videos").rglob(f"*_episode_{episode_index:06d}.mp4"):
        key = video_path.stem.replace(f"_episode_{episode_index:06d}", "")
        frames = _decode_video(video_path)
        if frames is not None:
            video_frames[key] = frames

    return df, states, actions, video_frames


def _decode_video(path: Path) -> Optional[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame[:, :, ::-1])  # BGR → RGB
    cap.release()
    return np.array(frames, dtype=np.uint8) if frames else None


def print_dataset_summary(dataset_dir: Path) -> None:
    """Print a human-readable summary of the dataset."""
    info_path = dataset_dir / "meta" / "info.json"
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"

    if not info_path.exists():
        print(f"No meta/info.json found in {dataset_dir}")
        return

    with open(info_path) as f:
        info = json.load(f)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_dir}")
    print(f"  Total episodes : {info.get('total_episodes', '?')}")
    print(f"  Total frames   : {info.get('total_frames', '?')}")
    print(f"  FPS            : {info.get('fps', '?')}")
    print(f"  Tasks          : {info.get('tasks', {})}")
    print(f"\nFeatures:")
    for name, feat in info.get("features", {}).items():
        print(f"  {name}: shape={feat.get('shape')}, dtype={feat.get('dtype')}")

    if episodes_path.exists():
        print(f"\nEpisodes:")
        with open(episodes_path) as f:
            for i, line in enumerate(f):
                ep = json.loads(line)
                print(f"  [{ep['episode_index']:4d}] length={ep['length']:5d} steps  tasks={ep['tasks']}")
                if i >= 9:
                    remaining = info.get("total_episodes", 0) - 10
                    if remaining > 0:
                        print(f"  ... and {remaining} more")
                    break
    print(f"{'='*60}\n")


def plot_episode(states: np.ndarray, actions: np.ndarray, episode_index: int) -> None:
    """Plot state and action trajectories for one episode."""
    import matplotlib.pyplot as plt

    n_dofs = states.shape[1]
    t = np.arange(len(states))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Episode {episode_index}: State and Action Trajectories")

    ax_state, ax_action = axes

    for j in range(n_dofs):
        ax_state.plot(t, states[:, j], label=f"joint {j}")
    ax_state.set_ylabel("State (rad)")
    ax_state.legend(loc="upper right", fontsize="small")
    ax_state.grid(True, alpha=0.3)

    for j in range(actions.shape[1]):
        ax_action.plot(t, actions[:, j], label=f"joint {j}")
    ax_action.set_ylabel("Action (rad)")
    ax_action.set_xlabel("Step")
    ax_action.legend(loc="upper right", fontsize="small")
    ax_action.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def play_episode(
    video_frames: dict[str, np.ndarray],
    fps: int,
    export_gif: Optional[str] = None,
) -> None:
    """Display video frames in an OpenCV window or export as GIF."""
    import cv2

    if not video_frames:
        print("No video frames to display.")
        return

    # Pick the wrist camera preferentially
    key = next((k for k in video_frames if "wrist" in k), next(iter(video_frames)))
    frames = video_frames[key]
    n = len(frames)
    print(f"Playing '{key}': {n} frames @ {fps} fps  (press 'q' to quit)")

    dt_ms = int(1000 / fps)

    if export_gif:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            export_gif,
            save_all=True,
            append_images=pil_frames[1:],
            duration=dt_ms,
            loop=0,
        )
        print(f"GIF saved to {export_gif}")
        return

    for frame in frames:
        bgr = frame[:, :, ::-1]
        cv2.imshow(key, bgr)
        if cv2.waitKey(dt_ms) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    dataset_dir = Path(args.dataset_dir)

    print_dataset_summary(dataset_dir)

    if args.episode_index is not None:
        df, states, actions, video_frames = load_episode(dataset_dir, args.episode_index)
        print(f"Episode {args.episode_index}: {len(df)} steps, "
              f"video keys: {list(video_frames.keys())}")

        plot_episode(states, actions, args.episode_index)
        play_episode(video_frames, fps=args.fps, export_gif=args.export_gif)


if __name__ == "__main__":
    main(tyro.cli(Args))
