"""Collect teleoperation demonstrations and write them in LeRobot v3 format.

Usage::

    python grasp_lab/collection/collect_demos.py \\
        --follower-can can0 \\
        --leader-can can1 \\
        --gripper crank_4310 \\
        --task grasp_mug \\
        --dataset-dir ./data/grasp_mug \\
        --num-episodes 20

    # Smoke-test without hardware
    python grasp_lab/collection/collect_demos.py --dry-run

Recording protocol
------------------
- Press the teaching-handle button (io_inputs[0] > 0.5) to **start** an episode.
- Move the leader arm to demonstrate the task.
- Press the button again to **stop / save** the episode.
- A second button press within 0.5 s after stopping **discards** the episode.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import tyro

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTROL_HZ = 30
STATE_DIM = 7   # 6 arm joints + 1 gripper
ACTION_DIM = 7  # same


# ---------------------------------------------------------------------------
# LeRobot v3 writer helpers
# ---------------------------------------------------------------------------

class LeRobotV3Writer:
    """Incrementally writes a LeRobot v3 dataset to disk.

    Directory layout::

        <root>/
          data/
            chunk-000/
              episode_000000.parquet
              ...
          videos/
            chunk-000/
              observation.images.wrist_episode_000000.mp4
              observation.images.top_episode_000000.mp4  (if present)
          meta/
            info.json
            episodes.jsonl
            stats.json          (written on finalise())
    """

    CHUNK_SIZE = 1000  # episodes per chunk

    def __init__(
        self,
        root: Path,
        task: str,
        fps: int = CONTROL_HZ,
        image_keys: Optional[list[str]] = None,
        image_shape: tuple[int, int, int] = (480, 640, 3),
    ):
        self.root = root
        self.task = task
        self.fps = fps
        self.image_keys = image_keys or ["observation.images.wrist"]
        self.image_shape = image_shape

        self._episode_idx = 0
        self._step_idx = 0
        self._episode_rows: list[dict] = []
        self._episode_meta: list[dict] = []
        self._video_writers: dict[str, object] = {}

        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "data").mkdir(exist_ok=True)
        (self.root / "videos").mkdir(exist_ok=True)
        (self.root / "meta").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    def begin_episode(self) -> None:
        """Start recording a new episode."""
        self._episode_rows = []
        self._video_writers = {}
        chunk_dir = self._chunk_dir("videos")
        for key in self.image_keys:
            video_path = chunk_dir / f"{key}_episode_{self._episode_idx:06d}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            self._video_writers[key] = _VideoWriter(video_path, fps=self.fps)

    def record_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        images: dict[str, np.ndarray],
        depth: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a single control step."""
        if timestamp is None:
            timestamp = self._step_idx / self.fps

        row = {
            "episode_index": self._episode_idx,
            "frame_index": len(self._episode_rows),
            "timestamp": float(timestamp),
            "index": self._step_idx,
            "task_index": 0,
        }
        row["observation.state"] = state.tolist()
        row["action"] = action.tolist()

        for key, img in images.items():
            if key in self._video_writers:
                self._video_writers[key].write(img)

        self._episode_rows.append(row)
        self._step_idx += 1

    def end_episode(self, discard: bool = False) -> None:
        """Finish the current episode."""
        for vw in self._video_writers.values():
            vw.close()

        if discard or len(self._episode_rows) == 0:
            logger.info(f"Episode {self._episode_idx} discarded.")
            # Remove partially-written video files
            chunk_dir = self._chunk_dir("videos")
            for key in self.image_keys:
                p = chunk_dir / f"{key}_episode_{self._episode_idx:06d}.mp4"
                if p.exists():
                    p.unlink()
            return

        self._write_episode_parquet()
        self._episode_meta.append(
            {
                "episode_index": self._episode_idx,
                "tasks": [self.task],
                "length": len(self._episode_rows),
            }
        )
        logger.info(f"Episode {self._episode_idx} saved ({len(self._episode_rows)} steps).")
        self._episode_idx += 1

    def finalise(self) -> None:
        """Write metadata files and compute normalization statistics."""
        self._write_episodes_jsonl()
        self._write_info_json()
        self._write_stats_json()
        logger.info(f"Dataset finalised: {self._episode_idx} episodes → {self.root}")

    # ------------------------------------------------------------------
    def _chunk_dir(self, kind: str) -> Path:
        chunk = self._episode_idx // self.CHUNK_SIZE
        d = self.root / kind / f"chunk-{chunk:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_episode_parquet(self) -> None:
        import pandas as pd

        df = pd.DataFrame(self._episode_rows)
        chunk_dir = self._chunk_dir("data")
        path = chunk_dir / f"episode_{self._episode_idx:06d}.parquet"
        df.to_parquet(path, index=False)

    def _write_episodes_jsonl(self) -> None:
        path = self.root / "meta" / "episodes.jsonl"
        with open(path, "w") as f:
            for meta in self._episode_meta:
                f.write(json.dumps(meta) + "\n")

    def _write_info_json(self) -> None:
        total_steps = sum(m["length"] for m in self._episode_meta)
        info = {
            "codebase_version": "v2.0",
            "fps": self.fps,
            "robot_type": "yam",
            "total_episodes": self._episode_idx,
            "total_frames": total_steps,
            "total_tasks": 1,
            "tasks": {0: self.task},
            "features": {
                "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": None},
                "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": None},
                **{
                    key: {
                        "dtype": "video",
                        "shape": list(self.image_shape),
                        "names": ["height", "width", "channel"],
                        "video_info": {"video.fps": self.fps, "video.codec": "mp4v"},
                    }
                    for key in self.image_keys
                },
            },
            "encoding": {"video": {"vcodec": "libx264", "pix_fmt": "yuv420p", "g": 2, "crf": 22}},
        }
        with open(self.root / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=2)

    def _write_stats_json(self) -> None:
        """Compute per-feature mean/std/min/max from collected episodes."""
        if not self._episode_meta:
            return

        import pandas as pd

        all_states, all_actions = [], []
        for chunk_path in sorted((self.root / "data").rglob("*.parquet")):
            df = pd.read_parquet(chunk_path, columns=["observation.state", "action"])
            all_states.extend(df["observation.state"].tolist())
            all_actions.extend(df["action"].tolist())

        def _stats(arr):
            a = np.array(arr)
            return {
                "mean": a.mean(axis=0).tolist(),
                "std": a.std(axis=0).tolist(),
                "min": a.min(axis=0).tolist(),
                "max": a.max(axis=0).tolist(),
            }

        stats = {
            "observation.state": _stats(all_states),
            "action": _stats(all_actions),
        }
        with open(self.root / "meta" / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)


class _VideoWriter:
    """Thin wrapper around cv2.VideoWriter with lazy import."""

    def __init__(self, path: Path, fps: int):
        self.path = path
        self.fps = fps
        self._writer = None
        self._fourcc = None

    def write(self, rgb: np.ndarray) -> None:
        import cv2

        if self._writer is None:
            h, w = rgb.shape[:2]
            self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), self._fourcc, self.fps, (w, h))
        bgr = rgb[:, :, ::-1]
        self._writer.write(bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class Args:
    follower_can: str = "can0"
    """CAN channel for the follower (controlled) arm."""
    leader_can: str = "can1"
    """CAN channel for the leader (teaching) arm."""
    gripper: str = "crank_4310"
    """Gripper type for the follower arm. One of: crank_4310, linear_3507, linear_4310, no_gripper."""
    camera_serial: Optional[str] = None
    """RealSense serial number. None = first available device."""
    task: str = "grasp_task"
    """Task name stored in dataset metadata."""
    dataset_dir: str = "./data/grasp_task"
    """Output directory for the LeRobot v3 dataset."""
    num_episodes: int = 20
    """Number of episodes to collect."""
    dry_run: bool = False
    """Smoke-test mode: simulate without real hardware."""
    image_width: int = 640
    image_height: int = 480
    enable_top_camera: bool = False
    """Record a second 'top' camera (requires two RealSense devices)."""
    top_camera_serial: Optional[str] = None


def _fake_leader():
    """Dummy leader that returns random joint pos + unpressed button."""
    class _Fake:
        def get_info(self):
            qpos = np.zeros(STATE_DIM, dtype=np.float32)
            io = [0.0]
            return qpos, io
    return _Fake()


def _fake_follower():
    """Dummy follower that accepts commands and returns zero state."""
    class _Fake:
        def get_observations(self):
            return {"joint_pos": np.zeros(STATE_DIM, dtype=np.float32)}
        def command_joint_pos(self, pos):
            pass
    return _Fake()


def _fake_camera(width, height):
    class _Fake:
        def read(self):
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            depth = np.zeros((height, width), dtype=np.float32)
            from grasp_lab.data.realsense import CameraIntrinsics
            intr = CameraIntrinsics(width, height, 600.0, 600.0, width/2, height/2, [0]*5)
            return rgb, depth, intr
        def connect(self): pass
        def disconnect(self): pass
    return _Fake()


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # ------------------------------------------------------------------
    # Hardware setup
    # ------------------------------------------------------------------
    if args.dry_run:
        logger.info("DRY-RUN mode: using fake hardware.")
        follower = _fake_follower()
        leader = _fake_leader()
        wrist_cam = _fake_camera(args.image_width, args.image_height)
        top_cam = _fake_camera(args.image_width, args.image_height) if args.enable_top_camera else None
    else:
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType
        from scripts.minimum_gello import YAMLeaderRobot
        from grasp_lab.data.realsense import RealSenseCamera

        gripper_type = GripperType.from_string_name(args.gripper)

        logger.info("Initialising follower arm …")
        follower_robot = get_yam_robot(channel=args.follower_can, gripper_type=gripper_type)
        follower = follower_robot

        logger.info("Initialising leader arm …")
        leader_robot = get_yam_robot(
            channel=args.leader_can,
            gripper_type=GripperType.YAM_TEACHING_HANDLE,
            zero_gravity_mode=True,
        )
        leader = YAMLeaderRobot(leader_robot)

        logger.info("Connecting wrist camera …")
        wrist_cam = RealSenseCamera(
            serial=args.camera_serial,
            width=args.image_width,
            height=args.image_height,
            fps=CONTROL_HZ,
        )
        wrist_cam.connect()

        top_cam = None
        if args.enable_top_camera:
            top_cam = RealSenseCamera(
                serial=args.top_camera_serial,
                width=args.image_width,
                height=args.image_height,
                fps=CONTROL_HZ,
            )
            top_cam.connect()

    # ------------------------------------------------------------------
    # Dataset writer
    # ------------------------------------------------------------------
    image_keys = ["observation.images.wrist"]
    if top_cam is not None:
        image_keys.append("observation.images.top")

    writer = LeRobotV3Writer(
        root=Path(args.dataset_dir),
        task=args.task,
        fps=CONTROL_HZ,
        image_keys=image_keys,
        image_shape=(args.image_height, args.image_width, 3),
    )

    # ------------------------------------------------------------------
    # Collection loop
    # ------------------------------------------------------------------
    dt = 1.0 / CONTROL_HZ
    collected = 0

    print("\nReady. Press the teaching-handle button to start an episode.")
    print("Press again to stop and save.  Press quickly a second time to discard.\n")

    try:
        while collected < args.num_episodes:
            # ---- wait for button press to start ----
            if not args.dry_run:
                while True:
                    _, io = leader.get_info()
                    if io[0] > 0.5:
                        break
                    time.sleep(0.02)
                # debounce
                while True:
                    _, io = leader.get_info()
                    if io[0] <= 0.5:
                        break
                    time.sleep(0.02)

            logger.info(f"Recording episode {collected + 1}/{args.num_episodes} …")
            writer.begin_episode()
            t0 = time.time()
            step_count = 0

            while True:
                step_start = time.time()

                # Read leader (action) and follower (state)
                if args.dry_run:
                    leader_info = leader.get_info()
                    action, io = leader_info
                    state = follower.get_observations()["joint_pos"]
                else:
                    action, io = leader.get_info()
                    state = follower.get_observations()["joint_pos"]
                    follower.command_joint_pos(action)

                # Read cameras
                wrist_rgb, wrist_depth, _ = wrist_cam.read()
                images = {"observation.images.wrist": wrist_rgb}
                if top_cam is not None:
                    top_rgb, _, _ = top_cam.read()
                    images["observation.images.top"] = top_rgb

                writer.record_step(
                    state=np.array(state, dtype=np.float32),
                    action=np.array(action, dtype=np.float32),
                    images=images,
                    timestamp=time.time() - t0,
                )
                step_count += 1

                # Check for stop signal
                if args.dry_run:
                    # In dry-run, record 30 frames per episode
                    if step_count >= 30:
                        break
                else:
                    if io[0] > 0.5:
                        break

                # Rate limiting
                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

            # ---- episode ended ----
            if args.dry_run:
                discard = False
            else:
                # Allow a quick second press to discard
                discard = False
                t_stop = time.time()
                # debounce stop button
                while True:
                    _, io = leader.get_info()
                    if io[0] <= 0.5:
                        break
                    time.sleep(0.02)
                # short window: second press = discard
                while time.time() - t_stop < 0.8:
                    _, io = leader.get_info()
                    if io[0] > 0.5:
                        discard = True
                        break
                    time.sleep(0.02)

            writer.end_episode(discard=discard)

            if not discard:
                collected += 1
                logger.info(f"Collected {collected}/{args.num_episodes} episodes ({step_count} steps).")
            else:
                logger.info("Episode discarded.")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    finally:
        writer.finalise()
        if not args.dry_run:
            wrist_cam.disconnect()
            if top_cam is not None:
                top_cam.disconnect()

    logger.info(f"Done. Dataset written to {args.dataset_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
