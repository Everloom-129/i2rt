"""Live 3D point cloud visualisation from a RealSense depth stream.

Usage::

    # Live point cloud viewer
    python grasp_lab/visualization/realsense_3d.py

    # Replay a stored episode with 3D visualisation
    python grasp_lab/visualization/realsense_3d.py \\
        --replay-episode ./data/grasp_mug \\
        --episode-index 0

Both modes display an Open3D interactive window.
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


@dataclass
class Args:
    camera_serial: Optional[str] = None
    """RealSense serial number (None = first available)."""
    width: int = 640
    height: int = 480
    fps: int = 30
    replay_episode: Optional[str] = None
    """Dataset dir for replay mode. If set, replays a stored episode."""
    episode_index: int = 0
    """Episode index to replay (used with --replay-episode)."""
    depth_trunc: float = 2.0
    """Discard depth points beyond this distance (metres)."""


def live_point_cloud(args: Args) -> None:
    """Stream live RGB-D frames from a RealSense and display as 3D point cloud."""
    import open3d as o3d
    from grasp_lab.data.realsense import RealSenseCamera

    cam = RealSenseCamera(
        serial=args.camera_serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    cam.connect()

    vis = o3d.visualization.Visualizer()
    vis.create_window("Live RealSense Point Cloud")

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 2.0

    print("Streaming point cloud. Close the window to exit.")
    try:
        while vis.poll_events():
            rgb, depth, intrinsics = cam.read()
            if depth is None:
                continue

            new_pcd = cam.get_point_cloud(depth, intrinsics, depth_trunc=args.depth_trunc)
            # Colour the cloud with the RGB image
            h, w = rgb.shape[:2]
            o3d_rgb = o3d.geometry.Image(rgb)
            o3d_depth = o3d.geometry.Image((depth * 1000).astype(np.uint16))
            o3d_intr = o3d.camera.PinholeCameraIntrinsic(
                width=intrinsics.width,
                height=intrinsics.height,
                fx=intrinsics.fx,
                fy=intrinsics.fy,
                cx=intrinsics.ppx,
                cy=intrinsics.ppy,
            )
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_rgb,
                o3d_depth,
                depth_scale=1000.0,
                depth_trunc=args.depth_trunc,
                convert_rgb_to_intensity=False,
            )
            new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intr)

            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors
            vis.update_geometry(pcd)
            vis.update_renderer()
    except KeyboardInterrupt:
        pass
    finally:
        cam.disconnect()
        vis.destroy_window()


def replay_episode_3d(dataset_dir: Path, episode_index: int, depth_trunc: float = 2.0) -> None:
    """Replay a stored episode with synchronized 3D point cloud visualisation.

    Requires that depth frames were saved (not yet supported in collect_demos.py;
    this function uses the wrist MP4 as RGB and generates a dummy flat depth).
    """
    import cv2
    import open3d as o3d

    # Load video
    video_pattern = f"*wrist*_episode_{episode_index:06d}.mp4"
    matches = list((dataset_dir / "videos").rglob(video_pattern))
    if not matches:
        logger.error(f"No wrist video found for episode {episode_index} in {dataset_dir}")
        return

    cap = cv2.VideoCapture(str(matches[0]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps

    # Dummy intrinsics (replace with real if available)
    from grasp_lab.data.realsense import CameraIntrinsics, RealSenseCamera
    intr = CameraIntrinsics(
        width=640, height=480,
        fx=600.0, fy=600.0,
        ppx=320.0, ppy=240.0,
        coeffs=[0] * 5,
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window(f"Episode {episode_index} Replay")
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 2.0

    print(f"Replaying episode {episode_index}. Close the window to exit.")
    try:
        while cap.isOpened() and vis.poll_events():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            rgb = frame[:, :, ::-1]

            # Use a flat depth plane at 1 m (placeholder — replace with real depth)
            depth = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32)

            o3d_rgb = o3d.geometry.Image(rgb)
            o3d_depth = o3d.geometry.Image((depth * 1000).astype(np.uint16))
            o3d_intr = o3d.camera.PinholeCameraIntrinsic(
                intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
            )
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_rgb, o3d_depth,
                depth_scale=1000.0,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )
            new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intr)
            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors

            vis.update_geometry(pcd)
            vis.update_renderer()

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        vis.destroy_window()


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)

    if args.replay_episode is not None:
        replay_episode_3d(Path(args.replay_episode), args.episode_index, args.depth_trunc)
    else:
        live_point_cloud(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
