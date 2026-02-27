"""LeRobot Robot subclass wrapping the i2rt YAM arm + RealSense cameras.

This module implements the Robot protocol expected by the LeRobot evaluation
loop (lerobot/common/robot_devices/robots/utils.py).

Observation features
--------------------
- ``observation.state``              : (7,) float32 — follower joint positions
- ``observation.images.wrist``       : (H, W, 3) uint8 — wrist RGB
- ``observation.images.top``         : (H, W, 3) uint8 — top RGB (optional)

Action features
---------------
- ``action``                         : (7,) float32 — target joint positions
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default image resolution
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
STATE_DIM = 7
ACTION_DIM = 7


class I2RTRobotEnv:
    """LeRobot-compatible robot environment wrapping the i2rt YAM arm.

    Parameters
    ----------
    follower_can:
        CAN channel for the follower arm (e.g. ``"can0"``).
    gripper:
        Gripper type string (e.g. ``"crank_4310"``).
    wrist_camera_serial:
        RealSense serial for the wrist camera (None = first available).
    top_camera_serial:
        RealSense serial for the top-down camera (None = disabled).
    image_width, image_height:
        Camera resolution.
    """

    # Feature descriptors expected by LeRobot's training loop
    observation_features: Dict[str, Any] = {
        "observation.state": (STATE_DIM,),
        "observation.images.wrist": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
    }
    action_features: Dict[str, Any] = {
        "action": (ACTION_DIM,),
    }

    def __init__(
        self,
        follower_can: str = "can0",
        gripper: str = "crank_4310",
        wrist_camera_serial: Optional[str] = None,
        top_camera_serial: Optional[str] = None,
        image_width: int = IMAGE_WIDTH,
        image_height: int = IMAGE_HEIGHT,
    ):
        self.follower_can = follower_can
        self.gripper = gripper
        self.wrist_camera_serial = wrist_camera_serial
        self.top_camera_serial = top_camera_serial
        self.image_width = image_width
        self.image_height = image_height

        self._robot = None
        self._wrist_cam = None
        self._top_cam = None

        if top_camera_serial is not None:
            self.observation_features["observation.images.top"] = (image_height, image_width, 3)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Initialise hardware connections."""
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType
        from grasp_lab.data.realsense import RealSenseCamera

        gripper_type = GripperType.from_string_name(self.gripper)
        logger.info(f"Connecting follower arm on {self.follower_can} …")
        self._robot = get_yam_robot(channel=self.follower_can, gripper_type=gripper_type)

        logger.info("Connecting wrist camera …")
        self._wrist_cam = RealSenseCamera(
            serial=self.wrist_camera_serial,
            width=self.image_width,
            height=self.image_height,
        )
        self._wrist_cam.connect()

        if self.top_camera_serial is not None:
            logger.info("Connecting top camera …")
            self._top_cam = RealSenseCamera(
                serial=self.top_camera_serial,
                width=self.image_width,
                height=self.image_height,
            )
            self._top_cam.connect()

        logger.info("I2RTRobotEnv connected.")

    def disconnect(self) -> None:
        """Release hardware connections."""
        if self._wrist_cam is not None:
            self._wrist_cam.disconnect()
        if self._top_cam is not None:
            self._top_cam.disconnect()
        logger.info("I2RTRobotEnv disconnected.")

    def __enter__(self) -> "I2RTRobotEnv":
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Robot protocol
    # ------------------------------------------------------------------

    def get_observation(self) -> Dict[str, np.ndarray]:
        """Return a dict of observations matching ``observation_features``."""
        state = np.array(self._robot.get_observations()["joint_pos"], dtype=np.float32)

        wrist_rgb, _, _ = self._wrist_cam.read()

        obs = {
            "observation.state": state,
            "observation.images.wrist": wrist_rgb,
        }

        if self._top_cam is not None:
            top_rgb, _, _ = self._top_cam.read()
            obs["observation.images.top"] = top_rgb

        return obs

    def send_action(self, action: np.ndarray) -> None:
        """Send a joint-position command to the follower arm.

        Args:
            action: (7,) float32 target joint positions in radians.
        """
        self._robot.command_joint_pos(action.astype(np.float32))

    # LeRobot Robot protocol compatibility
    def get_obs(self) -> Dict[str, np.ndarray]:
        return self.get_observation()

    def num_dofs(self) -> int:
        return ACTION_DIM


# ---------------------------------------------------------------------------
# Quick smoke-test (no hardware required with --dry-run)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("I2RTRobotEnv feature shapes:")
        for k, v in I2RTRobotEnv.observation_features.items():
            print(f"  {k}: {v}")
        print("action_features:", I2RTRobotEnv.action_features)
        print("Smoke-test passed.")
    else:
        env = I2RTRobotEnv()
        env.connect()
        obs = env.get_observation()
        print("Observation keys:", list(obs.keys()))
        for k, v in obs.items():
            print(f"  {k}: {v.shape}")
        env.disconnect()
