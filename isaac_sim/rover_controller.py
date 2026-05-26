"""
rover_controller.py
===================
6-wheel differential drive controller for the ARES rover.

Subscribes to /rover/cmd_vel (geometry_msgs/Twist) and converts
linear/angular velocity commands into individual wheel joint velocities
using the standard differential drive formula — applied to all 6 wheels
simultaneously (3 left, 3 right).

Why Python instead of OmniGraph:
  OmniGraph's DifferentialController outputs 2 wheel velocities.
  Our rocker-bogie has 6 driven wheels. The Python approach is simpler,
  more debuggable, and easier to extend (ramp rates, slip limits, etc.).

Drive formula:
  left_vel  = (linear - angular * track/2) / wheel_radius   [rad/s]
  right_vel = (linear + angular * track/2) / wheel_radius   [rad/s]

Called once per simulation step from setup_scene.py's main loop.
"""

from __future__ import annotations
import math
from typing import List

try:
    from pxr import UsdPhysics
    _PXR_AVAILABLE = True
except ImportError:
    _PXR_AVAILABLE = False

try:
    import rclpy
    from geometry_msgs.msg import Twist
    _RCLPY_AVAILABLE = True
except ImportError:
    _RCLPY_AVAILABLE = False


class RoverWheelController:
    """
    Reads /rover/cmd_vel and drives all 6 wheels each simulation step.

    Parameters
    ----------
    stage         Active Usd.Stage
    rover_result  dict returned by build_ares_rover()
    """

    WHEEL_RADIUS  = 0.2625   # metres (matches RoverDims.WHEEL_R)
    TRACK_WIDTH   = 2.78     # metres (left-centre to right-centre)
    MAX_WHEEL_VEL = 10.0     # rad/s (~0.5 m/s linear at wheel radius)
    RAMP_RATE     = 2.0      # rad/s² — soft acceleration limit

    def __init__(self, stage, rover_result: dict):
        self._stage      = stage
        self._root       = rover_result["root_path"]

        # Drive joint paths (DriveAPI applied on Y axis)
        self._left_joints: List[str] = [
            f"{self._root}/SuspL/WheelF_DriveJoint",
            f"{self._root}/SuspL/WheelM_DriveJoint",
            f"{self._root}/SuspL/WheelR_DriveJoint",
        ]
        self._right_joints: List[str] = [
            f"{self._root}/SuspR/WheelF_DriveJoint",
            f"{self._root}/SuspR/WheelM_DriveJoint",
            f"{self._root}/SuspR/WheelR_DriveJoint",
        ]

        # Latest command (updated by ROS2 callback)
        self._cmd_linear  = 0.0
        self._cmd_angular = 0.0

        # Current wheel velocities (ramped)
        self._cur_left  = 0.0
        self._cur_right = 0.0

        # ROS2
        self._ros_node = None
        self._init_ros2()

        print(f"[rover_controller] Ready — 6 wheels on {self._root}")
        print(f"[rover_controller] Listening on /rover/cmd_vel")

    # ── ROS2 setup ────────────────────────────────────────────────────────────

    def _init_ros2(self) -> None:
        if not _RCLPY_AVAILABLE:
            print("[rover_controller] rclpy not available — cmd_vel disabled")
            return
        try:
            if not rclpy.ok():
                rclpy.init()
            self._ros_node = rclpy.create_node("rover_wheel_controller")
            self._ros_node.create_subscription(
                Twist, "/rover/cmd_vel", self._on_cmd_vel, 10
            )
            print("[rover_controller] ROS2 subscription /rover/cmd_vel OK")
        except Exception as e:
            print(f"[rover_controller] ROS2 init warning: {e} — proceeding without cmd_vel")
            self._ros_node = None

    def _on_cmd_vel(self, msg: "Twist") -> None:
        self._cmd_linear  = float(msg.linear.x)
        self._cmd_angular = float(msg.angular.z)

    # ── Per-step update ───────────────────────────────────────────────────────

    def step(self, dt: float = 1.0 / 60.0) -> None:
        """
        Call once per simulation step.
        Polls ROS2, ramps wheel velocities, writes DriveAPI targets.
        """
        # Poll ROS2 (non-blocking)
        if self._ros_node is not None:
            try:
                rclpy.spin_once(self._ros_node, timeout_sec=0.0)
            except Exception:
                pass

        # Compute target wheel velocities from differential drive
        half_track = self.TRACK_WIDTH / 2.0
        tgt_left  = (self._cmd_linear - self._cmd_angular * half_track) \
                    / self.WHEEL_RADIUS
        tgt_right = (self._cmd_linear + self._cmd_angular * half_track) \
                    / self.WHEEL_RADIUS

        # Clamp to max
        tgt_left  = max(-self.MAX_WHEEL_VEL, min(self.MAX_WHEEL_VEL, tgt_left))
        tgt_right = max(-self.MAX_WHEEL_VEL, min(self.MAX_WHEEL_VEL, tgt_right))

        # Soft ramp (prevents wheel spin / sudden jerks on Mars soil)
        max_delta = self.RAMP_RATE * dt
        self._cur_left  += max(-max_delta, min(max_delta, tgt_left  - self._cur_left))
        self._cur_right += max(-max_delta, min(max_delta, tgt_right - self._cur_right))

        # Write to USD DriveAPI
        for jp in self._left_joints:
            self._set_joint_vel(jp, self._cur_left)
        for jp in self._right_joints:
            self._set_joint_vel(jp, self._cur_right)

    def _set_joint_vel(self, joint_path: str, vel_rad_s: float) -> None:
        if not _PXR_AVAILABLE:
            return
        prim = self._stage.GetPrimAtPath(joint_path)
        if not prim.IsValid():
            return
        drive = UsdPhysics.DriveAPI.Get(prim, "Y")
        if drive:
            drive.GetTargetVelocityAttr().Set(float(vel_rad_s))

    # ── Emergency stop ────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Immediately stop all wheels."""
        self._cmd_linear  = 0.0
        self._cmd_angular = 0.0
        for jp in self._left_joints + self._right_joints:
            self._set_joint_vel(jp, 0.0)
        print("[rover_controller] STOP")
