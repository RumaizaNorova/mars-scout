"""
rover_controller.py
===================
6-wheel differential drive controller for the ARES rover.

Takes /rover/cmd_vel commands from a read_command callable (setup_scene.py
reads them from Isaac's ROS bridge) and converts linear/angular velocity
into individual wheel joint velocities using the standard differential
drive formula — applied to all 6 wheels
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
from typing import Callable, List, Optional, Tuple

try:
    from pxr import UsdPhysics
    _PXR_AVAILABLE = True
except ImportError:
    _PXR_AVAILABLE = False


class RoverWheelController:
    """
    Drives all 6 wheels each simulation step.

    Parameters
    ----------
    stage         Active Usd.Stage
    rover_result  dict returned by build_ares_rover()
    read_command  Returns the latest (linear m/s, angular rad/s)
    """

    WHEEL_RADIUS  = 0.2625   # metres (matches RoverDims.WHEEL_R)
    TRACK_WIDTH   = 2.78     # metres (left-centre to right-centre)
    MAX_WHEEL_VEL = 10.0     # rad/s (~0.5 m/s linear at wheel radius)
    RAMP_RATE     = 2.0      # rad/s² — soft acceleration limit
    SINKAGE_DRAG  = 0.15     # 15% velocity loss due to soil sinkage (Golombek 2018)
    SLIP_ONSET    = 0.08     # slip starts if accel > 8% per step (loose regolith threshold)
    SLIP_RECOVERY = 0.5      # slip decays at 50% per second (friction regains grip)

    def __init__(self, stage, rover_result: dict,
                 read_command: Callable[[], Tuple[float, float]]):
        self._stage        = stage
        self._root         = rover_result["root_path"]
        self._read_command = read_command
        self._override: Optional[Tuple[float, float]] = None

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

        # Wheel slip state (0=no slip, 1=full slip); decays over time
        self._slip_left  = 0.0
        self._slip_right = 0.0

        print(f"[rover_controller] Ready — 6 wheels on {self._root}")

    def set_override(self, linear: float, angular: float) -> None:
        """Ignore read_command and hold this command (used by the smoke test)."""
        self._override = (float(linear), float(angular))

    # ── Per-step update ───────────────────────────────────────────────────────

    def step(self, dt: float = 1.0 / 60.0) -> None:
        """
        Call once per simulation step.
        Reads the command, ramps wheel velocities, writes DriveAPI targets.
        """
        self._cmd_linear, self._cmd_angular = (
            self._override if self._override is not None else self._read_command()
        )

        # Compute target wheel velocities from differential drive
        half_track = self.TRACK_WIDTH / 2.0
        tgt_left  = (self._cmd_linear - self._cmd_angular * half_track) \
                    / self.WHEEL_RADIUS
        tgt_right = (self._cmd_linear + self._cmd_angular * half_track) \
                    / self.WHEEL_RADIUS

        # Clamp to max
        tgt_left  = max(-self.MAX_WHEEL_VEL, min(self.MAX_WHEEL_VEL, tgt_left))
        tgt_right = max(-self.MAX_WHEEL_VEL, min(self.MAX_WHEEL_VEL, tgt_right))

        # Apply sinkage drag (soil resistance during wheel penetration)
        drag_factor = 1.0 - self.SINKAGE_DRAG
        tgt_left  *= drag_factor
        tgt_right *= drag_factor

        # Soft ramp (prevents wheel spin / sudden jerks on Mars soil)
        max_delta = self.RAMP_RATE * dt
        self._cur_left  += max(-max_delta, min(max_delta, tgt_left  - self._cur_left))
        self._cur_right += max(-max_delta, min(max_delta, tgt_right - self._cur_right))

        # Wheel slip model: slip increases if wheel velocity change is large, decays over time
        slip_onset_threshold = self.SLIP_ONSET * self.MAX_WHEEL_VEL
        left_accel  = abs((tgt_left  - self._cur_left) / max(dt, 0.001))
        right_accel = abs((tgt_right - self._cur_right) / max(dt, 0.001))

        # Slip dynamics: onset if accel > threshold, recovery otherwise
        if left_accel > slip_onset_threshold:
            self._slip_left = min(1.0, self._slip_left + 0.05)
        else:
            self._slip_left *= (1.0 - self.SLIP_RECOVERY * dt)

        if right_accel > slip_onset_threshold:
            self._slip_right = min(1.0, self._slip_right + 0.05)
        else:
            self._slip_right *= (1.0 - self.SLIP_RECOVERY * dt)

        # Apply slip to actual commanded velocities
        cmd_left  = self._cur_left  * (1.0 - self._slip_left)
        cmd_right = self._cur_right * (1.0 - self._slip_right)

        # Write to USD DriveAPI
        for jp in self._left_joints:
            self._set_joint_vel(jp, cmd_left)
        for jp in self._right_joints:
            self._set_joint_vel(jp, cmd_right)

    def _set_joint_vel(self, joint_path: str, vel_rad_s: float) -> None:
        if not _PXR_AVAILABLE:
            return
        prim = self._stage.GetPrimAtPath(joint_path)
        if not prim.IsValid():
            raise RuntimeError(f"Wheel joint not found: {joint_path}")
        UsdPhysics.DriveAPI.Get(prim, "Y").GetTargetVelocityAttr().Set(float(vel_rad_s))

    # ── Emergency stop ────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Immediately stop all wheels."""
        self._override    = (0.0, 0.0)
        self._cmd_linear  = 0.0
        self._cmd_angular = 0.0
        for jp in self._left_joints + self._right_joints:
            self._set_joint_vel(jp, 0.0)
        print("[rover_controller] STOP")
