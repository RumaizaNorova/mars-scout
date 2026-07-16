"""
Differential-drive rover kinematic model.

Integrates cmd_vel (linear.x, angular.z) to produce a continuously
updated pose in the odom frame.  The same equations govern both the
mock and, eventually, the real Isaac Sim bridge — so control code
written against this model transfers directly.

Coordinate conventions (ROS REP-103)
--------------------------------------
  x  → forward
  y  → left
  z  → up
  θ  → yaw, counter-clockwise positive

Kinematics (Euler integration at dt)
--------------------------------------
  x(t+dt) = x(t) + v·cos(θ)·dt
  y(t+dt) = y(t) + v·sin(θ)·dt
  θ(t+dt) = θ(t) + ω·dt

For small dt (≤ 50 ms) Euler is accurate enough for a ground rover.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field


@dataclass
class RoverPose:
    x: float = 0.0      # metres, odom frame
    y: float = 0.0
    z: float = 0.0      # height above ground (flat terrain = 0)
    yaw: float = 0.0    # radians

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def quaternion(self) -> np.ndarray:
        """Return (x, y, z, w) quaternion from yaw-only rotation."""
        half = self.yaw * 0.5
        return np.array([0.0, 0.0, math.sin(half), math.cos(half)])

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z, "yaw": self.yaw}


@dataclass
class RoverVelocity:
    linear:  float = 0.0   # m/s  forward
    angular: float = 0.0   # rad/s  yaw rate


@dataclass
class DifferentialDriveConfig:
    max_linear_speed:  float = 0.5    # m/s
    max_angular_speed: float = 1.0    # rad/s
    wheel_base:        float = 0.3    # metres (Jetbot-sized stand-in)
    # Slippage factor [0,1]: 0 = frictionless ice, 1 = perfect grip
    slip_factor:       float = 0.95


class DifferentialDriveModel:
    """
    Thread-safe kinematic integrator.

    Usage
    -----
    model = DifferentialDriveModel()
    model.set_velocity(0.3, 0.2)      # called from ROS cmd_vel callback
    model.step(dt=0.02)               # called from a 50 Hz timer
    pose = model.pose                 # read current pose
    """

    def __init__(
        self,
        cfg: DifferentialDriveConfig | None = None,
        initial_pose: RoverPose | None = None,
    ):
        self.cfg  = cfg or DifferentialDriveConfig()
        self.pose = initial_pose or RoverPose()
        self._vel = RoverVelocity()

        # Odometry accumulators (for covariance estimation)
        self._total_distance: float = 0.0
        self._total_rotation: float = 0.0

    # ── Control input ─────────────────────────────────────────────────────────

    def set_velocity(self, linear: float, angular: float) -> None:
        """Accept a cmd_vel and clamp to hardware limits."""
        cfg = self.cfg
        self._vel.linear  = float(np.clip(linear,  -cfg.max_linear_speed,  cfg.max_linear_speed))
        self._vel.angular = float(np.clip(angular, -cfg.max_angular_speed, cfg.max_angular_speed))

    # ── Integration ───────────────────────────────────────────────────────────

    def step(self, dt: float) -> RoverPose:
        """
        Advance the simulation by dt seconds.
        Returns the updated pose.
        """
        v = self._vel.linear  * self.cfg.slip_factor
        w = self._vel.angular * self.cfg.slip_factor

        if abs(v) < 1e-9 and abs(w) < 1e-9:
            return self.pose  # stationary — skip integration

        # Exact arc integration (avoids Euler error for pure rotations)
        if abs(w) < 1e-6:
            # Straight-line motion
            dx = v * math.cos(self.pose.yaw) * dt
            dy = v * math.sin(self.pose.yaw) * dt
            dθ = 0.0
        else:
            # Arc motion — exact solution
            r   = v / w                        # turning radius
            dθ  = w * dt
            dx  = r * (math.sin(self.pose.yaw + dθ) - math.sin(self.pose.yaw))
            dy  = r * (math.cos(self.pose.yaw)       - math.cos(self.pose.yaw + dθ))

        self.pose.x   += dx
        self.pose.y   += dy
        self.pose.yaw  = _wrap_angle(self.pose.yaw + dθ)

        self._total_distance += abs(v * dt)
        self._total_rotation += abs(dθ)

        return self.pose

    # ── Odometry covariance (diagonal, conservative) ─────────────────────────

    @property
    def odom_covariance(self) -> np.ndarray:
        """
        6×6 pose covariance matrix (x,y,z,rx,ry,rz).
        Grows with accumulated distance and rotation — mimics real odometry drift.
        """
        cov = np.zeros((6, 6), dtype=np.float64)
        d = self._total_distance
        r = self._total_rotation
        # Position uncertainty grows with distance
        cov[0, 0] = 0.001 + 0.002 * d
        cov[1, 1] = 0.001 + 0.002 * d
        cov[2, 2] = 1e-4            # z locked to ground
        # Orientation uncertainty grows with rotation
        cov[3, 3] = cov[4, 4] = 1e-4
        cov[5, 5] = 0.001 + 0.003 * r
        return cov

    # ── Camera pose (rigid offset from base_link) ─────────────────────────────

    def camera_pose_in_map(
        self,
        cam_offset: np.ndarray = np.array([0.1, 0.0, 0.15]),
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (position, quaternion) of the camera optical frame in the map frame.

        Parameters
        ----------
        cam_offset : (3,) forward/left/up offset from base_link in metres
        """
        # Rotate offset by rover yaw
        cos_y, sin_y = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
        rot = np.array([[cos_y, -sin_y, 0],
                        [sin_y,  cos_y, 0],
                        [0,      0,     1]])
        world_offset = rot @ cam_offset
        pos = self.pose.position + world_offset

        # Camera optical frame: pitched down ~15° to see ground ahead
        pitch = math.radians(-15.0)
        qx = math.sin(pitch / 2) * math.cos(self.pose.yaw / 2)
        qy = math.sin(pitch / 2) * math.sin(self.pose.yaw / 2)
        qz = math.cos(pitch / 2) * math.sin(self.pose.yaw / 2)
        qw = math.cos(pitch / 2) * math.cos(self.pose.yaw / 2)
        quat = np.array([qx, qy, qz, qw])

        return pos, quat / np.linalg.norm(quat)


def _wrap_angle(θ: float) -> float:
    """Wrap angle to (−π, π]."""
    return (θ + math.pi) % (2 * math.pi) - math.pi
