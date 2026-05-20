"""
Pure-pursuit waypoint controller for differential-drive rover.

Given the rover's current pose (x, y, yaw) and a target waypoint (tx, ty),
computes linear and angular velocity commands that steer the rover to the goal.

Algorithm
---------
1. Compute bearing error:  heading to goal minus current yaw
2. Compute distance:       Euclidean distance to goal
3. Angular velocity:       proportional to bearing error (P controller)
4. Linear velocity:        proportional to distance, scaled down when turning
                           hard (so the rover doesn't overshoot)

The controller is stateless — call compute() every tick.
"""

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class ControllerConfig:
    # Proportional gains
    k_angular: float = 1.2     # rad/s per rad of heading error
    k_linear:  float = 0.4     # m/s per metre of distance

    # Limits
    max_linear:  float = 0.5   # m/s
    max_angular: float = 0.8   # rad/s

    # Slow down when heading error exceeds this (radians)
    heading_slowdown_thresh: float = 0.3

    # Stop driving forward if heading error is larger than this
    max_heading_error: float = 1.2   # ~70°


@dataclass
class ControlOutput:
    linear:  float   # m/s
    angular: float   # rad/s
    distance_m:     float
    bearing_error:  float
    aligned:        bool     # True when heading error < slowdown threshold


class PurePursuitController:
    """
    Stateless P-controller for waypoint pursuit.
    """

    def __init__(self, cfg: ControllerConfig | None = None):
        self.cfg = cfg or ControllerConfig()

    def compute(
        self,
        rover_x:   float,
        rover_y:   float,
        rover_yaw: float,
        goal_x:    float,
        goal_y:    float,
    ) -> ControlOutput:
        cfg = self.cfg

        dx = goal_x - rover_x
        dy = goal_y - rover_y
        distance = math.hypot(dx, dy)

        # Angle from rover to goal in world frame
        angle_to_goal = math.atan2(dy, dx)

        # Heading error: wrap to (−π, π)
        heading_error = _wrap(angle_to_goal - rover_yaw)

        # Angular velocity — proportional control
        angular = float(
            max(-cfg.max_angular, min(cfg.max_angular, cfg.k_angular * heading_error))
        )

        # Linear velocity — only drive forward if roughly aligned
        aligned = abs(heading_error) < cfg.heading_slowdown_thresh
        if abs(heading_error) > cfg.max_heading_error:
            linear = 0.0   # turn in place
        else:
            # Scale down linearly as heading error grows
            alignment_factor = max(0.0, 1.0 - abs(heading_error) / cfg.max_heading_error)
            linear = float(
                min(cfg.max_linear, cfg.k_linear * distance) * alignment_factor
            )

        return ControlOutput(
            linear=linear,
            angular=angular,
            distance_m=distance,
            bearing_error=heading_error,
            aligned=aligned,
        )


def _wrap(angle: float) -> float:
    """Wrap angle to (−π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi
