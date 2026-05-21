"""
Tests for mars_scout_sim_bridge — pure Python, no ROS needed.

These tests validate:
  1. TF quaternion maths (camera transform)
  2. Throttle logic
  3. Topic-parameter defaults
  4. Yaw extraction from quaternion (for RoverState)
"""

from __future__ import annotations
import math
import sys
from pathlib import Path

# Make the package importable without a colcon build
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers extracted from sim_bridge_node for pure-Python testing ─────────────

def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


def _camera_transform_quaternion(pitch_deg: float = -15.0):
    """Returns (qy, qw) for a pure Y-axis rotation of pitch_deg."""
    pitch = math.radians(pitch_deg)
    qy = math.sin(pitch / 2.0)
    qw = math.cos(pitch / 2.0)
    return qy, qw


def _throttle_check(last_t: float, now: float, min_dt: float) -> tuple[bool, float]:
    """Returns (ok, updated_last_t)."""
    if now - last_t >= min_dt:
        return True, now
    return False, last_t


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestYawExtraction:
    def test_zero_yaw(self):
        yaw = _yaw_from_quaternion(0, 0, 0, 1)
        assert abs(yaw) < 1e-9

    def test_90_deg_yaw(self):
        # 90 deg around Z: q = (0, 0, sin45, cos45)
        s = math.sin(math.pi / 4)
        yaw = _yaw_from_quaternion(0, 0, s, s)
        assert abs(yaw - math.pi / 2) < 1e-6

    def test_minus_90_deg_yaw(self):
        s = math.sin(math.pi / 4)
        yaw = _yaw_from_quaternion(0, 0, -s, s)
        assert abs(yaw + math.pi / 2) < 1e-6

    def test_180_deg_yaw(self):
        yaw = _yaw_from_quaternion(0, 0, 1, 0)
        assert abs(abs(yaw) - math.pi) < 1e-6

    def test_arbitrary_yaw(self):
        angle = 1.23
        s = math.sin(angle / 2)
        c = math.cos(angle / 2)
        yaw = _yaw_from_quaternion(0, 0, s, c)
        assert abs(yaw - angle) < 1e-6


class TestCameraTransform:
    def test_unit_quaternion(self):
        """Camera transform must be a unit quaternion."""
        qy, qw = _camera_transform_quaternion(-15.0)
        magnitude = math.sqrt(qy**2 + qw**2)
        assert abs(magnitude - 1.0) < 1e-9

    def test_pitch_angle_correct(self):
        """Reconstructed pitch from quaternion should match input."""
        for pitch_deg in [-5.0, -15.0, -30.0, 0.0]:
            qy, qw = _camera_transform_quaternion(pitch_deg)
            # recover pitch: 2 * asin(qy)
            recovered = math.degrees(2.0 * math.asin(max(-1.0, min(1.0, qy))))
            assert abs(recovered - pitch_deg) < 0.01, (
                f"pitch_deg={pitch_deg}, recovered={recovered}"
            )

    def test_default_is_downward(self):
        """Default camera pitch should be negative (pointing down)."""
        qy, qw = _camera_transform_quaternion()
        pitch = 2.0 * math.asin(qy)
        assert pitch < 0.0


class TestThrottle:
    def test_first_call_always_passes(self):
        ok, _ = _throttle_check(last_t=0.0, now=1.0, min_dt=0.1)
        assert ok

    def test_too_soon_blocked(self):
        ok, _ = _throttle_check(last_t=1.0, now=1.05, min_dt=0.1)
        assert not ok

    def test_exactly_on_boundary_passes(self):
        ok, _ = _throttle_check(last_t=1.0, now=1.1, min_dt=0.1)
        assert ok

    def test_updates_last_t_on_pass(self):
        _, new_t = _throttle_check(last_t=0.0, now=5.0, min_dt=0.1)
        assert new_t == 5.0

    def test_does_not_update_last_t_on_block(self):
        _, new_t = _throttle_check(last_t=1.0, now=1.05, min_dt=0.1)
        assert new_t == 1.0

    def test_zero_min_dt_always_passes(self):
        """min_dt=0 means no throttling."""
        for t in [0.0, 0.001, 0.1, 100.0]:
            ok, _ = _throttle_check(last_t=t, now=t, min_dt=0.0)
            assert ok, f"should pass at t={t}"


class TestTopicDefaults:
    """Verify the default topic name strings are what Isaac Sim uses."""

    DEFAULTS = {
        "isaac_rgb_topic":    "/isaac_camera/rgb",
        "isaac_depth_topic":  "/isaac_camera/depth",
        "isaac_info_topic":   "/isaac_camera/camera_info",
        "isaac_odom_topic":   "/odom",
        "isaac_cmdvel_topic": "/cmd_vel",
    }

    def test_defaults_present(self):
        """All required parameter keys are documented."""
        src = Path(__file__).parent.parent / "mars_scout_sim_bridge" / "sim_bridge_node.py"
        text = src.read_text()
        for key in self.DEFAULTS:
            assert key in text, f"Parameter '{key}' missing from sim_bridge_node.py"

    def test_default_values_in_source(self):
        """Default values match what we document."""
        src = Path(__file__).parent.parent / "mars_scout_sim_bridge" / "sim_bridge_node.py"
        text = src.read_text()
        for key, val in self.DEFAULTS.items():
            assert val in text, f"Default value '{val}' for '{key}' missing from source"


class TestMarsScoutOutputTopics:
    """Verify all required output topics are referenced in the node source."""

    REQUIRED_PUBS = [
        "/rover/camera/image_raw",
        "/rover/camera/depth/image_raw",
        "/rover/camera/camera_info",
        "/rover/odom",
        "/rover/state",
        "/rover/cmd_vel",
    ]

    def test_all_output_topics_present(self):
        src = Path(__file__).parent.parent / "mars_scout_sim_bridge" / "sim_bridge_node.py"
        text = src.read_text()
        for topic in self.REQUIRED_PUBS:
            assert topic in text, f"Output topic '{topic}' missing from sim_bridge_node.py"
