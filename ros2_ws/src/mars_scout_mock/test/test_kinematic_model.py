import math
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mars_scout_mock.kinematic_model import (
    DifferentialDriveModel, DifferentialDriveConfig, RoverPose, _wrap_angle,
)


def test_stationary_no_movement():
    m = DifferentialDriveModel()
    m.set_velocity(0.0, 0.0)
    for _ in range(100):
        m.step(0.02)
    assert m.pose.x == pytest.approx(0.0, abs=1e-9)
    assert m.pose.y == pytest.approx(0.0, abs=1e-9)


def test_straight_line_forward():
    """
    Driving at max linear speed for 1 s should move
    max_linear_speed × slip_factor metres in x.
    """
    m = DifferentialDriveModel()
    v = m.cfg.max_linear_speed        # use the actual limit, don't exceed it
    m.set_velocity(v, 0.0)
    for _ in range(50):               # 50 × 0.02 s = 1 s
        m.step(0.02)
    expected = v * m.cfg.slip_factor
    assert m.pose.x == pytest.approx(expected, rel=0.01)
    assert m.pose.y == pytest.approx(0.0, abs=0.01)


def test_pure_rotation():
    """
    Rotating at max angular speed for a full 2π/ω seconds should
    complete one revolution (accounting for slip).
    """
    m = DifferentialDriveModel()
    omega = m.cfg.max_angular_speed           # stay within the limit
    effective_omega = omega * m.cfg.slip_factor
    period = 2 * math.pi / effective_omega    # exact time for one revolution with slip
    dt = 0.005
    m.set_velocity(0.0, omega)
    steps = int(period / dt)
    for _ in range(steps):
        m.step(dt)
    assert abs(_wrap_angle(m.pose.yaw)) < 0.05


def test_circular_arc_returns_near_origin():
    """
    Driving in a perfect circle (slip_factor=1) for one full period
    should return the rover near its starting position.
    """
    cfg = DifferentialDriveConfig(slip_factor=1.0,
                                  max_linear_speed=1.0,
                                  max_angular_speed=2.0)
    m = DifferentialDriveModel(cfg=cfg)
    v = omega = 0.5                        # radius = v/ω = 1 m
    period = 2 * math.pi / omega
    dt = 0.005
    m.set_velocity(v, omega)
    for _ in range(int(period / dt)):
        m.step(dt)
    assert m.pose.x == pytest.approx(0.0, abs=0.05)
    assert m.pose.y == pytest.approx(0.0, abs=0.05)


def test_velocity_clamping():
    """set_velocity should clamp to hardware limits."""
    m = DifferentialDriveModel()
    m.set_velocity(999.0, 999.0)
    assert m._vel.linear  <= m.cfg.max_linear_speed
    assert m._vel.angular <= m.cfg.max_angular_speed


def test_odom_covariance_grows_with_distance():
    """Positional covariance should increase as the rover travels further."""
    m = DifferentialDriveModel()
    cov_start = m.odom_covariance[0, 0]
    m.set_velocity(0.5, 0.0)
    for _ in range(500):
        m.step(0.02)
    cov_end = m.odom_covariance[0, 0]
    assert cov_end > cov_start


def test_quaternion_norm():
    """Pose quaternion should always be unit length."""
    m = DifferentialDriveModel()
    m.set_velocity(0.3, 0.4)
    for _ in range(100):
        m.step(0.02)
        q = m.pose.quaternion
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-6)


def test_wrap_angle():
    assert _wrap_angle(0.0)        == pytest.approx(0.0)
    assert _wrap_angle(math.pi)    == pytest.approx(-math.pi, abs=1e-9)
    assert _wrap_angle(3 * math.pi) == pytest.approx(-math.pi, abs=1e-9)
    assert _wrap_angle(-math.pi)   == pytest.approx(-math.pi, abs=1e-9)
