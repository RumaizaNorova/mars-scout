import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mars_scout_control.controller import PurePursuitController, ControllerConfig


@pytest.fixture
def ctrl():
    return PurePursuitController()


def test_goal_directly_ahead(ctrl):
    """Goal straight ahead → small angular, positive linear."""
    out = ctrl.compute(0, 0, 0.0, 5.0, 0.0)
    assert out.linear  > 0.0
    assert abs(out.angular) < 0.05


def test_goal_directly_behind(ctrl):
    """Goal behind → large heading error → no forward motion, turn in place."""
    out = ctrl.compute(0, 0, 0.0, -5.0, 0.0)
    assert out.linear == pytest.approx(0.0, abs=0.05)
    assert abs(out.angular) > 0.5


def test_goal_to_left(ctrl):
    """Goal to the left → turn left (positive angular in ROS convention)."""
    out = ctrl.compute(0, 0, 0.0, 0.0, 5.0)  # goal at (0, 5) = 90° left
    assert out.angular > 0.0


def test_goal_to_right(ctrl):
    """Goal to the right → turn right (negative angular)."""
    out = ctrl.compute(0, 0, 0.0, 0.0, -5.0)
    assert out.angular < 0.0


def test_speed_scales_with_distance(ctrl):
    """Closer goal → lower speed."""
    out_far  = ctrl.compute(0, 0, 0.0, 10.0, 0.0)
    out_near = ctrl.compute(0, 0, 0.0,  0.5, 0.0)
    assert out_near.linear < out_far.linear


def test_output_within_limits(ctrl):
    """All outputs should respect hardware limits."""
    cfg = ctrl.cfg
    for gx, gy in [(5, 3), (-2, 4), (0.1, 0.1), (10, -10)]:
        out = ctrl.compute(0, 0, 0.0, gx, gy)
        assert abs(out.linear)  <= cfg.max_linear  + 1e-6
        assert abs(out.angular) <= cfg.max_angular + 1e-6


def test_aligned_flag(ctrl):
    """aligned=True when heading error < threshold."""
    out = ctrl.compute(0, 0, 0.0, 5.0, 0.0)   # straight ahead
    assert out.aligned is True
    out2 = ctrl.compute(0, 0, 0.0, 0.0, 5.0)  # 90° off
    assert out2.aligned is False


def test_distance_computed_correctly(ctrl):
    out = ctrl.compute(0, 0, 0.0, 3.0, 4.0)
    assert out.distance_m == pytest.approx(5.0, abs=1e-6)
