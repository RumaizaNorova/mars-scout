"""
Unit tests for CameraModel back-projection.
Run with: pytest ros2_ws/src/mars_scout_geometry/test/
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mars_scout_geometry.camera_model import CameraModel, CameraIntrinsics


@pytest.fixture
def cam():
    intr = CameraIntrinsics.isaac_sim_default(width=1280, height=720)
    return CameraModel(intr)


def test_principal_point_projects_forward(cam):
    """A point at the principal point (cx, cy) should project straight forward (+z)."""
    cx, cy = cam.K.cx, cam.K.cy
    pt = cam.backproject(cx, cy, 5.0)
    assert pytest.approx(pt[0], abs=1e-9) == 0.0   # x ≈ 0
    assert pytest.approx(pt[1], abs=1e-9) == 0.0   # y ≈ 0
    assert pytest.approx(pt[2], abs=1e-9) == 5.0   # z = depth


def test_depth_scales_linearly(cam):
    """Doubling depth should double the 3-D distance."""
    u, v = cam.K.cx + 100, cam.K.cy + 50
    pt1 = cam.backproject(u, v, 3.0)
    pt2 = cam.backproject(u, v, 6.0)
    np.testing.assert_allclose(pt2, pt1 * 2.0, rtol=1e-6)


def test_backproject_batch(cam):
    """Batch back-projection should match per-point back-projection."""
    rng = np.random.default_rng(42)
    us = rng.uniform(0, cam.K.width,  50)
    vs = rng.uniform(0, cam.K.height, 50)
    ds = rng.uniform(0.5, 10.0,       50)

    batch = cam.backproject(us, vs, ds)
    for i in range(50):
        single = cam.backproject(us[i], vs[i], ds[i])
        np.testing.assert_allclose(batch[i], single, rtol=1e-6)


def test_backproject_bbox_returns_valid_point(cam):
    """backproject_bbox should handle a normal depth image without raising."""
    depth = np.ones((720, 1280), dtype=np.float32) * 5.0
    # Add small noise
    depth += np.random.default_rng(0).normal(0, 0.01, depth.shape).astype(np.float32)
    pt, d, std = cam.backproject_bbox(0.5, 0.5, depth, patch_radius=5)
    assert np.isfinite(pt).all()
    assert 4.9 < d < 5.1
    assert std >= 0


def test_backproject_bbox_no_valid_depth(cam):
    """backproject_bbox should raise when depth is all zeros."""
    depth = np.zeros((720, 1280), dtype=np.float32)
    with pytest.raises(ValueError, match="No valid depth"):
        cam.backproject_bbox(0.5, 0.5, depth)


def test_unproject_ray_unit_length(cam):
    """Unprojected rays should be unit vectors."""
    for u, v in [(0, 0), (640, 360), (1279, 719)]:
        ray = cam.unproject_ray(u, v)
        assert pytest.approx(np.linalg.norm(ray), abs=1e-9) == 1.0
