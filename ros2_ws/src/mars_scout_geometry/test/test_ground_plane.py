"""
Unit tests for RANSAC ground plane fitting.
Run with: pytest ros2_ws/src/mars_scout_geometry/test/
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mars_scout_geometry.ground_plane import RANSACGroundFit, Plane


def flat_ground(n=500, noise=0.01, n_outliers=50, rng_seed=0) -> np.ndarray:
    """Generate a flat y=0 ground plane with some rock outliers."""
    rng = np.random.default_rng(rng_seed)
    xs = rng.uniform(-5, 5, n)
    zs = rng.uniform(0.5, 8, n)
    ys = rng.normal(0, noise, n)           # ground points near y=0

    # Outliers: rocks sticking up
    ox = rng.uniform(-5, 5, n_outliers)
    oz = rng.uniform(0.5, 8, n_outliers)
    oy = rng.uniform(0.2, 1.5, n_outliers)  # elevated

    pts = np.stack([
        np.concatenate([xs, ox]),
        np.concatenate([ys, oy]),
        np.concatenate([zs, oz]),
    ], axis=1)
    return pts


def test_ransac_finds_flat_ground():
    """RANSAC should recover y≈0 ground plane despite 10% outlier rocks."""
    pts = flat_ground(n=500, n_outliers=50)
    fitter = RANSACGroundFit(n_iters=200, inlier_thresh=0.05)
    plane, inliers = fitter.fit(pts)

    assert plane is not None, "RANSAC failed to find a plane"
    # Normal should be roughly [0, ±1, 0]
    assert abs(plane.normal[1]) > 0.9, f"Normal not pointing up: {plane.normal}"
    # Offset ≈ 0
    assert abs(plane.offset) < 0.1, f"Plane offset too large: {plane.offset}"
    # Most inliers should be the ground points
    assert inliers.sum() >= 450


def test_plane_distance():
    """Points on the plane should have zero distance."""
    plane = Plane(normal=np.array([0.0, 1.0, 0.0]), offset=0.0)
    pts_on = np.array([[1.0, 0.0, 2.0], [-3.0, 0.0, 5.0]])
    np.testing.assert_allclose(plane.distance(pts_on), 0.0, atol=1e-9)


def test_plane_project_onto():
    """Projected points should lie on the plane."""
    plane = Plane(normal=np.array([0.0, 1.0, 0.0]), offset=0.5)
    pts = np.array([[1.0, 2.0, 3.0], [-1.0, -1.0, 0.5]])
    projected = plane.project_onto(pts)
    np.testing.assert_allclose(plane.distance(projected), 0.0, atol=1e-9)


def test_ray_plane_intersection():
    """Ray fired straight down should hit y=0 plane directly below origin."""
    plane = Plane(normal=np.array([0.0, 1.0, 0.0]), offset=0.0)
    origin = np.array([3.0, 5.0, 2.0])
    direction = np.array([0.0, -1.0, 0.0])
    pt = plane.intersect_ray(origin, direction)
    assert pt is not None
    np.testing.assert_allclose(pt, [3.0, 0.0, 2.0], atol=1e-9)


def test_ray_parallel_to_plane_returns_none():
    """Ray parallel to the plane should return None."""
    plane = Plane(normal=np.array([0.0, 1.0, 0.0]), offset=0.0)
    pt = plane.intersect_ray(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert pt is None


def test_ransac_too_few_points():
    """RANSAC should gracefully return None for < 3 points."""
    fitter = RANSACGroundFit()
    plane, inliers = fitter.fit(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]))
    assert plane is None
