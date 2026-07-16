"""
Pinhole camera model with distortion support.

Implements the standard back-projection pipeline:
  (u, v, depth) → 3-D point in camera optical frame

Reference: Hartley & Zisserman, "Multiple View Geometry", Ch. 6
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""
    fx: float   # focal length in pixels (x)
    fy: float   # focal length in pixels (y)
    cx: float   # principal point x
    cy: float   # principal point y
    width: int
    height: int

    # Radial + tangential distortion coefficients (k1, k2, p1, p2, k3)
    distortion: np.ndarray = None

    def __post_init__(self):
        if self.distortion is None:
            self.distortion = np.zeros(5)

    @property
    def K(self) -> np.ndarray:
        """3×3 intrinsic matrix."""
        return np.array([
            [self.fx,  0.0,     self.cx],
            [0.0,      self.fy, self.cy],
            [0.0,      0.0,     1.0   ],
        ])

    @property
    def K_inv(self) -> np.ndarray:
        return np.linalg.inv(self.K)

    # ── Isaac Sim defaults (can be overridden via ROS params) ─────────────────
    @classmethod
    def isaac_sim_default(cls, width: int = 1280, height: int = 720) -> "CameraIntrinsics":
        """
        Default intrinsics for Isaac Sim's synthetic camera.
        Horizontal FOV = 90°.
        """
        fx = fy = width / (2.0 * np.tan(np.radians(45.0)))
        return cls(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height)


class CameraModel:
    """
    Stateless back-projection utilities.

    All operations are vectorised over N points using NumPy so they're
    fast enough to run at sensor rate without Cython or C extensions.
    """

    def __init__(self, intrinsics: CameraIntrinsics):
        self.K = intrinsics
        self._K_inv = intrinsics.K_inv

    # ── Core back-projection ──────────────────────────────────────────────────

    def backproject(
        self,
        u: np.ndarray,
        v: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        """
        Back-project pixel coordinates + depth into 3-D camera frame.

        Parameters
        ----------
        u, v  : pixel coordinates (float or N-array)
        depth : metric depth in metres (same shape as u, v)

        Returns
        -------
        points : (N, 3) array in camera optical frame  [x_right, y_down, z_forward]
        """
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        depth = np.asarray(depth, dtype=np.float64)

        x_c = (u - self.K.cx) * depth / self.K.fx
        y_c = (v - self.K.cy) * depth / self.K.fy
        z_c = depth

        return np.stack([x_c, y_c, z_c], axis=-1)

    def backproject_bbox(
        self,
        bbox_cx_norm: float,
        bbox_cy_norm: float,
        depth_image: np.ndarray,
        patch_radius: int = 5,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Back-project a normalised bounding-box centre using the median depth
        of a small patch around the centre (robust to depth holes).

        Returns
        -------
        point_3d   : (3,) in camera frame
        depth_used : scalar depth value used
        depth_std  : std-dev of the depth patch (proxy for depth confidence)
        """
        u = bbox_cx_norm * self.K.width
        v = bbox_cy_norm * self.K.height

        # Sample a patch — clamp to image bounds
        u_lo = max(0, int(u) - patch_radius)
        u_hi = min(self.K.width,  int(u) + patch_radius + 1)
        v_lo = max(0, int(v) - patch_radius)
        v_hi = min(self.K.height, int(v) + patch_radius + 1)

        patch = depth_image[v_lo:v_hi, u_lo:u_hi]
        valid = patch[(patch > 0.05) & (patch < 50.0)]  # filter out noise / sky

        if valid.size == 0:
            raise ValueError("No valid depth in target patch — likely occluded or out of range.")

        d_median = float(np.median(valid))
        d_std    = float(np.std(valid))

        point = self.backproject(u, v, d_median)
        return point, d_median, d_std

    def unproject_ray(self, u: float, v: float) -> np.ndarray:
        """
        Return the unit ray direction for pixel (u, v) in camera frame.
        Useful for ray–plane intersection when depth is unavailable.
        """
        ray = self._K_inv @ np.array([u, v, 1.0])
        return ray / np.linalg.norm(ray)
