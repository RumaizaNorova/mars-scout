"""
Ground plane estimation from a depth image using RANSAC.

Why RANSAC?
-----------
On Mars terrain the "floor" isn't flat — there are rocks, slopes, craters.
Least-squares plane fitting would pull the estimate toward outlier rocks.
RANSAC discards those outliers and fits only the dominant flat surface,
giving us a reliable z=0 reference for waypoint projection.

Reference: Fischler & Bolles, "Random Sample Consensus", CACM 1981.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Plane:
    """
    Plane in normal form:  n · x = d
    where n is a unit normal vector.
    """
    normal: np.ndarray   # (3,)  unit normal in camera frame
    offset: float        # scalar d

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from each point (N,3) to the plane."""
        return points @ self.normal - self.offset

    def project_onto(self, points: np.ndarray) -> np.ndarray:
        """Orthogonal projection of points (N,3) onto the plane."""
        dists = self.distance(points)
        return points - np.outer(dists, self.normal)

    def intersect_ray(self, origin: np.ndarray, direction: np.ndarray) -> Optional[np.ndarray]:
        """
        Ray–plane intersection.
        Returns the 3-D intersection point, or None if ray is parallel.
        """
        denom = self.normal @ direction
        if abs(denom) < 1e-9:
            return None
        t = (self.offset - self.normal @ origin) / denom
        if t < 0:
            return None   # intersection is behind the camera
        return origin + t * direction


class RANSACGroundFit:
    """
    Fit a ground plane to a point cloud derived from the depth image.

    Parameters
    ----------
    n_iters        : RANSAC iterations
    inlier_thresh  : max point-to-plane distance to count as inlier (metres)
    min_inlier_frac: minimum fraction of points that must be inliers
    subsample      : random subsample size to keep runtime bounded
    """

    def __init__(
        self,
        n_iters: int = 100,
        inlier_thresh: float = 0.05,
        min_inlier_frac: float = 0.4,
        subsample: int = 2000,
    ):
        self.n_iters = n_iters
        self.inlier_thresh = inlier_thresh
        self.min_inlier_frac = min_inlier_frac
        self.subsample = subsample

    def fit(self, points: np.ndarray) -> Tuple[Optional[Plane], np.ndarray]:
        """
        Fit a plane to points (N, 3).

        Returns
        -------
        plane   : best Plane found, or None if fitting failed
        inliers : boolean mask of shape (N,)
        """
        N = len(points)
        if N < 3:
            return None, np.zeros(N, dtype=bool)

        # Subsample for speed
        if N > self.subsample:
            idx = np.random.choice(N, self.subsample, replace=False)
            pts = points[idx]
        else:
            pts = points
            idx = np.arange(N)

        best_plane: Optional[Plane] = None
        best_inlier_count = 0
        best_inliers = np.zeros(len(pts), dtype=bool)

        for _ in range(self.n_iters):
            # 1. Sample 3 random non-collinear points
            sample_idx = np.random.choice(len(pts), 3, replace=False)
            p0, p1, p2 = pts[sample_idx]

            normal = np.cross(p1 - p0, p2 - p0)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-9:
                continue   # degenerate / collinear sample
            normal /= norm_len

            # Ground plane should have its normal pointing roughly upward
            # In camera frame that means |normal_y| is large (y points down)
            if abs(normal[1]) < 0.5:
                continue

            offset = float(normal @ p0)
            plane = Plane(normal=normal, offset=offset)

            # 2. Count inliers
            dists = np.abs(plane.distance(pts))
            inliers = dists < self.inlier_thresh
            count = int(inliers.sum())

            if count > best_inlier_count:
                best_inlier_count = count
                best_plane = plane
                best_inliers = inliers

        if best_plane is None or best_inlier_count < self.min_inlier_frac * len(pts):
            return None, np.zeros(N, dtype=bool)

        # 3. Refit using all inliers (least-squares polish)
        inlier_pts = pts[best_inliers]
        best_plane = self._refit(inlier_pts)

        # Map inlier mask back to original point array
        full_inliers = np.zeros(N, dtype=bool)
        full_inliers[idx[best_inliers]] = True

        return best_plane, full_inliers

    @staticmethod
    def _refit(points: np.ndarray) -> Plane:
        """Least-squares plane through a set of inlier points (SVD method)."""
        centroid = points.mean(axis=0)
        _, _, Vt = np.linalg.svd(points - centroid)
        normal = Vt[-1]  # smallest singular value → normal direction
        if normal[1] > 0:   # ensure normal points upward in camera frame
            normal = -normal
        offset = float(normal @ centroid)
        return Plane(normal=normal, offset=offset)
