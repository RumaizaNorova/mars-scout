"""
Synthetic depth map generation from Mars terrain RGB images.

Why this works
--------------
Real Mars surface images follow predictable depth cues:
  1. Perspective ground plane — rows near the bottom of the image
     correspond to terrain close to the rover; rows near the horizon
     are farther away.  We model this as a linear depth ramp.
  2. Rock / obstacle elevation — rocks have high local edge density
     (Sobel gradient magnitude) and high local intensity variance.
     We detect these regions and add a positive depth offset
     (they stick up above the ground plane toward the camera).
  3. Sky mask — the top fraction of the image has no valid depth
     (infinite / max range).  We detect sky by low saturation + high
     brightness and assign NaN / max_depth there.

The resulting depth map is physically plausible enough that
mars_scout_geometry's RANSAC ground-plane fitter and back-projector
behave exactly as they will with Isaac Sim's real depth sensor.
"""

from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DepthSynthesisConfig:
    min_depth: float = 0.4      # metres — rover footprint (~40 cm)
    max_depth: float = 15.0     # metres — far-field clip
    horizon_frac: float = 0.45  # fraction of image height where horizon sits
    sky_frac: float = 0.35      # top fraction treated as sky (no depth)
    rock_max_height: float = 0.8  # max rock elevation above ground (metres)
    rock_blur_px: int = 7       # smoothing kernel for rock mask
    noise_sigma: float = 0.03   # sensor noise std-dev (metres)


class DepthSynthesiser:
    """
    Converts a Mars terrain RGB image (H×W×3, uint8) to a
    synthetic float32 depth image (H×W, metres).
    """

    def __init__(self, cfg: DepthSynthesisConfig | None = None):
        self.cfg = cfg or DepthSynthesisConfig()

    def synthesise(self, rgb: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        rgb : (H, W, 3) uint8  BGR or RGB — colour order doesn't matter here

        Returns
        -------
        depth : (H, W) float32  metres, NaN where no valid depth
        """
        H, W = rgb.shape[:2]
        cfg = self.cfg

        # ── 1. Perspective ground-plane ramp ──────────────────────────────────
        # row 0 (top)    → max_depth
        # horizon row    → max_depth   (flat ramp above horizon)
        # bottom row     → min_depth
        depth = self._ground_ramp(H, W)

        # ── 2. Rock / obstacle elevation ──────────────────────────────────────
        rock_mask, rock_height = self._detect_rocks(rgb)
        depth = depth - rock_height   # rocks are closer (smaller z)

        # ── 3. Sky mask → NaN ─────────────────────────────────────────────────
        sky_mask = self._sky_mask(rgb, H, W)
        depth[sky_mask] = np.nan

        # ── 4. Gaussian sensor noise ──────────────────────────────────────────
        noise = np.random.normal(0.0, cfg.noise_sigma, (H, W)).astype(np.float32)
        depth = depth + noise

        # ── 5. Clip to valid range ────────────────────────────────────────────
        depth = np.clip(depth, cfg.min_depth, cfg.max_depth)
        depth[sky_mask] = np.nan

        return depth.astype(np.float32)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ground_ramp(self, H: int, W: int) -> np.ndarray:
        """Linear depth ramp: bottom row = min_depth, horizon row = max_depth."""
        cfg = self.cfg
        horizon_row = int(H * cfg.horizon_frac)
        depth = np.full((H, W), cfg.max_depth, dtype=np.float32)

        # Below horizon: linear ramp from max_depth → min_depth
        for row in range(horizon_row, H):
            frac = (row - horizon_row) / max(H - 1 - horizon_row, 1)
            depth[row, :] = cfg.max_depth - frac * (cfg.max_depth - cfg.min_depth)

        return depth

    def _detect_rocks(self, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect rock regions using Sobel edge density + local intensity variance.

        Returns
        -------
        rock_mask   : (H, W) bool
        rock_height : (H, W) float32  elevation above ground (metres)
        """
        cfg = self.cfg
        H, W = rgb.shape[:2]

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Edge density via Sobel
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.sqrt(gx**2 + gy**2)
        edge_mag = cv2.GaussianBlur(edge_mag, (cfg.rock_blur_px, cfg.rock_blur_px), 0)

        # Local variance (rocks have varied texture)
        mean  = cv2.GaussianBlur(gray, (15, 15), 0)
        mean2 = cv2.GaussianBlur(gray**2, (15, 15), 0)
        local_var = np.clip(mean2 - mean**2, 0, None)

        # Combine: high edge density AND high local variance → rock
        rock_score = (edge_mag / (edge_mag.max() + 1e-6)) * 0.6 + \
                     (local_var  / (local_var.max()  + 1e-6)) * 0.4

        # Threshold — top 25% of score are rocks
        thresh = np.percentile(rock_score, 75)
        rock_mask = rock_score > thresh

        # Rock height proportional to score (taller rocks = stronger edges)
        rock_height = (rock_score * cfg.rock_max_height).astype(np.float32)
        rock_height[~rock_mask] = 0.0

        # Smooth the height field so depth edges aren't pixel-sharp
        rock_height = cv2.GaussianBlur(rock_height, (cfg.rock_blur_px * 2 + 1, cfg.rock_blur_px * 2 + 1), 0)

        return rock_mask, rock_height

    def _sky_mask(self, rgb: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        Detect sky pixels: top fraction of image + high brightness / low saturation.
        Returns bool mask (True = sky / no valid depth).
        """
        cfg = self.cfg
        mask = np.zeros((H, W), dtype=bool)

        # Always mask the top sky_frac rows
        sky_row = int(H * cfg.sky_frac)
        mask[:sky_row, :] = True

        # Additionally detect bright, low-saturation pixels (sky on Mars is pinkish)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation = hsv[:, :, 1] / 255.0
        value      = hsv[:, :, 2] / 255.0
        sky_pixels = (saturation < 0.25) & (value > 0.65)
        mask |= sky_pixels

        return mask
