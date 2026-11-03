"""
scene_checks.py
===============
Scene startup helpers and smoke-test rules for ARES.

No Isaac Sim imports, so these run in CI without a GPU.
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np

MARS_GRAVITY_M_S2     = 3.72
GRAVITY_TOLERANCE     = 0.15    # fraction of MARS_GRAVITY_M_S2
SETTLE_SPEED_M_S      = 0.05    # rover counts as settled below this
SMOKE_DRIVE_SPEED_M_S = 0.1
SMOKE_DRIVE_SECONDS   = 10.0
SMOKE_MIN_TRAVEL_M    = 0.2
CAMERA_MIN_MEAN       = 5.0     # 0-255: darker than this is a black frame
CAMERA_MIN_STD        = 2.0     # 0-255: flatter than this is a blank frame


def spawn_height(
    elevation: np.ndarray,
    width_m: float,
    depth_m: float,
    half_extent_m: float = 2.5,
    clearance_m: float = 1.0,
) -> float:
    """
    Spawn Z that clears the terrain under a rover centred on the scene origin.

    Uses the highest point in a square footprint, so a rover placed on a slope
    still starts in free air and only falls a short way.
    """
    ny, nx = elevation.shape
    half_rows = max(1, int(np.ceil(half_extent_m / (depth_m / (ny - 1)))))
    half_cols = max(1, int(np.ceil(half_extent_m / (width_m / (nx - 1)))))
    cy, cx = ny // 2, nx // 2
    window = elevation[
        max(0, cy - half_rows): cy + half_rows + 1,
        max(0, cx - half_cols): cx + half_cols + 1,
    ]
    return float(window.max()) + clearance_m


def gravity_from_drop(drop_m: float, steps: int, dt: float) -> float:
    """
    Gravity implied by a free-fall drop over `steps` physics steps.

    The solver integrates velocity before position, so the drop after n steps
    is g * dt^2 * n(n+1)/2 rather than the continuous g * t^2 / 2.
    """
    if steps <= 0 or dt <= 0.0:
        raise ValueError(f"steps and dt must be positive, got {steps}, {dt}")
    return 2.0 * drop_m / (dt * dt * steps * (steps + 1))


def image_stats(rgb: np.ndarray) -> dict:
    """Mean and standard deviation of an (H, W, 3 or 4) uint8 image, ignoring alpha."""
    if rgb is None or rgb.size == 0 or rgb.ndim != 3:
        return {"mean": 0.0, "std": 0.0}
    channels = rgb[:, :, :3].astype(np.float32)
    return {"mean": float(channels.mean()), "std": float(channels.std())}


def evaluate_smoke(report: dict) -> List[str]:
    """Return every failed check. An empty list means the scene passed."""
    failures = []

    configured = report.get("gravity_configured_m_s2", 0.0)
    if abs(configured - MARS_GRAVITY_M_S2) > 1e-3:
        failures.append(f"configured gravity is {configured}, expected {MARS_GRAVITY_M_S2}")

    measured = report.get("gravity_measured_m_s2", 0.0)
    if abs(measured - MARS_GRAVITY_M_S2) > GRAVITY_TOLERANCE * MARS_GRAVITY_M_S2:
        failures.append(f"rover fell at {measured:.2f} m/s2, expected about {MARS_GRAVITY_M_S2}")

    if not report.get("settled", False):
        failures.append("rover never came to rest on the terrain")

    camera = report.get("camera", {})
    if camera.get("mean", 0.0) < CAMERA_MIN_MEAN or camera.get("std", 0.0) < CAMERA_MIN_STD:
        failures.append(f"camera frame is black or blank: {camera}")

    travel = report.get("drive_travel_m", 0.0)
    if travel < SMOKE_MIN_TRAVEL_M:
        failures.append(f"rover moved {travel:.3f} m when driven, expected at least {SMOKE_MIN_TRAVEL_M}")

    return failures


def write_report(report: dict, path: str) -> None:
    """Write the smoke report atomically so a crash cannot leave a passing file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, path)
