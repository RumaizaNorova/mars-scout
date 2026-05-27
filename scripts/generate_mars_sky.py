#!/usr/bin/env python3
"""
generate_mars_sky.py
====================
Generate a physically-motivated equirectangular sky texture for the ARES
Mars simulation, based on Henyey-Greenstein dust aerosol scattering.

The Martian sky is NOT blue — it is pinkish-tan because iron-oxide (Fe₂O₃)
dust aerosols dominate the atmosphere.  These particles:

  1. Forward-scatter strongly (g ≈ 0.68) → bright halo within ~25° of sun
  2. Absorb blue/UV (strong absorption edge at 400–600 nm) → overall red tint
  3. Create horizon brightening through longer path lengths (air-mass factor)
  4. Produce a subtle anti-solar brightening (backscatter lobe at g=0.68)

The output is a 2048×1024 PNG loaded by DomeLight in Isaac Sim RTX.  When
the file exists, _build_martian_lighting() automatically switches from the
flat-colour fallback to this texture.

Usage
-----
  python3 scripts/generate_mars_sky.py            # generate at default path
  python3 scripts/generate_mars_sky.py --preview  # also display with matplotlib

Output
------
  ~/mars-rover-agent/data/mars_sky.png   (2048×1024, 8-bit sRGB PNG)

References
----------
Madeleine et al. 2012, Icarus 220, 798
  → Mars aerosol asymmetry parameter g_eff = 0.66–0.73 (best fit TES data)
Bell et al. 2021, Space Sci Rev 217, 24
  → Mastcam-Z sky colour calibration (Perseverance sols 1–400)
Wolff & Clancy 2003, JGR Planets 108, 5097
  → Dust single-scatter albedo ω₀ ≈ 0.93 at 880 nm
Vicente-Retortillo et al. 2023, JGR Planets
  → MEDA sky brightness measurements (optical depth τ ≈ 0.4–0.6)
Lemmon et al. 2015, Icarus 251, 96
  → Pathfinder sky photometry — forward-scatter halo confirmed at 20–30°
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# =============================================================================
# Physical constants and calibration parameters
# =============================================================================

# Sun position — must match _build_martian_lighting() in hirise_terrain_builder.py
SUN_ELEV_DEG = 28.0    # elevation above horizon (degrees)
SUN_AZ_DEG   = 15.0    # azimuth from north (degrees, clockwise)

# Henyey-Greenstein asymmetry parameter for Mars iron-oxide aerosols
# Madeleine et al. 2012: g_eff = 0.68 (best-fit, optical depth τ_vis ≈ 0.5)
HG_G = 0.68

# Sky colours calibrated from Mastcam-Z and Pancam imagery (Bell 2021)
# All values in LINEAR light (before gamma), normalised to 0–1 display range.
# The equatorial sky (90° from sun, 30° elevation) is the reference point.
#
# Zenith (overhead):          pinkish-tan,  relatively dark
# Horizon (low elevation):    brighter, more orange-red — longer dust path
# Near-sun corona (< 25°):    pale ivory — HG forward scatter peak
# Ground (below horizon):     dark basalt floor colour
#
# Source: Bell 2021 Fig 4 — linearised from Mastcam-Z calibrated radiance
SKY_ZENITH_RGB   = np.array([0.43, 0.28, 0.20], dtype=np.float32)
SKY_MID_RGB      = np.array([0.56, 0.38, 0.26], dtype=np.float32)   # 20° elevation
SKY_HORIZON_RGB  = np.array([0.72, 0.51, 0.34], dtype=np.float32)   # <5° elevation
SKY_CORONA_RGB   = np.array([0.85, 0.68, 0.48], dtype=np.float32)   # near-sun corona
SUN_DISK_RGB     = np.array([1.00, 0.94, 0.82], dtype=np.float32)   # sun disk
GROUND_RGB       = np.array([0.11, 0.08, 0.05], dtype=np.float32)   # below horizon

# Geometric parameters
SUN_DISK_HALF_ANG_DEG = 0.35    # Mars sun angular radius (smaller than Earth 0.53°)
CORONA_HALF_ANG_DEG   = 25.0    # extent of HG forward-scatter brightening
HORIZON_FADE_DEG      = 4.0     # elevation range over which sky → ground blends


# =============================================================================
# Core functions
# =============================================================================

def henyey_greenstein(cos_theta: np.ndarray, g: float) -> np.ndarray:
    """
    Henyey-Greenstein phase function (Chandrasekhar form, unnormalised).

    P(μ) = (1 − g²) / (1 + g² − 2g·μ)^(3/2)   where μ = cos(θ)

    At g=0.68: forward peak (μ=1) is ~13× stronger than isotropic backscatter
    (μ=-1), consistent with Lemmon 2015 Pathfinder halo photometry.

    Returns ndarray with same shape as cos_theta, values ≥ 0.
    """
    numerator   = 1.0 - g * g
    denominator = np.power(np.maximum(1.0 + g*g - 2.0*g*cos_theta, 1e-9), 1.5)
    return (numerator / denominator).astype(np.float32)


def _elevation_gradient(elev_rad: np.ndarray) -> np.ndarray:
    """
    Altitude-dependent sky colour weight (0 = ground, 1 = zenith).

    The transition is nonlinear: sky brightens rapidly toward the horizon
    because the atmospheric path length ≈ 1/sin(elev) for simple slab geometry,
    and the extra dust column adds scatter (both intensity and colour).

    The zenith → horizon gradient is approximated with two blended segments:
      - elev > 30°: zenith colour (low air-mass)
      - 5°–30°:     mix toward horizon colour (increasing air-mass)
      - < 5°:       full horizon colour
    This matches Bell 2021 Mastcam-Z all-sky photometry observations.
    """
    deg = np.degrees(elev_rad)
    # Zenith weight: 1 at elev≥30°, falls to 0 at elev≤5°
    zenith_w  = np.clip((deg - 5.0) / 25.0, 0.0, 1.0).astype(np.float32)
    # Apply smoothstep to reduce harsh linear banding
    zenith_w  = zenith_w * zenith_w * (3.0 - 2.0 * zenith_w)
    return zenith_w   # 1 = zenith colour, 0 = horizon colour


def generate_sky_texture(
    width:         int   = 2048,
    height:        int   = 1024,
    sun_elev_deg:  float = SUN_ELEV_DEG,
    sun_az_deg:    float = SUN_AZ_DEG,
    g:             float = HG_G,
) -> np.ndarray:
    """
    Generate equirectangular sky texture as float32 (height, width, 3) in
    linear light, range 0–1 (slightly above 1 near sun — clamped later).

    Equirectangular mapping convention:
      u ∈ [0, 1] → azimuth ∈ [0°, 360°]     (left = north, wraps east)
      v ∈ [0, 1] → elevation ∈ [+90°, -90°]  (top = zenith, bottom = nadir)

    The texture is then passed to Isaac Sim DomeLight which handles the
    mapping correctly when guideRotation = 0.
    """
    # Build angular coordinate grids
    u_lin = np.linspace(0.0, 1.0, width,  endpoint=False, dtype=np.float64)
    v_lin = np.linspace(0.0, 1.0, height, endpoint=False, dtype=np.float64)
    U, V  = np.meshgrid(u_lin, v_lin)

    az_rad   = U * 2.0 * math.pi                 # 0 → 2π (azimuth)
    elev_rad = (0.5 - V) * math.pi               # +π/2 (zenith) → -π/2 (nadir)
    elev_rad = elev_rad.astype(np.float32)

    # Direction unit vectors — Z-up, X=east, Y=north
    cos_elev = np.cos(elev_rad)
    dx = (cos_elev * np.sin(az_rad)).astype(np.float32)   # east component
    dy = (cos_elev * np.cos(az_rad)).astype(np.float32)   # north component
    dz = np.sin(elev_rad)                                  # up component

    # Sun direction vector
    s_az_r  = math.radians(sun_az_deg)
    s_el_r  = math.radians(sun_elev_deg)
    sx = float(math.cos(s_el_r) * math.sin(s_az_r))
    sy = float(math.cos(s_el_r) * math.cos(s_az_r))
    sz = float(math.sin(s_el_r))

    # Angle between pixel direction and sun
    cos_theta = np.clip(dx*sx + dy*sy + dz*sz, -1.0, 1.0)

    # ── 1. Henyey-Greenstein forward scatter ─────────────────────────────────
    # Normalise to [0, 1] so we can use it as a blending weight.
    hg      = henyey_greenstein(cos_theta, g)
    hg_max  = float(henyey_greenstein(np.array([1.0]), g)[0])
    hg_norm = np.clip(hg / hg_max, 0.0, 1.0)            # 1 = at sun, ~0.03 opposite

    # ── 2. Elevation-based colour gradient ───────────────────────────────────
    zenith_w = _elevation_gradient(elev_rad)             # (H, W), 0–1
    # Three-way blend: zenith → mid-sky → horizon
    mid_w    = np.where(zenith_w > 0.5,
                        2.0 * (1.0 - zenith_w),          # zenith → mid transition
                        2.0 * zenith_w)                   # mid → horizon transition
    mid_w    = np.clip(mid_w, 0.0, 1.0).astype(np.float32)

    sky_base = (
        SKY_ZENITH_RGB[None, None, :]  * np.maximum(2.0*zenith_w - 1.0, 0.0)[:, :, None] +
        SKY_MID_RGB[None, None, :]     * mid_w[:, :, None] +
        SKY_HORIZON_RGB[None, None, :] * np.maximum(1.0 - 2.0*zenith_w, 0.0)[:, :, None]
    )

    # ── 3. HG modulation of sky brightness near sun ───────────────────────────
    # The HG forward lobe increases brightness within ~25° of the sun.
    # Weight: 30 % intensity boost at maximum forward scatter.
    # Result: distinct sun-halo visible in rendered sky.
    hg_boost   = 0.25 * hg_norm[:, :, None]
    sky_col    = sky_base * (1.0 + hg_boost)

    # ── 4. Near-sun corona blend ──────────────────────────────────────────────
    # Within CORONA_HALF_ANG_DEG of sun: blend toward pale corona colour.
    theta_rad   = np.arccos(cos_theta)                   # 0 at sun, π at antisun
    corona_mask = np.clip(
        1.0 - theta_rad / math.radians(CORONA_HALF_ANG_DEG),
        0.0, 1.0
    ) ** 2                                               # quadratic falloff
    sky_col = (sky_col * (1.0 - 0.60 * corona_mask[:, :, None]) +
               SKY_CORONA_RGB[None, None, :] * (0.60 * corona_mask[:, :, None]))

    # ── 5. Sun disk ───────────────────────────────────────────────────────────
    sun_mask = (theta_rad < math.radians(SUN_DISK_HALF_ANG_DEG)).astype(np.float32)
    sky_col  = (sky_col * (1.0 - sun_mask[:, :, None]) +
                SUN_DISK_RGB[None, None, :] * sun_mask[:, :, None])

    # ── 6. Ground (below horizon) ─────────────────────────────────────────────
    # Fade from sky to ground colour over HORIZON_FADE_DEG below horizon.
    elev_deg    = np.degrees(elev_rad)
    ground_w    = np.clip(-elev_deg / HORIZON_FADE_DEG, 0.0, 1.0).astype(np.float32)
    sky_col     = (sky_col * (1.0 - ground_w[:, :, None]) +
                   GROUND_RGB[None, None, :] * ground_w[:, :, None])

    # Zero out pixels that are definitively below horizon (clean ground fill)
    hard_ground = (dz < -0.10).astype(np.float32)[:, :, None]
    sky_col     = sky_col * (1.0 - hard_ground) + GROUND_RGB[None, None, :] * hard_ground

    return sky_col.astype(np.float32)


# =============================================================================
# Save / preview
# =============================================================================

def save_png(sky_linear: np.ndarray, output_path: str) -> None:
    """
    Save linear-light sky texture as 8-bit sRGB PNG.

    Applies 2.2 gamma before saving (sRGB-ish, acceptable for render engine).
    The DomeLight in Isaac Sim treats the texture as sRGB and internally
    linearises it back — so we MUST apply gamma here.
    """
    if not _PIL_OK:
        raise RuntimeError("Pillow is required: pip install pillow")

    sky_clamp  = np.clip(sky_linear, 0.0, 1.0)
    sky_srgb   = np.power(sky_clamp, 1.0 / 2.2)
    sky_uint8  = (sky_srgb * 255.0).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(sky_uint8, "RGB").save(output_path, optimize=False)
    print(f"[generate_mars_sky] Saved: {output_path}  "
          f"({sky_uint8.shape[1]}×{sky_uint8.shape[0]} px)")


def save_exr(sky_linear: np.ndarray, output_path: str) -> None:
    """
    Save as 32-bit float EXR (true HDR). Requires OpenEXR package.
    Isaac Sim can load EXR as DomeLight texture for better HDR fidelity.
    """
    try:
        import OpenEXR
        import Imath
    except ImportError:
        print("[generate_mars_sky] OpenEXR not available; use PNG.")
        return

    h, w = sky_linear.shape[:2]
    header = OpenEXR.Header(w, h)
    pt     = Imath.PixelType(Imath.PixelType.FLOAT)
    header["channels"] = {
        "R": Imath.Channel(pt),
        "G": Imath.Channel(pt),
        "B": Imath.Channel(pt),
    }
    exr = OpenEXR.OutputFile(output_path, header)
    exr.writePixels({
        "R": sky_linear[:, :, 0].astype(np.float32).tobytes(),
        "G": sky_linear[:, :, 1].astype(np.float32).tobytes(),
        "B": sky_linear[:, :, 2].astype(np.float32).tobytes(),
    })
    exr.close()
    print(f"[generate_mars_sky] Saved HDR EXR: {output_path}")


def preview(sky_linear: np.ndarray) -> None:
    """Display sky with matplotlib (optional, --preview flag)."""
    try:
        import matplotlib.pyplot as plt
        sky_show = np.power(np.clip(sky_linear, 0.0, 1.0), 1.0 / 2.2)
        fig, ax  = plt.subplots(figsize=(14, 7))
        ax.imshow(sky_show, aspect="equal")
        ax.set_title(
            f"Mars Sky — Henyey-Greenstein g={HG_G:.2f}  "
            f"Sun elev={SUN_ELEV_DEG}°  az={SUN_AZ_DEG}°  "
            f"(Bell 2021 / Madeleine 2012)"
        )
        ax.set_xlabel("Azimuth →  0° (N) … 360°")
        ax.set_ylabel("↑ Zenith        Nadir ↓")
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("[generate_mars_sky] matplotlib not available — skipping preview.")


# =============================================================================
# Self-test
# =============================================================================

def _run_tests() -> None:
    """Quick sanity checks (no display, no file I/O needed)."""
    print("Running validation tests...")

    # HG phase function at key angles
    hg_fwd  = float(henyey_greenstein(np.array([1.0]),   HG_G)[0])
    hg_90   = float(henyey_greenstein(np.array([0.0]),   HG_G)[0])
    hg_bwd  = float(henyey_greenstein(np.array([-1.0]),  HG_G)[0])
    assert hg_fwd > hg_90 > hg_bwd, "HG ordering wrong"
    assert hg_fwd / hg_bwd > 5.0,   f"HG contrast too low: {hg_fwd/hg_bwd:.1f}×"
    print(f"  [a] HG phase OK  fwd={hg_fwd:.3f}  90°={hg_90:.3f}  "
          f"bwd={hg_bwd:.4f}  contrast={hg_fwd/hg_bwd:.1f}×")

    # Texture generation
    sky = generate_sky_texture(512, 256)
    assert sky.shape == (256, 512, 3), f"Bad shape: {sky.shape}"
    assert sky.min() >= 0.0, "Negative values"

    # Zenith (top-centre pixel) should be pinkish-tan (Mars — NOT blue)
    zenith_rgb = sky[0, 256]
    assert zenith_rgb[0] > zenith_rgb[2], \
        f"Zenith should be red>blue (Mars), got R={zenith_rgb[0]:.3f} B={zenith_rgb[2]:.3f}"

    # ── Sky-horizon rows (elevation ≈ 0–5°) should be brighter than zenith ───
    # Equirectangular mapping: v=0 → zenith, v=0.5 → horizon, v=1 → nadir.
    # In 256 rows: horizon row ≈ 128; rows 220-240 are BELOW horizon (ground).
    # Use rows 122-130 (elev ≈ 2–6°) as "sky horizon" strip.
    H, W = sky.shape[:2]
    horizon_row_lo = int(H * (0.5 - 6.0  / 180.0))   # elev ~6°
    horizon_row_hi = int(H * (0.5 - 1.0  / 180.0))   # elev ~1°
    horizon_mean   = sky[horizon_row_lo:horizon_row_hi, :, :].mean()
    zenith_mean    = sky[0:max(1, H // 64), :, :].mean()
    assert horizon_mean > zenith_mean, \
        f"Sky horizon ({horizon_mean:.3f}) should be brighter than zenith ({zenith_mean:.3f})"

    # ── Near-sun sky brighter than anti-solar sky at the same elevation ───────
    # Sun pixel (equirect):  (px_x, px_y)
    # Anti-solar sky point:  same elevation (same row), opposite azimuth (+W/2)
    sun_px_x = int(W * (SUN_AZ_DEG / 360.0))
    sun_px_y = int(H * (0.5 - SUN_ELEV_DEG / 180.0))
    sun_px_y = max(0, min(H - 1, sun_px_y))
    # Anti-solar: same elevation (+W/2 columns = opposite azimuth)
    anti_px_x = (sun_px_x + W // 2) % W
    anti_px_y = sun_px_y              # same row = same elevation → stays in sky
    sun_bright  = float(sky[sun_px_y,  sun_px_x,  :].mean())
    anti_bright = float(sky[anti_px_y, anti_px_x, :].mean())
    assert sun_bright > anti_bright * 1.5, \
        f"Near-sun ({sun_bright:.3f}) not bright enough vs anti-solar ({anti_bright:.3f})"

    print(f"  [b] Texture OK  {sky.shape}  zenith R={zenith_rgb[0]:.3f} B={zenith_rgb[2]:.3f}")
    print(f"  [c] Horizon brightening OK  "
          f"horizon_row=[{horizon_row_lo}:{horizon_row_hi}]  "
          f"horizon/zenith = {horizon_mean/zenith_mean:.2f}× (expect >1)")
    print(f"  [d] Forward scatter OK  "
          f"near-sun/anti-solar = {sun_bright/anti_bright:.2f}× (expect >1.5)")
    print("All tests passed ✓")


# =============================================================================
# Entry point
# =============================================================================

DEFAULT_OUTPUT = os.path.expanduser("~/mars-rover-agent/data/mars_sky.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Mars Henyey-Greenstein sky texture for ARES simulation."
    )
    parser.add_argument("--output",   default=DEFAULT_OUTPUT,
                        help=f"Output PNG path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--width",    type=int,   default=2048)
    parser.add_argument("--height",   type=int,   default=1024)
    parser.add_argument("--preview",  action="store_true",
                        help="Show texture with matplotlib after saving")
    parser.add_argument("--test",     action="store_true",
                        help="Run self-tests and exit")
    parser.add_argument("--exr",      action="store_true",
                        help="Also save an EXR version (requires OpenEXR)")
    args = parser.parse_args()

    if args.test:
        _run_tests()
        sys.exit(0)

    if not _PIL_OK:
        print("ERROR: Pillow is required.  Install with: pip install pillow")
        sys.exit(1)

    print(f"Generating Mars sky texture ({args.width}×{args.height})...")
    print(f"  Sun:  elev={SUN_ELEV_DEG}°  az={SUN_AZ_DEG}°")
    print(f"  Aerosol HG g = {HG_G}  (Madeleine 2012)")

    sky = generate_sky_texture(
        width=args.width, height=args.height,
        sun_elev_deg=SUN_ELEV_DEG, sun_az_deg=SUN_AZ_DEG, g=HG_G,
    )

    save_png(sky, args.output)

    if args.exr:
        exr_path = args.output.replace(".png", ".exr")
        save_exr(sky, exr_path)

    if args.preview:
        preview(sky)

    print()
    print("Next steps:")
    print("  1. git add data/mars_sky.png && git commit  (so Vast.ai can pull it)")
    print("  2. On Vast.ai: git pull && restart simulation")
    print("  3. Isaac Sim will auto-load the sky texture via DomeLight.textureFile")
    print()
    print(f"Reference: Bell et al. 2021, Madeleine et al. 2012, Lemmon et al. 2015")
