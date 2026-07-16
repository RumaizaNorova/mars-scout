#!/usr/bin/env python3
"""
validate_rendering.py
=====================
Quantitative validation of ARES rendered images against real
Perseverance Mastcam-Z images from Jezero Crater.

Usage (on Vast.ai, after simulation is running):
  # Save a frame from the simulation:
  ros2 run image_tools showimage --ros-args -r image:=/isaac_camera/rgb &
  # Or capture via the camera_viewer dashboard screenshot

  # Then run:
  python3 scripts/validate_rendering.py \
      --rendered  /tmp/ares_frame.png \
      --reference data/mastcam_ref/mcz_sol0100_example.jpg

What it measures
----------------
1. Luminance histogram intersection — overall brightness distribution match
2. Hue histogram intersection — color distribution match
3. Mean albedo ratio — is our scene the right overall brightness?
4. Red/blue ratio — is the red slope correct?

A score >0.80 on histogram intersection means the images are plausibly
from the same scene type. This is reported as a validation metric in the paper.

Reference images
----------------
Downloaded by: scripts/download_mastcam.sh
Source: NASA PDS raw image archive (public domain, no license required)
  https://pds-imaging.jpl.nasa.gov/data/mars2020/
"""

from __future__ import annotations
import argparse
import sys
import os
import json
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError:
    print("Install deps: pip install numpy pillow")
    sys.exit(1)


# =============================================================================
# Histogram intersection metric
# =============================================================================

def histogram_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Histogram intersection similarity ∈ [0, 1].
    = sum(min(h1[i], h2[i])) where both are normalised to sum=1.
    1.0 = identical distributions, 0.0 = no overlap.
    """
    h1 = h1.astype(float) / (h1.sum() + 1e-9)
    h2 = h2.astype(float) / (h2.sum() + 1e-9)
    return float(np.minimum(h1, h2).sum())


def compute_histograms(img_rgb: np.ndarray, bins: int = 64) -> dict:
    """Compute luminance and hue histograms from an RGB image array."""
    # Luminance (Y from sRGB, approximate)
    Y = 0.2126 * img_rgb[:,:,0] + 0.7152 * img_rgb[:,:,1] + 0.0722 * img_rgb[:,:,2]
    lum_hist, _ = np.histogram(Y.ravel(), bins=bins, range=(0, 255))

    # Convert to HSV for hue histogram (skip low-saturation pixels)
    img_float = img_rgb.astype(float) / 255.0
    r, g, b = img_float[:,:,0], img_float[:,:,1], img_float[:,:,2]
    Cmax = np.maximum(np.maximum(r, g), b)
    Cmin = np.minimum(np.minimum(r, g), b)
    delta = Cmax - Cmin

    saturation = np.where(Cmax > 0, delta / Cmax, 0)

    # Only compute hue where saturation > 0.1 (avoid grey/sky)
    mask = saturation > 0.10
    hue = np.zeros_like(r)
    with np.errstate(invalid='ignore', divide='ignore'):
        m = mask & (Cmax == r)
        hue[m] = (60 * ((g[m] - b[m]) / delta[m]) % 360)
        m = mask & (Cmax == g)
        hue[m] = (60 * ((b[m] - r[m]) / delta[m]) + 120)
        m = mask & (Cmax == b)
        hue[m] = (60 * ((r[m] - g[m]) / delta[m]) + 240)

    hue_hist, _ = np.histogram(hue[mask].ravel(), bins=bins, range=(0, 360))

    return {
        "luminance_hist": lum_hist,
        "hue_hist": hue_hist,
        "mean_luminance": float(Y.mean()),
        "mean_r": float(img_rgb[:,:,0].mean()),
        "mean_g": float(img_rgb[:,:,1].mean()),
        "mean_b": float(img_rgb[:,:,2].mean()),
    }


def load_image(path: str) -> np.ndarray:
    """Load image as uint8 RGB array, crop sky (top 30%) for ground comparison."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    # Crop top 30% (sky) — we're comparing ground surface, not sky
    h = arr.shape[0]
    return arr[int(h * 0.30):, :, :]


def validate(rendered_path: str, reference_path: str) -> dict:
    """
    Compare rendered image to Mastcam-Z reference.
    Returns a dict of metrics and an overall score.
    """
    print(f"\nRendered:  {rendered_path}")
    print(f"Reference: {reference_path}\n")

    rendered  = load_image(rendered_path)
    reference = load_image(reference_path)

    h_rend = compute_histograms(rendered)
    h_ref  = compute_histograms(reference)

    lum_score = histogram_intersection(h_rend["luminance_hist"],
                                       h_ref["luminance_hist"])
    hue_score = histogram_intersection(h_rend["hue_hist"],
                                       h_ref["hue_hist"])

    # Albedo ratio: how close is our mean brightness to real images?
    albedo_ratio = h_rend["mean_luminance"] / (h_ref["mean_luminance"] + 1e-9)

    # Red slope ratio: (R/B) in rendered vs reference
    r_b_rend = h_rend["mean_r"] / (h_rend["mean_b"] + 1e-9)
    r_b_ref  = h_ref["mean_r"]  / (h_ref["mean_b"]  + 1e-9)
    red_slope_ratio = r_b_rend / (r_b_ref + 1e-9)

    # Overall score: weighted average
    overall = 0.45 * lum_score + 0.35 * hue_score + \
              0.10 * max(0, 1 - abs(1 - albedo_ratio)) + \
              0.10 * max(0, 1 - abs(1 - red_slope_ratio))

    results = {
        "luminance_histogram_intersection": round(lum_score, 4),
        "hue_histogram_intersection":       round(hue_score, 4),
        "albedo_ratio_rendered_to_real":    round(albedo_ratio, 4),
        "red_blue_slope_ratio":             round(red_slope_ratio, 4),
        "overall_score":                    round(overall, 4),
        "rendered_mean_luminance":          round(h_rend["mean_luminance"], 2),
        "reference_mean_luminance":         round(h_ref["mean_luminance"],  2),
    }

    # ── Report ────────────────────────────────────────────────────────────────
    print("="*55)
    print("ARES Rendering Validation vs Mastcam-Z Reference")
    print("="*55)
    print(f"  Luminance histogram intersection: {lum_score:.4f}  (target >0.80)")
    print(f"  Hue histogram intersection:       {hue_score:.4f}  (target >0.75)")
    print(f"  Albedo ratio (rendered/real):      {albedo_ratio:.4f}  (target 0.85–1.15)")
    print(f"  Red/blue slope ratio:              {red_slope_ratio:.4f}  (target 0.90–1.10)")
    print(f"")
    print(f"  OVERALL SCORE:  {overall:.4f}  ({'PASS ✓' if overall > 0.75 else 'NEEDS WORK ✗'})")
    print("="*55)

    if albedo_ratio < 0.70:
        print(f"\n⚠ Scene is too DARK ({albedo_ratio:.2f}×). Increase sun intensity.")
    elif albedo_ratio > 1.30:
        print(f"\n⚠ Scene is too BRIGHT ({albedo_ratio:.2f}×). Decrease sun intensity.")
    else:
        print(f"\n✓ Albedo within acceptable range ({albedo_ratio:.2f}×).")

    if red_slope_ratio < 0.85:
        print(f"⚠ Red slope too WEAK. Terrain may look too blue/neutral.")
    elif red_slope_ratio > 1.15:
        print(f"⚠ Red slope too STRONG. Terrain may look too orange.")
    else:
        print(f"✓ Red/blue ratio matches reference ({red_slope_ratio:.2f}×).")

    print()

    return results


# =============================================================================
# Calibration target analysis (standalone mode)
# =============================================================================

def analyze_calibration_target(rendered_path: str) -> None:
    """
    Analyze a rendered image containing the calibration target panels.
    The calibration target has 4 horizontal panels with known albedo:
      Row 0 (top):    albedo = 0.04  (near-black)
      Row 1:          albedo = 0.12
      Row 2:          albedo = 0.20
      Row 3 (bottom): albedo = 0.40  (light grey)

    Measures rendered pixel values for each panel and computes the
    Isaac Sim intensity multiplier needed for correct calibration.

    Target is placed at /World/CalibrationTarget in the scene.
    """
    print("\nCalibration target analysis mode")
    print("Expected: 4 horizontal grey panels with albedo 0.04, 0.12, 0.20, 0.40")
    print()

    img = np.array(Image.open(rendered_path).convert("RGB")).astype(float) / 255.0
    h, w = img.shape[:2]
    panel_h = h // 4

    known_albedos = [0.04, 0.12, 0.20, 0.40]
    print(f"{'Panel':<8} {'Albedo':<10} {'Rendered_Y':<14} {'Expected_Y':<14} {'Ratio'}")
    print("-"*55)

    ratios = []
    for i, alb in enumerate(known_albedos):
        row = img[i*panel_h : (i+1)*panel_h, :, :]
        Y = 0.2126*row[:,:,0] + 0.7152*row[:,:,1] + 0.0722*row[:,:,2]
        rendered_y = float(Y.mean())
        expected_y = alb  # for a perfectly calibrated renderer
        ratio = rendered_y / (expected_y + 1e-9)
        ratios.append(ratio)
        print(f"{i:<8} {alb:<10.2f} {rendered_y:<14.4f} {expected_y:<14.4f} {ratio:.4f}")

    mean_ratio = float(np.mean(ratios))
    print("-"*55)
    print(f"{'Mean ratio:':<34} {mean_ratio:.4f}")
    print()
    print(f"→ Set DistantLight intensity × {1.0/mean_ratio:.3f} to correct calibration")
    print(f"  (current intensity × {1.0/mean_ratio:.3f} = calibrated intensity)")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARES rendering validation")
    parser.add_argument("--rendered",   required=True, help="Path to rendered ARES frame (PNG/JPG)")
    parser.add_argument("--reference",  help="Path to Mastcam-Z reference image (PNG/JPG)")
    parser.add_argument("--calib-mode", action="store_true",
                        help="Analyze calibration target panels instead of comparing to reference")
    parser.add_argument("--save-json",  help="Save results to JSON file")
    args = parser.parse_args()

    if not os.path.exists(args.rendered):
        print(f"Error: rendered image not found: {args.rendered}")
        sys.exit(1)

    if args.calib_mode:
        analyze_calibration_target(args.rendered)
    elif args.reference:
        if not os.path.exists(args.reference):
            print(f"Error: reference image not found: {args.reference}")
            sys.exit(1)
        results = validate(args.rendered, args.reference)
        if args.save_json:
            with open(args.save_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.save_json}")
    else:
        print("Provide --reference for comparison, or --calib-mode for calibration target analysis.")
        parser.print_help()
