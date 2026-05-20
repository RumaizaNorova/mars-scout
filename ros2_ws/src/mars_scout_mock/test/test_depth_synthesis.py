import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mars_scout_mock.depth_synthesis import DepthSynthesiser, DepthSynthesisConfig


@pytest.fixture
def synth():
    return DepthSynthesiser(DepthSynthesisConfig(noise_sigma=0.0))  # no noise for determinism


def mars_image(h=720, w=1280):
    """Realistic Mars-coloured test image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 2] = 160   # reddish
    img[:, :, 1] = 80
    img[:, :, 0] = 60
    # Add some rocks (darker blobs)
    img[400:450, 600:660] = (40, 50, 80)
    img[500:530, 300:340] = (35, 45, 70)
    return img


def test_output_shape(synth):
    img = mars_image()
    depth = synth.synthesise(img)
    assert depth.shape == (720, 1280)


def test_output_dtype(synth):
    depth = synth.synthesise(mars_image())
    assert depth.dtype == np.float32


def test_depth_range(synth):
    """All valid (non-NaN) depths should be within [min_depth, max_depth]."""
    cfg = synth.cfg
    depth = synth.synthesise(mars_image())
    valid = depth[~np.isnan(depth)]
    assert valid.min() >= cfg.min_depth - 1e-4
    assert valid.max() <= cfg.max_depth + 1e-4


def test_ground_ramp_bottom_closer_than_top(synth):
    """Bottom rows should be closer (smaller depth) than middle rows."""
    depth = synth.synthesise(mars_image())
    valid_bottom = depth[680:720, :][~np.isnan(depth[680:720, :])]
    valid_mid    = depth[400:450, :][~np.isnan(depth[400:450, :])]
    assert np.nanmedian(valid_bottom) < np.nanmedian(valid_mid)


def test_sky_is_nan(synth):
    """Top rows should be NaN (sky mask)."""
    depth = synth.synthesise(mars_image())
    top_rows = depth[:int(720 * synth.cfg.sky_frac * 0.8), :]
    nan_frac = np.isnan(top_rows).mean()
    assert nan_frac > 0.5, f"Expected mostly NaN sky, got {nan_frac:.2%} NaN"


def test_some_valid_depth_exists(synth):
    """At least 30% of pixels should have valid depth."""
    depth = synth.synthesise(mars_image())
    valid_frac = (~np.isnan(depth)).mean()
    assert valid_frac > 0.30, f"Too few valid depth pixels: {valid_frac:.2%}"


def test_noise_adds_variation():
    """Synthesiser with noise should produce non-identical consecutive frames."""
    synth_noisy = DepthSynthesiser(DepthSynthesisConfig(noise_sigma=0.05))
    img = mars_image()
    d1 = synth_noisy.synthesise(img)
    d2 = synth_noisy.synthesise(img)
    # They should differ (noise is random)
    valid = ~(np.isnan(d1) | np.isnan(d2))
    assert not np.allclose(d1[valid], d2[valid])
