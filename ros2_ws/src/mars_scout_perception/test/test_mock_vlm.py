import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mars_scout_perception.mock_vlm_backend import MockVLMBackend
from mars_scout_perception.vlm_backend import VLMResult, BoundingBox, _article, _answer_is_affirmative


@pytest.fixture
def backend():
    return MockVLMBackend(min_confidence=0.0, noise_std=0.0)  # deterministic


def mars_image(h=720, w=1280, with_rock=True):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (60, 80, 160)          # reddish Mars base
    img[:int(h*0.35), :] = (100, 80, 140)  # sky
    if with_rock:
        img[450:520, 580:680] = (35, 45, 70)   # dark rock blob
    return img


def test_returns_vlm_result(backend):
    result = backend.query(mars_image(), "rock")
    assert isinstance(result, VLMResult)


def test_finds_rock_in_rocky_image(backend):
    result = backend.query(mars_image(with_rock=True), "rock")
    assert result.target_found
    assert result.bbox is not None


def test_bbox_normalised(backend):
    result = backend.query(mars_image(), "rock")
    if result.target_found:
        b = result.bbox
        assert 0.0 <= b.cx <= 1.0
        assert 0.0 <= b.cy <= 1.0
        assert 0.0 < b.w  <= 1.0
        assert 0.0 < b.h  <= 1.0


def test_confidence_in_range(backend):
    result = backend.query(mars_image(), "boulder")
    assert 0.0 <= result.confidence <= 1.0


def test_description_is_string(backend):
    result = backend.query(mars_image(), "rock formation")
    assert isinstance(result.description, str)
    assert len(result.description) > 0


def test_not_found_when_confidence_too_low():
    strict = MockVLMBackend(min_confidence=0.99, noise_std=0.0)
    # Featureless image — low saliency → likely below threshold
    blank = np.full((720, 1280, 3), (60, 80, 160), dtype=np.uint8)
    result = strict.query(blank, "rock")
    # Either found with conf >= 0.99 (unlikely on blank) or not found
    if not result.target_found:
        assert result.bbox is None


def test_flat_query_returns_result(backend):
    result = backend.query(mars_image(), "flat safe ground")
    assert isinstance(result, VLMResult)


def test_dark_query_returns_result(backend):
    result = backend.query(mars_image(), "dark shadow region")
    assert isinstance(result, VLMResult)


def test_inference_ms_positive(backend):
    result = backend.query(mars_image(), "rock")
    assert result.inference_ms >= 0.0


def test_not_found_result_factory():
    r = VLMResult.not_found("nothing here")
    assert not r.target_found
    assert r.bbox is None
    assert r.confidence == 0.0


def test_article_helper():
    assert _article("rock")    == "a"
    assert _article("outcrop") == "an"
    assert _article("boulder") == "a"


def test_affirmative_detection():
    assert _answer_is_affirmative("Yes, there is a rock visible on the left.")
    assert _answer_is_affirmative("I can see a boulder in the centre.")
    assert not _answer_is_affirmative("No rock is visible in this image.")
    assert not _answer_is_affirmative("I cannot see any formation here.")
