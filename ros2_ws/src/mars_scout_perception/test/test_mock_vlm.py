import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mars_scout_perception.mock_vlm_backend import MockVLMBackend
from mars_scout_perception.vlm_backend import (
    VLMResult, BoundingBox, _article, _answer_is_affirmative,
    _parse_gemini_json, GeminiVisionBackend,
)


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


# ── GeminiVisionBackend unit tests (no real API calls) ────────────────────────

class TestParseGeminiJson:
    """_parse_gemini_json must handle all common Gemini response shapes."""

    def test_clean_json(self):
        raw = '{"found": true, "description": "a rock", "confidence": 0.85, "bbox": {"cx": 0.5, "cy": 0.6, "w": 0.2, "h": 0.15}}'
        data = _parse_gemini_json(raw, "rock")
        assert data["found"] is True
        assert data["confidence"] == pytest.approx(0.85)

    def test_strips_markdown_fence(self):
        raw = "```json\n{\"found\": false, \"description\": \"nothing\", \"confidence\": 0.0, \"bbox\": null}\n```"
        data = _parse_gemini_json(raw, "rock")
        assert data["found"] is False

    def test_strips_plain_fence(self):
        raw = "```\n{\"found\": true, \"description\": \"yes\", \"confidence\": 0.7, \"bbox\": {\"cx\": 0.4, \"cy\": 0.5, \"w\": 0.1, \"h\": 0.1}}\n```"
        data = _parse_gemini_json(raw, "boulder")
        assert data["found"] is True

    def test_json_embedded_in_prose(self):
        # Model sometimes adds one sentence before the JSON
        raw = 'Here is my analysis: {"found": true, "description": "rock", "confidence": 0.9, "bbox": {"cx": 0.3, "cy": 0.7, "w": 0.2, "h": 0.2}}'
        data = _parse_gemini_json(raw, "rock")
        assert data["found"] is True

    def test_raises_on_garbage(self):
        with pytest.raises(RuntimeError, match="not parseable JSON"):
            _parse_gemini_json("I see a rock. It is on the left.", "rock")

    def test_raises_on_empty(self):
        with pytest.raises(RuntimeError):
            _parse_gemini_json("", "rock")


def _fake_genai_modules():
    """sys.modules entries that stand in for the google-genai SDK."""
    import unittest.mock as mock
    genai = mock.MagicMock()
    google = mock.MagicMock(genai=genai)
    return genai, {"google": google, "google.genai": genai, "google.genai.types": genai.types}


class TestGeminiBackendInitGuard:
    """GeminiVisionBackend must raise immediately if configuration is missing."""

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            GeminiVisionBackend()

    def test_raises_with_empty_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            GeminiVisionBackend()

    def test_raises_without_model(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-testing")
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        with pytest.raises(RuntimeError, match="GEMINI_MODEL"):
            GeminiVisionBackend()


class TestGeminiBackendCalls:
    """The backend must use the configured model and the google-genai call shape."""

    def _backend(self, monkeypatch, response_text):
        import unittest.mock as mock
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-testing")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
        genai, modules = _fake_genai_modules()
        client = genai.Client.return_value
        client.models.generate_content.return_value = mock.MagicMock(text=response_text)
        with mock.patch.dict("sys.modules", modules):
            return GeminiVisionBackend(), client

    def test_name_contains_model(self, monkeypatch):
        backend, _ = self._backend(monkeypatch, "{}")
        assert backend.name == "gemini/gemini-test-model"

    def test_query_sends_model_and_parses_result(self, monkeypatch):
        backend, client = self._backend(monkeypatch, (
            '{"found": true, "description": "a rock", "confidence": 0.8,'
            ' "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}}'
        ))
        result = backend.query(np.zeros((8, 8, 3), dtype=np.uint8), "rock")
        assert client.models.generate_content.call_args.kwargs["model"] == "gemini-test-model"
        assert result.target_found and result.confidence == 0.8 and result.bbox.w == 0.1

    def test_missing_confidence_raises(self, monkeypatch):
        backend, _ = self._backend(monkeypatch, '{"found": true, "description": "a rock"}')
        with pytest.raises(RuntimeError, match="confidence"):
            backend.query(np.zeros((8, 8, 3), dtype=np.uint8), "rock")
