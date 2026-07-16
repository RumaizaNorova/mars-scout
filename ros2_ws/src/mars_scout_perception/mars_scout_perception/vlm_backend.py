"""
Abstract VLM backend interface + backend implementations.

Design
------
All VLM backends share one method signature:

    query(image, query_text) -> VLMResult

This makes it trivial to swap backends — the perception node never knows
which backend it's talking to.

Backends
--------
- MockVLMBackend      : OpenCV-based, runs on CPU, no model download needed
- Moondream2Backend   : real Moondream2 inference, requires CUDA GPU
- GeminiVisionBackend : Google Gemini 1.5 Flash via API, requires GOOGLE_API_KEY
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import os
import json
import re
import textwrap
import time


@dataclass
class BoundingBox:
    """Normalised [0,1] bounding box in image space."""
    cx: float   # centre x
    cy: float   # centre y
    w:  float   # width
    h:  float   # height

    def area(self) -> float:
        return self.w * self.h

    def __repr__(self):
        return f"BBox(cx={self.cx:.3f} cy={self.cy:.3f} w={self.w:.3f} h={self.h:.3f})"


@dataclass
class VLMResult:
    target_found:   bool
    bbox:           Optional[BoundingBox]   # None if not found
    description:    str                     # raw VLM answer
    confidence:     float                   # [0, 1]
    inference_ms:   float = 0.0             # wall-clock inference time

    @classmethod
    def not_found(cls, description: str = "Target not found in current view.") -> "VLMResult":
        return cls(target_found=False, bbox=None,
                   description=description, confidence=0.0)


class VLMBackend(ABC):
    """Interface every backend must implement."""

    @abstractmethod
    def query(self, image_bgr, query_text: str) -> VLMResult:
        """
        Parameters
        ----------
        image_bgr  : np.ndarray  (H, W, 3) uint8, BGR colour order
        query_text : str         natural-language description of the target

        Returns
        -------
        VLMResult
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name for logging."""


# ── Moondream2 backend ────────────────────────────────────────────────────────

class Moondream2Backend(VLMBackend):
    """
    Real Moondream2 inference.
    Lazy-loads the model on first call so the node starts immediately
    even before the model is downloaded.
    """

    MODEL_ID  = "vikhyatk/moondream2"
    REVISION  = "2024-08-26"

    # Prompt templates
    _DETECT_PROMPT = (
        "Is there {article} {target} visible in this image? "
        "If yes, describe where it is (left / centre / right, near / far) "
        "and estimate roughly what fraction of the image it occupies."
    )
    _BBOX_PROMPT = (
        "Where exactly is {article} {target} in the image? "
        "Reply with the bounding box as: cx=<0-1> cy=<0-1> w=<0-1> h=<0-1> "
        "where values are normalised image fractions. "
        "Reply ONLY with those four numbers."
    )

    def __init__(self):
        self._model     = None
        self._tokenizer = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, revision=self.REVISION
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID, trust_remote_code=True, revision=self.REVISION,
        ).to(device).eval()

    @property
    def name(self) -> str:
        return "moondream2"

    def query(self, image_bgr, query_text: str) -> VLMResult:
        import cv2
        from PIL import Image as PILImage

        if self._model is None:
            self._load()

        pil_img = PILImage.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        enc = self._model.encode_image(pil_img)

        article = _article(query_text)
        t0 = time.perf_counter()

        # Step 1 — is it there?
        detect_q = self._DETECT_PROMPT.format(article=article, target=query_text)
        description = self._model.answer_question(enc, detect_q, self._tokenizer)
        found = _answer_is_affirmative(description)

        if not found:
            return VLMResult.not_found(description)

        # Step 2 — where exactly?
        bbox_q = self._BBOX_PROMPT.format(article=article, target=query_text)
        bbox_answer = self._model.answer_question(enc, bbox_q, self._tokenizer)
        bbox = _parse_bbox_answer(bbox_answer)

        inference_ms = (time.perf_counter() - t0) * 1000
        confidence = _confidence_from_description(description)

        return VLMResult(
            target_found=True,
            bbox=bbox,
            description=description,
            confidence=confidence,
            inference_ms=inference_ms,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _article(text: str) -> str:
    return "an" if text and text[0].lower() in "aeiou" else "a"


def _answer_is_affirmative(text: str) -> bool:
    t = text.lower()
    negative = any(w in t for w in ["no ", "not ", "don't", "cannot", "can't", "none", "absent"])
    positive = any(w in t for w in ["yes", "visible", "present", "i can see", "there is", "there are"])
    return positive and not negative


def _parse_bbox_answer(text: str) -> Optional[BoundingBox]:
    """Parse 'cx=0.5 cy=0.6 w=0.2 h=0.3' from VLM output."""
    import re
    vals = {}
    for key in ("cx", "cy", "w", "h"):
        m = re.search(rf"{key}\s*=\s*([0-9]*\.?[0-9]+)", text)
        if m:
            vals[key] = float(m.group(1))
    if len(vals) == 4:
        return BoundingBox(**vals)
    return None


def _confidence_from_description(text: str) -> float:
    """Heuristic confidence from description keywords."""
    t = text.lower()
    if any(w in t for w in ["clearly", "definitely", "obviously", "large"]):
        return 0.9
    if any(w in t for w in ["appears", "looks like", "seems", "small"]):
        return 0.6
    if any(w in t for w in ["possibly", "maybe", "might", "faint"]):
        return 0.35
    return 0.7


def _parse_gemini_json(raw: str, query_text: str) -> dict:
    """
    Parse Gemini's JSON response.

    Strips markdown code fences first, then tries json.loads.
    If that fails, searches for any {...} block in the text.
    Raises RuntimeError (never silently degrades) if nothing parses.
    """
    # Strip ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$",       "", text, flags=re.MULTILINE).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Last chance: find the first {...} block anywhere in the output
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    raise RuntimeError(
        f"GeminiVisionBackend: response is not parseable JSON.\n"
        f"query={query_text!r}\n"
        f"raw response:\n{raw}"
    )


# ── Gemini Vision backend ─────────────────────────────────────────────────────

class GeminiVisionBackend(VLMBackend):
    """
    Google Gemini 1.5 Flash vision backend.

    Requirements
    ------------
    - pip install google-generativeai
    - GOOGLE_API_KEY environment variable set before node launch.

    Design
    ------
    Single API call per frame: the model receives the rover image and a
    natural-language query, and returns structured JSON:

        {
          "found":       true | false,
          "description": "<1-2 sentence observation>",
          "confidence":  0.0 – 1.0,
          "bbox": { "cx": 0-1, "cy": 0-1, "w": 0-1, "h": 0-1 }
        }

    bbox uses normalised image-space coordinates (0,0 = top-left).
    If "found" is false the bbox field may be absent or null.

    Error handling
    --------------
    - Missing GOOGLE_API_KEY → RuntimeError at __init__ (no silent fallback).
    - Non-JSON response      → RuntimeError with raw text (no silent fallback).
    - Malformed bbox fields  → RuntimeError with context (no silent fallback).
    - API / network errors   → propagated as-is (no silent fallback).
    """

    MODEL_ID = "gemini-1.5-flash"

    _SYSTEM_PROMPT = textwrap.dedent("""\
        You are a Mars terrain analysis system processing images from a rover camera
        (similar to Perseverance Mastcam-Z). You will receive a query about a terrain
        feature or science target and must analyse the image carefully.

        Respond ONLY with valid JSON — no prose, no markdown, no code fences.
        Use exactly this schema:
        {
          "found":       <true|false>,
          "description": "<1-2 sentences: what you see and where in the frame>",
          "confidence":  <float 0.0-1.0, reflecting how clearly the target is visible>,
          "bbox": {
            "cx": <normalised x of bounding box centre, 0=left  1=right>,
            "cy": <normalised y of bounding box centre, 0=top   1=bottom>,
            "w":  <normalised width  of bounding box, 0-1>,
            "h":  <normalised height of bounding box, 0-1>
          }
        }
        If the target is not found: set found=false, confidence=0.0, bbox=null.
        Bounding box values must be in [0, 1]. Confidence must be in [0, 1].
    """)

    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY environment variable is not set or is empty. "
                "Export it before launching the perception node:\n"
                "  export GOOGLE_API_KEY=<your-key>"
            )

        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            self.MODEL_ID,
            system_instruction=self._SYSTEM_PROMPT,
        )

    @property
    def name(self) -> str:
        return f"gemini/{self.MODEL_ID}"

    def query(self, image_bgr, query_text: str) -> VLMResult:
        import cv2
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        article = _article(query_text)
        prompt  = (
            f"Is there {article} {query_text} visible in this Mars rover image? "
            f"Analyse carefully and respond with JSON only."
        )

        t0 = time.perf_counter()
        response = self._model.generate_content([prompt, pil_img])
        inference_ms = (time.perf_counter() - t0) * 1000

        raw  = response.text.strip()
        data = _parse_gemini_json(raw, query_text)

        if not data.get("found", False):
            return VLMResult.not_found(
                description=data.get("description", "Target not found in current view."),
            )

        # Parse bounding box — raise if present but malformed
        bbox: Optional[BoundingBox] = None
        raw_bbox = data.get("bbox")
        if raw_bbox:
            try:
                bbox = BoundingBox(
                    cx=float(raw_bbox["cx"]),
                    cy=float(raw_bbox["cy"]),
                    w =float(raw_bbox["w"]),
                    h =float(raw_bbox["h"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"GeminiVisionBackend: malformed bbox in response.\n"
                    f"bbox={raw_bbox!r}\nfull response:\n{raw}"
                ) from exc

        confidence = float(data.get("confidence", 0.7))
        description = data.get("description", raw)

        return VLMResult(
            target_found=True,
            bbox=bbox,
            description=description,
            confidence=confidence,
            inference_ms=inference_ms,
        )
