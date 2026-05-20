"""
Abstract VLM backend interface + Moondream2 implementation.

Design
------
All VLM backends share one method signature:

    query(image, query_text) -> VLMResult

This makes it trivial to swap Moondream2 for GPT-4V, LLaVA, or anything
else — the perception node never knows which backend it's talking to.

Backends
--------
- MockVLMBackend     : OpenCV-based, runs on CPU, no model download needed
- Moondream2Backend  : real Moondream2 inference, requires CUDA GPU
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
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
