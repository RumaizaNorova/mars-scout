"""
MockVLMBackend — CPU-only, OpenCV-based terrain analyser.

This is NOT a dummy that returns random outputs.
It actually analyses the image using the same rock-detection logic
as the depth synthesiser, finds the most prominent terrain feature,
and returns a real bounding box.

The query_text is used to bias which kind of region is returned:
  - "rock" / "boulder" / "formation"  → largest high-edge-density blob
  - "flat" / "safe" / "smooth"        → lowest-edge-density region
  - "dark" / "shadow"                 → darkest region
  - anything else                     → most salient region (highest contrast)

This means the full pipeline — mock_bridge → perception → geometry →
control — produces meaningful, query-dependent behaviour on a Mac with
zero GPU and zero model download.
"""

from __future__ import annotations
import time
import numpy as np
import cv2

from mars_scout_perception.vlm_backend import VLMBackend, VLMResult, BoundingBox


# ── Keyword → strategy mapping ────────────────────────────────────────────────
_ROCK_KEYWORDS    = {"rock", "boulder", "formation", "cliff", "outcrop", "skull", "stone", "crater"}
_FLAT_KEYWORDS    = {"flat", "safe", "smooth", "ground", "path", "clear", "open"}
_DARK_KEYWORDS    = {"dark", "shadow", "cave", "hole", "depression"}


class MockVLMBackend(VLMBackend):
    """
    OpenCV-based mock that genuinely analyses the terrain image.

    Parameters
    ----------
    min_confidence : float   Never report confidence below this threshold.
    noise_std      : float   Jitter added to bbox to simulate VLM imprecision.
    """

    def __init__(self, min_confidence: float = 0.45, noise_std: float = 0.02):
        self.min_confidence = min_confidence
        self.noise_std      = noise_std
        self._rng           = np.random.default_rng(0)

    @property
    def name(self) -> str:
        return "mock_opencv"

    def query(self, image_bgr: np.ndarray, query_text: str) -> VLMResult:
        t0 = time.perf_counter()

        strategy = self._classify_query(query_text)
        saliency_map = self._compute_saliency(image_bgr, strategy)

        # Mask out sky (top 35%) — rover shouldn't navigate to the sky
        h = image_bgr.shape[0]
        saliency_map[: int(h * 0.35), :] = 0.0

        if saliency_map.max() < 1e-6:
            return VLMResult.not_found("No matching terrain feature found.")

        bbox, confidence = self._extract_best_region(saliency_map, image_bgr)

        if confidence < self.min_confidence:
            return VLMResult.not_found(
                f"No {query_text} found with sufficient confidence "
                f"(best={confidence:.2f} < {self.min_confidence:.2f})."
            )

        # Add small bbox jitter to simulate VLM output variance
        bbox = self._jitter_bbox(bbox)
        description = self._generate_description(query_text, bbox, confidence, strategy)

        return VLMResult(
            target_found  = True,
            bbox          = bbox,
            description   = description,
            confidence    = confidence,
            inference_ms  = (time.perf_counter() - t0) * 1000,
        )

    # ── Strategy selection ────────────────────────────────────────────────────

    @staticmethod
    def _classify_query(query_text: str) -> str:
        words = set(query_text.lower().split())
        if words & _ROCK_KEYWORDS:
            return "rock"
        if words & _FLAT_KEYWORDS:
            return "flat"
        if words & _DARK_KEYWORDS:
            return "dark"
        return "salient"

    # ── Saliency maps ─────────────────────────────────────────────────────────

    def _compute_saliency(self, bgr: np.ndarray, strategy: str) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        if strategy == "rock":
            return self._rock_saliency(gray)
        elif strategy == "flat":
            return self._flat_saliency(gray)
        elif strategy == "dark":
            return 1.0 - gray
        else:
            return self._salient_saliency(bgr, gray)

    @staticmethod
    def _rock_saliency(gray: np.ndarray) -> np.ndarray:
        """High edge density + high local variance = rock."""
        gx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = np.sqrt(gx**2 + gy**2)
        edges = cv2.GaussianBlur(edges, (21, 21), 0)

        mean   = cv2.GaussianBlur(gray,    (21, 21), 0)
        mean_sq= cv2.GaussianBlur(gray**2, (21, 21), 0)
        var    = np.clip(mean_sq - mean**2, 0, None)
        var    = cv2.GaussianBlur(var, (11, 11), 0)

        sal = 0.6 * edges / (edges.max() + 1e-6) + \
              0.4 * var   / (var.max()   + 1e-6)
        return sal.astype(np.float32)

    @staticmethod
    def _flat_saliency(gray: np.ndarray) -> np.ndarray:
        """Low edge density + uniform intensity = flat ground."""
        gx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = np.sqrt(gx**2 + gy**2)
        edges = cv2.GaussianBlur(edges, (21, 21), 0)
        flat = 1.0 - (edges / (edges.max() + 1e-6))
        return flat.astype(np.float32)

    @staticmethod
    def _salient_saliency(bgr: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """Spectral residual saliency — perceptually salient regions."""
        log_spec = np.log(np.abs(np.fft.fft2(gray)) + 1e-6)
        avg_log  = cv2.GaussianBlur(log_spec.astype(np.float32), (3, 3), 0)
        residual = log_spec - avg_log
        sal_img  = np.abs(np.fft.ifft2(np.exp(residual))) ** 2
        sal      = cv2.GaussianBlur(sal_img.astype(np.float32), (7, 7), 0)
        return sal / (sal.max() + 1e-6)

    # ── Region extraction ─────────────────────────────────────────────────────

    def _extract_best_region(
        self,
        saliency: np.ndarray,
        bgr: np.ndarray,
    ) -> tuple[BoundingBox, float]:
        """
        Find the best connected region in the saliency map.
        Returns (BoundingBox, confidence).
        """
        H, W = saliency.shape

        # Threshold at 70th percentile
        thresh_val = float(np.percentile(saliency, 70))
        _, binary  = cv2.threshold(saliency, thresh_val, 1.0, cv2.THRESH_BINARY)
        binary_u8  = (binary * 255).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return BoundingBox(0.5, 0.5, 0.1, 0.1), 0.0

        # Pick contour with highest mean saliency × area
        best_score = -1.0
        best_cnt   = contours[0]
        for cnt in contours:
            mask = np.zeros_like(binary_u8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_sal = float(saliency[mask > 0].mean()) if mask.any() else 0.0
            area     = cv2.contourArea(cnt) / (H * W)
            score    = mean_sal * (area ** 0.3)    # prefer large + salient
            if score > best_score:
                best_score = score
                best_cnt   = cnt

        x, y, w, h = cv2.boundingRect(best_cnt)
        bbox = BoundingBox(
            cx = (x + w / 2) / W,
            cy = (y + h / 2) / H,
            w  = w / W,
            h  = h / H,
        )

        # Confidence from mean saliency in the region
        mask = np.zeros_like(binary_u8)
        cv2.drawContours(mask, [best_cnt], -1, 255, -1)
        mean_sal   = float(saliency[mask > 0].mean()) if mask.any() else 0.0
        confidence = float(np.clip(0.4 + mean_sal * 0.6, 0.0, 1.0))

        return bbox, confidence

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _jitter_bbox(self, bbox: BoundingBox) -> BoundingBox:
        s = self.noise_std
        return BoundingBox(
            cx = float(np.clip(bbox.cx + self._rng.normal(0, s), 0.05, 0.95)),
            cy = float(np.clip(bbox.cy + self._rng.normal(0, s), 0.05, 0.95)),
            w  = float(np.clip(bbox.w  + self._rng.normal(0, s * 0.5), 0.02, 0.8)),
            h  = float(np.clip(bbox.h  + self._rng.normal(0, s * 0.5), 0.02, 0.8)),
        )

    @staticmethod
    def _generate_description(
        query_text: str,
        bbox: BoundingBox,
        confidence: float,
        strategy: str,
    ) -> str:
        side  = "left"  if bbox.cx < 0.4 else ("right" if bbox.cx > 0.6 else "centre")
        dist  = "nearby" if bbox.cy > 0.65 else ("mid-range" if bbox.cy > 0.45 else "far")
        size  = "large"  if bbox.area() > 0.04 else ("medium" if bbox.area() > 0.01 else "small")
        conf_ = "clearly" if confidence > 0.75 else ("appears to be" if confidence > 0.55 else "possibly")

        return (
            f"Yes — {conf_} a {size} {query_text} visible on the {side}, "
            f"{dist} from the rover. "
            f"[mock/{strategy} conf={confidence:.2f}]"
        )
