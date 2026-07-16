"""
VLM Prompt Engineering Tool
============================

Systematically tests different prompt strategies against Mars terrain images
and scores which prompts produce the most useful outputs for navigation.

Why this matters
----------------
Moondream2 is a small (1.87B param) vision-language model.  Its outputs
are highly sensitive to prompt wording.  A bad prompt like "is there a rock?"
gives yes/no.  A good prompt like the ones below gives spatial location,
size, and confidence — exactly what the geometry node needs.

Run on Mac (CPU, mock images):
  python3 benchmark/prompt_eval.py --mock

Run on GPU instance (real Moondream2):
  python3 benchmark/prompt_eval.py --moondream2 --image /path/to/mars.jpg

Output: a scored table + saved results in benchmark/results/prompts_*.json
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR   = BENCHMARK_DIR / "results"


# ── Prompt catalogue ──────────────────────────────────────────────────────────
# Each entry: (id, template)
# {target} is replaced with the actual query at runtime.

PROMPT_STRATEGIES = [
    (
        "v1_simple",
        "Is there {article} {target} in this image?",
    ),
    (
        "v2_location",
        "Is there {article} {target} visible? "
        "If yes, is it on the left, centre, or right of the image, "
        "and is it near or far?",
    ),
    (
        "v3_bbox",
        "Locate {article} {target} in this image. "
        "Give the bounding box as cx=<0-1> cy=<0-1> w=<0-1> h=<0-1> "
        "where values are normalised fractions of image size.",
    ),
    (
        "v4_structured",
        "You are a Mars rover navigation assistant. "
        "Look at this terrain image and answer: "
        "(1) Is there {article} {target} visible? "
        "(2) If yes: which side (left/centre/right)? "
        "(3) How far (near <2m / mid 2-5m / far >5m)? "
        "(4) Confidence: high/medium/low.",
    ),
    (
        "v5_two_step_detect",
        "Does this Mars terrain image contain {article} {target}? "
        "Answer only YES or NO.",
    ),
    (
        "v5_two_step_locate",
        # Only sent if v5_two_step_detect returns YES
        "Where exactly is the {target} in this image? "
        "Describe its position as (left|centre|right), (near|mid|far), "
        "and give an approximate bounding box cx cy w h as 0-1 fractions.",
    ),
    (
        "v6_mars_expert",
        "This is a camera image from a Mars surface rover. "
        "Identify any {target} in the scene. "
        "For each one found, state: "
        "position (left/centre/right of frame), "
        "distance estimate (near/mid/far), "
        "size (small/medium/large), "
        "and a confidence score 0-10.",
    ),
]

TARGET_PHRASES = [
    ("rock",          "a"),
    ("boulder",       "a"),
    ("rock formation","a"),
    ("flat terrain",  ""),
    ("dark shadow",   "a"),
]


# ── Scoring ───────────────────────────────────────────────────────────────────

@dataclass
class PromptScore:
    strategy_id:   str
    target:        str
    response:      str
    has_location:  bool    # mentions left/centre/right
    has_distance:  bool    # mentions near/far/mid or metres
    has_bbox:      bool    # contains cx= cy= or similar
    has_confidence: bool   # mentions confidence / score
    is_affirmative: bool   # says yes / visible / present
    length:        int     # response word count
    latency_ms:    float

    @property
    def score(self) -> float:
        """Composite usefulness score [0, 1]."""
        return (
            0.25 * self.has_location  +
            0.20 * self.has_distance  +
            0.25 * self.has_bbox      +
            0.15 * self.has_confidence +
            0.15 * (20 <= self.length <= 80)  # penalise too short or too verbose
        )


def _score_response(strategy_id: str, target: str, response: str, latency_ms: float) -> PromptScore:
    r = response.lower()
    return PromptScore(
        strategy_id    = strategy_id,
        target         = target,
        response       = response,
        has_location   = any(w in r for w in ["left", "centre", "center", "right"]),
        has_distance   = any(w in r for w in ["near", "far", "mid", "metre", "meter", "m away", "close"]),
        has_bbox       = any(w in r for w in ["cx=", "cy=", "0.", "bounding"]),
        has_confidence = any(w in r for w in ["confidence", "certain", "sure", "/10", "high", "medium", "low"]),
        is_affirmative = any(w in r for w in ["yes", "visible", "present", "i can see", "there is"]) and
                         not any(w in r for w in ["no ", "not ", "cannot"]),
        length         = len(response.split()),
        latency_ms     = latency_ms,
    )


# ── Backends ──────────────────────────────────────────────────────────────────

def _run_mock(image_bgr: np.ndarray, prompts_and_targets: list) -> list[PromptScore]:
    """
    Mock mode: uses the mock VLM backend's description as the 'response'.
    Evaluates prompt structure rather than actual Moondream2 behaviour.
    """
    sys.path.insert(0, str(BENCHMARK_DIR.parent / "ros2_ws/src/mars_scout_perception"))
    from mars_scout_perception.mock_vlm_backend import MockVLMBackend

    vlm = MockVLMBackend(min_confidence=0.0)
    scores = []
    for strategy_id, prompt_template, target, article in prompts_and_targets:
        t0 = time.perf_counter()
        result = vlm.query(image_bgr, target)
        latency_ms = (time.perf_counter() - t0) * 1000
        scores.append(_score_response(strategy_id, target, result.description, latency_ms))
    return scores


def _run_moondream2(image_bgr: np.ndarray, prompts_and_targets: list) -> list[PromptScore]:
    """Real Moondream2 inference — requires GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from PIL import Image as PILImage

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading Moondream2 on {device}…")
    tok   = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision="2024-08-26")
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2", trust_remote_code=True, revision="2024-08-26"
    ).to(device).eval()

    pil_img = PILImage.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    enc     = model.encode_image(pil_img)

    scores = []
    for strategy_id, prompt_template, target, article in prompts_and_targets:
        article_str = article if article else ("an" if target[0] in "aeiou" else "a")
        prompt = prompt_template.format(target=target, article=article_str)

        t0 = time.perf_counter()
        response = model.answer_question(enc, prompt, tok)
        latency_ms = (time.perf_counter() - t0) * 1000
        scores.append(_score_response(strategy_id, target, response, latency_ms))
        print(f"    [{strategy_id}] '{target}' → {response[:60]}…")

    return scores


# ── Reporting ─────────────────────────────────────────────────────────────────

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _bar(score: float, width: int = 10) -> str:
    filled = int(score * width)
    colour = _GREEN if score >= 0.6 else (_YELLOW if score >= 0.35 else _RED)
    return colour + "█" * filled + "░" * (width - filled) + _RESET


def print_results(scores: list[PromptScore]):
    print(f"\n{_BOLD}{'═'*80}{_RESET}")
    print(f"{_BOLD}  VLM Prompt Evaluation Results{_RESET}")
    print(f"{'═'*80}")
    print(f"  {'Strategy':<26} {'Target':<18} {'Score':>5}  "
          f"{'Loc':>3} {'Dst':>3} {'BBox':>4} {'Conf':>4}  {'ms':>6}")
    print(f"  {'─'*26} {'─'*18} {'─'*5}  {'─'*3} {'─'*3} {'─'*4} {'─'*4}  {'─'*6}")

    # Group by strategy, show mean score
    by_strategy: dict[str, list[PromptScore]] = {}
    for s in scores:
        by_strategy.setdefault(s.strategy_id, []).append(s)

    ranked = sorted(by_strategy.items(), key=lambda kv: -sum(s.score for s in kv[1]))

    for strategy_id, strat_scores in ranked:
        mean_score = sum(s.score for s in strat_scores) / len(strat_scores)
        for s in strat_scores:
            loc  = f"{_GREEN}✓{_RESET}" if s.has_location   else f"{_RED}✗{_RESET}"
            dst  = f"{_GREEN}✓{_RESET}" if s.has_distance   else f"{_RED}✗{_RESET}"
            bbox = f"{_GREEN}✓{_RESET}" if s.has_bbox       else f"{_RED}✗{_RESET}"
            conf = f"{_GREEN}✓{_RESET}" if s.has_confidence else f"{_RED}✗{_RESET}"
            print(
                f"  {s.strategy_id:<26} {s.target:<18} "
                f"{_bar(s.score)}  "
                f"{loc}   {dst}   {bbox}    {conf}   {s.latency_ms:>5.0f}"
            )
        print(f"  {'':26} {'mean score':>18} {mean_score:.2f}")
        print()

    # Winner
    best_id, best_scores = ranked[0]
    best_mean = sum(s.score for s in best_scores) / len(best_scores)
    print(f"{_BOLD}  🏆  Best strategy: {_GREEN}{best_id}{_RESET}{_BOLD}  "
          f"(mean score {best_mean:.2f}){_RESET}")
    print(f"{'═'*80}\n")


def save_results(scores: list[PromptScore], mode: str):
    from datetime import datetime
    RESULTS_DIR.mkdir(exist_ok=True)
    ts  = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out = RESULTS_DIR / f"prompts_{mode}_{ts}.json"
    with open(out, "w") as f:
        json.dump([asdict(s) for s in scores], f, indent=2)
    print(f"  Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _make_test_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:, :] = (60, 80, 160)
    img[:252, :] = (100, 80, 140)
    for _ in range(20):
        cx = int(rng.integers(100, 1180))
        cy = int(rng.integers(300, 680))
        r  = int(rng.integers(15, 70))
        cv2.circle(img, (cx, cy), r, (int(rng.integers(30,70)),)*3, -1)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock",       action="store_true", default=True)
    parser.add_argument("--moondream2", action="store_true")
    parser.add_argument("--image",      type=str, default=None,
                        help="Path to a Mars image (JPG/PNG). Uses procedural if omitted.")
    parser.add_argument("--target",     type=str, default=None,
                        help="Test a single target (e.g. 'rock')")
    parser.add_argument("--save",       action="store_true")
    args = parser.parse_args()

    if args.moondream2:
        args.mock = False

    # Load image
    if args.image and Path(args.image).exists():
        image_bgr = cv2.imread(args.image)
        print(f"  Using image: {args.image}  {image_bgr.shape}")
    else:
        print("  Using procedural Mars terrain image.")
        image_bgr = _make_test_image()

    # Build (strategy, prompt, target, article) tuples
    targets = [(args.target, "a")] if args.target else TARGET_PHRASES
    combos = [
        (sid, tmpl, tgt, art)
        for sid, tmpl in PROMPT_STRATEGIES[:6]  # skip two-step for mock
        for tgt, art in targets
    ]

    mode = "moondream2" if args.moondream2 else "mock"
    print(f"\n{_BOLD}Running {len(combos)} prompt × target combos in {mode.upper()} mode…{_RESET}\n")

    scores = _run_moondream2(image_bgr, combos) if args.moondream2 else _run_mock(image_bgr, combos)

    print_results(scores)

    if args.save:
        save_results(scores, mode)


if __name__ == "__main__":
    main()
