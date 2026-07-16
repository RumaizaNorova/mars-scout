"""
Mars Scout Benchmark Runner
===========================

Fires every phrase in phrases_v1.yaml against the running stack,
records results, and prints a scored table.

Works in two modes
------------------
--mock   (default)  Uses the mock VLM backend — runs on Mac, no GPU.
                    Tests pipeline correctness, not VLM quality.
--live              Sends real NavigateToTarget actions to a running stack.
                    Use on the GPU instance with Isaac Sim running.

Usage
-----
  # Mock mode (Mac)
  python3 benchmark/run_benchmark.py --mock

  # Live mode (GPU instance, stack already running)
  python3 benchmark/run_benchmark.py --live --timeout 60

  # Single phrase
  python3 benchmark/run_benchmark.py --mock --phrase skull_like
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import yaml

BENCHMARK_DIR = Path(__file__).parent
PHRASES_FILE  = BENCHMARK_DIR / "phrases_v1.yaml"
SCENES_FILE   = BENCHMARK_DIR / "scenes_v1.yaml"
RESULTS_DIR   = BENCHMARK_DIR / "results"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class PhraseResult:
    phrase_id:      str
    phrase_text:    str
    mode:           str           # mock | live
    success:        bool
    outcome:        str           # ARRIVED | TIMEOUT | ABORTED | ERROR | MOCK_OK
    elapsed_sec:    float
    vlm_confidence: float
    vlm_description: str
    error_message:  Optional[str] = None


@dataclass
class BenchmarkReport:
    timestamp:    str
    mode:         str
    scene_id:     str
    results:      list[PhraseResult] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_success(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_total if self.n_total else 0.0

    @property
    def mean_elapsed(self) -> float:
        vals = [r.elapsed_sec for r in self.results if r.success]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_confidence(self) -> float:
        vals = [r.vlm_confidence for r in self.results]
        return sum(vals) / len(vals) if vals else 0.0


# ── Mock runner (no ROS needed) ───────────────────────────────────────────────

def run_mock(phrases: list[dict], timeout: float) -> list[PhraseResult]:
    """
    Runs each phrase through the mock VLM backend directly (no ROS).
    Validates that: backend returns a result, confidence is reasonable,
    description is non-empty.  Does NOT test navigation.
    """
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent /
                           "ros2_ws/src/mars_scout_perception"))
    import numpy as np
    from mars_scout_perception.mock_vlm_backend import MockVLMBackend

    # Generate a procedural Mars image for consistent testing
    import cv2
    rng = np.random.default_rng(42)
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:, :] = (60, 80, 160)
    img[:252, :] = (100, 80, 140)    # sky
    for _ in range(15):
        cx = rng.integers(100, 1180)
        cy = rng.integers(300, 680)
        r  = rng.integers(20, 80)
        cv2.circle(img, (cx, cy), r, (int(rng.integers(30,70)),)*3, -1)

    vlm = MockVLMBackend(min_confidence=0.0)
    results = []

    for p in phrases:
        t0 = time.perf_counter()
        try:
            result = vlm.query(img, p["text"])
            elapsed = time.perf_counter() - t0
            success = result.target_found and result.confidence > 0.3
            results.append(PhraseResult(
                phrase_id       = p["id"],
                phrase_text     = p["text"],
                mode            = "mock",
                success         = success,
                outcome         = "MOCK_OK" if success else "LOW_CONFIDENCE",
                elapsed_sec     = elapsed,
                vlm_confidence  = result.confidence,
                vlm_description = result.description,
            ))
        except Exception as e:
            results.append(PhraseResult(
                phrase_id       = p["id"],
                phrase_text     = p["text"],
                mode            = "mock",
                success         = False,
                outcome         = "ERROR",
                elapsed_sec     = time.perf_counter() - t0,
                vlm_confidence  = 0.0,
                vlm_description = "",
                error_message   = str(e),
            ))

    return results


# ── Live runner (full ROS stack) ──────────────────────────────────────────────

def run_live(phrases: list[dict], timeout: float) -> list[PhraseResult]:
    """Send real NavigateToTarget actions to a running stack."""
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from mars_scout_msgs.action import NavigateToTarget

    rclpy.init()
    node = Node("benchmark_runner")
    client = ActionClient(node, NavigateToTarget, "navigate_to_target")

    if not client.wait_for_server(timeout_sec=10.0):
        print("✗  Action server not available.")
        rclpy.shutdown()
        sys.exit(1)

    results = []
    for p in phrases:
        print(f"  → '{p['text']}'")
        goal = NavigateToTarget.Goal()
        goal.query_text       = p["text"]
        goal.timeout_sec      = timeout
        goal.arrival_radius_m = 0.5
        goal.min_confidence   = 0.4

        t0 = time.perf_counter()
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, future)
        handle = future.result()

        if not handle.accepted:
            results.append(PhraseResult(
                phrase_id=p["id"], phrase_text=p["text"], mode="live",
                success=False, outcome="REJECTED",
                elapsed_sec=time.perf_counter() - t0,
                vlm_confidence=0.0, vlm_description="",
            ))
            continue

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future)
        res = result_future.result().result
        elapsed = time.perf_counter() - t0

        results.append(PhraseResult(
            phrase_id       = p["id"],
            phrase_text     = p["text"],
            mode            = "live",
            success         = res.success,
            outcome         = res.outcome,
            elapsed_sec     = elapsed,
            vlm_confidence  = 0.0,
            vlm_description = res.vlm_description,
        ))

    node.destroy_node()
    rclpy.shutdown()
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def print_report(report: BenchmarkReport):
    print(f"\n{_BOLD}{'═'*72}{_RESET}")
    print(f"{_BOLD}  Mars Scout Benchmark — {report.mode.upper()} mode  "
          f"[{report.timestamp}]{_RESET}")
    print(f"{'═'*72}")
    print(f"  {'ID':<22} {'Outcome':<16} {'Conf':>6}  {'Time':>6}  Description")
    print(f"  {'─'*22} {'─'*16} {'─'*6}  {'─'*6}  {'─'*30}")

    for r in report.results:
        icon    = f"{_GREEN}✓{_RESET}" if r.success else f"{_RED}✗{_RESET}"
        conf    = f"{r.vlm_confidence:.0%}"
        elapsed = f"{r.elapsed_sec:.2f}s"
        desc    = r.vlm_description[:38] + "…" if len(r.vlm_description) > 38 else r.vlm_description
        print(f"  {icon} {r.phrase_id:<21} {r.outcome:<16} {conf:>6}  {elapsed:>6}  {desc}")

    print(f"{'─'*72}")
    rate_col = _GREEN if report.success_rate >= 0.7 else (_YELLOW if report.success_rate >= 0.4 else _RED)
    print(f"  {_BOLD}Success: {rate_col}{report.n_success}/{report.n_total} "
          f"({report.success_rate:.0%}){_RESET}   "
          f"mean conf: {report.mean_confidence:.0%}   "
          f"mean time: {report.mean_elapsed:.2f}s")
    print(f"{'═'*72}\n")


def save_report(report: BenchmarkReport):
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"benchmark_{report.mode}_{report.timestamp.replace(':', '-')}.json"
    with open(out, "w") as f:
        data = asdict(report)
        json.dump(data, f, indent=2)
    print(f"  Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mock", action="store_true", default=True,
                       help="Mock VLM backend — no ROS needed (default)")
    group.add_argument("--live", action="store_true",
                       help="Live ROS action — requires running stack")
    parser.add_argument("--timeout",  type=float, default=60.0)
    parser.add_argument("--phrase",   type=str,   default=None,
                        help="Run a single phrase by ID")
    parser.add_argument("--save",     action="store_true", help="Save JSON report")
    args = parser.parse_args()

    if args.live:
        args.mock = False

    # Load phrases
    with open(PHRASES_FILE) as f:
        all_phrases = yaml.safe_load(f)["phrases"]

    if args.phrase:
        phrases = [p for p in all_phrases if p["id"] == args.phrase]
        if not phrases:
            print(f"Phrase '{args.phrase}' not found.")
            sys.exit(1)
    else:
        phrases = all_phrases

    mode = "live" if args.live else "mock"
    print(f"\n{_BOLD}Running {len(phrases)} phrase(s) in {mode.upper()} mode…{_RESET}\n")

    t0 = time.perf_counter()
    results = run_live(phrases, args.timeout) if args.live else run_mock(phrases, args.timeout)
    total_time = time.perf_counter() - t0

    from datetime import datetime
    report = BenchmarkReport(
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        mode      = mode,
        scene_id  = "mars_flat_v1",
        results   = results,
    )

    print_report(report)
    print(f"  Total wall time: {total_time:.1f}s")

    if args.save:
        save_report(report)


if __name__ == "__main__":
    main()
