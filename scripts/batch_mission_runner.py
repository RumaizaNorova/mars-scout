#!/usr/bin/env python3
"""
Batch Mission Runner — populates all three telemetry systems without ROS2.

Simulates N complete missions using the pure-Python kinematic model, FSM,
and controller.  Each mission writes observations to MongoDB, Elasticsearch,
and Arize Phoenix.  After all missions, runs analytics and prints a report.

This is how we build the data volume that justifies using these tools:
  50 missions × ~120 observations = 6,000 documents in MongoDB/Elastic
  2,000+ terrain features in the geospatial catalog
  Calibration curves visible in Arize Phoenix

Usage
-----
  python3 scripts/batch_mission_runner.py
  python3 scripts/batch_mission_runner.py --n-missions 100
  python3 scripts/batch_mission_runner.py --n-missions 20 --use-gemini

Environment required
--------------------
  MONGODB_URI              MongoDB Atlas or local URI
  ELASTICSEARCH_URL        Elasticsearch endpoint
  ELASTICSEARCH_API_KEY    Elastic API key (if using cloud)
  PHOENIX_COLLECTOR_ENDPOINT  or ARIZE_API_KEY + ARIZE_SPACE_ID
  GOOGLE_API_KEY           for Gemini VLM (if --use-gemini) and Arize evals

"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

# ── Make project root importable ─────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "mars_scout_control"))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "mars_scout_perception"))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "mars_scout_mock"))

from mars_scout_control.fsm        import AgentFSM, FSMConfig, State
from mars_scout_control.controller import PurePursuitController
from mars_scout_mock.kinematic_model import DifferentialDriveModel, RoverPose
from mars_scout_perception.mock_vlm_backend import MockVLMBackend
from mars_scout_perception.vlm_backend      import VLMResult

from ground_control.terrain_memory  import TerrainMemory
from ground_control.vlm_quality     import VLMQualityMonitor
from ground_control.fleet_monitor   import FleetMonitor
from ground_control.mission_planner import MissionPlanner, _classify_feature
from ground_control._embedder       import embed, embed_batch

import numpy as np
import cv2


# ── Mission scenario library ──────────────────────────────────────────────────
# Realistic diversity: different queries, terrain units, goal distances, outcomes

SCENARIOS = [
    # (query_text, terrain_unit, goal_dist_m, p_success_base)
    ("vesicular basalt rock formation",    "maaz",        8.0,  0.75),
    ("dark olivine-rich boulder",          "seitah",      6.0,  0.70),
    ("skull rock formation",               "maaz",        10.0, 0.65),
    ("polygon crack network",              "jezero_floor",5.0,  0.80),
    ("light-toned carbonate outcrop",      "delta_front", 12.0, 0.60),
    ("aeolian ripple crest",               "jezero_floor",4.0,  0.85),
    ("dark shadow region",                 "seitah",      7.0,  0.55),
    ("flat safe traversable ground",       "jezero_floor",3.0,  0.90),
    ("impact crater rim",                  "maaz",        15.0, 0.50),
    ("columnar jointed basalt",            "maaz",        9.0,  0.68),
    ("lag pavement pebbles",               "jezero_floor",4.0,  0.78),
    ("bright sulfate vein",                "delta_front", 7.0,  0.62),
    ("red dust-coated rock",               "maaz",        5.0,  0.82),
    ("cliff face with layered stratigraphy","delta_front",14.0, 0.45),
    ("smooth bedrock slab",                "seitah",      6.0,  0.72),
]


# ── Procedural image synthesis ────────────────────────────────────────────────

def _synth_image(
    rover_x:    float,
    rover_y:    float,
    rover_yaw:  float,
    goal_x:     float,
    goal_y:     float,
    rng:        random.Random,
    target_found_prob: float,
    seed: int = 0,
) -> np.ndarray:
    """
    Synthesise a 640×360 Mars terrain image appropriate for the current pose.
    Varies with yaw and position to give MockVLMBackend something meaningful.
    """
    H, W = 360, 640
    img  = np.zeros((H, W, 3), dtype=np.uint8)

    # Base: reddish-brown Martian surface
    img[:, :] = (55, 75, 155)

    # Sky (top 35%)
    sky_h = int(H * 0.35)
    for r in range(sky_h):
        frac = 1.0 - r / sky_h
        img[r, :, 0] = int(95  + frac * 55)
        img[r, :, 1] = int(65  + frac * 35)
        img[r, :, 2] = int(115 + frac * 55)

    # Terrain texture (seeded per scene for variety)
    np_rng = np.random.default_rng(abs(seed ^ int(rover_x * 100) ^ int(rover_y * 100)))
    for scale, amp in [(32, 30), (16, 15), (8, 8)]:
        nh, nw = H // scale + 2, W // scale + 2
        noise  = np_rng.integers(0, 255, (nh, nw), dtype=np.uint8)
        noise_up = cv2.resize(noise, (W, H), interpolation=cv2.INTER_CUBIC)
        for c in range(3):
            layer = (noise_up.astype(np.float32) / 255.0 * amp).astype(np.int16)
            img[:, :, c] = np.clip(
                img[:, :, c].astype(np.int16) + layer - amp // 2, 0, 255
            ).astype(np.uint8)

    # Place "target" object in image if rover is within range AND prob triggers
    dx   = goal_x - rover_x
    dy   = goal_y - rover_y
    dist = math.hypot(dx, dy)
    angle_to_goal  = math.atan2(dy, dx)
    heading_error  = (angle_to_goal - rover_yaw + math.pi) % (2 * math.pi) - math.pi

    in_fov = abs(heading_error) < 0.6           # ~35° camera half-FOV
    near   = dist < 6.0
    visible = in_fov and (rng.random() < target_found_prob)

    if visible:
        # Place a dark rock blob at image position proportional to angle + distance
        cx_frac = 0.5 - heading_error / 1.2 * 0.5
        cy_frac = 0.45 + (1.0 - min(dist / 12.0, 1.0)) * 0.35
        cx_px   = int(np.clip(cx_frac, 0.1, 0.9) * W)
        cy_px   = int(np.clip(cy_frac, 0.4, 0.85) * H)
        size    = max(15, int((1.0 - dist / 12.0) * 80))
        colour  = (int(35 + rng.randint(0, 20)),) * 3
        cv2.circle(img, (cx_px, cy_px), size, colour, -1)
        # Add bright specular highlight to make it stand out
        cv2.circle(img, (cx_px - size // 4, cy_px - size // 4),
                   max(3, size // 4), (200, 180, 160), -1)

    return img


# ── Single mission simulation ─────────────────────────────────────────────────

@dataclass
class SimResult:
    mission_id:   str
    query_text:   str
    terrain_unit: str
    outcome:      str
    elapsed_sec:  float
    n_obs:        int
    n_detections: int
    n_fp:         int
    final_pos:    tuple[float, float]
    observations: list[dict] = field(default_factory=list)


def run_one_mission(
    scenario:     tuple,
    planner:      MissionPlanner,
    vlm:          MockVLMBackend,
    rng:          random.Random,
    mission_idx:  int,
    dt:           float = 0.1,
) -> SimResult:
    query_text, terrain_unit, goal_dist, p_success = scenario

    # Random start and goal positions
    start_x = rng.uniform(-20.0, 20.0)
    start_y = rng.uniform(-20.0, 20.0)
    angle   = rng.uniform(0, 2 * math.pi)
    goal_x  = start_x + goal_dist * math.cos(angle)
    goal_y  = start_y + goal_dist * math.sin(angle)

    # Plan mission (queries MongoDB + VLMQualityMonitor)
    plan = planner.plan_mission(
        query_text=query_text,
        start_pos=(start_x, start_y),
        terrain_unit=terrain_unit,
    )
    mission_id = plan.mission_id

    # Override goal with seeded waypoint if available
    if plan.seeded_goal:
        goal_x, goal_y = plan.seeded_goal[0] + rng.gauss(0, 1.5), \
                         plan.seeded_goal[1] + rng.gauss(0, 1.5)

    # Build rover simulation
    rover      = DifferentialDriveModel(initial_pose=RoverPose(x=start_x, y=start_y))
    fsm        = AgentFSM(FSMConfig(
        min_confidence   = plan.recommended_confidence,
        timeout_sec      = plan.timeout_hint_sec,
        arrival_radius_m = 0.5,
    ))
    controller = PurePursuitController()
    fsm.goal_received(
        timeout_sec      = plan.timeout_hint_sec,
        arrival_radius_m = 0.5,
    )

    prev_state:         str    = State.SEARCHING.value
    goal_set_x:         float  = float("nan")
    goal_set_y:         float  = float("nan")
    observations:       list   = []
    false_positive_seqs: list  = []
    seed = mission_idx * 1000

    for step in range(int(plan.timeout_hint_sec / dt)):
        pose = rover.pose

        # Synthesise camera image
        effective_p = p_success * min(1.0, 2.0 / max(
            math.hypot(goal_x - pose.x, goal_y - pose.y), 0.5
        ))
        img = _synth_image(
            pose.x, pose.y, pose.yaw,
            goal_x, goal_y,
            rng, effective_p, seed + step,
        )

        # VLM inference
        vlm_result: VLMResult = vlm.query(img, query_text)

        # Update navigation goal
        if vlm_result.target_found and not math.isnan(goal_set_x):
            pass  # keep goal
        elif vlm_result.target_found:
            goal_set_x, goal_set_y = goal_x, goal_y

        dist = (
            math.hypot(goal_set_x - pose.x, goal_set_y - pose.y)
            if not math.isnan(goal_set_x) else float("nan")
        )

        # FSM tick
        status = fsm.update(
            dt=dt,
            target_found=vlm_result.target_found,
            confidence=vlm_result.confidence,
            distance_m=dist,
            description=vlm_result.description,
        )

        # Track FSM transitions (false positive detection)
        curr_state = status.state.value
        if prev_state == State.VERIFYING.value and curr_state == State.APPROACHING.value:
            false_positive_seqs.append(step)
            planner.record_fsm_transition(mission_id, prev_state, curr_state)
        prev_state = curr_state

        # Controller
        if status.state in (State.APPROACHING, State.VERIFYING) and not math.isnan(goal_set_x):
            ctrl = controller.compute(pose.x, pose.y, pose.yaw, goal_set_x, goal_set_y)
            rover.set_velocity(ctrl.linear, ctrl.angular)
        elif status.state == State.SEARCHING:
            rover.set_velocity(0.0, 0.35)
        else:
            rover.set_velocity(0.0, 0.0)

        rover.step(dt)

        # Build observation document
        obs = {
            "seq":               step,
            "timestamp":         datetime.now(timezone.utc) + timedelta(seconds=step * dt),
            "fsm_state":         status.state.value,
            "target_found":      vlm_result.target_found,
            "confidence":        float(vlm_result.confidence),
            "description":       vlm_result.description,
            "distance_to_goal_m":float(dist) if not math.isnan(dist) else -1.0,
            "rover_position":    {"x": pose.x, "y": pose.y},
            "inference_ms":      float(vlm_result.inference_ms),
            "query_text":        query_text,
            "vlm_backend":       "mock_opencv",
        }
        observations.append(obs)

        # Log every 5 ticks to planner (→ MongoDB + Elastic background writes)
        if step % 5 == 0:
            planner.log_tick(mission_id, obs)

        if fsm.is_terminal:
            break

    outcome      = "ARRIVED" if fsm.state == State.ARRIVED else (
        "TIMEOUT" if status.outcome and status.outcome.value == "TIMEOUT" else "CANCELLED"
    )
    elapsed      = step * dt
    final_pos    = (rover.pose.x, rover.pose.y)
    n_detections = sum(1 for o in observations if o["target_found"])

    # Close mission across all three systems
    planner.close_mission(mission_id, outcome, elapsed, final_pos)

    return SimResult(
        mission_id=mission_id,
        query_text=query_text,
        terrain_unit=terrain_unit,
        outcome=outcome,
        elapsed_sec=elapsed,
        n_obs=len(observations),
        n_detections=n_detections,
        n_fp=len(false_positive_seqs),
        final_pos=final_pos,
        observations=observations,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch Mars mission simulator")
    parser.add_argument("--n-missions",  type=int, default=50,
                        help="Number of missions to simulate (default: 50)")
    parser.add_argument("--use-gemini",  action="store_true",
                        help="Use real Gemini Vision instead of mock VLM "
                             "(requires GOOGLE_API_KEY, much slower)")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    print("=" * 60)
    print("  ARES Mars Terrain Intelligence — Batch Mission Runner")
    print(f"  Missions: {args.n_missions}   Seed: {args.seed}")
    print("=" * 60)

    # ── Connect to all three services ─────────────────────────────────────────
    print("\n[1/4] Connecting to telemetry services…")
    mem   = TerrainMemory()
    qual  = VLMQualityMonitor()
    fleet = FleetMonitor()
    planner = MissionPlanner(mem, qual, fleet,
                             vlm_backend_name="mock_opencv")

    # ── VLM backend ───────────────────────────────────────────────────────────
    if args.use_gemini:
        from mars_scout_perception.vlm_backend import GeminiVisionBackend
        vlm = GeminiVisionBackend()
        print(f"[2/4] VLM backend: {vlm.name}")
    else:
        vlm = MockVLMBackend(min_confidence=0.0, noise_std=0.02)
        print("[2/4] VLM backend: mock_opencv (fast, no API calls)")

    # ── Run missions ──────────────────────────────────────────────────────────
    print(f"\n[3/4] Running {args.n_missions} missions…\n")
    rng     = random.Random(args.seed)
    results: list[SimResult] = []
    t_start  = time.perf_counter()

    for i in range(args.n_missions):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        try:
            result = run_one_mission(
                scenario=scenario,
                planner=planner,
                vlm=vlm,
                rng=rng,
                mission_idx=i,
            )
            results.append(result)
            icon = "✓" if result.outcome == "ARRIVED" else "✗"
            print(f"  [{i+1:3d}/{args.n_missions}] {icon} {result.outcome:9s} "
                  f"{result.elapsed_sec:5.1f}s  "
                  f"det={result.n_detections:3d} fp={result.n_fp}  "
                  f"'{result.query_text[:35]}'")
        except Exception as exc:
            import traceback
            print(f"  [{i+1:3d}/{args.n_missions}] ERROR: {exc}")
            traceback.print_exc()
            continue

    elapsed_total = time.perf_counter() - t_start

    # ── Analytics report ──────────────────────────────────────────────────────
    print(f"\n[4/4] Analytics  ({elapsed_total:.1f}s for {len(results)} missions)\n")
    print("─" * 60)

    # Mission outcomes
    from collections import Counter
    outcomes = Counter(r.outcome for r in results)
    print("Outcomes:")
    for outcome, count in outcomes.most_common():
        pct = count / len(results) * 100
        print(f"  {outcome:12s}  {count:3d}  ({pct:.0f}%)")

    # MongoDB analytics
    print("\nMongoDB mission analytics:")
    try:
        analytics = mem.mission_analytics()
        for row in analytics:
            print(f"  {row['_id']:12s}  count={row['count']:3d}  "
                  f"avg_elapsed={row['avg_elapsed_sec']:.1f}s  "
                  f"avg_fp={row['avg_false_positives']:.2f}")
    except Exception as e:
        print(f"  (analytics error: {e})")

    # VLM precision/recall
    print("\nVLM precision by query type (from Arize calibration):")
    try:
        report = qual.get_precision_recall_report()
        for qt, stats in sorted(report.items(), key=lambda x: x[1]["precision"]):
            print(f"  [{stats['precision']:.2f} prec / {stats['recommended_conf']:.2f} rec_conf]  "
                  f"n={stats['n_detections']:4d}  fp={stats['n_false_positives']:3d}  "
                  f"'{qt[:40]}'")
    except Exception as e:
        print(f"  (report error: {e})")

    # Elasticsearch latency stats
    print("\nElasticsearch observation latency stats:")
    try:
        # Give the async writer a moment to flush
        time.sleep(2.0)
        latency = fleet.mission_latency_stats()
        pcts = latency.get("percentiles", {})
        print(f"  P50={pcts.get('50.0',0):.1f}ms  "
              f"P95={pcts.get('95.0',0):.1f}ms  "
              f"P99={pcts.get('99.0',0):.1f}ms")
    except Exception as e:
        print(f"  (latency error: {e})")

    # Terrain catalog size
    print("\nTerrain catalog:")
    try:
        from pymongo import MongoClient
        n_features = mem._features.count_documents({})
        n_missions  = mem._missions.count_documents({})
        print(f"  {n_features} features  {n_missions} missions in MongoDB")
    except Exception as e:
        print(f"  (catalog error: {e})")

    print("\n" + "─" * 60)
    print("Batch run complete.  Check:")
    endpoint = os.environ.get(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
    ).replace("/v1/traces", "")
    print(f"  Phoenix UI:  {endpoint}")
    print(f"  Kibana:      {os.environ.get('ELASTICSEARCH_URL','<ELASTICSEARCH_URL>')}"
          ":5601")
    print(f"  MongoDB:     Atlas UI → Browse Collections → mars_scout")
    print("─" * 60)

    # Graceful shutdown
    fleet.close()
    mem.close()


if __name__ == "__main__":
    main()
