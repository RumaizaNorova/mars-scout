"""
MissionPlanner — orchestrates all three telemetry systems.

Before a mission starts:
  1. query TerrainMemory for known features nearby the start position
  2. find similar past missions → seed initial waypoint and timeout
  3. get adaptive min_confidence from VLMQualityMonitor

During a mission:
  observations are logged to all three systems every N ticks via
  the non-blocking TelemetryQueue (see below)

After a mission ends:
  1. add confirmed detections to terrain_memory as terrain features
  2. run post-mission Arize evals
  3. index mission summary to Elasticsearch
  4. return updated FSMConfig for next mission

The ROS2 agent_node calls this as:

    plan = planner.plan_mission(query_text, start_pos, terrain_unit)
    # set FSM config from plan
    ...
    # during mission loop (every 5 ticks):
    planner.log_tick(plan.mission_id, obs)
    # on terminal state:
    planner.close_mission(plan.mission_id, outcome, observations, ...)
"""

from __future__ import annotations

import threading
import queue
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class MissionPlan:
    mission_id:              str
    query_text:              str
    recommended_confidence:  float          # from VLMQualityMonitor calibration
    seeded_goal:             Optional[tuple[float, float]]  # from similar past mission
    timeout_hint_sec:        float          # based on past mission durations
    known_features_nearby:   list[dict]     # from TerrainMemory.query_nearby()
    terrain_unit:            str


class MissionPlanner:
    """
    Central orchestrator.  One instance shared across the system lifetime.
    Thread-safe: all methods safe to call from the ROS2 executor thread.
    """

    # Only log every Nth tick to avoid hammering the databases
    LOG_EVERY_N_TICKS = 5

    def __init__(
        self,
        terrain_memory,    # TerrainMemory instance
        vlm_quality,       # VLMQualityMonitor instance
        fleet_monitor,     # FleetMonitor instance
        vlm_backend_name: str = "gemini/gemini-1.5-flash",
    ):
        self._mem   = terrain_memory
        self._qual  = vlm_quality
        self._fleet = fleet_monitor
        self._vlm_backend = vlm_backend_name

        # Per-mission bookkeeping (keyed by mission_id)
        self._active: dict[str, _MissionState] = {}
        self._lock   = threading.Lock()

        # Non-blocking telemetry write queue
        self._tq: queue.Queue = queue.Queue(maxsize=1000)
        self._tq_thread = threading.Thread(
            target=self._tq_drain, daemon=True, name="telemetry-queue"
        )
        self._tq_thread.start()

    # ── Pre-mission ───────────────────────────────────────────────────────────

    def plan_mission(
        self,
        query_text:   str,
        start_pos:    tuple[float, float],
        terrain_unit: str = "unknown",
    ) -> MissionPlan:
        """
        Query all three systems and return a MissionPlan.
        Called by agent_node before accepting a NavigateToTarget goal.
        """
        mission_id = str(uuid.uuid4())

        # 1. Known features nearby
        nearby = self._mem.query_nearby(start_pos[0], start_pos[1], radius_m=60.0)

        # 2. Similar past missions → seeded waypoint + timeout hint
        similar = self._mem.find_similar_missions(query_text, terrain_unit)
        seeded_goal:   Optional[tuple[float, float]] = None
        timeout_hint:  float                         = 120.0

        if similar:
            best = similar[0]
            fp   = best.get("final_position")
            if fp:
                seeded_goal = (fp["x"], fp["y"])
                print(f"[MissionPlanner] Seeding goal from mission "
                      f"{best['mission_id'][:8]} → ({fp['x']:.1f}, {fp['y']:.1f})")

            # Use 1.2× the average elapsed time of similar missions as timeout hint
            elapsed_vals = [m["elapsed_sec"] for m in similar if m.get("elapsed_sec")]
            if elapsed_vals:
                timeout_hint = min(300.0, max(60.0, 1.2 * sum(elapsed_vals) / len(elapsed_vals)))

        # 3. Adaptive confidence threshold
        rec_conf = self._qual.get_recommended_confidence(query_text)
        if rec_conf > 0.45:
            print(f"[MissionPlanner] Raising min_confidence "
                  f"→ {rec_conf:.2f} (calibration from past missions)")

        # 4. Open in MongoDB
        self._mem.open_mission(
            query_text=query_text,
            start_pos=start_pos,
            arrival_radius_m=0.5,
            vlm_backend=self._vlm_backend,
            terrain_unit=terrain_unit,
            mission_id=mission_id,
        )

        # 5. Register active mission state
        with self._lock:
            self._active[mission_id] = _MissionState(
                mission_id=mission_id,
                query_text=query_text,
            )

        return MissionPlan(
            mission_id=mission_id,
            query_text=query_text,
            recommended_confidence=rec_conf,
            seeded_goal=seeded_goal,
            timeout_hint_sec=timeout_hint,
            known_features_nearby=nearby,
            terrain_unit=terrain_unit,
        )

    # ── During mission ────────────────────────────────────────────────────────

    def log_tick(self, mission_id: str, obs: dict) -> None:
        """
        Called every control tick from agent_node.
        Internally throttled: only writes to databases every LOG_EVERY_N_TICKS.
        Non-blocking: enqueues to background thread.
        """
        with self._lock:
            state = self._active.get(mission_id)
            if state is None:
                return
            state.tick_count  += 1
            state.observations.append(obs)
            if obs.get("target_found"):
                state.embedding_queue.append(obs)  # embed later in batch

            should_write = (state.tick_count % self.LOG_EVERY_N_TICKS == 0)

        if should_write:
            self._tq.put_nowait(("obs", mission_id, obs))

    def record_fsm_transition(
        self,
        mission_id: str,
        from_state: str,
        to_state:   str,
    ) -> None:
        """
        Call when the FSM changes state.
        VERIFYING→APPROACHING = confirmed false positive → logged to MongoDB.
        """
        is_fp = (from_state == "VERIFYING" and to_state == "APPROACHING")
        if is_fp:
            with self._lock:
                state = self._active.get(mission_id)
                if state:
                    state.false_positive_seqs.append(
                        state.observations[-1]["seq"] if state.observations else -1
                    )
            self._mem.log_false_positive(mission_id)

        self._qual.record_fsm_event(
            {"root_span": _noop_span(), "mission_id": mission_id,
             "query_text": "", "tracer": _noop_tracer()},
            from_state, to_state,
        )

    # ── Post-mission ──────────────────────────────────────────────────────────

    def close_mission(
        self,
        mission_id:     str,
        outcome:        str,
        elapsed_sec:    float,
        final_position: tuple[float, float],
    ) -> dict:
        """
        Called when agent_node reaches a terminal FSM state.
        Returns eval summary dict.
        Runs post-mission work in a background thread — returns immediately.
        """
        with self._lock:
            state = self._active.pop(mission_id, None)
        if state is None:
            return {}

        # Close MongoDB mission document synchronously (fast write)
        self._mem.close_mission(mission_id, outcome, elapsed_sec, final_position)

        # Post-mission work in background (evals are slow — Gemini API calls)
        t = threading.Thread(
            target=self._post_mission_work,
            args=(mission_id, state, outcome, elapsed_sec, final_position),
            daemon=True,
            name=f"post-mission-{mission_id[:8]}",
        )
        t.start()

        return {"mission_id": mission_id, "outcome": outcome, "status": "post_mission_started"}

    def _post_mission_work(
        self,
        mission_id:     str,
        state:          "_MissionState",
        outcome:        str,
        elapsed_sec:    float,
        final_position: tuple[float, float],
    ) -> None:
        """Runs in background thread after mission ends."""
        from ground_control._embedder import embed, embed_batch

        obs_list = state.observations
        query    = state.query_text

        # 1. Batch-embed all detection descriptions
        detections = [o for o in obs_list if o.get("target_found")]
        if detections:
            descs      = [o.get("description", "") for o in detections]
            embeddings = embed_batch(descs)
        else:
            embeddings = []

        # 2. Add confirmed detections to terrain feature catalog
        #    (only use observations that are NOT false positives)
        fp_set = set(state.false_positive_seqs)
        for obs, emb in zip(detections, embeddings):
            if obs["seq"] in fp_set:
                continue
            pos  = obs.get("rover_position", {})
            x, y = pos.get("x", 0.0), pos.get("y", 0.0)
            is_new = self._mem.add_feature(
                mission_id=mission_id,
                x=x, y=y,
                feature_type=_classify_feature(query),
                description=obs.get("description", ""),
                confidence=float(obs.get("confidence", 0.0)),
                embedding=emb,
            )
            # Index to Elasticsearch (terrain features index)
            self._fleet.index_terrain_feature(
                mission_id=mission_id,
                x=x, y=y,
                feature={"feature_type": _classify_feature(query),
                         "description": obs.get("description", ""),
                         "confidence":  obs.get("confidence", 0.0)},
                embedding=emb,
            )

        # 3. Run Arize evals
        eval_summary = self._qual.run_post_mission_evals(
            mission_id=mission_id,
            observations=obs_list,
            outcome=outcome,
            false_positive_seqs=state.false_positive_seqs,
            query_text=query,
        )

        # 4. Index mission summary to Elasticsearch
        self._fleet.index_mission_summary({
            "mission_id":        mission_id,
            "query_text":        query,
            "terrain_unit":      state.terrain_unit if hasattr(state, "terrain_unit") else "",
            "outcome":           outcome,
            "elapsed_sec":       elapsed_sec,
            "n_observations":    len(obs_list),
            "n_detections":      len(detections),
            "n_false_positives": len(state.false_positive_seqs),
            "started_at":        state.started_at,
            "completed_at":      datetime.now(timezone.utc),
            "final_position":    {"x": final_position[0], "y": final_position[1]},
            "precision":         eval_summary.get("precision", 0.0),
            "hallucination_score": eval_summary.get("hallucination_score"),
        })

        print(f"[MissionPlanner] Post-mission complete for {mission_id[:8]}: "
              f"outcome={outcome}  precision={eval_summary.get('precision',0):.2f}  "
              f"rec_conf={eval_summary.get('recommended_min_confidence',0.45):.2f}")

    # ── Background telemetry drain ────────────────────────────────────────────

    def _tq_drain(self) -> None:
        """Drain the telemetry queue: write observations to MongoDB + Elastic."""
        while True:
            try:
                item = self._tq.get(timeout=1.0)
                tag, mission_id, obs = item
                if tag == "obs":
                    try:
                        self._mem.log_observation(mission_id, obs)
                        self._fleet.index_observation(mission_id, obs)
                    except Exception as exc:
                        print(f"[MissionPlanner] Telemetry write error: {exc}")
            except queue.Empty:
                pass


# ── Internal state per active mission ─────────────────────────────────────────

@dataclass
class _MissionState:
    mission_id:           str
    query_text:           str
    tick_count:           int       = 0
    observations:         list      = field(default_factory=list)
    false_positive_seqs:  list[int] = field(default_factory=list)
    embedding_queue:      list      = field(default_factory=list)
    started_at:           datetime  = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classify_feature(query_text: str) -> str:
    """Rough feature type from query keywords."""
    q = query_text.lower()
    if any(w in q for w in ["rock", "boulder", "formation", "outcrop", "skull"]):
        return "rock"
    if any(w in q for w in ["crater", "rim", "impact"]):
        return "crater"
    if any(w in q for w in ["crack", "polygon", "fracture"]):
        return "crack"
    if any(w in q for w in ["ripple", "dune", "sand"]):
        return "ripple"
    if any(w in q for w in ["flat", "safe", "smooth", "ground"]):
        return "flat_terrain"
    return "unknown"


class _NoopSpan:
    def add_event(self, *a, **kw): pass
    def set_status(self, *a, **kw): pass


class _NoopTracer:
    from contextlib import contextmanager
    @contextmanager
    def start_as_current_span(self, *a, **kw):
        yield _NoopSpan()


def _noop_span():  return _NoopSpan()
def _noop_tracer(): return _NoopTracer()
