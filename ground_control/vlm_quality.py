"""
VLM quality monitoring with Arize Phoenix.

What this does
--------------
Every Gemini Vision call is recorded as an OpenTelemetry span with the full
OpenInference semantic conventions so Phoenix understands it as an LLM call:

  span attributes
  ├─ input.value          → query_text
  ├─ output.value         → VLM description
  ├─ llm.model_name       → gemini/gemini-1.5-flash
  ├─ llm.output_messages  → description
  ├─ metadata.fsm_state   → SEARCHING | APPROACHING | VERIFYING
  ├─ metadata.confidence  → float [0,1]
  ├─ metadata.target_found→ bool
  ├─ metadata.distance_m  → float
  └─ metadata.mission_id  → uuid

Each NavigateToTarget action is a root Mission span containing all VLM
child spans.  FSM state transitions are logged as span events.

Post-mission evals (Phoenix run_evals)
--------------------------------------
After each mission we run two evaluators over all observations:

  HallucinationEvaluator
    query    = query_text
    response = vlm_description
    Grades whether the VLM described something real or confabulated.

  RelevanceEvaluator
    query    = query_text
    response = vlm_description
    Grades whether the description is actually about the query target.

  ConfidenceCalibrationEval  (custom, ground-truth from FSM)
    For each VERIFYING→APPROACHING regression (confirmed false positive),
    we tag that observation's VLM call as a precision failure.
    Over many missions, precision-by-confidence-bin gives a calibration
    curve: if the model says 0.8 conf but only 50% are correct, it's
    overconfident.

Adaptive threshold
------------------
get_recommended_confidence(query_text) returns a per-query-type
min_confidence, raised from the FSM default when the calibration curve
shows systematic overconfidence.

Environment
-----------
  PHOENIX_COLLECTOR_ENDPOINT   default http://localhost:6006/v1/traces
  ARIZE_API_KEY                if set, sends to Arize cloud instead
  ARIZE_SPACE_ID               required with ARIZE_API_KEY
  GOOGLE_API_KEY               used by Phoenix eval model (Gemini)
"""

from __future__ import annotations

import os
import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, Optional

# ── OpenTelemetry + OpenInference setup ───────────────────────────────────────

def _build_tracer_provider():
    """
    Returns a configured TracerProvider pointing at Phoenix or Arize cloud.
    Raises RuntimeError if neither endpoint is reachable / configured.
    """
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError as exc:
        raise ImportError(
            "opentelemetry packages missing. Install:\n"
            "  pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
        ) from exc

    arize_key  = os.environ.get("ARIZE_API_KEY", "").strip()
    arize_space = os.environ.get("ARIZE_SPACE_ID", "").strip()

    if arize_key and arize_space:
        endpoint = "https://otlp.arize.com/v1/traces"
        headers  = {
            "Authorization": f"Bearer {arize_key}",
            "space-id":      arize_space,
        }
        print(f"[VLMQuality] Sending traces → Arize cloud (space {arize_space[:8]}…)")
    else:
        endpoint = os.environ.get(
            "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
        )
        headers  = {}
        print(f"[VLMQuality] Sending traces → Phoenix at {endpoint}")

    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


# ── Calibration tracking ──────────────────────────────────────────────────────

@dataclass
class _QueryStats:
    """Per-query-type running statistics for adaptive threshold."""
    n_detections:    int   = 0
    n_false_positives: int = 0
    # Confidence histogram: 10 bins × [0.0,0.1), [0.1,0.2), …, [0.9,1.0]
    conf_bins:       list[int]  = field(default_factory=lambda: [0] * 10)
    tp_bins:         list[int]  = field(default_factory=lambda: [0] * 10)  # true positives

    @property
    def precision(self) -> float:
        if self.n_detections == 0:
            return 1.0
        return (self.n_detections - self.n_false_positives) / self.n_detections

    def calibration_curve(self) -> list[tuple[float, float]]:
        """Returns (confidence_bin_centre, empirical_precision) pairs."""
        result = []
        for i in range(10):
            if self.conf_bins[i] == 0:
                continue
            centre    = i * 0.1 + 0.05
            precision = self.tp_bins[i] / self.conf_bins[i]
            result.append((centre, precision))
        return result

    def recommended_min_confidence(self, base: float = 0.45) -> float:
        """
        Raise min_confidence if precision < 0.65 to suppress false positives.
        Maximum adjustment: +0.20 (cap at 0.80).
        """
        p = self.precision
        if self.n_detections < 10:
            return base   # not enough data yet
        if p >= 0.80:
            return base
        if p >= 0.65:
            return min(0.80, base + 0.05)
        if p >= 0.50:
            return min(0.80, base + 0.10)
        return min(0.80, base + 0.20)


# ── VLMQualityMonitor ─────────────────────────────────────────────────────────

class VLMQualityMonitor:
    """
    Arize Phoenix integration for VLM call tracing and post-mission evaluation.

    Usage
    -----
    monitor = VLMQualityMonitor()

    with monitor.mission_trace(mission_id, query_text) as ctx:
        ...
        with monitor.vlm_span(ctx, obs) as _:
            result = vlm_backend.query(image, query_text)
        monitor.record_fsm_event(ctx, "SEARCHING", "APPROACHING")
        ...

    monitor.run_post_mission_evals(mission_id, observations, outcome,
                                   false_positive_seqs)
    """

    BASE_MIN_CONFIDENCE = 0.45

    def __init__(self):
        self._provider = _build_tracer_provider()

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            trace.set_tracer_provider(self._provider)
            self._tracer = trace.get_tracer("mars.scout.vlm")
        except ImportError as exc:
            raise ImportError(
                "opentelemetry-api missing. Install: pip install opentelemetry-api"
            ) from exc

        self._stats:  dict[str, _QueryStats] = {}  # query_text → stats
        self._lock   = threading.Lock()

    # ── Mission root span ─────────────────────────────────────────────────────

    @contextmanager
    def mission_trace(
        self,
        mission_id: str,
        query_text: str,
    ) -> Iterator[dict]:
        """
        Context manager that wraps the entire NavigateToTarget execution.
        Yields a context dict that child spans and event loggers consume.
        """
        from opentelemetry import trace
        with self._tracer.start_as_current_span(
            f"mission:{query_text[:40]}",
            attributes={
                "mission.id":          mission_id,
                "mission.query":       query_text,
                "session.id":          mission_id,
            }
        ) as root_span:
            ctx = {
                "root_span":  root_span,
                "mission_id": mission_id,
                "query_text": query_text,
                "tracer":     self._tracer,
            }
            try:
                yield ctx
            except Exception:
                root_span.set_status(
                    trace.StatusCode.ERROR, "Mission raised exception"
                )
                raise
            finally:
                pass  # span closes automatically

    # ── VLM call child span ───────────────────────────────────────────────────

    @contextmanager
    def vlm_span(self, ctx: dict, obs: dict) -> Iterator[None]:
        """
        Context manager for one VLM inference call.
        obs keys used: fsm_state, confidence, target_found, description,
                       distance_to_goal_m, inference_ms, seq
        """
        from openinference.semconv.trace import SpanAttributes
        attrs = {
            SpanAttributes.LLM_MODEL_NAME:    ctx.get("vlm_backend", "gemini/gemini-1.5-flash"),
            SpanAttributes.INPUT_VALUE:        ctx["query_text"],
            SpanAttributes.OUTPUT_VALUE:       obs.get("description", ""),
            "metadata.mission_id":             ctx["mission_id"],
            "metadata.seq":                    obs.get("seq", -1),
            "metadata.fsm_state":              obs.get("fsm_state", ""),
            "metadata.confidence":             float(obs.get("confidence", 0.0)),
            "metadata.target_found":           bool(obs.get("target_found", False)),
            "metadata.distance_m":             float(obs.get("distance_to_goal_m", -1.0)),
            "metadata.inference_ms":           float(obs.get("inference_ms", 0.0)),
        }
        with ctx["tracer"].start_as_current_span("vlm.query", attributes=attrs):
            yield

    # ── FSM event logging ─────────────────────────────────────────────────────

    def record_fsm_event(
        self,
        ctx:        dict,
        from_state: str,
        to_state:   str,
    ) -> None:
        """Log a state transition as an event on the root mission span."""
        ctx["root_span"].add_event(
            "fsm.transition",
            attributes={
                "from_state": from_state,
                "to_state":   to_state,
                "is_regression": (
                    from_state == "VERIFYING" and to_state == "APPROACHING"
                ),
            }
        )

    # ── Post-mission evals ────────────────────────────────────────────────────

    def run_post_mission_evals(
        self,
        mission_id:           str,
        observations:         list[dict],
        outcome:              str,
        false_positive_seqs:  list[int],   # seq numbers of confirmed false positives
        query_text:           str,
    ) -> dict:
        """
        Run Phoenix evaluators on all observations from this mission.

        Returns a summary dict with keys:
          hallucination_score, relevance_score, precision, calibration_curve,
          recommended_min_confidence
        """
        detections = [o for o in observations if o.get("target_found")]

        # ── Update calibration stats ───────────────────────────────────────────
        fp_seq_set = set(false_positive_seqs)
        with self._lock:
            stats = self._stats.setdefault(query_text, _QueryStats())
            for obs in detections:
                stats.n_detections += 1
                conf = float(obs.get("confidence", 0.0))
                bin_i = min(int(conf * 10), 9)
                stats.conf_bins[bin_i] += 1
                if obs["seq"] not in fp_seq_set:
                    stats.tp_bins[bin_i] += 1
            stats.n_false_positives += len(false_positive_seqs)
            calibration = stats.calibration_curve()
            precision   = stats.precision
            rec_conf    = stats.recommended_min_confidence(self.BASE_MIN_CONFIDENCE)

        # ── Phoenix LLM evals (if enough detections) ─────────────────────────
        hallucination_score = None
        relevance_score     = None

        if len(detections) >= 3:
            try:
                hallucination_score, relevance_score = self._run_phoenix_evals(
                    mission_id, query_text, detections, fp_seq_set
                )
            except Exception as exc:
                print(f"[VLMQuality] Phoenix evals failed (non-fatal): {exc}")

        summary = {
            "mission_id":                 mission_id,
            "outcome":                    outcome,
            "n_detections":               len(detections),
            "n_false_positives":          len(false_positive_seqs),
            "precision":                  round(precision, 3),
            "calibration_curve":          calibration,
            "recommended_min_confidence": round(rec_conf, 3),
            "hallucination_score":        hallucination_score,
            "relevance_score":            relevance_score,
        }
        return summary

    def _run_phoenix_evals(
        self,
        mission_id:  str,
        query_text:  str,
        detections:  list[dict],
        fp_seq_set:  set[int],
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Build a DataFrame and run Phoenix's built-in evaluators.
        Returns (avg_hallucination_score, avg_relevance_score).
        """
        try:
            import pandas as pd
            from phoenix.evals import (
                HallucinationEvaluator,
                RelevanceEvaluator,
                run_evals,
                GeminiModel,
            )
        except ImportError as exc:
            raise ImportError(
                "arize-phoenix and pandas required for evals.\n"
                "Install: pip install arize-phoenix pandas"
            ) from exc

        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY required for Phoenix LLM evals."
            )

        eval_model = GeminiModel(
            model="gemini-1.5-flash",
            api_key=api_key,
        )

        df = pd.DataFrame([
            {
                "input":     query_text,
                "output":    obs.get("description", ""),
                "context":   (
                    f"Mars rover camera image. "
                    f"FSM state: {obs.get('fsm_state','?')}. "
                    f"Reported confidence: {obs.get('confidence',0):.2f}. "
                    f"Distance to goal: {obs.get('distance_to_goal_m',-1):.1f}m."
                ),
                "label":     "hallucination" if obs["seq"] in fp_seq_set else "factual",
                "reference": (
                    f"The rover is looking for: {query_text}. "
                    f"The target was {'NOT actually present' if obs['seq'] in fp_seq_set else 'confirmed present'}."
                ),
            }
            for obs in detections
        ])

        evals = run_evals(
            dataframe=df,
            evaluators=[
                HallucinationEvaluator(eval_model),
                RelevanceEvaluator(eval_model),
            ],
            provide_explanation=True,
        )

        hall_df = evals[0]
        rel_df  = evals[1]

        # score column is 1.0=correct, 0.0=wrong
        hall_score = float(hall_df["score"].mean()) if "score" in hall_df.columns else None
        rel_score  = float(rel_df["score"].mean())  if "score" in rel_df.columns  else None
        return hall_score, rel_score

    # ── Adaptive threshold ────────────────────────────────────────────────────

    def get_recommended_confidence(self, query_text: str) -> float:
        """
        Return the adaptive min_confidence for this query type.
        Falls back to the global default if we have < 10 observations.
        """
        with self._lock:
            stats = self._stats.get(query_text)
            if stats is None:
                return self.BASE_MIN_CONFIDENCE
            return stats.recommended_min_confidence(self.BASE_MIN_CONFIDENCE)

    def get_precision_recall_report(self) -> dict[str, dict]:
        """Return precision stats for all observed query types."""
        with self._lock:
            return {
                qt: {
                    "precision":           round(s.precision, 3),
                    "n_detections":        s.n_detections,
                    "n_false_positives":   s.n_false_positives,
                    "calibration_curve":   s.calibration_curve(),
                    "recommended_conf":    round(
                        s.recommended_min_confidence(self.BASE_MIN_CONFIDENCE), 3
                    ),
                }
                for qt, s in self._stats.items()
            }
