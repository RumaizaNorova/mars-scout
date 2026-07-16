"""
Fleet-wide observability with Elasticsearch + Elastic APM.

What this does
--------------
Every observation, terrain feature, and mission summary from every rover
run is indexed into Elasticsearch.  Two indices:

  rover-observations
    One document per VLM call.  Includes:
      - rover_position as geo_point  → spatial queries
      - description_embedding as dense_vector(384) → KNN semantic search
      - All FSM metadata

  rover-terrain-features
    One document per confirmed feature.  Includes:
      - location as geo_point
      - description_embedding for KNN

  rover-missions
    One document per completed mission.  Summary statistics.

Elastic APM
-----------
APM agent is installed on perception_node and agent_node Python processes.
Every ROS2 callback that calls into Gemini is wrapped in an APM transaction.
This gives us:
  - Gemini inference latency (P50/P95/P99) in APM
  - Distribution traces: perception → geometry → control per tick
  - Automatic anomaly detection via Elastic ML jobs (latency spikes)

Kibana dashboard
----------------
A saved Kibana dashboard JSON is committed to the repo at:
  kibana/mars_scout_dashboard.ndjson

It contains:
  - Geo map of all terrain features (rover-terrain-features)
  - VLM confidence timeline per mission
  - Mission success rate pie chart
  - Inference latency histogram from APM

Import it: Stack Management → Saved Objects → Import → mars_scout_dashboard.ndjson

Async write queue
-----------------
All Elasticsearch writes go through a background thread queue so the
ROS2 control loop is never blocked waiting for network I/O.
Queue capacity: 500 documents.  If the queue is full, oldest items are
dropped and a WARNING is logged — the rover keeps running.

Environment
-----------
  ELASTICSEARCH_URL        required (e.g. https://my-cluster.es.io:443)
  ELASTICSEARCH_API_KEY    required (Base64 encoded id:api_key)
  ELASTIC_APM_SERVER_URL   optional (defaults to http://localhost:8200)
  ELASTIC_APM_SECRET_TOKEN optional
"""

from __future__ import annotations

import os
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from ground_control._embedder import DIM

# ── Index names ───────────────────────────────────────────────────────────────

IDX_OBS      = "rover-observations"
IDX_FEATURES = "rover-terrain-features"
IDX_MISSIONS = "rover-missions"

# ── Index mappings ────────────────────────────────────────────────────────────

_OBS_MAPPING = {
    "mappings": {
        "properties": {
            "mission_id":          {"type": "keyword"},
            "seq":                 {"type": "integer"},
            "timestamp":           {"type": "date"},
            "fsm_state":           {"type": "keyword"},
            "query_text":          {"type": "text", "fields": {"kw": {"type": "keyword"}}},
            "target_found":        {"type": "boolean"},
            "confidence":          {"type": "float"},
            "description":         {"type": "text"},
            "distance_to_goal_m":  {"type": "float"},
            "rover_position":      {"type": "geo_point"},
            "inference_ms":        {"type": "float"},
            "vlm_backend":         {"type": "keyword"},
            "knn_similarity_score":{"type": "float"},   # cosine similarity of terrain memory KNN match
            "description_embedding": {
                "type":       "dense_vector",
                "dims":       DIM,
                "index":      True,
                "similarity": "cosine",
            },
        }
    }
}

_FEAT_MAPPING = {
    "mappings": {
        "properties": {
            "mission_id":       {"type": "keyword"},
            "feature_type":     {"type": "keyword"},
            "feature_class":    {"type": "keyword"},    # boulder / cobble / ripple / crater / outcrop
            "geological_unit":  {"type": "keyword"},    # maaz / seitah / delta_front / jezero_floor
            "location":         {"type": "geo_point"},
            "description":      {"type": "text"},
            "confidence":       {"type": "float"},
            "detected_at":      {"type": "date"},
            "discovered_at":    {"type": "date"},       # alias for Kibana time filter
            "observation_count":{"type": "integer"},
            "description_embedding": {
                "type":       "dense_vector",
                "dims":       DIM,
                "index":      True,
                "similarity": "cosine",
            },
        }
    }
}

_MISSION_MAPPING = {
    "mappings": {
        "properties": {
            "mission_id":               {"type": "keyword"},
            "query_text":               {"type": "text",  "fields": {"kw": {"type": "keyword"}}},
            "terrain_unit":             {"type": "keyword"},
            "outcome":                  {"type": "keyword"},
            # dashboard aliases
            "success":                  {"type": "boolean"},       # outcome == "success"
            "duration_sec":             {"type": "float"},         # alias for elapsed_sec
            "end_time":                 {"type": "date"},          # alias for completed_at
            "path_efficiency":          {"type": "float"},         # actual_dist / straight_line_dist
            "final_distance_to_goal_m": {"type": "float"},         # metres at mission end
            "false_positive_rate":      {"type": "float"},         # n_false_positives / n_detections
            # core fields
            "elapsed_sec":              {"type": "float"},
            "n_observations":           {"type": "integer"},
            "n_detections":             {"type": "integer"},
            "n_false_positives":        {"type": "integer"},
            "started_at":               {"type": "date"},
            "completed_at":             {"type": "date"},
            "start_position":           {"type": "geo_point"},
            "final_position":           {"type": "geo_point"},
            "vlm_backend":              {"type": "keyword"},
            "precision":                {"type": "float"},
            "hallucination_score":      {"type": "float"},
        }
    }
}


# ── FleetMonitor ──────────────────────────────────────────────────────────────

class FleetMonitor:
    """
    Non-blocking Elasticsearch indexer + Elastic APM wrapper.

    All index_* methods enqueue documents to a background writer thread.
    If Elasticsearch is slow, documents are dropped (oldest-first) rather
    than blocking the caller.  Dropped count is logged.
    """

    QUEUE_CAPACITY = 500

    def __init__(self):
        self._es   = _build_es_client()
        self._apm  = _build_apm_client()   # may be None if APM not configured
        self._q: queue.Queue = queue.Queue(maxsize=self.QUEUE_CAPACITY)
        self._dropped = 0
        self._lock    = threading.Lock()

        # Ensure indices exist with correct mappings
        _ensure_index(self._es, IDX_OBS,      _OBS_MAPPING)
        _ensure_index(self._es, IDX_FEATURES, _FEAT_MAPPING)
        _ensure_index(self._es, IDX_MISSIONS, _MISSION_MAPPING)

        # Background writer thread
        self._stop  = threading.Event()
        self._writer = threading.Thread(
            target=self._write_loop,
            daemon=True,
            name="elastic-writer",
        )
        self._writer.start()

        print(f"[FleetMonitor] Elasticsearch connected.  "
              f"APM {'active' if self._apm else 'disabled'}.")

    # ── Public index methods ──────────────────────────────────────────────────

    def index_observation(
        self,
        mission_id:  str,
        obs:         dict,
        embedding:   Optional[list[float]] = None,
    ) -> None:
        """
        Index one VLM observation.
        rover_position must be {"x": float, "y": float} in obs.
        """
        pos = obs.get("rover_position", {})
        doc = {
            "mission_id":          mission_id,
            "seq":                 obs.get("seq", -1),
            "timestamp":           _to_iso(obs.get("timestamp")),
            "fsm_state":           obs.get("fsm_state", ""),
            "query_text":          obs.get("query_text", ""),
            "target_found":        bool(obs.get("target_found", False)),
            "confidence":          float(obs.get("confidence", 0.0)),
            "description":         obs.get("description", ""),
            "distance_to_goal_m":  float(obs.get("distance_to_goal_m", -1.0)),
            "rover_position":      f"{pos.get('y', 0.0)},{pos.get('x', 0.0)}",
            "inference_ms":        float(obs.get("inference_ms", 0.0)),
            "vlm_backend":         obs.get("vlm_backend", ""),
            "knn_similarity_score": float(obs.get("knn_similarity_score", 0.0)),
        }
        if embedding is not None:
            doc["description_embedding"] = embedding
        self._enqueue(IDX_OBS, doc)

    def index_terrain_feature(
        self,
        mission_id: str,
        x:          float,
        y:          float,
        feature:    dict,
        embedding:  Optional[list[float]] = None,
    ) -> None:
        now_iso = _to_iso(datetime.now(timezone.utc))
        doc = {
            "mission_id":        mission_id,
            "feature_type":      feature.get("feature_type", "unknown"),
            "feature_class":     feature.get("feature_class", feature.get("feature_type", "unknown")),
            "geological_unit":   feature.get("geological_unit", "unknown"),
            "location":          f"{y},{x}",   # geo_point: lat,lon
            "description":       feature.get("description", ""),
            "confidence":        float(feature.get("confidence", 0.0)),
            "detected_at":       now_iso,
            "discovered_at":     now_iso,
            "observation_count": feature.get("observation_count", 1),
        }
        if embedding is not None:
            doc["description_embedding"] = embedding
        self._enqueue(IDX_FEATURES, doc)

    def index_mission_summary(self, mission: dict) -> None:
        sp = mission.get("start_position", {})
        fp = mission.get("final_position",  {})
        outcome      = mission.get("outcome", "")
        elapsed_sec  = float(mission.get("elapsed_sec") or 0.0)
        n_det        = int(mission.get("n_detections", 0))
        n_fp         = int(mission.get("n_false_positives", 0))
        fp_rate      = n_fp / n_det if n_det > 0 else 0.0
        completed_at = mission.get("completed_at")

        # path_efficiency: actual dist / straight-line dist (1.0 = optimal)
        path_eff = mission.get("path_efficiency")
        if path_eff is None:
            actual   = float(mission.get("actual_path_length_m") or 0.0)
            straight = float(mission.get("straight_line_dist_m") or 0.0)
            path_eff = actual / straight if straight > 0.5 else 1.0

        doc = {
            "mission_id":               mission.get("mission_id", ""),
            "query_text":               mission.get("query_text", ""),
            "terrain_unit":             mission.get("terrain_unit", ""),
            "outcome":                  outcome,
            "success":                  outcome == "success",
            "duration_sec":             elapsed_sec,
            "end_time":                 _to_iso(completed_at),
            "path_efficiency":          float(path_eff),
            "final_distance_to_goal_m": float(mission.get("final_distance_to_goal_m") or 0.0),
            "false_positive_rate":      fp_rate,
            "elapsed_sec":              elapsed_sec,
            "n_observations":           int(mission.get("n_observations", 0)),
            "n_detections":             n_det,
            "n_false_positives":        n_fp,
            "started_at":               _to_iso(mission.get("started_at")),
            "completed_at":             _to_iso(completed_at),
            "start_position":           f"{sp.get('y',0)},{sp.get('x',0)}",
            "final_position":           f"{fp.get('y',0)},{fp.get('x',0)}" if fp else None,
            "vlm_backend":              mission.get("vlm_backend", ""),
            "precision":                float(mission.get("precision") or 0.0),
            "hallucination_score":      mission.get("hallucination_score"),
        }
        self._enqueue(IDX_MISSIONS, doc)

    # ── Search methods ────────────────────────────────────────────────────────

    def search_similar_observations(
        self,
        embedding: list[float],
        k:         int = 5,
        filter_found: bool = True,
    ) -> list[dict]:
        """KNN search on observation embeddings — find past similar detections."""
        knn = {
            "field":          "description_embedding",
            "query_vector":   embedding,
            "k":              k,
            "num_candidates": k * 10,
        }
        body: dict = {"knn": knn, "size": k}
        if filter_found:
            body["query"] = {"term": {"target_found": True}}

        resp = self._es.search(index=IDX_OBS, body=body)
        return [h["_source"] for h in resp["hits"]["hits"]]

    def search_features_near(
        self,
        x:        float,
        y:        float,
        radius_m: float = 10.0,
        limit:    int   = 20,
    ) -> list[dict]:
        """Geo-distance query: terrain features within radius_m of (x, y)."""
        body = {
            "query": {
                "geo_distance": {
                    "distance":  f"{radius_m}m",
                    "location":  f"{y},{x}",
                }
            },
            "size": limit,
            "sort": [{"_geo_distance": {"location": f"{y},{x}", "order": "asc"}}],
        }
        resp = self._es.search(index=IDX_FEATURES, body=body)
        return [h["_source"] for h in resp["hits"]["hits"]]

    def get_mission_timeline(self, mission_id: str) -> list[dict]:
        """All observations for a mission, sorted by seq."""
        body = {
            "query": {"term": {"mission_id": mission_id}},
            "sort":  [{"seq": {"order": "asc"}}],
            "size":  2000,
        }
        resp = self._es.search(index=IDX_OBS, body=body)
        return [h["_source"] for h in resp["hits"]["hits"]]

    def mission_latency_stats(self) -> dict:
        """P50/P95/P99 inference latency across all observations."""
        body = {
            "aggs": {
                "latency_pct": {
                    "percentiles": {
                        "field":    "inference_ms",
                        "percents": [50, 95, 99],
                    }
                },
                "by_backend": {
                    "terms": {"field": "vlm_backend"},
                    "aggs":  {"avg_ms": {"avg": {"field": "inference_ms"}}},
                }
            },
            "size": 0,
        }
        resp = self._es.search(index=IDX_OBS, body=body)
        aggs = resp.get("aggregations", {})
        return {
            "percentiles": aggs.get("latency_pct", {}).get("values", {}),
            "by_backend":  [
                {"backend": b["key"], "avg_ms": b["avg_ms"]["value"]}
                for b in aggs.get("by_backend", {}).get("buckets", [])
            ],
        }

    # ── APM context manager ───────────────────────────────────────────────────

    def apm_transaction(self, name: str, tx_type: str = "ros2_callback"):
        """
        Context manager that wraps a ROS2 callback in an APM transaction.

        Usage:
            with fleet_monitor.apm_transaction("perception_node.inference"):
                result = vlm.query(image, query)
        """
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            if self._apm is None:
                yield
                return
            self._apm.begin_transaction(tx_type)
            try:
                yield
                self._apm.end_transaction(name, result="success")
            except Exception:
                self._apm.end_transaction(name, result="error")
                raise

        return _ctx()

    # ── Background writer ─────────────────────────────────────────────────────

    def _enqueue(self, index: str, doc: dict) -> None:
        try:
            self._q.put_nowait((index, doc))
        except queue.Full:
            with self._lock:
                self._dropped += 1
                if self._dropped % 50 == 1:
                    print(f"[FleetMonitor] WARNING: write queue full — "
                          f"{self._dropped} documents dropped total. "
                          "Elasticsearch may be slow.")
            # Drop oldest to make room
            try:
                self._q.get_nowait()
                self._q.put_nowait((index, doc))
            except queue.Empty:
                pass

    def _write_loop(self) -> None:
        """Background thread: drain queue and bulk-index to Elasticsearch."""
        from elasticsearch.helpers import bulk

        batch:   list[dict] = []
        timeout: float      = 0.5   # seconds to accumulate before flushing

        while not self._stop.is_set():
            try:
                index, doc = self._q.get(timeout=timeout)
                batch.append({"_index": index, "_source": doc})

                # Flush every 50 documents or when queue drains
                if len(batch) >= 50 or self._q.empty():
                    self._flush_batch(bulk, batch)
                    batch = []
            except queue.Empty:
                if batch:
                    from elasticsearch.helpers import bulk as _bulk
                    self._flush_batch(_bulk, batch)
                    batch = []

        # Final flush on shutdown
        if batch:
            from elasticsearch.helpers import bulk as _bulk
            self._flush_batch(_bulk, batch)

    def _flush_batch(self, bulk_fn, batch: list[dict]) -> None:
        if not batch:
            return
        try:
            ok, errors = bulk_fn(self._es, batch, raise_on_error=False)
            if errors:
                print(f"[FleetMonitor] {len(errors)} bulk index errors: "
                      f"{errors[0]}")
        except Exception as exc:
            print(f"[FleetMonitor] Bulk write failed: {exc}")

    def close(self) -> None:
        self._stop.set()
        self._writer.join(timeout=5.0)
        self._es.close()


# ── Client builders ───────────────────────────────────────────────────────────

def _build_es_client():
    url     = os.environ.get("ELASTICSEARCH_URL", "").strip()
    api_key = os.environ.get("ELASTICSEARCH_API_KEY", "").strip()

    if not url:
        raise RuntimeError(
            "ELASTICSEARCH_URL environment variable is not set.\n"
            "Example: export ELASTICSEARCH_URL=https://my-cluster.es.io:443"
        )

    try:
        from elasticsearch import Elasticsearch
    except ImportError as exc:
        raise ImportError(
            "elasticsearch package missing. Install: pip install elasticsearch"
        ) from exc

    kwargs: dict = {"hosts": [url]}
    if api_key:
        kwargs["api_key"] = api_key

    client = Elasticsearch(**kwargs, request_timeout=10)
    info = client.info()
    print(f"[FleetMonitor] Elasticsearch {info['version']['number']} @ {url}")
    return client


def _build_apm_client():
    server_url = os.environ.get("ELASTIC_APM_SERVER_URL", "").strip()
    if not server_url:
        print("[FleetMonitor] ELASTIC_APM_SERVER_URL not set — APM disabled.")
        return None

    try:
        import elasticapm
    except ImportError:
        print("[FleetMonitor] elastic-apm not installed — APM disabled. "
              "Install: pip install elastic-apm")
        return None

    token = os.environ.get("ELASTIC_APM_SECRET_TOKEN", "")
    client = elasticapm.Client(
        service_name="mars-scout",
        server_url=server_url,
        secret_token=token or None,
        environment="simulation",
    )
    elasticapm.instrument()
    print(f"[FleetMonitor] Elastic APM → {server_url}")
    return client


def _ensure_index(es, name: str, mapping: dict) -> None:
    if not es.indices.exists(index=name):
        es.indices.create(index=name, body=mapping)
        print(f"[FleetMonitor] Created index '{name}'")
    else:
        print(f"[FleetMonitor] Index '{name}' exists.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_iso(dt) -> Optional[str]:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)
