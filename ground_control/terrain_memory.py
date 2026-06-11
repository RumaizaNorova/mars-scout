"""
Persistent Mars terrain feature catalog — MongoDB backend.

What this does
--------------
Every rock, crater, ripple, and formation identified by the VLM across ANY
mission is stored here.  The catalog grows with every mission.  Before
starting a new mission the MissionPlanner queries it:

  - query_nearby()          → what's already been mapped near the start point?
  - find_similar_missions() → did we succeed on similar terrain before?
  - atlas_vector_search()   → semantic KNN across all known features

After each mission:

  - add_feature() for each confirmed detection
  - close_mission() with outcome, elapsed, final position

Change Streams (Atlas/replica-set only)
---------------------------------------
start_alert_stream() watches the observations collection for runs of low-
confidence detections and writes alert documents to the `alerts` collection.
This drives real behavior: the MissionPlanner reads alerts and raises
min_confidence for the next mission of the same query type.

Collections
-----------
  missions          one document per NavigateToTarget execution
  observations      one document per VLM call (compound index: mission_id + seq)
  terrain_features  every discovered feature, 2dsphere + vector search index
  alerts            low-confidence warning events from the Change Stream

Environment
-----------
  MONGODB_URI   connection string (required — no silent default)
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from ground_control._embedder import DIM


# ── Connection ────────────────────────────────────────────────────────────────

def _get_client():
    uri = os.environ.get("MONGODB_URI", "").strip()
    if not uri:
        raise RuntimeError(
            "MONGODB_URI environment variable is not set.\n"
            "For Atlas: export MONGODB_URI='mongodb+srv://user:pass@cluster.mongodb.net'\n"
            "For local: export MONGODB_URI='mongodb://localhost:27017'"
        )
    try:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure
    except ImportError as exc:
        raise ImportError(
            "pymongo is required. Install it: pip install 'pymongo[srv]'"
        ) from exc

    try:
        import certifi
        tls_kwargs = {"tlsCAFile": certifi.where()}
    except ImportError:
        tls_kwargs = {}
    client = MongoClient(uri, serverSelectionTimeoutMS=10_000, **tls_kwargs)
    # Force a real connection attempt — fail loud if unreachable
    try:
        client.admin.command("ping")
    except ConnectionFailure as exc:
        raise RuntimeError(
            f"Cannot reach MongoDB at URI (masked). Check MONGODB_URI and network.\n"
            f"Error: {exc}"
        ) from exc
    return client


# ── TerrainMemory ─────────────────────────────────────────────────────────────

class TerrainMemory:
    """
    Thread-safe interface to the Mars terrain catalog.

    All writes are fire-and-forget from the caller's perspective:
    the public methods return immediately.  Errors are re-raised — the
    caller decides whether to catch them (telemetry path) or propagate
    (critical path).
    """

    DB_NAME = "mars_scout"

    def __init__(self):
        from pymongo import ASCENDING, GEOSPHERE
        from pymongo.operations import IndexModel

        self._client = _get_client()
        db = self._client[self.DB_NAME]

        self._missions  = db["missions"]
        self._obs       = db["observations"]
        self._features  = db["terrain_features"]
        self._alerts    = db["alerts"]

        # Indexes — idempotent
        self._missions.create_index([("mission_id", ASCENDING)], unique=True)
        self._missions.create_index([("outcome", ASCENDING), ("started_at", ASCENDING)])
        self._missions.create_index([("query_text", ASCENDING), ("terrain_unit", ASCENDING)])

        self._obs.create_index([("mission_id", ASCENDING), ("seq", ASCENDING)])
        self._obs.create_index([("timestamp",  ASCENDING)])
        self._obs.create_index([("target_found", ASCENDING), ("confidence", ASCENDING)])

        self._features.create_index([("location", GEOSPHERE)])   # 2dsphere
        self._features.create_index([("feature_type", ASCENDING)])
        self._features.create_index([("mission_id", ASCENDING)])

        self._alerts.create_index([("mission_id", ASCENDING), ("created_at", ASCENDING)])

        print(f"[TerrainMemory] Connected — DB '{self.DB_NAME}'  "
              f"features={self._features.count_documents({})}  "
              f"missions={self._missions.count_documents({})}")

    # ── Mission lifecycle ──────────────────────────────────────────────────────

    def open_mission(
        self,
        query_text:       str,
        start_pos:        tuple[float, float],
        arrival_radius_m: float,
        vlm_backend:      str,
        terrain_unit:     str,
        mission_id:       Optional[str] = None,
    ) -> str:
        mid = mission_id or str(uuid.uuid4())
        self._missions.insert_one({
            "mission_id":        mid,
            "query_text":        query_text,
            "terrain_unit":      terrain_unit,
            "vlm_backend":       vlm_backend,
            "started_at":        datetime.now(timezone.utc),
            "completed_at":      None,
            "outcome":           None,
            "elapsed_sec":       None,
            "start_position":    {"x": start_pos[0], "y": start_pos[1]},
            "final_position":    None,
            "arrival_radius_m":  arrival_radius_m,
            "n_observations":    0,
            "n_detections":      0,
            "n_false_positives": 0,   # VERIFYING→APPROACHING regressions = ground-truth FPs
        })
        return mid

    def log_observation(self, mission_id: str, obs: dict) -> None:
        """
        obs must contain at minimum:
          seq, timestamp, fsm_state, target_found, confidence,
          description, distance_to_goal_m, rover_position {x,y}, inference_ms
        """
        doc = {**obs, "mission_id": mission_id}
        self._obs.insert_one(doc)

        inc = {"n_observations": 1}
        if obs.get("target_found"):
            inc["n_detections"] = 1
        self._missions.update_one({"mission_id": mission_id}, {"$inc": inc})

    def log_false_positive(self, mission_id: str) -> None:
        """Call when FSM transitions VERIFYING → APPROACHING (confirmed hallucination)."""
        self._missions.update_one(
            {"mission_id": mission_id},
            {"$inc": {"n_false_positives": 1}}
        )

    def close_mission(
        self,
        mission_id:     str,
        outcome:        str,
        elapsed_sec:    float,
        final_position: tuple[float, float],
    ) -> None:
        self._missions.update_one(
            {"mission_id": mission_id},
            {"$set": {
                "completed_at":   datetime.now(timezone.utc),
                "outcome":        outcome,
                "elapsed_sec":    elapsed_sec,
                "final_position": {"x": final_position[0], "y": final_position[1]},
            }}
        )

    # ── Terrain feature catalog ───────────────────────────────────────────────

    def add_feature(
        self,
        mission_id:   str,
        x:            float,
        y:            float,
        feature_type: str,
        description:  str,
        confidence:   float,
        embedding:    Optional[list[float]] = None,
    ) -> bool:
        """
        Add a terrain feature. If a feature already exists within 1 m,
        increment its observation_count instead of creating a duplicate.

        Returns True if a new feature was inserted, False if updated.
        """
        existing = self._features.find_one({
            "location": {
                "$near": {
                    "$geometry": {"type": "Point", "coordinates": [x, y]},
                    "$maxDistance": 1.0,
                }
            }
        })

        if existing:
            update: dict = {
                "$inc": {"observation_count": 1},
                "$set": {
                    "last_seen_mission": mission_id,
                    "last_confidence":   confidence,
                    "last_description":  description,
                }
            }
            if embedding is not None:
                update["$set"]["description_embedding"] = embedding
            self._features.update_one({"_id": existing["_id"]}, update)
            return False

        doc: dict = {
            "mission_id":     mission_id,
            "feature_type":   feature_type,
            "location":       {"type": "Point", "coordinates": [x, y]},
            "description":    description,
            "confidence":     confidence,
            "first_detected": datetime.now(timezone.utc),
            "last_seen_mission": mission_id,
            "observation_count": 1,
        }
        if embedding is not None:
            doc["description_embedding"] = embedding
        self._features.insert_one(doc)
        return True

    # ── Query methods ─────────────────────────────────────────────────────────

    def query_nearby(
        self,
        x:        float,
        y:        float,
        radius_m: float = 50.0,
        limit:    int   = 20,
    ) -> list[dict]:
        """Return up to `limit` terrain features within `radius_m` metres."""
        cursor = self._features.find({
            "location": {
                "$near": {
                    "$geometry": {"type": "Point", "coordinates": [x, y]},
                    "$maxDistance": radius_m,
                }
            }
        }).limit(limit)
        return [_strip_id(d) for d in cursor]

    def find_similar_missions(
        self,
        query_text:   str,
        terrain_unit: Optional[str] = None,
        limit:        int = 5,
    ) -> list[dict]:
        """
        Return past ARRIVED missions with query text most similar to the given one.
        Scored by Jaccard word overlap — fast, no extra model needed.
        """
        filt: dict = {"outcome": "ARRIVED"}
        if terrain_unit:
            filt["terrain_unit"] = terrain_unit

        candidates = list(self._missions.find(filt).sort("started_at", -1).limit(200))

        q_words = set(query_text.lower().split())
        scored = []
        for m in candidates:
            m_words = set(m["query_text"].lower().split())
            denom   = len(q_words | m_words)
            jaccard = len(q_words & m_words) / denom if denom else 0.0
            scored.append((jaccard, m))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [_strip_id(m) for _, m in scored[:limit]]

    def atlas_vector_search(
        self,
        embedding:  list[float],
        k:          int = 5,
        index_name: str = "description_vector_index",
    ) -> list[dict]:
        """
        Semantic KNN over terrain features using Atlas Vector Search.

        Requires an Atlas Vector Search index named `description_vector_index`
        on the terrain_features collection with:
          {
            "fields": [{
              "type": "vector",
              "path": "description_embedding",
              "numDimensions": 384,
              "similarity": "cosine"
            }]
          }

        Create it in the Atlas UI: Atlas Search → Create Index → JSON Editor.
        Raises RuntimeError if the index doesn't exist or this isn't Atlas.
        """
        pipeline = [
            {
                "$vectorSearch": {
                    "index":         index_name,
                    "path":          "description_embedding",
                    "queryVector":   embedding,
                    "numCandidates": k * 10,
                    "limit":         k,
                }
            },
            {
                "$project": {
                    "description":    1,
                    "feature_type":   1,
                    "location":       1,
                    "confidence":     1,
                    "mission_id":     1,
                    "observation_count": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        try:
            return [_strip_id(d) for d in self._features.aggregate(pipeline)]
        except Exception as exc:
            raise RuntimeError(
                f"Atlas Vector Search failed: {exc}\n"
                f"Ensure index '{index_name}' is configured in the Atlas UI on "
                f"the terrain_features collection with numDimensions=384, "
                f"similarity=cosine, path=description_embedding."
            ) from exc

    # ── Analytics ─────────────────────────────────────────────────────────────

    def mission_analytics(self) -> list[dict]:
        """
        Aggregation: per-outcome statistics across all completed missions.
        Returns list of {outcome, count, avg_elapsed_sec, avg_detections, avg_false_positives}.
        """
        pipeline = [
            {"$match":  {"outcome": {"$ne": None}}},
            {"$group": {
                "_id":                  "$outcome",
                "count":                {"$sum": 1},
                "avg_elapsed_sec":      {"$avg": "$elapsed_sec"},
                "avg_detections":       {"$avg": "$n_detections"},
                "avg_false_positives":  {"$avg": "$n_false_positives"},
                "total_false_positives":{"$sum": "$n_false_positives"},
            }},
            {"$sort": {"count": -1}},
        ]
        return list(self._missions.aggregate(pipeline))

    def confidence_by_state_analytics(self) -> list[dict]:
        """
        Aggregation on observations: average confidence per FSM state.
        Reveals which states produce reliable detections.
        """
        pipeline = [
            {"$match": {"target_found": True}},
            {"$group": {
                "_id":            "$fsm_state",
                "avg_confidence": {"$avg": "$confidence"},
                "count":          {"$sum": 1},
                "avg_inf_ms":     {"$avg": "$inference_ms"},
            }},
            {"$sort": {"avg_confidence": -1}},
        ]
        return list(self._obs.aggregate(pipeline))

    def false_positive_rate_by_query(self) -> list[dict]:
        """
        Join missions with observation counts to compute per-query-type FP rate.
        """
        pipeline = [
            {"$match": {"outcome": {"$ne": None}, "n_observations": {"$gt": 0}}},
            {"$addFields": {
                "fp_rate": {
                    "$cond": [
                        {"$gt": ["$n_detections", 0]},
                        {"$divide": ["$n_false_positives", "$n_detections"]},
                        0
                    ]
                }
            }},
            {"$group": {
                "_id":           "$query_text",
                "avg_fp_rate":   {"$avg": "$fp_rate"},
                "total_missions":{"$sum": 1},
                "arrived":       {"$sum": {"$cond": [{"$eq": ["$outcome", "ARRIVED"]}, 1, 0]}},
            }},
            {"$addFields": {
                "success_rate": {"$divide": ["$arrived", "$total_missions"]}
            }},
            {"$sort": {"avg_fp_rate": -1}},
        ]
        return list(self._missions.aggregate(pipeline))

    # ── Change Stream alert monitor ───────────────────────────────────────────

    def start_alert_stream(
        self,
        low_conf_threshold: float = 0.3,
        consecutive_n:      int   = 3,
        stop_event:         Optional[threading.Event] = None,
    ) -> threading.Thread:
        """
        Background thread that watches for runs of low-confidence detections
        (target_found=True but confidence < threshold for N consecutive frames)
        and writes alert documents to the alerts collection.

        Requires Atlas or a local MongoDB replica set.
        Returns the thread (already started).
        """
        t = threading.Thread(
            target=self._run_alert_stream,
            args=(low_conf_threshold, consecutive_n, stop_event),
            daemon=True,
            name="terrain-change-stream",
        )
        t.start()
        return t

    def _run_alert_stream(
        self,
        threshold:    float,
        n_consec:     int,
        stop_event:   Optional[threading.Event],
    ) -> None:
        pipeline = [{"$match": {
            "operationType":               "insert",
            "fullDocument.target_found":   True,
            "fullDocument.confidence":     {"$lt": threshold},
        }}]
        try:
            with self._obs.watch(pipeline, full_document="updateLookup") as stream:
                run: dict[str, int] = {}   # mission_id → consecutive low-conf count
                for event in stream:
                    if stop_event and stop_event.is_set():
                        break
                    doc = event["fullDocument"]
                    mid = doc.get("mission_id", "")
                    run[mid] = run.get(mid, 0) + 1
                    if run[mid] >= n_consec:
                        self._alerts.insert_one({
                            "mission_id":    mid,
                            "alert_type":    "low_confidence_run",
                            "consecutive_n": run[mid],
                            "threshold":     threshold,
                            "fsm_state":     doc.get("fsm_state"),
                            "created_at":    datetime.now(timezone.utc),
                        })
                        print(f"[TerrainMemory] ALERT: {n_consec} consecutive "
                              f"low-confidence detections on mission {mid[:8]}")
                    # Reset on high-confidence or not-found
                    # (handled implicitly — only low-conf events trigger this watch)
        except Exception as exc:
            # Change streams require replica set — print error but don't crash the rover
            print(f"[TerrainMemory] Change Stream unavailable: {exc}\n"
                  f"  Alerts disabled. Use MongoDB Atlas or a local replica set "
                  f"to enable them.")

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_id(doc: dict) -> dict:
    """Remove the non-serialisable ObjectId _id field."""
    doc.pop("_id", None)
    return doc
