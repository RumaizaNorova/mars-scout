"""
setup_atlas_indexes.py — one-shot script to create all MongoDB Atlas indexes
needed by the mars-scout Ground Control stack.

Run once after creating your Atlas M0 cluster:
    MONGODB_URI="mongodb+srv://..." python scripts/setup_atlas_indexes.py

What this creates
-----------------
Standard indexes (always created):
  missions        : compound (terrain_unit, created_at)
  observations    : compound (mission_id, seq), single (fsm_state)
  terrain_features: 2dsphere on location, compound (feature_type, observation_count)
  alerts          : TTL index (expires_at), single (mission_id)

Atlas Vector Search index (requires Atlas M10+ OR free-tier with preview):
  terrain_features: "terrain_features_vector" on embedding[384]

  NOTE — M0 free tier does NOT support programmatic vector search index creation
  via the Data API. If you are on M0, follow the manual steps printed below.
"""

from __future__ import annotations
import os
import sys
import time

REQUIRED = ("pymongo",)
for pkg in REQUIRED:
    try:
        __import__(pkg)
    except ImportError:
        sys.exit(f"Missing package: {pkg}. Run: pip install pymongo[srv]")

import pymongo
from pymongo import MongoClient, GEOSPHERE, ASCENDING, DESCENDING
from pymongo.operations import IndexModel

# ── Connection ────────────────────────────────────────────────────────────────

MONGODB_URI = os.environ.get("MONGODB_URI", "")
if not MONGODB_URI:
    sys.exit("Set MONGODB_URI before running this script.")

print(f"Connecting to Atlas…  ({MONGODB_URI[:40]}…)")
client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10_000)
try:
    client.admin.command("ping")
except Exception as exc:
    sys.exit(f"Could not reach MongoDB: {exc}")

db = client["mars_scout"]
print("Connected.\n")

# ── Standard indexes ──────────────────────────────────────────────────────────

def create_standard_indexes():
    print("=== Standard indexes ===")

    # missions
    col = db["missions"]
    col.create_indexes([
        IndexModel([("terrain_unit", ASCENDING), ("created_at", DESCENDING)]),
        IndexModel([("outcome", ASCENDING)]),
        IndexModel([("mission_id", ASCENDING)], unique=True),
    ])
    print("  missions: OK")

    # observations
    col = db["observations"]
    col.create_indexes([
        IndexModel([("mission_id", ASCENDING), ("seq", ASCENDING)]),
        IndexModel([("fsm_state", ASCENDING)]),
        IndexModel([("timestamp", DESCENDING)]),
    ])
    print("  observations: OK")

    # terrain_features — 2dsphere + standard
    col = db["terrain_features"]
    col.create_indexes([
        IndexModel([("location", GEOSPHERE)]),
        IndexModel([("feature_type", ASCENDING), ("observation_count", DESCENDING)]),
        IndexModel([("last_seen", DESCENDING)]),
    ])
    print("  terrain_features (2dsphere + compound): OK")

    # alerts — TTL (auto-delete after expires_at)
    col = db["alerts"]
    col.create_indexes([
        IndexModel([("expires_at", ASCENDING)], expireAfterSeconds=0),
        IndexModel([("mission_id", ASCENDING)]),
        IndexModel([("created_at", DESCENDING)]),
    ])
    print("  alerts (TTL): OK")

    print()

# ── Atlas Vector Search index ─────────────────────────────────────────────────

VECTOR_INDEX_DEF = {
    "name": "terrain_features_vector",
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 384,
                "similarity": "cosine",
            },
            # Filter fields for pre-filtering by terrain_unit / feature_type
            {"type": "filter", "path": "terrain_unit"},
            {"type": "filter", "path": "feature_type"},
        ]
    },
}

def create_vector_search_index():
    print("=== Atlas Vector Search index ===")

    # Check if the index already exists via listSearchIndexes command
    try:
        existing = list(db.command(
            "listSearchIndexes",
            "terrain_features",
        ).get("cursor", {}).get("firstBatch", []))
        names = [idx.get("name") for idx in existing]
        if "terrain_features_vector" in names:
            print("  terrain_features_vector: already exists — skipping.")
            return
    except Exception as exc:
        # M0 free tier does not support this command; fall through to manual instructions
        if "not supported" in str(exc).lower() or "AtlasError" in type(exc).__name__:
            _print_manual_instructions()
            return
        # Unexpected error — try createSearchIndexes anyway
        print(f"  listSearchIndexes not available ({exc}), attempting createSearchIndexes…")

    try:
        db.command(
            "createSearchIndexes",
            "terrain_features",
            indexes=[VECTOR_INDEX_DEF],
        )
        print("  terrain_features_vector: creation submitted (Atlas builds async — wait ~60 s).")
        _wait_for_vector_index()
    except Exception as exc:
        if "not supported" in str(exc).lower() or "requires" in str(exc).lower():
            _print_manual_instructions()
        else:
            print(f"  ERROR: {exc}")
            _print_manual_instructions()


def _wait_for_vector_index(timeout_sec: int = 120):
    print("  Waiting for index to become READY…", end="", flush=True)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        time.sleep(5)
        try:
            result = list(db.command(
                "listSearchIndexes",
                "terrain_features",
            ).get("cursor", {}).get("firstBatch", []))
            for idx in result:
                if idx.get("name") == "terrain_features_vector":
                    status = idx.get("status", "UNKNOWN")
                    print(f" {status}", end="", flush=True)
                    if status == "READY":
                        print()
                        return
        except Exception:
            pass
    print("\n  Timed out — check Atlas UI for index status.")


def _print_manual_instructions():
    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Atlas M0 does not support programmatic Vector Search index creation.   │
  │  Create the index manually in 3 clicks:                                 │
  │                                                                         │
  │  1. Atlas UI → your cluster → "Search" tab → "Create Search Index"     │
  │  2. Choose "JSON Editor"                                                 │
  │  3. Database: mars_scout   Collection: terrain_features                 │
  │  4. Index name: terrain_features_vector                                 │
  │  5. Paste this definition:                                               │
  │                                                                         │
  │  {                                                                       │
  │    "fields": [                                                           │
  │      {                                                                   │
  │        "type": "vector",                                                 │
  │        "path": "embedding",                                              │
  │        "numDimensions": 384,                                             │
  │        "similarity": "cosine"                                            │
  │      },                                                                  │
  │      { "type": "filter", "path": "terrain_unit" },                      │
  │      { "type": "filter", "path": "feature_type" }                       │
  │    ]                                                                     │
  │  }                                                                       │
  │                                                                         │
  │  Click "Create" — index builds in ~60 s.                                │
  └─────────────────────────────────────────────────────────────────────────┘
""")


# ── Elasticsearch index setup (bonus — connects to local ES if reachable) ────

def setup_elasticsearch():
    es_url = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
    print(f"=== Elasticsearch indexes ({es_url}) ===")
    try:
        import elasticsearch
        from elasticsearch import Elasticsearch, BadRequestError
    except ImportError:
        print("  elasticsearch package not installed — skip (pip install elasticsearch).")
        return

    es = Elasticsearch(es_url, request_timeout=5)
    try:
        if not es.ping():
            print(f"  Not reachable at {es_url} — skip (start with: docker compose up elasticsearch -d).")
            return
    except Exception as exc:
        print(f"  Not reachable: {exc} — skip.")
        return

    MAPPINGS = {
        "rover-observations": {
            "mappings": {
                "properties": {
                    "mission_id":      {"type": "keyword"},
                    "seq":             {"type": "integer"},
                    "timestamp":       {"type": "date"},
                    "fsm_state":       {"type": "keyword"},
                    "target_found":    {"type": "boolean"},
                    "confidence":      {"type": "float"},
                    "description":     {"type": "text"},
                    "distance_to_goal_m": {"type": "float"},
                    "rover_position":  {"type": "geo_point"},
                    "inference_ms":    {"type": "float"},
                    "query_text":      {"type": "text"},
                    "terrain_unit":    {"type": "keyword"},
                    "vlm_backend":     {"type": "keyword"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }
        },
        "rover-terrain-features": {
            "mappings": {
                "properties": {
                    "feature_type":       {"type": "keyword"},
                    "terrain_unit":       {"type": "keyword"},
                    "location":           {"type": "geo_point"},
                    "description":        {"type": "text"},
                    "observation_count":  {"type": "integer"},
                    "first_seen":         {"type": "date"},
                    "last_seen":          {"type": "date"},
                    "avg_confidence":     {"type": "float"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }
        },
        "rover-missions": {
            "mappings": {
                "properties": {
                    "mission_id":       {"type": "keyword"},
                    "query_text":       {"type": "text"},
                    "terrain_unit":     {"type": "keyword"},
                    "outcome":          {"type": "keyword"},
                    "elapsed_sec":      {"type": "float"},
                    "n_ticks":          {"type": "integer"},
                    "n_detections":     {"type": "integer"},
                    "n_false_positives":{"type": "integer"},
                    "precision":        {"type": "float"},
                    "started_at":       {"type": "date"},
                    "completed_at":     {"type": "date"},
                    "final_position":   {"type": "geo_point"},
                }
            }
        },
    }

    for idx_name, body in MAPPINGS.items():
        try:
            es.indices.create(index=idx_name, body=body)
            print(f"  {idx_name}: created")
        except BadRequestError as exc:
            if "resource_already_exists_exception" in str(exc):
                print(f"  {idx_name}: already exists — skip")
            else:
                print(f"  {idx_name}: ERROR {exc}")
        except Exception as exc:
            print(f"  {idx_name}: ERROR {exc}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    create_standard_indexes()
    create_vector_search_index()
    setup_elasticsearch()

    print("Done.")
    print()
    print("Next steps:")
    print("  1. If on Atlas M0, follow the manual Vector Search instructions above.")
    print("  2. Start local stack:  docker compose up elasticsearch kibana phoenix -d")
    print("  3. Run batch seeding:  MONGODB_URI=$MONGODB_URI python scripts/batch_mission_runner.py --n-missions 50")
    print("  4. Open Kibana:        http://localhost:5601")
    print("  5. Open Phoenix:       http://localhost:6006")
