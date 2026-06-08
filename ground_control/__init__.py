"""
ARES Ground Control — terrain intelligence layer.

Three telemetry systems:
  TerrainMemory   → MongoDB Atlas (persistent terrain feature catalog)
  VLMQualityMonitor → Arize Phoenix (LLM eval + calibration)
  FleetMonitor    → Elasticsearch + Elastic APM (observability + geo search)

Orchestrated by MissionPlanner, which agent_node calls before/during/after
every NavigateToTarget execution.
"""

from ground_control.terrain_memory  import TerrainMemory
from ground_control.vlm_quality     import VLMQualityMonitor
from ground_control.fleet_monitor   import FleetMonitor
from ground_control.mission_planner import MissionPlanner, MissionPlan

__all__ = [
    "TerrainMemory",
    "VLMQualityMonitor",
    "FleetMonitor",
    "MissionPlanner",
    "MissionPlan",
]
