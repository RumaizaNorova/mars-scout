"""
ARES — Autonomous Rover Exploration System
Isaac Sim Scene Setup
======================

Builds the full simulation:
  1. 500×500m Jezero Crater terrain (real HiRISE DTM, 1m/px)
  2. 450 geologically-placed rocks (CFA size-frequency distribution)
  3. Perseverance-class rover with rocker-bogie suspension
  4. Stereo MastCam-Z + chase camera (3rd-person view)
  5. ROS2 topics for the full pipeline

Usage:
  python3 isaac_sim/setup_scene.py            # headless
  python3 isaac_sim/setup_scene.py --headed   # GUI via VNC (port 6080)

What gets published:
  /isaac_camera/rgb       MastCam RGB  (what rover sees)
  /isaac_camera/depth     MastCam depth
  /isaac_camera/chase     Chase camera (3rd-person view)
  /odom                   Rover odometry
  Subscribes: /rover/cmd_vel  → 6-wheel drive

Then in a new terminal:
  ros2 launch mars_scout_sim_bridge sim_bridge.launch.py
"""

from __future__ import annotations
import argparse
import os
import sys

# ── Parse args BEFORE SimulationApp (it hijacks sys.argv) ────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--headed",   action="store_true", help="Run with GUI (requires VNC)")
parser.add_argument("--headless", action="store_true", help="Run headless (default)")
_args, _remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining
_headless = not _args.headed   # default is headless

# ── SimulationApp (must be first Isaac Sim call) ──────────────────────────────
try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({
    "headless": _headless,
    "width":  1280,
    "height":  720,
})

print(f"[setup_scene] ARES Isaac Sim starting ({'headless' if _headless else 'GUI'})")

# ── Deferred imports (after SimulationApp) ────────────────────────────────────
import numpy as np
import omni
import omni.usd
import omni.graph.core as og
import omni.replicator.core as rep
from omni.isaac.core.utils.extensions import enable_extension

# Local builders (same directory)
sys.path.insert(0, os.path.dirname(__file__))
from hirise_terrain_builder import build_mars_scene
from ares_rover_builder      import build_ares_rover
from rover_controller        import RoverWheelController

# Isaac Sim core
try:
    from isaacsim.core.api import World
except ImportError:
    from omni.isaac.core import World

# ── Helpers ───────────────────────────────────────────────────────────────────
def _clean_graphs(paths: list) -> None:
    """Remove stale OmniGraph prims before recreating them."""
    _s = omni.usd.get_context().get_stage()
    for p in paths:
        if _s.GetPrimAtPath(p).IsValid():
            _s.RemovePrim(p)

# ── Enable ROS2 bridge ────────────────────────────────────────────────────────
enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()
print("[setup_scene] ROS2 bridge enabled")

# ── World + stage ─────────────────────────────────────────────────────────────
world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

# ── Mars gravity ──────────────────────────────────────────────────────────────
# Perseverance experiences 3.72 m/s² on Mars (vs 9.81 m/s² on Earth)
try:
    from pxr import UsdPhysics
    physics_scene_path = "/physicsScene"
    if not stage.GetPrimAtPath(physics_scene_path).IsValid():
        UsdPhysics.Scene.Define(stage, physics_scene_path)
    physics_scene = UsdPhysics.Scene(stage.GetPrimAtPath(physics_scene_path))
    physics_scene.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr().Set(3.72)
    print("[setup_scene] Mars gravity: 3.72 m/s²")
except Exception as _e:
    print(f"[setup_scene] Gravity warning: {_e} (defaulting to Earth gravity)")

# =============================================================================
# 1. Terrain
# =============================================================================
_HIRISE_PATH = os.path.expanduser("~/mars-rover-agent/data/jezero_hirise.tif")
terrain_result = build_mars_scene(
    stage,
    terrain_path       = "/World/MarsTerrain",
    rocks_root         = "/World/Rocks",
    terrain_width      = 500.0,
    terrain_depth      = 500.0,
    terrain_nx         = 512,
    terrain_ny         = 512,
    terrain_amplitude  = 0.70,
    replace_existing   = True,
    hirise_dtm_path    = _HIRISE_PATH if os.path.exists(_HIRISE_PATH) else None,
    hirise_patch_size  = 500.0,
)
print(f"[setup_scene] Terrain: {terrain_result['dtm_source']} | "
      f"{len(terrain_result['rock_paths'])} rocks | "
      f"elev range {terrain_result['elevation_min']:.1f}–"
      f"{terrain_result['elevation_max']:.1f}m")

# =============================================================================
# 2. Rover
# =============================================================================
# Spawn 30m above terrain minimum so rover drops cleanly onto terrain.
# Under Mars gravity (3.72 m/s²), 30m drop ≈ 4 seconds — handled by warmup.
rover_result = build_ares_rover(
    stage,
    root_path  = "/World/Rover",
    spawn_pos  = (0.0, 0.0, 30.0),
    spawn_yaw  = 0.0,   # facing +X (toward rocks)
)

# =============================================================================
# 3. OmniGraph: Odometry publisher
# =============================================================================
_rover_path = rover_result["root_path"]
_clean_graphs(["/World/OdomGraph"])
try:
    og.Controller.edit(
        {"graph_path": "/World/OdomGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick",   "omni.graph.action.OnTick"),
                ("CompOdom", "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("PubOdom",  "isaacsim.ros2.bridge.ROS2PublishOdometry"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("CompOdom.inputs:chassisPrim",   [_rover_path]),
                ("PubOdom.inputs:topicName",      "/odom"),
                ("PubOdom.inputs:odomFrameId",    "odom"),
                ("PubOdom.inputs:chassisFrameId", "base_link"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick",             "CompOdom.inputs:execIn"),
                ("CompOdom.outputs:execOut",        "PubOdom.inputs:execIn"),
                ("CompOdom.outputs:position",       "PubOdom.inputs:position"),
                ("CompOdom.outputs:orientation",    "PubOdom.inputs:orientation"),
                ("CompOdom.outputs:linearVelocity", "PubOdom.inputs:linearVelocity"),
                ("CompOdom.outputs:angularVelocity","PubOdom.inputs:angularVelocity"),
            ],
        },
    )
    print("[setup_scene] OmniGraph: odometry → /odom")
except Exception as _e:
    print(f"[setup_scene] Odometry graph warning: {_e}")

# =============================================================================
# 4. Physics init + renderer warmup
# =============================================================================
try:
    world.reset()
    print("[setup_scene] Physics initialised")
except Exception as _e:
    print(f"[setup_scene] Physics warning: {_e} — continuing")

print("[setup_scene] Warming up renderer (letting rover drop onto terrain)...")
for _ in range(60):   # 60 steps ≈ 1 sim-second — rover lands and settles
    world.step(render=True)
print("[setup_scene] Renderer warm, rover settled on terrain")

# =============================================================================
# 5. Render products (MUST be after world.reset + warmup)
# =============================================================================
_mastcam_path = rover_result["mastcam_L"]   # /World/Rover/Mast/MastCamL
_chase_path   = rover_result["chase_cam"]   # /World/Rover/ChaseCamera

_rp_mast  = ""
_rp_chase = ""

try:
    _rp_mast  = rep.create.render_product(_mastcam_path,  (1280, 720)).path
    print(f"[setup_scene] Render product: MastCamL → {_rp_mast}")
except Exception as _e:
    print(f"[setup_scene] MastCam render product failed: {_e}")

try:
    _rp_chase = rep.create.render_product(_chase_path, (1280, 720)).path
    print(f"[setup_scene] Render product: ChaseCamera → {_rp_chase}")
except Exception as _e:
    print(f"[setup_scene] Chase cam render product failed: {_e}")

# =============================================================================
# 6. OmniGraph: Camera publishers (post-reset)
# =============================================================================
def _wire_camera(graph_path: str, render_product: str,
                 topic: str, cam_type: str) -> None:
    if not render_product:
        return
    _clean_graphs([graph_path])
    try:
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnTick",    "omni.graph.action.OnTick"),
                    ("CamHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("CamHelper.inputs:renderProductPath",    render_product),
                    ("CamHelper.inputs:topicName",            topic),
                    ("CamHelper.inputs:type",                 cam_type),
                    ("CamHelper.inputs:frameId",              "camera_optical_frame"),
                    ("CamHelper.inputs:enableSemanticLabels", False),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnTick.outputs:tick", "CamHelper.inputs:execIn"),
                ],
            },
        )
        print(f"[setup_scene] OmniGraph: {cam_type} → {topic}")
    except Exception as _e:
        print(f"[setup_scene] Camera graph {topic} warning: {_e}")

# MastCam — what rover sees
_wire_camera("/World/CamRgbGraph",   _rp_mast,  "/isaac_camera/rgb",   "rgb")
_wire_camera("/World/CamDepthGraph", _rp_mast,  "/isaac_camera/depth", "depth")

# Chase camera — 3rd-person view
_wire_camera("/World/CamChaseGraph", _rp_chase, "/isaac_camera/chase", "rgb")

# =============================================================================
# 7. Rover wheel controller
# =============================================================================
controller = RoverWheelController(stage, rover_result)

# =============================================================================
# 8. Main loop
# =============================================================================
print("\n" + "="*60)
print("[setup_scene] ARES simulation running.")
print(f"[setup_scene] Terrain: {terrain_result['dtm_source'].upper()} "
      f"({'real Jezero Crater' if terrain_result['dtm_source'] == 'hirise' else 'procedural'})")
print(f"[setup_scene] Rover: ARES Perseverance-class at /World/Rover")
print(f"[setup_scene] Cameras: MastCam → /isaac_camera/rgb | "
      f"Chase → /isaac_camera/chase")
if not _headless:
    print(f"[setup_scene] GUI: open browser → http://localhost:6080/vnc.html")
print("[setup_scene] ROS2 stack:")
print("  ros2 launch mars_scout_sim_bridge sim_bridge.launch.py")
print("="*60 + "\n")

while simulation_app.is_running():
    world.step(render=True)
    controller.step(dt=1.0 / 60.0)
    try:
        rep.orchestrator.step(pause_timeline=False)
    except Exception:
        pass

controller.stop()
simulation_app.close()
