"""
ARES — Autonomous Rover Exploration System
Isaac Sim Scene Setup
======================

Builds the full simulation:
  1. 500×500m Jezero Crater terrain (real HiRISE DTM, 1m/px)
  2. Rocks placed by the CFA size-frequency distribution
  3. Perseverance-class rover with rocker-bogie suspension
  4. Stereo MastCam-Z + chase camera (3rd-person view)
  5. ROS2 topics for the full pipeline

Any failure stops the run: a scene that starts must be the scene we asked for.

Usage (inside the Isaac Sim container, see docker-compose.yml):
  isaac_sim/setup_scene.py                        # HiRISE terrain, runs until stopped
  isaac_sim/setup_scene.py --terrain procedural   # no DTM file needed
  isaac_sim/setup_scene.py --smoke-test           # run checks, write a report, exit

What gets published:
  /isaac_camera/rgb       MastCam RGB  (what rover sees)
  /isaac_camera/depth     MastCam depth
  /isaac_camera/chase     Chase camera (3rd-person view)
  /odom                   Rover odometry
  Subscribes: /rover/cmd_vel  → 6-wheel drive
"""

from __future__ import annotations
import argparse
import os
import sys

# ── Parse args BEFORE SimulationApp (it hijacks sys.argv) ────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--headed", action="store_true", help="Run with GUI (not in the container)")
parser.add_argument("--terrain", choices=["hirise", "procedural"], default="hirise")
parser.add_argument("--smoke-test", action="store_true", help="Run checks, write a report, exit")
parser.add_argument("--report", default=os.path.join(_REPO_ROOT, "outputs", "smoke_report.json"))
_args, _remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

# Local helpers with no Isaac dependency (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_checks

_HIRISE_PATH = os.path.join(_REPO_ROOT, "data", "jezero_hirise.tif")
if _args.terrain == "hirise" and not os.path.isfile(_HIRISE_PATH):
    sys.exit(
        f"[setup_scene] HiRISE DTM not found at {_HIRISE_PATH}.\n"
        "Run scripts/download_hirise.sh, or pass --terrain procedural."
    )

# ── SimulationApp (must be first Isaac Sim call) ──────────────────────────────
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": not _args.headed,
    "width":  1280,
    "height":  720,
})

# ── Deferred imports (after SimulationApp) ────────────────────────────────────
from isaacsim.core.utils.extensions import enable_extension

enable_extension("omni.replicator.core")
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()
print("[setup_scene] ROS2 bridge enabled")

import numpy as np
import omni.graph.core as og
import omni.physx
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from pxr import Gf, UsdPhysics

from hirise_terrain_builder import build_mars_scene
from ares_rover_builder      import build_ares_rover
from rover_controller        import RoverWheelController

# ── Helpers ───────────────────────────────────────────────────────────────────
PHYSICS_DT = 1.0 / 60.0


def _clean_graphs(paths: list) -> None:
    """Remove stale OmniGraph prims before recreating them."""
    _s = omni.usd.get_context().get_stage()
    for p in paths:
        if _s.GetPrimAtPath(p).IsValid():
            _s.RemovePrim(p)


def _set_mars_gravity(stage) -> float:
    """Set gravity on the stage's physics scene and return the value read back."""
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        raise RuntimeError(
            f"Expected one physics scene, found {[str(p.GetPath()) for p in scenes]}"
        )
    scene = UsdPhysics.Scene(scenes[0])
    scene.GetGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.GetGravityMagnitudeAttr().Set(scene_checks.MARS_GRAVITY_M_S2)
    return float(scene.GetGravityMagnitudeAttr().Get())


def _body_position(body_path: str) -> np.ndarray:
    """Rover body position read from PhysX, not from possibly stale USD."""
    pose = omni.physx.get_physx_interface().get_rigidbody_transformation(body_path)
    if not pose["ret_val"]:
        raise RuntimeError(f"PhysX has no rigid body at {body_path}")
    return np.array(pose["position"], dtype=float)


# ── World + stage ─────────────────────────────────────────────────────────────
world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT)
stage = omni.usd.get_context().get_stage()
# Use World's own physics scene: a second /physicsScene causes TGS solver crashes.
world.get_physics_context()
_gravity_configured = _set_mars_gravity(stage)
print(f"[setup_scene] Mars gravity: {_gravity_configured} m/s²")

# =============================================================================
# 1. Terrain
# =============================================================================
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
    hirise_dtm_path    = _HIRISE_PATH if _args.terrain == "hirise" else None,
    hirise_patch_size  = 500.0,
)
print(f"[setup_scene] Terrain: {terrain_result['dtm_source']} | "
      f"{len(terrain_result['rock_paths'])} rocks | "
      f"elev range {terrain_result['elevation_min']:.1f}–"
      f"{terrain_result['elevation_max']:.1f}m")

# =============================================================================
# 2. Rover
# =============================================================================
_spawn_z = scene_checks.spawn_height(terrain_result["elevation"], 500.0, 500.0)
rover_result = build_ares_rover(
    stage,
    root_path  = "/World/Rover",
    spawn_pos  = (0.0, 0.0, _spawn_z),
    spawn_yaw  = 0.0,   # facing +X (toward rocks)
)
_rover_path = rover_result["root_path"]
_body_path  = f"{_rover_path}/Body"

# =============================================================================
# 3. OmniGraph: Odometry publisher + cmd_vel subscriber
# =============================================================================
_clean_graphs(["/World/OdomGraph", "/World/CmdVelGraph"])
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

# Rover commands arrive through Isaac's ROS bridge, so Isaac's Python never
# imports rclpy and never has to match the ROS install's Python.
_TWIST_NODE = "/World/CmdVelGraph/SubTwist"
og.Controller.edit(
    {"graph_path": "/World/CmdVelGraph", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnTick",   "omni.graph.action.OnTick"),
            ("SubTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
        ],
        og.Controller.Keys.SET_VALUES: [
            ("SubTwist.inputs:topicName", "/rover/cmd_vel"),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnTick.outputs:tick", "SubTwist.inputs:execIn"),
        ],
    },
)
print("[setup_scene] OmniGraph: /rover/cmd_vel → wheels")


def _read_cmd_vel() -> tuple:
    """Latest (linear.x, angular.z) from the ROS bridge."""
    linear  = og.Controller.get(og.Controller.attribute(f"{_TWIST_NODE}.outputs:linearVelocity"))
    angular = og.Controller.get(og.Controller.attribute(f"{_TWIST_NODE}.outputs:angularVelocity"))
    return float(linear[0]), float(angular[2])


# =============================================================================
# 4. Physics init, gravity measurement, settle
# =============================================================================
world.reset()
print("[setup_scene] Physics initialised")

# The first steps are free fall, so they measure the gravity physics really uses.
_FREE_FALL_STEPS = 12
_z_start = _body_position(_body_path)[2]
for _ in range(_FREE_FALL_STEPS):
    world.step(render=False)
_gravity_measured = scene_checks.gravity_from_drop(
    _z_start - _body_position(_body_path)[2], _FREE_FALL_STEPS, PHYSICS_DT
)
print(f"[setup_scene] Measured gravity: {_gravity_measured:.2f} m/s²")

_settled = False
_calm_steps = 0
_last_pos = _body_position(_body_path)
for _ in range(1200):   # up to 20 sim-seconds
    world.step(render=True)
    _pos = _body_position(_body_path)
    _speed = float(np.linalg.norm(_pos - _last_pos)) / PHYSICS_DT
    _last_pos = _pos
    _calm_steps = _calm_steps + 1 if _speed < scene_checks.SETTLE_SPEED_M_S else 0
    if _calm_steps >= 30:
        _settled = True
        break
if not _settled:
    raise RuntimeError("Rover did not come to rest on the terrain within 20 sim-seconds")
print("[setup_scene] Rover settled on terrain")

# =============================================================================
# 5. Render products (MUST be after world.reset + settle)
# =============================================================================
_mastcam_path = rover_result["mastcam_L"]   # /World/Rover/Mast/MastCamL
_chase_path   = rover_result["chase_cam"]   # /World/Rover/ChaseCamera

_rp_mast_obj = rep.create.render_product(_mastcam_path, (1280, 720))
_rp_mast     = _rp_mast_obj.path
_rp_chase    = rep.create.render_product(_chase_path, (1280, 720)).path
print(f"[setup_scene] Render products: {_rp_mast}, {_rp_chase}")

# =============================================================================
# 6. OmniGraph: Camera publishers (post-reset)
# =============================================================================
def _wire_camera(graph_path: str, render_product: str,
                 topic: str, cam_type: str) -> None:
    _clean_graphs([graph_path])
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick",    "omni.graph.action.OnTick"),
                ("CamHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("CamHelper.inputs:renderProductPath", render_product),
                ("CamHelper.inputs:topicName",         topic),
                ("CamHelper.inputs:type",              cam_type),
                ("CamHelper.inputs:frameId",           "camera_optical_frame"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick", "CamHelper.inputs:execIn"),
            ],
        },
    )
    print(f"[setup_scene] OmniGraph: {cam_type} → {topic}")

# MastCam — what rover sees
_wire_camera("/World/CamRgbGraph",   _rp_mast,  "/isaac_camera/rgb",   "rgb")
_wire_camera("/World/CamDepthGraph", _rp_mast,  "/isaac_camera/depth", "depth")

# Chase camera — 3rd-person view
_wire_camera("/World/CamChaseGraph", _rp_chase, "/isaac_camera/chase", "rgb")

# =============================================================================
# 7. Rover wheel controller
# =============================================================================
controller = RoverWheelController(stage, rover_result, read_command=_read_cmd_vel)

# =============================================================================
# 8. Smoke test or main loop
# =============================================================================
def _run_smoke_test() -> list:
    """Check the assembled scene, write the report, return any failures."""
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([_rp_mast_obj])
    for _ in range(30):     # let the renderer deliver frames
        world.step(render=True)
    camera = scene_checks.image_stats(np.asarray(annotator.get_data()))

    start = _body_position(_body_path)
    controller.set_override(scene_checks.SMOKE_DRIVE_SPEED_M_S, 0.0)
    for _ in range(int(scene_checks.SMOKE_DRIVE_SECONDS / PHYSICS_DT)):
        world.step(render=True)
        controller.step(dt=PHYSICS_DT)
    controller.stop()
    travel = float(np.linalg.norm((_body_position(_body_path) - start)[:2]))

    report = {
        "dtm_source":              terrain_result["dtm_source"],
        "gravity_configured_m_s2": _gravity_configured,
        "gravity_measured_m_s2":   _gravity_measured,
        "settled":                 _settled,
        "camera":                  camera,
        "drive_travel_m":          travel,
    }
    failures = scene_checks.evaluate_smoke(report)
    report["failures"] = failures
    report["passed"]   = not failures
    scene_checks.write_report(report, _args.report)
    print(f"[setup_scene] Smoke report: {report}")
    return failures


if _args.smoke_test:
    _failures = _run_smoke_test()
    simulation_app.close()
    # A crash before this point leaves no report, which also counts as a failure.
    sys.exit(1 if _failures else 0)

print("\n" + "="*60)
print("[setup_scene] ARES simulation running.")
print(f"[setup_scene] Terrain: {terrain_result['dtm_source'].upper()}")
print(f"[setup_scene] Rover: ARES Perseverance-class at {_rover_path}")
print("[setup_scene] Cameras: MastCam → /isaac_camera/rgb | "
      "Chase → /isaac_camera/chase")
print("="*60 + "\n")

while simulation_app.is_running():
    world.step(render=True)
    controller.step(dt=PHYSICS_DT)

controller.stop()
simulation_app.close()
