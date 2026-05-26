"""
Isaac Sim Mars Terrain Scene
==============================

Run this as a standalone Isaac Sim app (not inside the GUI script editor).
It creates the Mars terrain, places the Jetbot rover, wires up the
omni.isaac.ros2_bridge so the sim_bridge_node can consume topics, and
starts the simulation loop.

Usage (on the Vast.ai GPU instance):
  cd /root/mars-rover-agent
  python3 isaac_sim/setup_scene.py                  # headed (VNC)
  python3 isaac_sim/setup_scene.py --headless        # no GUI

What this publishes (via omni.isaac.ros2_bridge):
  /isaac_camera/rgb           sensor_msgs/Image (BGR8)
  /isaac_camera/depth         sensor_msgs/Image (32FC1, metres)
  /isaac_camera/camera_info   sensor_msgs/CameraInfo
  /odom                       nav_msgs/Odometry
  Subscribes: /cmd_vel        geometry_msgs/Twist

These are exactly what sim_bridge_node.py expects by default.
"""

from __future__ import annotations
import argparse
import sys

# ── Args before Isaac Sim boots (SimulationApp consumes sys.argv) ────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--headless", action="store_true")
_parsed, _remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

# Isaac Sim 6.x (pip install) uses isaacsim.SimulationApp
# Isaac Sim 4.x and older used omni.isaac.kit.SimulationApp
try:
    from isaacsim import SimulationApp  # noqa: E402  (Isaac Sim 6.x pip)
except ImportError:
    from omni.isaac.kit import SimulationApp  # noqa: E402  (Isaac Sim 4.x)

simulation_app = SimulationApp({
    "headless": _parsed.headless,
    "width":    1280,
    "height":   720,
})

# ── Deferred imports (must come after SimulationApp init) ─────────────────────
import numpy as np
import omni
import omni.usd
from hirise_terrain_builder import build_mars_scene as _build_mars_scene
from omni.isaac.core import World
try:
    from isaacsim.core.utils.nucleus import get_assets_root_path  # 5.x
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.prims import define_prim
except ImportError:
    from omni.isaac.core.utils.nucleus import get_assets_root_path  # 4.x
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.prims import define_prim
try:
    from isaacsim.robot.wheeled_robots.robots import WheeledRobot  # 5.x
except ImportError:
    from omni.isaac.wheeled_robots.robots import WheeledRobot  # 4.x

import omni.graph.core as og
from omni.isaac.core.utils.extensions import enable_extension

# Enable ROS2 bridge extension — must be done before importing
enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()  # let the extension load

# Now safe to import bridge classes (extension is loaded)
try:
    from omni.isaac.ros2_bridge import ROS2Camera, ROS2Odometry  # 4.x / 5.x
except ImportError:
    # Isaac Sim 5.x pip variant may place them differently; OmniGraph fallback below
    ROS2Camera = None
    ROS2Odometry = None


# ── World ──────────────────────────────────────────────────────────────────────
world = World(stage_units_in_meters=1.0)
assets_root = get_assets_root_path()

# ── Mars terrain, rocks, and lighting (procedural — no remote assets) ──────────
# Replaces: TERRAIN_CANDIDATES / add_reference_to_stage / add_default_ground_plane
#           ROCK_CONFIG spheres / DistantLight added later in original code
# Kept intact: camera prim + OmniGraph (lines 182-350 of original)
_mars_stage = omni.usd.get_context().get_stage()
import os as _os
_HIRISE_PATH = _os.path.expanduser("~/mars-rover-agent/data/jezero_hirise.tif")
_mars_result = _build_mars_scene(
    _mars_stage,
    terrain_path      = "/World/MarsTerrain",
    rocks_root        = "/World/Rocks",
    terrain_width     = 500.0,
    terrain_depth     = 500.0,
    terrain_nx        = 512,
    terrain_ny        = 512,
    terrain_amplitude = 0.70,
    replace_existing  = True,
    hirise_dtm_path   = _HIRISE_PATH if _os.path.exists(_HIRISE_PATH) else None,
    hirise_patch_size = 500.0,
)
print(f"[setup_scene] Procedural Mars scene built: "
      f"{len(_mars_result['rock_paths'])} rocks, "
      f"terrain at /World/MarsTerrain.")

# ── Rover ─────────────────────────────────────────────────────────────────────
ROVER_USD = f"{assets_root}/Isaac/Robots/Jetbot/jetbot.usd"
rover_prim_path = "/World/Rover"

# Stage the USD first (non-fatal — may fail if Nucleus is unreachable)
try:
    add_reference_to_stage(usd_path=ROVER_USD, prim_path=rover_prim_path)
    print(f"[setup_scene] Jetbot USD staged at {rover_prim_path}")
except Exception as _e:
    print(f"[setup_scene] Jetbot USD staging failed ({_e}); rover will be a kinematic prim.")

# Register articulation — use create_robot=False so we don't re-add the USD
rover = None
try:
    rover = world.scene.add(
        WheeledRobot(
            prim_path       = rover_prim_path,
            name            = "rover",
            wheel_dof_names = ["left_wheel_joint", "right_wheel_joint"],
            create_robot    = False,
            position        = np.array([0.0, 0.0, 0.1]),
        )
    )
    print("[setup_scene] WheeledRobot articulation registered.")
except Exception as _e:
    print(f"[setup_scene] WheeledRobot skipped ({_e}); rover is pose-only.")
    rover = None
print(f"[setup_scene] Rover prim at {rover_prim_path}")

# ── Rocks: now handled by mars_terrain_builder (33 rocks, 3 materials) ────────
# The ROCK_CONFIG sphere loop has been replaced.  Rock prims live under
# /World/Rocks/ and were already created by _build_mars_scene() above.
stage = omni.usd.get_context().get_stage()
print(f"[setup_scene] {len(_mars_result['rock_paths'])} rock prims at /World/Rocks/.")

# ── Lighting: now handled by mars_terrain_builder ─────────────────────────────
# SunLight (DistantLight, 3500, amber, 28° elevation) and
# SkyDome  (DomeLight,  600, pink-peach) were created by _build_mars_scene().
# Nothing to do here — keeping the comment block for traceability.
print("[setup_scene] Martian lighting already applied by mars_terrain_builder.")

# ── Camera prim (top-level, not tied to Rover chassis which may not have loaded) ─
camera_prim_path = "/World/Camera"
define_prim(camera_prim_path, "Camera")
camera_prim = stage.GetPrimAtPath(camera_prim_path)
try:
    from pxr import UsdGeom, Gf
    _xf = UsdGeom.Xformable(camera_prim)
    _xf.AddTranslateOp().Set((0.0, 0.0, 1.5))      # 1.5 m — rover mast height
    # Isaac Sim camera default faces -Z (straight down in Z-up world).
    # rotateXYZ(75, 0, -90):
    #   rotX(75°): -Z → (0, sin75, -cos75) = mostly +Y, slightly -Z
    #   rotZ(-90°): swings +Y → +X  →  camera now faces +X with 15° nose-down ✓
    _xf.AddRotateXYZOp().Set((75.0, 0.0, -90.0))
    camera_prim.GetAttribute("focalLength").Set(18.0)   # wider FOV to see more terrain
    camera_prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.01, 10000.0))
    print(f"[setup_scene] Camera prim at {camera_prim_path} (facing +X, 15° down, f/18)")
except Exception as _e:
    print(f"[setup_scene] Camera transform warning: {_e}")

# Render product and camera OmniGraph are set up AFTER world.reset() below.
# Creating them before reset attaches them to an uninitialised pipeline → black frames.
_render_product_path = ""
import omni.replicator.core as rep

# ── OmniGraph: odometry publisher ─────────────────────────────────────────────
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
                ("CompOdom.inputs:chassisPrim",   [rover_prim_path]),
                ("PubOdom.inputs:topicName",      "/odom"),
                ("PubOdom.inputs:odomFrameId",    "odom"),
                ("PubOdom.inputs:chassisFrameId", "base_link"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick",               "CompOdom.inputs:execIn"),
                ("CompOdom.outputs:execOut",          "PubOdom.inputs:execIn"),
                ("CompOdom.outputs:position",         "PubOdom.inputs:position"),
                ("CompOdom.outputs:orientation",      "PubOdom.inputs:orientation"),
                ("CompOdom.outputs:linearVelocity",   "PubOdom.inputs:linearVelocity"),
                ("CompOdom.outputs:angularVelocity",  "PubOdom.inputs:angularVelocity"),
            ],
        },
    )
    print("[setup_scene] OmniGraph odometry → /odom")
except Exception as _e:
    print(f"[setup_scene] Odometry OmniGraph failed (non-fatal): {_e}")

# ── OmniGraph: cmd_vel subscriber → differential drive ───────────────────────
# ROS2SubscribeTwist outputs linearVelocity / angularVelocity as double3.
# DifferentialController expects scalar doubles, so we extract x (linear) and z (angular).
# Clear any stale graph from a previous run first.
_cmdvel_stage = omni.usd.get_context().get_stage()
if _cmdvel_stage.GetPrimAtPath("/World/CmdVelGraph").IsValid():
    _cmdvel_stage.RemovePrim("/World/CmdVelGraph")

_cmdvel_wired = False
for _sub, _diff, _artic in [
    # Isaac Sim 5.x names (try first)
    ("isaacsim.ros2.bridge.ROS2SubscribeTwist",
     "isaacsim.robot.wheeled_robots.DifferentialController",
     "isaacsim.core.nodes.IsaacArticulationController"),
    # Isaac Sim 4.x names (fallback)
    ("omni.isaac.ros2_bridge.ROS2SubscribeTwist",
     "omni.isaac.wheeled_robots.DifferentialController",
     "omni.isaac.core_nodes.IsaacArticulationController"),
]:
    try:
        og.Controller.edit(
            {"graph_path": "/World/CmdVelGraph", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnTick",    "omni.graph.action.OnTick"),
                    ("ROS2Sub",   _sub),
                    # Break double3 → scalar x / z for the differential controller
                    ("BrkLin",    "omni.graph.nodes.BreakVector3"),
                    ("BrkAng",    "omni.graph.nodes.BreakVector3"),
                    ("DiffDrive", _diff),
                    ("ArticCtrl", _artic),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("ROS2Sub.inputs:topicName",        "/cmd_vel"),
                    ("DiffDrive.inputs:wheelRadius",    0.0325),
                    ("DiffDrive.inputs:wheelDistance",  0.1125),
                    ("DiffDrive.inputs:maxLinearSpeed", 0.5),
                    ("ArticCtrl.inputs:robotPath",      "/World/Rover"),
                    ("ArticCtrl.inputs:jointNames",     ["left_wheel_joint",
                                                         "right_wheel_joint"]),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnTick.outputs:tick",               "ROS2Sub.inputs:execIn"),
                    ("ROS2Sub.outputs:execOut",           "DiffDrive.inputs:execIn"),
                    # linear.x → forward speed
                    ("ROS2Sub.outputs:linearVelocity",    "BrkLin.inputs:tuple"),
                    ("BrkLin.outputs:x",                  "DiffDrive.inputs:linearVelocity"),
                    # angular.z → yaw rate
                    ("ROS2Sub.outputs:angularVelocity",   "BrkAng.inputs:tuple"),
                    ("BrkAng.outputs:z",                  "DiffDrive.inputs:angularVelocity"),
                    ("DiffDrive.outputs:execOut",         "ArticCtrl.inputs:execIn"),
                    ("DiffDrive.outputs:velocityCommand", "ArticCtrl.inputs:velocityCommands"),
                ],
            },
        )
        print(f"[setup_scene] OmniGraph cmd_vel → diff drive ({_sub.split('.')[0]}.*)")
        _cmdvel_wired = True
        break
    except Exception as _e:
        # Clear the partial graph before trying the next variant
        if _cmdvel_stage.GetPrimAtPath("/World/CmdVelGraph").IsValid():
            _cmdvel_stage.RemovePrim("/World/CmdVelGraph")

if not _cmdvel_wired:
    print("[setup_scene] cmd_vel OmniGraph: all variants failed — wire manually in Isaac Sim GUI.")

# ── Simulate ───────────────────────────────────────────────────────────────────
try:
    world.reset()
    print("[setup_scene] Physics initialised successfully.")
except Exception as _e:
    print(f"[setup_scene] world.reset() warning: {_e}")
    print("[setup_scene] Continuing — rover articulation may be unavailable, camera + OmniGraph still work.")

# ── Warm-up + render product (MUST be after world.reset) ─────────────────────
# world.reset() initialises the rendering pipeline. Creating the render product
# before reset returns a dead Hydra texture → zero pixel frames. We step a few
# times first so the pipeline is fully live before attaching the camera target.
print("[setup_scene] Warming up renderer before creating render product...")
for _ in range(10):
    world.step(render=True)

try:
    _rp = rep.create.render_product(camera_prim_path, (1280, 720))
    _render_product_path = _rp.path
    print(f"[setup_scene] Render product ready (post-reset): {_render_product_path}")
except Exception as _e:
    print(f"[setup_scene] Render product failed ({_e}) — camera will be skipped.")

# ── OmniGraph: camera RGB + depth publisher (post-reset) ─────────────────────
if _render_product_path:
    for _cam_type, _topic in [("rgb",   "/isaac_camera/rgb"),
                               ("depth", "/isaac_camera/depth")]:
        try:
            og.Controller.edit(
                {"graph_path": f"/World/Cam{_cam_type.capitalize()}Graph",
                 "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnTick",    "omni.graph.action.OnTick"),
                        ("CamHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("CamHelper.inputs:renderProductPath",    _render_product_path),
                        ("CamHelper.inputs:topicName",            _topic),
                        ("CamHelper.inputs:type",                 _cam_type),
                        ("CamHelper.inputs:frameId",              "camera_optical_frame"),
                        ("CamHelper.inputs:enableSemanticLabels", False),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnTick.outputs:tick", "CamHelper.inputs:execIn"),
                    ],
                },
            )
            print(f"[setup_scene] OmniGraph camera '{_cam_type}' → {_topic}")
        except Exception as _e:
            print(f"[setup_scene] Camera {_cam_type} OmniGraph failed (non-fatal): {_e}")
else:
    print("[setup_scene] Skipping camera OmniGraph (no render product).")

print("\n[setup_scene] Simulation running.")
print("[setup_scene] In a new terminal, run:")
print("  ros2 launch mars_scout_sim_bridge sim_bridge.launch.py")
print()

while simulation_app.is_running():
    world.step(render=True)
    # Step the replicator orchestrator so the camera annotator flushes each frame
    try:
        rep.orchestrator.step(pause_timeline=False)
    except Exception:
        pass

simulation_app.close()
