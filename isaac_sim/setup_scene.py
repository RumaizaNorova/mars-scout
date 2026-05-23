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

# ── Mars terrain ───────────────────────────────────────────────────────────────
# Primary: NVIDIA's Moon surface (closest to Mars in the asset library).
# If you have a custom Mars DEM USD, swap the path here.
TERRAIN_CANDIDATES = [
    f"{assets_root}/Isaac/Environments/Moon_Surface/moon_surface.usd",
    f"{assets_root}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",  # fallback
    f"{assets_root}/Isaac/Environments/Simple_Room/simple_room.usd",          # last resort
]
terrain_loaded = False
for usd_path in TERRAIN_CANDIDATES:
    try:
        add_reference_to_stage(usd_path=usd_path, prim_path="/World/Terrain")
        print(f"[setup_scene] Terrain loaded: {usd_path}")
        terrain_loaded = True
        break
    except Exception as e:
        print(f"[setup_scene] Could not load {usd_path}: {e}")

if not terrain_loaded:
    world.scene.add_default_ground_plane()
    print("[setup_scene] Using default ground plane (no terrain USD found).")

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

# ── Rocks (interest targets) ──────────────────────────────────────────────────
ROCK_CONFIG = [
    # (x,    y,     z,    radius,  name_hint)
    ( 2.0,   0.5,  0.08,  0.25,  "boulder"),
    (-1.5,   1.8,  0.06,  0.15,  "pebble"),
    ( 3.5,  -1.2,  0.10,  0.35,  "large_rock"),
    ( 0.8,   3.0,  0.07,  0.20,  "rock"),
    (-2.8,  -0.5,  0.12,  0.45,  "outcrop"),
]
stage = omni.usd.get_context().get_stage()
for i, (x, y, z, r, hint) in enumerate(ROCK_CONFIG):
    prim_path = f"/World/Rock_{i}_{hint}"
    define_prim(prim_path, "Sphere")
    prim = stage.GetPrimAtPath(prim_path)
    prim.GetAttribute("radius").Set(r)
    # Set dark reddish-brown colour (Mars rock approximation)
    if prim.HasAttribute("primvars:displayColor"):
        prim.GetAttribute("primvars:displayColor").Set([(0.35, 0.20, 0.10)])
    xform = prim.GetAttribute("xformOp:translate")
    if not xform:
        from pxr import UsdGeom
        xformable = UsdGeom.Xformable(prim)
        xformable.AddTranslateOp().Set((x, y, z))
    else:
        xform.Set((x, y, z))

print(f"[setup_scene] {len(ROCK_CONFIG)} rock prims placed.")

# ── Camera prim (top-level, not tied to Rover chassis which may not have loaded) ─
camera_prim_path = "/World/Camera"
define_prim(camera_prim_path, "Camera")
camera_prim = stage.GetPrimAtPath(camera_prim_path)
try:
    from pxr import UsdGeom, Gf
    _xf = UsdGeom.Xformable(camera_prim)
    _xf.AddTranslateOp().Set((0.0, 0.0, 0.3))     # 30 cm above ground
    _xf.AddRotateXYZOp().Set((-15.0, 0.0, 0.0))   # 15° nose-down
    camera_prim.GetAttribute("focalLength").Set(24.0)
    camera_prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.01, 10000.0))
    print(f"[setup_scene] Camera prim at {camera_prim_path} (15° tilt, f/24)")
except Exception as _e:
    print(f"[setup_scene] Camera transform warning: {_e}")

# ── Render product (needed for OmniGraph camera publisher) ────────────────────
_render_product_path = ""
try:
    import omni.replicator.core as rep
    _rp = rep.create.render_product(camera_prim_path, (1280, 720))
    _render_product_path = _rp.path
    print(f"[setup_scene] Render product ready: {_render_product_path}")
except Exception as _e:
    print(f"[setup_scene] Render product failed ({_e}) — camera OmniGraph will be skipped.")

# ── OmniGraph: camera RGB + depth publisher ───────────────────────────────────
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

print("\n[setup_scene] Simulation running.")
print("[setup_scene] In a new terminal, run:")
print("  ros2 run mars_scout_sim_bridge topic_inspector")
print("  ros2 launch mars_scout_sim_bridge sim_bridge.launch.py")
print()

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
