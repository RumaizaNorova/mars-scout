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
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.wheeled_robots.robots import WheeledRobot

# ROS2 bridge extension
from omni.isaac.ros2_bridge import ROS2Camera, ROS2Odometry, ROS2Subscriber
from geometry_msgs.msg import Twist
import omni.graph.core as og


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
rover = world.scene.add(
    WheeledRobot(
        prim_path         = "/World/Rover",
        name              = "rover",
        wheel_dof_names   = ["left_wheel_joint", "right_wheel_joint"],
        create_robot      = True,
        usd_path          = ROVER_USD,
        position          = np.array([0.0, 0.0, 0.1]),
    )
)
print(f"[setup_scene] Rover loaded at /World/Rover")

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

# ── ROS2 camera bridge ────────────────────────────────────────────────────────
# Attaches a camera to the rover at 15-degree downward pitch, publishes:
#   /isaac_camera/rgb, /isaac_camera/depth, /isaac_camera/camera_info
camera_prim_path = "/World/Rover/chassis/Camera"
define_prim(camera_prim_path, "Camera")
camera_prim = stage.GetPrimAtPath(camera_prim_path)

ros2_camera = ROS2Camera(
    prim_path           = camera_prim_path,
    name                = "rover_camera",
    frequency           = 15,
    resolution          = (1280, 720),
    rgb_topic           = "/isaac_camera/rgb",
    depth_topic         = "/isaac_camera/depth",
    camera_info_topic   = "/isaac_camera/camera_info",
)
world.scene.add(ros2_camera)
print("[setup_scene] ROS2Camera bridge ready on /isaac_camera/*")

# ── ROS2 odometry bridge ──────────────────────────────────────────────────────
ros2_odom = ROS2Odometry(
    prim_path         = "/World/Rover",
    name              = "rover_odom",
    frequency         = 30,
    linear_vel_topic  = "/cmd_vel",   # Isaac reads this for the differential drive
    odom_topic        = "/odom",
)
world.scene.add(ros2_odom)
print("[setup_scene] ROS2Odometry bridge ready on /odom")

# ── cmd_vel subscriber (ROS2 -> Isaac differential drive) ─────────────────────
# Action graph node subscribes /cmd_vel and drives the rover wheels.
# This uses OmniGraph to wire Twist.linear.x / Twist.angular.z -> wheel velocities.
try:
    og.Controller.edit(
        {"graph_path": "/World/CmdVelGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick",      "omni.graph.action.OnTick"),
                ("ROS2Sub",     "omni.isaac.ros2_bridge.ROS2SubscribeTwist"),
                ("DiffDrive",   "omni.isaac.wheeled_robots.DifferentialController"),
                ("ArticCtrl",   "omni.isaac.core_nodes.IsaacArticulationController"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("ROS2Sub.inputs:topicName",        "/cmd_vel"),
                ("DiffDrive.inputs:wheelRadius",    0.0325),
                ("DiffDrive.inputs:wheelDistance",  0.1125),
                ("DiffDrive.inputs:maxLinearSpeed",  0.5),
                ("ArticCtrl.inputs:robotPath",      "/World/Rover"),
                ("ArticCtrl.inputs:jointNames",     ["left_wheel_joint",
                                                     "right_wheel_joint"]),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick",             "ROS2Sub.inputs:execIn"),
                ("ROS2Sub.outputs:execOut",         "DiffDrive.inputs:execIn"),
                ("ROS2Sub.outputs:linearVelocity",  "DiffDrive.inputs:linearVelocity"),
                ("ROS2Sub.outputs:angularVelocity", "DiffDrive.inputs:angularVelocity"),
                ("DiffDrive.outputs:execOut",       "ArticCtrl.inputs:execIn"),
                ("DiffDrive.outputs:velocityCommand","ArticCtrl.inputs:velocityCommands"),
            ],
        },
    )
    print("[setup_scene] OmniGraph cmd_vel -> differential drive wired.")
except Exception as e:
    print(f"[setup_scene] OmniGraph wiring failed (non-fatal): {e}")
    print("[setup_scene] You can wire cmd_vel manually in the Isaac Sim GUI.")

# ── Simulate ───────────────────────────────────────────────────────────────────
world.reset()
print("\n[setup_scene] Simulation running.")
print("[setup_scene] In a new terminal, run:")
print("  ros2 run mars_scout_sim_bridge topic_inspector")
print("  ros2 launch mars_scout_sim_bridge sim_bridge.launch.py")
print()

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
