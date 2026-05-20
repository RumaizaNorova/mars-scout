"""
Isaac Sim Mars terrain scene setup.
Run this inside Isaac Sim's script editor or as a standalone app.
"""

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import omni
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.utils.prims import define_prim
import numpy as np

# ── World ──────────────────────────────────────────────────────────────────────
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

assets_root = get_assets_root_path()

# ── Mars terrain ───────────────────────────────────────────────────────────────
# Use NVIDIA's Moon/Mars surface asset; swap USD path if you have a custom DEM
TERRAIN_USD = f"{assets_root}/Isaac/Environments/Simple_Room/simple_room.usd"
add_reference_to_stage(usd_path=TERRAIN_USD, prim_path="/World/Terrain")

# ── Rover (Jetbot as stand-in; swap for a custom URDF rover if available) ──────
JETBOT_USD = f"{assets_root}/Isaac/Robots/Jetbot/jetbot.usd"
rover = world.scene.add(
    WheeledRobot(
        prim_path="/World/Rover",
        name="rover",
        wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
        create_robot=True,
        usd_path=JETBOT_USD,
        position=np.array([0.0, 0.0, 0.1]),
    )
)

# ── Rocks / obstacles ─────────────────────────────────────────────────────────
rock_positions = [
    [1.5,  0.5, 0.0],
    [-1.0, 1.2, 0.0],
    [2.0, -1.0, 0.0],
    [0.5,  2.0, 0.0],
]
for i, pos in enumerate(rock_positions):
    prim_path = f"/World/Rock_{i}"
    define_prim(prim_path, "Sphere")
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    prim.GetAttribute("radius").Set(0.15)
    prim.GetAttribute("xformOp:translate").Set(tuple(pos))

# ── Simulate ───────────────────────────────────────────────────────────────────
world.reset()

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
