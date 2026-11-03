"""
Full pipeline launch — ROS side only. Isaac Sim runs in its own container.

Usage
-----
  ros2 launch mars_scout_bringup full.launch.py
  ros2 launch mars_scout_bringup full.launch.py query:="find the skull rock"
  ros2 launch mars_scout_bringup full.launch.py backend:=mock

Requires
--------
  GOOGLE_API_KEY and GEMINI_MODEL set before launch (unless backend:=mock).
  The perception node raises RuntimeError at startup if either is missing.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    sim_bridge_launch = os.path.join(
        get_package_share_directory("mars_scout_sim_bridge"),
        "launch", "sim_bridge.launch.py",
    )
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(sim_bridge_launch)),
    ])
