"""
Launch RViz2 with the Mars Scout config.
Run alongside mock.launch.py or full.launch.py.

  ros2 launch mars_scout_bringup rviz.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    rviz_cfg = os.path.join(
        get_package_share_directory("mars_scout_bringup"),
        "config", "mars_scout.rviz",
    )
    return LaunchDescription([
        Node(
            package    = "rviz2",
            executable = "rviz2",
            name       = "rviz2",
            arguments  = ["-d", rviz_cfg],
            output     = "screen",
        )
    ])
