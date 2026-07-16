"""
Mock pipeline launch — runs entirely on CPU, no Isaac Sim, no GPU.

Nodes started
-------------
  mock_bridge       mars_scout_mock       — camera + depth + TF + odom
  perception_node   mars_scout_perception — VLM mock backend
  projection_node   mars_scout_geometry   — 2D→3D waypoint projection
  agent_node        mars_scout_control    — FSM + pursuit controller

Usage
-----
  ros2 launch mars_scout_bringup mock.launch.py
  ros2 launch mars_scout_bringup mock.launch.py query:="find the skull rock"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    query_arg = DeclareLaunchArgument(
        "query",
        default_value="rock formation",
        description="Natural-language target description",
    )
    backend_arg = DeclareLaunchArgument(
        "vlm_backend",
        default_value="mock",
        description="VLM backend: mock | moondream2",
    )

    mock_bridge = Node(
        package    = "mars_scout_mock",
        executable = "mock_bridge",
        name       = "mock_bridge",
        parameters = [{"camera_hz": 10.0, "odom_hz": 50.0}],
        output     = "screen",
    )

    perception = Node(
        package    = "mars_scout_perception",
        executable = "perception_node",
        name       = "perception_node",
        parameters = [{
            "backend":           LaunchConfiguration("vlm_backend"),
            "inference_rate_hz": 2.0,
            "min_confidence":    0.4,
            "default_query":     LaunchConfiguration("query"),
        }],
        output = "screen",
    )

    geometry = Node(
        package    = "mars_scout_geometry",
        executable = "projection_node",
        name       = "projection_node",
        parameters = [{
            "ransac_iters":    100,
            "ransac_inlier_m": 0.05,
            "camera_frame":    "camera_optical_frame",
            "map_frame":       "map",
        }],
        output = "screen",
    )

    agent = Node(
        package    = "mars_scout_control",
        executable = "agent_node",
        name       = "agent_node",
        parameters = [{
            "arrival_radius_m": 0.5,
        }],
        output = "screen",
    )

    return LaunchDescription([
        query_arg,
        backend_arg,
        mock_bridge,
        perception,
        geometry,
        agent,
    ])
