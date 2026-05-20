"""
Full pipeline launch — Isaac Sim + Moondream2 (GPU instance).

Usage
-----
  ros2 launch mars_scout_bringup full.launch.py
  ros2 launch mars_scout_bringup full.launch.py query:="find the skull rock"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    query_arg = DeclareLaunchArgument(
        "query", default_value="rock formation",
    )

    # Isaac Sim scene starts first — give it 20 s to load before ROS nodes
    isaac_sim = ExecuteProcess(
        cmd=["python3", "/root/mars-scout/isaac_sim/setup_scene.py"],
        output="screen",
        name="isaac_sim",
    )

    perception = Node(
        package    = "mars_scout_perception",
        executable = "perception_node",
        name       = "perception_node",
        parameters = [{
            "backend":           "moondream2",
            "inference_rate_hz": 1.0,
            "default_query":     LaunchConfiguration("query"),
        }],
        output = "screen",
    )

    geometry = Node(
        package    = "mars_scout_geometry",
        executable = "projection_node",
        name       = "projection_node",
        output     = "screen",
    )

    agent = Node(
        package    = "mars_scout_control",
        executable = "agent_node",
        name       = "agent_node",
        output     = "screen",
    )

    # Delay ROS nodes until Isaac Sim is up
    delayed_stack = TimerAction(
        period=20.0,
        actions=[perception, geometry, agent],
    )

    return LaunchDescription([query_arg, isaac_sim, delayed_stack])
