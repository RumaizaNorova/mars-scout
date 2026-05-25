"""
Launch the Isaac Sim bridge + full Mars Scout perception/control stack.

Usage (GPU instance, Isaac Sim already running):
  ros2 launch mars_scout_sim_bridge sim_bridge.launch.py
  ros2 launch mars_scout_sim_bridge sim_bridge.launch.py query:="skull-like rock"

  # If Isaac Sim uses non-default topic names:
  ros2 launch mars_scout_sim_bridge sim_bridge.launch.py \
      isaac_rgb_topic:=/rgb_camera/image_raw \
      isaac_depth_topic:=/depth_camera/image_raw

This launch file starts:
  1. sim_bridge_node      — Isaac Sim <-> Mars Scout topic translation
  2. perception_node      — VLM (Moondream2 on GPU)
  3. projection_node      — 2D bbox -> 3D waypoint
  4. agent_node           — FSM + controller + NavigateToTarget action server
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── launch args ───────────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument("query",               default_value="rock formation"),
        DeclareLaunchArgument("isaac_rgb_topic",     default_value="/isaac_camera/rgb"),
        DeclareLaunchArgument("isaac_depth_topic",   default_value="/isaac_camera/depth"),
        DeclareLaunchArgument("isaac_info_topic",    default_value="/isaac_camera/camera_info"),
        DeclareLaunchArgument("isaac_odom_topic",    default_value="/odom"),
        DeclareLaunchArgument("isaac_cmdvel_topic",  default_value="/cmd_vel"),
        DeclareLaunchArgument("publish_hz",          default_value="15.0"),
    ]

    # ── nodes ─────────────────────────────────────────────────────────────────
    sim_bridge = Node(
        package    = "mars_scout_sim_bridge",
        executable = "sim_bridge_node",
        name       = "sim_bridge_node",
        output     = "screen",
        parameters = [{
            "isaac_rgb_topic":    LaunchConfiguration("isaac_rgb_topic"),
            "isaac_depth_topic":  LaunchConfiguration("isaac_depth_topic"),
            "isaac_info_topic":   LaunchConfiguration("isaac_info_topic"),
            "isaac_odom_topic":   LaunchConfiguration("isaac_odom_topic"),
            "isaac_cmdvel_topic": LaunchConfiguration("isaac_cmdvel_topic"),
            "publish_hz":         LaunchConfiguration("publish_hz"),
        }],
    )

    perception = Node(
        package    = "mars_scout_perception",
        executable = "perception_node",
        name       = "perception_node",
        output     = "screen",
        parameters = [{"backend": "mock"}],   # switch to moondream2 once installed
    )

    projection = Node(
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

    return LaunchDescription(args + [sim_bridge, perception, projection, agent])
