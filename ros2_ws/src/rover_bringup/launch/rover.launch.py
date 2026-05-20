from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    target_arg = DeclareLaunchArgument(
        "target_description",
        default_value="large rock",
        description="Natural language description of what the rover should find",
    )

    vlm_node = Node(
        package="rover_vision",
        executable="vlm_node",
        name="vlm_node",
        parameters=[{
            "target_description": LaunchConfiguration("target_description"),
            "inference_rate_hz": 1.0,
        }],
        output="screen",
    )

    agent_node = Node(
        package="rover_nav",
        executable="agent_node",
        name="agent_node",
        parameters=[{
            "linear_speed": 0.3,
            "angular_speed": 0.4,
        }],
        output="screen",
    )

    return LaunchDescription([target_arg, vlm_node, agent_node])
