"""
Isaac Sim Topic Inspector
==========================

Run this on the GPU instance right after Isaac Sim starts to discover
exactly what topics it's publishing and what types they are.

Usage:
  ros2 run mars_scout_sim_bridge topic_inspector

Output: a neat table of all active topics + types + hz estimate,
with colour-coding to flag which ones the sim_bridge_node needs to
be pointed at via parameters.
"""

from __future__ import annotations
import time

import rclpy
from rclpy.node import Node

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

# Topics our bridge needs — we'll highlight them
_NEEDED = {
    "image", "rgb", "color",              # RGB camera
    "depth",                              # depth
    "camera_info",                        # intrinsics
    "odom", "odometry",                   # pose
    "cmd_vel",                            # velocity input
}


def _needs_bridge(topic: str) -> bool:
    t = topic.lower()
    return any(kw in t for kw in _NEEDED)


def main(args=None):
    rclpy.init(args=args)
    node = Node("topic_inspector")

    node.get_logger().info("Scanning active topics for 3 seconds…")
    deadline = time.time() + 3.0

    # Spin briefly to let the topic list populate
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    topic_types = node.get_topic_names_and_types()
    topic_types.sort(key=lambda x: x[0])

    print(f"\n{_BOLD}{'═'*72}{_RESET}")
    print(f"{_BOLD}  Isaac Sim Active Topics{_RESET}")
    print(f"{'═'*72}")
    print(f"  {'Topic':<44} Type")
    print(f"  {'─'*44} {'─'*25}")

    bridge_hints: dict[str, str] = {}

    for topic, types in topic_types:
        type_str = types[0] if types else "?"
        if _needs_bridge(topic):
            flag = f"{_GREEN}[bridge]{_RESET}"
            # heuristic mapping for bridge parameters
            tl = topic.lower()
            if any(w in tl for w in ["rgb", "color", "image_raw"]) and "depth" not in tl:
                bridge_hints["isaac_rgb_topic"] = topic
            elif "depth" in tl:
                bridge_hints["isaac_depth_topic"] = topic
            elif "camera_info" in tl:
                bridge_hints["isaac_info_topic"] = topic
            elif any(w in tl for w in ["odom", "odometry"]):
                bridge_hints["isaac_odom_topic"] = topic
            elif "cmd_vel" in tl:
                bridge_hints["isaac_cmdvel_topic"] = topic
        else:
            flag = "       "
        print(f"  {flag} {topic:<44} {type_str}")

    print(f"{'─'*72}")

    if bridge_hints:
        print(f"\n{_BOLD}  Suggested sim_bridge_node parameters:{_RESET}")
        for param, topic in bridge_hints.items():
            print(f"    --ros-args -p {_GREEN}{param}:={topic}{_RESET}")
        print()
        print(f"  Full launch command:")
        args_str = " ".join(f"-p {k}:={v}" for k, v in bridge_hints.items())
        print(f"  {_BOLD}ros2 run mars_scout_sim_bridge sim_bridge_node "
              f"--ros-args {args_str}{_RESET}")
    else:
        print(f"\n  {_YELLOW}No Isaac Sim topics detected. "
              f"Is Isaac Sim running with ros2_bridge enabled?{_RESET}")
        print(f"  Ensure your scene script includes:")
        print(f"    from omni.isaac.ros2_bridge import ROS2Camera, ROS2Odometry")

    print(f"{'═'*72}\n")

    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
