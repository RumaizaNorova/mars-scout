#!/usr/bin/env python3
"""
Prove ROS messages cross between the ROS container and a running Isaac Sim:
camera frames and odometry arrive, and a /rover/cmd_vel command moves the rover.
"""

import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

STARTUP_TIMEOUT_S = 900.0     # first Isaac start compiles shaders
DRIVE_SECONDS     = 10.0
DRIVE_SPEED_M_S   = 0.1
MIN_TRAVEL_M      = 0.2


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("ares_smoke_ros")
    latest = {"rgb": None, "depth": None, "odom": None}

    def keep(key):
        def callback(msg):
            latest[key] = msg
        return callback

    node.create_subscription(Image, "/isaac_camera/rgb", keep("rgb"), qos_profile_sensor_data)
    node.create_subscription(Image, "/isaac_camera/depth", keep("depth"), qos_profile_sensor_data)
    node.create_subscription(Odometry, "/odom", keep("odom"), qos_profile_sensor_data)
    cmd_pub = node.create_publisher(Twist, "/rover/cmd_vel", 10)

    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while None in latest.values():
        if time.monotonic() > deadline:
            missing = [k for k, v in latest.items() if v is None]
            print(f"[smoke_ros] FAIL: nothing received on {missing}", file=sys.stderr)
            return 1
        rclpy.spin_once(node, timeout_sec=0.1)
    print("[smoke_ros] Camera, depth and odometry are arriving")

    start = latest["odom"].pose.pose.position
    x0, y0 = start.x, start.y
    cmd = Twist()
    cmd.linear.x = DRIVE_SPEED_M_S
    stop_at = time.monotonic() + DRIVE_SECONDS
    while time.monotonic() < stop_at:
        cmd_pub.publish(cmd)
        rclpy.spin_once(node, timeout_sec=0.1)
    cmd_pub.publish(Twist())

    end = latest["odom"].pose.pose.position
    travel = math.hypot(end.x - x0, end.y - y0)
    print(f"[smoke_ros] Rover moved {travel:.3f} m")
    if travel < MIN_TRAVEL_M:
        print(f"[smoke_ros] FAIL: expected at least {MIN_TRAVEL_M} m", file=sys.stderr)
        return 1
    print("[smoke_ros] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
