"""
Agent node: reads VLM bearing → sends velocity commands to rover.

States
------
SEARCHING   rotate slowly, scan for target
APPROACHING move toward target bearing
ARRIVED     stop (target in center view for N consecutive frames)
"""

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Twist


class AgentState:
    SEARCHING = "SEARCHING"
    APPROACHING = "APPROACHING"
    ARRIVED = "ARRIVED"


class AgentNode(Node):
    def __init__(self):
        super().__init__("agent_node")

        # Tunable params
        self.declare_parameter("linear_speed", 0.3)
        self.declare_parameter("angular_speed", 0.4)
        self.declare_parameter("center_threshold_rad", 0.15)
        self.declare_parameter("arrived_frames", 5)

        self.linear_speed = self.get_parameter("linear_speed").value
        self.angular_speed = self.get_parameter("angular_speed").value
        self.center_thresh = self.get_parameter("center_threshold_rad").value
        self.arrived_frames = self.get_parameter("arrived_frames").value

        # State
        self.state = AgentState.SEARCHING
        self.center_count = 0
        self.latest_bearing = float("nan")

        # Subscribers
        self.bearing_sub = self.create_subscription(
            Float32, "/rover/vlm/target_bearing", self.bearing_callback, 10
        )
        self.desc_sub = self.create_subscription(
            String, "/rover/vlm/description", self.desc_callback, 10
        )

        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, "/rover/cmd_vel", 10)

        # Control loop at 10 Hz
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("Agent node ready. State: SEARCHING")

    def bearing_callback(self, msg: Float32):
        self.latest_bearing = msg.data

    def desc_callback(self, msg: String):
        self.get_logger().info(f"[Terrain] {msg.data}")

    def control_loop(self):
        twist = Twist()
        bearing = self.latest_bearing

        if self.state == AgentState.ARRIVED:
            self.get_logger().info("Target reached! Stopped.", once=True)
            self.cmd_vel_pub.publish(twist)  # zero velocity
            return

        # No target found → keep searching (slow rotation)
        if math.isnan(bearing):
            self.state = AgentState.SEARCHING
            twist.angular.z = self.angular_speed
            self.cmd_vel_pub.publish(twist)
            return

        self.state = AgentState.APPROACHING

        if abs(bearing) < self.center_thresh:
            # Target roughly centered → drive forward
            twist.linear.x = self.linear_speed
            self.center_count += 1
        else:
            # Steer toward target
            twist.angular.z = -bearing * 1.5  # proportional turn
            self.center_count = 0

        if self.center_count >= self.arrived_frames:
            self.state = AgentState.ARRIVED

        self.cmd_vel_pub.publish(twist)
        self.get_logger().info(
            f"State: {self.state} | bearing: {bearing:.2f} | "
            f"lin: {twist.linear.x:.2f} ang: {twist.angular.z:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
