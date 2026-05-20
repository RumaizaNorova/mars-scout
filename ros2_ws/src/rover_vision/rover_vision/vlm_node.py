"""
ROS2 node: subscribes to /rover/camera/image_raw,
runs Moondream2 on each frame, publishes terrain description
and a target bearing to /rover/vlm/description and /rover/vlm/target_bearing.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
from PIL import Image as PILImage
import cv2
import numpy as np

from rover_vision.moondream_client import MoondreamClient


class VLMNode(Node):
    def __init__(self):
        super().__init__("vlm_node")

        # Parameters
        self.declare_parameter("query", "What terrain features are visible? Is there a rock formation or obstacle ahead?")
        self.declare_parameter("target_description", "large rock")
        self.declare_parameter("inference_rate_hz", 1.0)

        self.query_prompt = self.get_parameter("query").value
        self.target_desc = self.get_parameter("target_description").value
        rate_hz = self.get_parameter("inference_rate_hz").value

        self.bridge = CvBridge()
        self.vlm = MoondreamClient()

        # Subscribers
        self.image_sub = self.create_subscription(
            RosImage, "/rover/camera/image_raw", self.image_callback, 10
        )

        # Publishers
        self.desc_pub = self.create_publisher(String, "/rover/vlm/description", 10)
        self.bearing_pub = self.create_publisher(Float32, "/rover/vlm/target_bearing", 10)

        # Timer-gated inference (don't run on every frame)
        self.latest_frame = None
        self.timer = self.create_timer(1.0 / rate_hz, self.run_inference)

        self.get_logger().info(f"VLM node ready. Target: '{self.target_desc}'")

    def image_callback(self, msg: RosImage):
        self.latest_frame = msg

    def run_inference(self):
        if self.latest_frame is None:
            return

        # Convert ROS image → PIL
        cv_img = self.bridge.imgmsg_to_cv2(self.latest_frame, desired_encoding="rgb8")
        pil_img = PILImage.fromarray(cv_img)

        # 1. General terrain description
        description = self.vlm.query(pil_img, self.query_prompt)
        msg = String()
        msg.data = description
        self.desc_pub.publish(msg)
        self.get_logger().info(f"VLM: {description}")

        # 2. Target detection — ask where the target is
        target_prompt = (
            f"Is there a {self.target_desc} visible? "
            "If yes, is it on the left, center, or right side of the image?"
        )
        target_answer = self.vlm.query(pil_img, target_prompt)
        bearing = self._parse_bearing(target_answer)
        bearing_msg = Float32()
        bearing_msg.data = bearing
        self.bearing_pub.publish(bearing_msg)
        self.get_logger().info(f"Target bearing: {bearing:.2f} rad ({target_answer})")

    def _parse_bearing(self, answer: str) -> float:
        """Convert VLM left/center/right answer to a bearing in radians."""
        answer_lower = answer.lower()
        if "left" in answer_lower:
            return -0.5   # turn left
        elif "right" in answer_lower:
            return 0.5    # turn right
        elif "center" in answer_lower or "yes" in answer_lower:
            return 0.0    # straight ahead
        else:
            return float("nan")  # not found


def main(args=None):
    rclpy.init(args=args)
    node = VLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
