"""
mars_scout_perception — PerceptionNode

Bridges the camera feed and VLM backend into the ROS2 graph.

Subscribed
----------
/rover/camera/image_raw       sensor_msgs/Image
/rover/vlm/query              mars_scout_msgs/TerrainQuery   (operator goal)

Published
---------
/rover/vlm/terrain_target     mars_scout_msgs/TerrainTarget

Parameters
----------
backend          : "mock" | "moondream2"   (default: "mock")
inference_rate_hz: float                   max VLM calls per second (default: 2.0)
min_confidence   : float                   suppress targets below this (default: 0.4)
"""

from __future__ import annotations
import uuid
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from mars_scout_msgs.msg import TerrainQuery, TerrainTarget
from mars_scout_perception.vlm_backend   import VLMResult, BoundingBox
from mars_scout_perception.mock_vlm_backend import MockVLMBackend


class PerceptionNode(Node):

    def __init__(self):
        super().__init__("perception_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("backend",           "mock")
        self.declare_parameter("inference_rate_hz", 2.0)
        self.declare_parameter("min_confidence",    0.4)
        self.declare_parameter("default_query",     "rock formation")

        backend_name = self.get_parameter("backend").value
        rate_hz      = self.get_parameter("inference_rate_hz").value
        self._min_conf  = self.get_parameter("min_confidence").value
        self._query     = self.get_parameter("default_query").value
        self._query_id  = str(uuid.uuid4())[:8]

        # ── VLM backend ───────────────────────────────────────────────────────
        self._vlm = self._build_backend(backend_name)
        self.get_logger().info(f"VLM backend: {self._vlm.name}")

        # ── State ─────────────────────────────────────────────────────────────
        self._bridge       = CvBridge()
        self._latest_frame = None

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Image,        "/rover/camera/image_raw", self._cb_image, 5)
        self.create_subscription(TerrainQuery, "/rover/vlm/query",        self._cb_query, 10)

        # ── Publisher ─────────────────────────────────────────────────────────
        self._pub = self.create_publisher(TerrainTarget, "/rover/vlm/terrain_target", 10)

        # ── Inference timer ───────────────────────────────────────────────────
        self.create_timer(1.0 / rate_hz, self._run_inference)

        self.get_logger().info(
            f"PerceptionNode ready  backend={self._vlm.name}  "
            f"rate={rate_hz}Hz  query='{self._query}'"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_image(self, msg: Image):
        self._latest_frame = msg

    def _cb_query(self, msg: TerrainQuery):
        self._query    = msg.query_text
        self._query_id = msg.query_id or str(uuid.uuid4())[:8]
        self.get_logger().info(f"New query: '{self._query}' (id={self._query_id})")

    def _run_inference(self):
        if self._latest_frame is None:
            return

        bgr = self._bridge.imgmsg_to_cv2(self._latest_frame, desired_encoding="bgr8")
        result: VLMResult = self._vlm.query(bgr, self._query)

        msg = self._result_to_msg(result, self._latest_frame.header)
        self._pub.publish(msg)

        self.get_logger().info(
            f"[{self._vlm.name}] found={result.target_found}  "
            f"conf={result.confidence:.2f}  {result.inference_ms:.0f}ms  "
            f"'{result.description[:60]}'"
        )

    # ── Conversion ────────────────────────────────────────────────────────────

    def _result_to_msg(self, result: VLMResult, header) -> TerrainTarget:
        msg = TerrainTarget()
        msg.header       = header
        msg.query_id     = self._query_id
        msg.target_found = result.target_found
        msg.vlm_description = result.description
        msg.vlm_confidence  = float(result.confidence)
        msg.depth_confidence  = 0.0
        msg.overall_confidence = float(result.confidence)

        if result.target_found and result.bbox is not None:
            msg.bbox_cx = float(result.bbox.cx)
            msg.bbox_cy = float(result.bbox.cy)
            msg.bbox_w  = float(result.bbox.w)
            msg.bbox_h  = float(result.bbox.h)

        return msg

    # ── Backend factory ───────────────────────────────────────────────────────

    def _build_backend(self, name: str):
        if name == "mock":
            return MockVLMBackend(
                min_confidence=self.get_parameter("min_confidence").value
            )
        elif name == "moondream2":
            from mars_scout_perception.vlm_backend import Moondream2Backend
            return Moondream2Backend()
        else:
            self.get_logger().warn(f"Unknown backend '{name}', falling back to mock.")
            return MockVLMBackend()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
