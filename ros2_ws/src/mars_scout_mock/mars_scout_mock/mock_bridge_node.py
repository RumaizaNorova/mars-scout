"""
mars_scout_mock — MockBridgeNode

A topic-perfect substitute for the Isaac Sim + ROS2 bridge.
Publishes exactly the same topics and TF frames that the real sim will publish,
so every downstream node (perception, geometry, control) runs unchanged.

Published topics (mirrors Isaac Sim bridge output)
---------------------------------------------------
/rover/camera/image_raw          sensor_msgs/Image          @ ~10 Hz
/rover/camera/depth/image_raw    sensor_msgs/Image          @ ~10 Hz
/rover/camera/camera_info        sensor_msgs/CameraInfo     @ ~10 Hz
/rover/odom                      nav_msgs/Odometry          @ 50 Hz
/rover/state                     mars_scout_msgs/RoverState @ 10 Hz

TF frames published
-------------------
  map → odom → base_link → camera_optical_frame

Subscribed topics
-----------------
/rover/cmd_vel   geometry_msgs/Twist   (consumed by kinematic model)

How the camera image works
--------------------------
A large Mars panorama is loaded at startup.  As the rover moves, a
1280×720 viewport is slid across the panorama proportional to the
rover's yaw angle — simulating a camera rotating with the rover body.
Depth is synthesised from the current RGB crop via DepthSynthesiser.
"""

from __future__ import annotations
import os
import math
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from sensor_msgs.msg  import Image, CameraInfo
from geometry_msgs.msg import Twist, TransformStamped, Quaternion, Point, Vector3
from nav_msgs.msg      import Odometry
from std_msgs.msg      import Header
from cv_bridge         import CvBridge
import tf2_ros

from mars_scout_mock.depth_synthesis  import DepthSynthesiser, DepthSynthesisConfig
from mars_scout_mock.kinematic_model  import DifferentialDriveModel, RoverPose

try:
    from mars_scout_msgs.msg import RoverState
    _HAS_MSGS = True
except ImportError:
    _HAS_MSGS = False   # allow running before msgs are built


# ── Constants ─────────────────────────────────────────────────────────────────
IMG_W, IMG_H = 1280, 720
CAMERA_FX = IMG_W / (2.0 * math.tan(math.radians(45.0)))
CAMERA_FY = CAMERA_FX
DEFAULT_PANORAMA = os.path.join(
    os.path.dirname(__file__), "..", "data", "mars_panorama.jpg"
)


class MockBridgeNode(Node):

    def __init__(self):
        super().__init__("mock_bridge_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("panorama_path",   DEFAULT_PANORAMA)
        self.declare_parameter("camera_hz",       10.0)
        self.declare_parameter("odom_hz",         50.0)
        self.declare_parameter("noise_sigma",     0.03)
        self.declare_parameter("fsm_state",       "SEARCHING")

        cam_hz  = self.get_parameter("camera_hz").value
        odom_hz = self.get_parameter("odom_hz").value
        pano_path = self.get_parameter("panorama_path").value

        # ── State ─────────────────────────────────────────────────────────────
        self._bridge   = CvBridge()
        self._rover    = DifferentialDriveModel()
        self._depth_synth = DepthSynthesiser(
            DepthSynthesisConfig(noise_sigma=self.get_parameter("noise_sigma").value)
        )
        self._panorama = self._load_panorama(pano_path)
        self._seq      = 0

        # ── TF broadcaster ────────────────────────────────────────────────────
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Twist, "/rover/cmd_vel", self._cb_cmd_vel, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self._pub_rgb    = self.create_publisher(Image,      "/rover/camera/image_raw",       sensor_qos)
        self._pub_depth  = self.create_publisher(Image,      "/rover/camera/depth/image_raw", sensor_qos)
        self._pub_info   = self.create_publisher(CameraInfo, "/rover/camera/camera_info",     sensor_qos)
        self._pub_odom   = self.create_publisher(Odometry,   "/rover/odom",                   10)
        if _HAS_MSGS:
            self._pub_state = self.create_publisher(RoverState, "/rover/state", 10)

        # ── Timers ────────────────────────────────────────────────────────────
        self.create_timer(1.0 / odom_hz, self._tick_odom)
        self.create_timer(1.0 / cam_hz,  self._tick_camera)

        self.get_logger().info(
            f"MockBridgeNode ready.  Panorama: {pano_path}  "
            f"cam@{cam_hz}Hz  odom@{odom_hz}Hz"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_cmd_vel(self, msg: Twist):
        self._rover.set_velocity(msg.linear.x, msg.angular.z)

    def _tick_odom(self):
        """50 Hz: integrate kinematics, publish odom + TF."""
        dt = 1.0 / self.get_parameter("odom_hz").value
        pose = self._rover.step(dt)
        stamp = self.get_clock().now().to_msg()
        self._publish_tf(pose, stamp)
        self._publish_odom(pose, stamp)

    def _tick_camera(self):
        """10 Hz: render camera view, synthesise depth, publish."""
        stamp = self.get_clock().now().to_msg()
        pose  = self._rover.pose

        rgb_crop = self._render_view(pose)
        depth    = self._depth_synth.synthesise(rgb_crop)

        self._publish_camera_info(stamp)
        self._publish_rgb(rgb_crop, stamp)
        self._publish_depth(depth, stamp)

        if _HAS_MSGS and hasattr(self, "_pub_state"):
            self._publish_rover_state(pose, stamp)

        self._seq += 1

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_view(self, pose: RoverPose) -> np.ndarray:
        """
        Crop a IMG_W×IMG_H viewport from the panorama.
        Viewport centre-x tracks rover yaw (full yaw range = full panorama width).
        Viewport centre-y is fixed at panorama mid-height.
        """
        pano_h, pano_w = self._panorama.shape[:2]

        # Map yaw (−π…π) to panorama x position
        yaw_norm  = (pose.yaw + math.pi) / (2 * math.pi)   # [0, 1]
        centre_x  = int(yaw_norm * pano_w)
        centre_y  = pano_h // 2

        x0 = centre_x - IMG_W // 2
        y0 = centre_y - IMG_H // 2

        # Tile horizontally so we never run out of panorama
        tiled = np.tile(self._panorama, (1, 3, 1))   # triple-wide
        x0 += pano_w   # offset into the middle tile

        crop = tiled[
            max(0, y0) : max(0, y0) + IMG_H,
            x0         : x0 + IMG_W,
        ]

        # Pad if crop is smaller than expected (edge of panorama)
        if crop.shape[:2] != (IMG_H, IMG_W):
            crop = cv2.resize(crop, (IMG_W, IMG_H))

        return crop

    # ── Publishers ────────────────────────────────────────────────────────────

    def _publish_rgb(self, bgr: np.ndarray, stamp):
        msg = self._bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        msg.header.stamp    = stamp
        msg.header.frame_id = "camera_optical_frame"
        self._pub_rgb.publish(msg)

    def _publish_depth(self, depth: np.ndarray, stamp):
        # Replace NaN with 0 for the ROS message (convention: 0 = invalid)
        d = depth.copy()
        d[np.isnan(d)] = 0.0
        msg = self._bridge.cv2_to_imgmsg(d, encoding="32FC1")
        msg.header.stamp    = stamp
        msg.header.frame_id = "camera_optical_frame"
        self._pub_depth.publish(msg)

    def _publish_camera_info(self, stamp):
        msg = CameraInfo()
        msg.header.stamp    = stamp
        msg.header.frame_id = "camera_optical_frame"
        msg.width  = IMG_W
        msg.height = IMG_H
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [
            CAMERA_FX, 0.0,       IMG_W / 2.0,
            0.0,       CAMERA_FY, IMG_H / 2.0,
            0.0,       0.0,       1.0,
        ]
        msg.r = [1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0]
        msg.p = [
            CAMERA_FX, 0.0,       IMG_W / 2.0, 0.0,
            0.0,       CAMERA_FY, IMG_H / 2.0, 0.0,
            0.0,       0.0,       1.0,          0.0,
        ]
        self._pub_info.publish(msg)

    def _publish_odom(self, pose: RoverPose, stamp):
        msg = Odometry()
        msg.header.stamp    = stamp
        msg.header.frame_id = "odom"
        msg.child_frame_id  = "base_link"

        msg.pose.pose.position    = Point(x=pose.x, y=pose.y, z=pose.z)
        q = pose.quaternion
        msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        msg.pose.covariance       = self._rover.odom_covariance.flatten().tolist()

        v = self._rover._vel
        msg.twist.twist.linear  = Vector3(x=v.linear,  y=0.0, z=0.0)
        msg.twist.twist.angular = Vector3(x=0.0, y=0.0, z=v.angular)

        self._pub_odom.publish(msg)

    def _publish_rover_state(self, pose: RoverPose, stamp):
        msg = RoverState()
        msg.header.stamp    = stamp
        msg.header.frame_id = "map"
        msg.fsm_state       = self.get_parameter("fsm_state").value
        self._pub_state.publish(msg)

    # ── TF ────────────────────────────────────────────────────────────────────

    def _publish_tf(self, pose: RoverPose, stamp):
        transforms = []

        # map → odom (static for mock; real bridge would account for loop-closure)
        transforms.append(self._make_tf("map", "odom", [0, 0, 0], [0, 0, 0, 1], stamp))

        # odom → base_link
        q = pose.quaternion
        transforms.append(self._make_tf(
            "odom", "base_link",
            [pose.x, pose.y, pose.z],
            q.tolist(), stamp,
        ))

        # base_link → camera_optical_frame  (fixed mount, pitched down 15°)
        cam_pos, cam_quat = self._rover.camera_pose_in_map()
        # Express as offset from base_link (simplified — use fixed offset)
        pitch = math.radians(-15.0)
        hp = pitch / 2
        cam_q_local = [math.sin(hp), 0.0, 0.0, math.cos(hp)]  # pitch rotation only
        transforms.append(self._make_tf(
            "base_link", "camera_optical_frame",
            [0.1, 0.0, 0.15],
            cam_q_local, stamp,
        ))

        self._tf_broadcaster.sendTransform(transforms)

    @staticmethod
    def _make_tf(parent, child, trans, rot, stamp) -> TransformStamped:
        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = parent
        t.child_frame_id  = child
        t.transform.translation.x = float(trans[0])
        t.transform.translation.y = float(trans[1])
        t.transform.translation.z = float(trans[2])
        t.transform.rotation.x = float(rot[0])
        t.transform.rotation.y = float(rot[1])
        t.transform.rotation.z = float(rot[2])
        t.transform.rotation.w = float(rot[3])
        return t

    # ── Panorama loading ──────────────────────────────────────────────────────

    def _load_panorama(self, path: str) -> np.ndarray:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                self.get_logger().info(f"Loaded panorama {img.shape} from {path}")
                return img

        self.get_logger().warn(
            f"Panorama not found at {path}. "
            "Using procedural Mars surface. Run data/download_samples.py for real imagery."
        )
        return self._generate_procedural_panorama()

    @staticmethod
    def _generate_procedural_panorama(width: int = 4096, height: int = 1024) -> np.ndarray:
        """
        Fallback: generate a procedural Mars-like terrain panorama using
        Perlin-style noise so the mock works with zero data dependencies.
        """
        rng = np.random.default_rng(42)

        # Base: reddish-brown Mars colour
        base = np.zeros((height, width, 3), dtype=np.uint8)
        base[:, :, 0] = 60   # B
        base[:, :, 1] = 80   # G
        base[:, :, 2] = 160  # R

        # Add terrain texture via layered noise
        for scale, amp in [(32, 40), (16, 20), (8, 10), (4, 5)]:
            noise_h = height // scale + 2
            noise_w = width  // scale + 2
            noise = rng.integers(0, 255, (noise_h, noise_w), dtype=np.uint8)
            noise_up = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
            for c in range(3):
                layer = (noise_up.astype(np.float32) / 255.0 * amp).astype(np.uint8)
                base[:, :, c] = np.clip(base[:, :, c].astype(np.int16) + layer - amp // 2, 0, 255).astype(np.uint8)

        # Scatter rocks (dark circular blobs)
        for _ in range(300):
            cx_ = rng.integers(0, width)
            cy_ = rng.integers(int(height * 0.4), height)
            r   = rng.integers(5, 40)
            colour = (int(rng.integers(30, 60)),) * 3
            cv2.circle(base, (cx_, cy_), r, colour, -1)

        # Sky gradient at the top (pinkish-purple Martian sky)
        sky_rows = int(height * 0.35)
        for row in range(sky_rows):
            frac = 1.0 - row / sky_rows
            base[row, :, 0] = int(100 + frac * 60)   # B
            base[row, :, 1] = int(60  + frac * 40)   # G
            base[row, :, 2] = int(120 + frac * 60)   # R

        return base


def main(args=None):
    rclpy.init(args=args)
    node = MockBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
