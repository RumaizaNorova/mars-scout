"""
Isaac Sim → Mars Scout ROS2 Bridge
====================================

Translates the topics that omni.isaac.ros2_bridge publishes into exactly
the same interface that mars_scout_mock publishes.  Every downstream node
(perception, geometry, control) is completely unaware of whether the source
is the mock or real Isaac Sim.

Isaac Sim topic layout (ros2_bridge defaults)
----------------------------------------------
  /isaac_camera/rgb          sensor_msgs/Image          BGR8
  /isaac_camera/depth        sensor_msgs/Image          32FC1 (metres)
  /isaac_camera/camera_info  sensor_msgs/CameraInfo
  /odom                      nav_msgs/Odometry          in odom frame
  /cmd_vel_input             geometry_msgs/Twist        (Isaac reads this)

Mars Scout standard interface (what this node publishes)
---------------------------------------------------------
  /rover/camera/image_raw           sensor_msgs/Image
  /rover/camera/depth/image_raw     sensor_msgs/Image
  /rover/camera/camera_info         sensor_msgs/CameraInfo
  /rover/odom                       nav_msgs/Odometry
  /rover/state                      mars_scout_msgs/RoverState
  TF: map -> odom -> base_link -> camera_optical_frame

This node subscribes to Isaac Sim, remaps, and re-publishes.  It also
subscribes to /rover/cmd_vel and forwards to whatever Isaac Sim listens on.

Configuration (ROS2 parameters)
--------------------------------
  isaac_rgb_topic       (default: /isaac_camera/rgb)
  isaac_depth_topic     (default: /isaac_camera/depth)
  isaac_info_topic      (default: /isaac_camera/camera_info)
  isaac_odom_topic      (default: /odom)
  isaac_cmdvel_topic    (default: /cmd_vel)
  publish_hz            (default: 15.0)   throttle if Isaac Sim runs faster

Usage
-----
  ros2 run mars_scout_sim_bridge sim_bridge_node
  ros2 run mars_scout_sim_bridge sim_bridge_node \
      --ros-args -p isaac_rgb_topic:=/my_cam/image_raw
"""

from __future__ import annotations
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import tf2_ros
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from builtin_interfaces.msg import Time as RosTime

from mars_scout_msgs.msg import RoverState


# ── QoS profiles ──────────────────────────────────────────────────────────────

_SENSOR_QOS = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

_RELIABLE_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


# ── Bridge node ───────────────────────────────────────────────────────────────

class SimBridgeNode(Node):
    """
    Remaps Isaac Sim topics to the Mars Scout standard interface.

    No image processing is done here — just header-fixing and republishing.
    The bridge is deliberately thin so it's easy to test offline.
    """

    def __init__(self):
        super().__init__("sim_bridge_node")

        # ── parameters ────────────────────────────────────────────────────────
        self.declare_parameter("isaac_rgb_topic",    "/isaac_camera/rgb")
        self.declare_parameter("isaac_depth_topic",  "/isaac_camera/depth")
        self.declare_parameter("isaac_info_topic",   "/isaac_camera/camera_info")
        self.declare_parameter("isaac_odom_topic",   "/odom")
        self.declare_parameter("isaac_cmdvel_topic", "/cmd_vel")
        self.declare_parameter("publish_hz",         15.0)

        rgb_in    = self.get_parameter("isaac_rgb_topic").value
        depth_in  = self.get_parameter("isaac_depth_topic").value
        info_in   = self.get_parameter("isaac_info_topic").value
        odom_in   = self.get_parameter("isaac_odom_topic").value
        cmdvel_out = self.get_parameter("isaac_cmdvel_topic").value

        # ── publishers (Mars Scout standard) ──────────────────────────────────
        self._pub_rgb   = self.create_publisher(Image,      "/rover/camera/image_raw",       _RELIABLE_QOS)
        self._pub_depth = self.create_publisher(Image,      "/rover/camera/depth/image_raw", _RELIABLE_QOS)
        self._pub_info  = self.create_publisher(CameraInfo, "/rover/camera/camera_info",     _RELIABLE_QOS)
        self._pub_odom  = self.create_publisher(Odometry,   "/rover/odom",                   _RELIABLE_QOS)
        self._pub_state = self.create_publisher(RoverState, "/rover/state",                  _RELIABLE_QOS)
        self._pub_chase = self.create_publisher(Image,      "/rover/chase_cam",              _RELIABLE_QOS)

        # ── cmd_vel passthrough (rover → Isaac Sim) ────────────────────────────
        self._pub_cmdvel = self.create_publisher(Twist, cmdvel_out, _RELIABLE_QOS)
        self.create_subscription(Twist, "/rover/cmd_vel",
                                 self._on_cmdvel, _RELIABLE_QOS)

        # ── subscribers (Isaac Sim) ────────────────────────────────────────────
        self.create_subscription(Image,      rgb_in,              self._on_rgb,   _SENSOR_QOS)
        self.create_subscription(Image,      depth_in,            self._on_depth, _SENSOR_QOS)
        self.create_subscription(CameraInfo, info_in,             self._on_info,  _SENSOR_QOS)
        self.create_subscription(Odometry,   odom_in,             self._on_odom,  _SENSOR_QOS)
        self.create_subscription(Image,      "/isaac_camera/chase", self._on_chase, _SENSOR_QOS)

        # ── TF broadcaster ────────────────────────────────────────────────────
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ── throttle state ────────────────────────────────────────────────────
        hz = self.get_parameter("publish_hz").value
        self._min_dt  = 1.0 / hz if hz > 0 else 0.0
        self._last_t  = {k: 0.0 for k in ("rgb", "depth", "info", "odom")}

        # track latest odom for state publishing
        self._latest_odom: Odometry | None = None

        hz_state = 5.0
        self.create_timer(1.0 / hz_state, self._publish_state)

        # Publish a static TF tree even before Isaac Sim odom arrives,
        # so downstream nodes (VLM, projection) can start immediately.
        self.create_timer(0.2, self._ensure_tf)

        # ── Camera intrinsics for fabricated CameraInfo ───────────────────────
        # Isaac Sim ROS2CameraHelper (type "rgb"/"depth") does NOT publish
        # camera_info.  We fabricate it from the known camera parameters:
        #   focal length 18 mm on a 1280×720 sensor.
        # Using the standard 36mm-wide sensor assumption for Isaac Sim:
        #   fx = fy = (focal_mm / sensor_width_mm) * image_width_px
        #           = (18 / 36) * 1280 = 640
        self._cam_width  = 1280
        self._cam_height = 720
        _fx = 640.0   # (18mm focal / 36mm sensor) * 1280px
        _fy = 640.0
        _cx = self._cam_width  / 2.0   # 640.0
        _cy = self._cam_height / 2.0   # 360.0
        self._fabricated_camera_info = self._make_camera_info(
            _fx, _fy, _cx, _cy, self._cam_width, self._cam_height
        )
        # Publish fabricated camera_info at a steady rate so projection_node
        # gets it even if Isaac Sim's OmniGraph never sends /isaac_camera/camera_info
        self.create_timer(0.1, self._publish_camera_info_heartbeat)

        self.get_logger().info(
            f"SimBridgeNode ready. "
            f"Subscribing Isaac Sim on: {rgb_in}, {depth_in}, {odom_in}. "
            f"Publishing Mars Scout on /rover/*"
        )

    # ── Isaac Sim → Mars Scout republishing ───────────────────────────────────

    def _throttle_ok(self, key: str) -> bool:
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_t[key] >= self._min_dt:
            self._last_t[key] = now
            return True
        return False

    def _fix_stamp(self, stamp: RosTime) -> RosTime:
        """
        Replace a zero timestamp (Isaac Sim publishes stamp=0 in headless mode)
        with the current wall-clock time.  This prevents TF_OLD_DATA warnings
        and downstream 'timestamp=0' complaints in all ROS2 nodes.
        """
        if stamp.sec == 0 and stamp.nanosec == 0:
            return self.get_clock().now().to_msg()
        return stamp

    def _on_rgb(self, msg: Image):
        if not self._throttle_ok("rgb"):
            return
        msg.header.frame_id = "camera_optical_frame"
        msg.header.stamp = self._fix_stamp(msg.header.stamp)
        self._pub_rgb.publish(msg)

    def _on_depth(self, msg: Image):
        if not self._throttle_ok("depth"):
            return
        msg.header.frame_id = "camera_optical_frame"
        msg.header.stamp = self._fix_stamp(msg.header.stamp)
        # Isaac Sim publishes depth as 32FC1 (metres) — so do we. No conversion.
        self._pub_depth.publish(msg)

    def _on_info(self, msg: CameraInfo):
        if not self._throttle_ok("info"):
            return
        msg.header.frame_id = "camera_optical_frame"
        msg.header.stamp = self._fix_stamp(msg.header.stamp)
        self._pub_info.publish(msg)

    def _on_odom(self, msg: Odometry):
        if not self._throttle_ok("odom"):
            return
        msg.header.frame_id     = "odom"
        msg.child_frame_id      = "base_link"
        msg.header.stamp = self._fix_stamp(msg.header.stamp)
        self._latest_odom = msg
        self._pub_odom.publish(msg)
        self._broadcast_tf(msg)

    # ── cmd_vel passthrough ───────────────────────────────────────────────────

    def _on_chase(self, msg: Image):
        """Republish chase camera — 3rd-person view of the rover."""
        msg.header.frame_id = "chase_cam_frame"
        msg.header.stamp = self._fix_stamp(msg.header.stamp)
        self._pub_chase.publish(msg)

    def _on_cmdvel(self, msg: Twist):
        """Forward rover velocity commands to whatever topic Isaac Sim reads."""
        self._pub_cmdvel.publish(msg)

    # ── TF broadcasting ───────────────────────────────────────────────────────

    def _broadcast_tf(self, odom: Odometry):
        """Publish TF from live Isaac Sim odometry."""
        p = odom.pose.pose.position
        r = odom.pose.pose.orientation
        self._broadcast_tf_at(
            odom.header.stamp,
            p.x, p.y, p.z,
            r.x, r.y, r.z, r.w,
        )

    def _broadcast_tf_at(self, stamp, px, py, pz, rx, ry, rz, rw):
        """
        Publish TF tree: map → odom (identity) → base_link → camera_optical_frame.
        Accepts explicit position/orientation so it works with or without live odom.
        """
        # map -> odom (identity; no localisation)
        t_map_odom = TransformStamped()
        t_map_odom.header.stamp    = stamp
        t_map_odom.header.frame_id = "map"
        t_map_odom.child_frame_id  = "odom"
        t_map_odom.transform.rotation.w = 1.0

        # odom -> base_link
        t_odom_base = TransformStamped()
        t_odom_base.header.stamp    = stamp
        t_odom_base.header.frame_id = "odom"
        t_odom_base.child_frame_id  = "base_link"
        t_odom_base.transform.translation.x = px
        t_odom_base.transform.translation.y = py
        t_odom_base.transform.translation.z = pz
        t_odom_base.transform.rotation.x = rx
        t_odom_base.transform.rotation.y = ry
        t_odom_base.transform.rotation.z = rz
        t_odom_base.transform.rotation.w = rw if rw != 0.0 else 1.0

        # base_link -> camera_optical_frame (fixed: 0.15m forward, 0.10m up, -15deg pitch)
        t_base_cam = TransformStamped()
        t_base_cam.header.stamp    = stamp
        t_base_cam.header.frame_id = "base_link"
        t_base_cam.child_frame_id  = "camera_optical_frame"
        t_base_cam.transform.translation.x = 0.15
        t_base_cam.transform.translation.z = 0.10
        pitch = -math.radians(15.0)
        t_base_cam.transform.rotation.y = math.sin(pitch / 2.0)
        t_base_cam.transform.rotation.w = math.cos(pitch / 2.0)

        self._tf_broadcaster.sendTransform([t_map_odom, t_odom_base, t_base_cam])

    # ── TF heartbeat (runs even before Isaac Sim sends odom) ─────────────────

    def _ensure_tf(self):
        """
        Keep the TF tree alive even when /odom isn't arriving.
        When real odom arrives, _broadcast_tf() takes over at its higher rate.
        """
        if self._latest_odom is not None:
            return  # _broadcast_tf is already handling it
        stamp = self.get_clock().now().to_msg()
        self._broadcast_tf_at(stamp, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    # ── camera_info helpers ───────────────────────────────────────────────────

    @staticmethod
    def _make_camera_info(fx: float, fy: float, cx: float, cy: float,
                          width: int, height: int) -> CameraInfo:
        """Build a CameraInfo message from pinhole intrinsics."""
        info = CameraInfo()
        info.header.frame_id = "camera_optical_frame"
        info.width  = width
        info.height = height
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        # K (3×3 row-major)
        info.k = [fx,  0.0, cx,
                  0.0, fy,  cy,
                  0.0, 0.0, 1.0]
        # R = identity
        info.r = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
        # P (3×4 row-major)
        info.p = [fx,  0.0, cx,  0.0,
                  0.0, fy,  cy,  0.0,
                  0.0, 0.0, 1.0, 0.0]
        return info

    def _publish_camera_info_heartbeat(self):
        """
        Publish fabricated CameraInfo at 10 Hz so projection_node always has it.
        If Isaac Sim sends a real camera_info via _on_info, that will overwrite
        the fabricated one; this heartbeat is the reliable fallback.
        """
        msg = self._fabricated_camera_info
        msg.header.stamp = self.get_clock().now().to_msg()
        self._pub_info.publish(msg)

    # ── rover state (derived from odometry) ──────────────────────────────────

    def _publish_state(self):
        """Publish rover state; emits zeros at origin if Isaac Sim odom is absent."""
        msg = RoverState()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.fsm_state       = "SEARCHING"   # sim bridge doesn't run the FSM
        # Populate active_query_text so the dashboard shows something meaningful
        # instead of "???".  The default query is the one set in the launch file.
        msg.active_query_text = "rock formation"
        msg.active_query_id   = "sim_bridge"
        msg.distance_to_goal  = math.nan
        if self._latest_odom is not None:
            msg.pose     = self._latest_odom.pose
            msg.velocity = self._latest_odom.twist
        self._pub_state.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
