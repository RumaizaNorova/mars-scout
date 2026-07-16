"""
ROS2 node: mars_scout_geometry/projection_node

Subscribes
----------
/rover/camera/image_raw          sensor_msgs/Image
/rover/camera/depth/image_raw    sensor_msgs/Image
/rover/camera/camera_info        sensor_msgs/CameraInfo
/rover/vlm/terrain_target        mars_scout_msgs/TerrainTarget  (2-D detections)

Publishes
---------
/rover/geometry/terrain_target   mars_scout_msgs/TerrainTarget  (with 3-D waypoints filled in)
/rover/geometry/ground_plane     visualization_msgs/Marker       (RViz ground plane)
/rover/geometry/waypoint_marker  visualization_msgs/Marker       (RViz target sphere)

Pipeline per detection
----------------------
1. Read normalised bbox centre from incoming TerrainTarget
2. Back-project using depth image + CameraModel → 3-D point in camera frame
3. Fit RANSAC ground plane to the full depth point cloud
4. Project 3-D point onto ground plane → driveable waypoint
5. Transform waypoint from camera frame → map frame (tf2)
6. Attach to TerrainTarget and republish
"""

from __future__ import annotations
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Point, Vector3
from visualization_msgs.msg import Marker
import tf2_ros
import tf2_geometry_msgs   # noqa: F401  (registers PointStamped transform support)

from mars_scout_msgs.msg import TerrainTarget
from mars_scout_geometry.camera_model import CameraModel, CameraIntrinsics
from mars_scout_geometry.ground_plane import RANSACGroundFit


class ProjectionNode(Node):

    def __init__(self):
        super().__init__("projection_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("ransac_iters",        100)
        self.declare_parameter("ransac_inlier_m",     0.05)
        self.declare_parameter("ransac_subsample",    2000)
        self.declare_parameter("depth_patch_radius",  5)
        self.declare_parameter("camera_frame",        "camera_optical_frame")
        self.declare_parameter("map_frame",           "map")

        self._cam_frame = self.get_parameter("camera_frame").value
        self._map_frame = self.get_parameter("map_frame").value

        # ── State ─────────────────────────────────────────────────────────────
        self._cam_model  = None   # built once CameraInfo arrives
        self._ransac     = RANSACGroundFit(
            n_iters       = self.get_parameter("ransac_iters").value,
            inlier_thresh = self.get_parameter("ransac_inlier_m").value,
            subsample     = self.get_parameter("ransac_subsample").value,
        )
        self._latest_depth: np.ndarray | None = None
        self._depth_pts:    np.ndarray | None = None   # (N,3) point cloud cache

        # ── TF2 ───────────────────────────────────────────────────────────────
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(CameraInfo,    "/rover/camera/camera_info",     self._cb_info,    1)
        self.create_subscription(Image,         "/rover/camera/depth/image_raw", self._cb_depth,   5)
        self.create_subscription(TerrainTarget, "/rover/vlm/terrain_target",     self._cb_target,  10)

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_target   = self.create_publisher(TerrainTarget, "/rover/geometry/terrain_target",  10)
        self._pub_plane    = self.create_publisher(Marker,        "/rover/geometry/ground_plane",    1)
        self._pub_waypoint = self.create_publisher(Marker,        "/rover/geometry/waypoint_marker", 10)

        self.get_logger().info("Projection node ready.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_info(self, msg: CameraInfo):
        if self._cam_model is not None:
            return  # already built
        K = msg.k   # row-major 3×3
        intr = CameraIntrinsics(
            fx=K[0], fy=K[4], cx=K[2], cy=K[5],
            width=msg.width, height=msg.height,
        )
        self._cam_model = CameraModel(intr)
        self.get_logger().info(f"Camera model built: {msg.width}×{msg.height}  "
                               f"fx={intr.fx:.1f} fy={intr.fy:.1f}")

    def _cb_depth(self, msg: Image):
        # Manual decode — no cv_bridge needed; 32FC1 = raw float32 bytes
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(
            (msg.height, msg.width)
        )
        self._latest_depth = depth.copy()

        # Pre-compute point cloud for RANSAC (runs once per depth frame)
        if self._cam_model is not None:
            self._depth_pts = self._build_pointcloud(self._latest_depth)

    def _cb_target(self, msg: TerrainTarget):
        if not msg.target_found:
            self._pub_target.publish(msg)
            return

        if self._cam_model is None or self._latest_depth is None:
            self.get_logger().warn("No camera info or depth yet — dropping target.")
            return

        try:
            point_3d, depth_used, depth_std = self._cam_model.backproject_bbox(
                msg.bbox_cx, msg.bbox_cy,
                self._latest_depth,
                patch_radius=self.get_parameter("depth_patch_radius").value,
            )
        except ValueError as e:
            self.get_logger().warn(f"Back-projection failed: {e}")
            return

        # Depth confidence: small std → confident, large std → noisy
        depth_conf = float(np.exp(-depth_std / 0.3))

        # Fill camera-frame position + uncertainty
        msg.position_camera = Point(x=point_3d[0], y=point_3d[1], z=point_3d[2])
        msg.position_sigma  = Vector3(x=depth_std, y=depth_std, z=depth_std * 0.5)
        msg.depth_confidence = depth_conf
        msg.overall_confidence = 0.6 * msg.vlm_confidence + 0.4 * depth_conf

        # Ground-plane waypoint
        waypoint_cam = self._ground_project(point_3d)
        waypoint_map = self._to_map_frame(waypoint_cam, msg.header.stamp)

        if waypoint_map is not None:
            msg.waypoint = waypoint_map
            self._publish_waypoint_marker(waypoint_map)

        self._pub_target.publish(msg)

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _build_pointcloud(self, depth: np.ndarray) -> np.ndarray:
        """Convert full depth image to (N,3) point cloud in camera frame."""
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        valid = (depth > 0.05) & (depth < 20.0)
        pts = self._cam_model.backproject(
            u[valid].astype(np.float64),
            v[valid].astype(np.float64),
            depth[valid].astype(np.float64),
        )
        return pts

    def _ground_project(self, point_cam: np.ndarray) -> np.ndarray:
        """
        Project a 3-D point onto the estimated ground plane.
        Falls back to setting y = 0 (Isaac Sim y-up convention) if RANSAC fails.
        """
        if self._depth_pts is not None and len(self._depth_pts) > 100:
            plane, inliers = self._ransac.fit(self._depth_pts)
            if plane is not None:
                self._publish_plane_marker(plane)
                projected = plane.project_onto(point_cam[np.newaxis])[0]
                return projected

        # Fallback: drop point to y=0 plane
        fallback = point_cam.copy()
        fallback[1] = 0.0
        return fallback

    def _to_map_frame(self, point_cam: np.ndarray, stamp) -> PointStamped | None:
        """Transform a point from camera_optical_frame → map using tf2."""
        ps = PointStamped()
        ps.header.frame_id = self._cam_frame
        ps.header.stamp    = stamp
        ps.point = Point(x=float(point_cam[0]),
                         y=float(point_cam[1]),
                         z=float(point_cam[2]))
        try:
            return self._tf_buffer.transform(ps, self._map_frame,
                                             timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            return None

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _publish_waypoint_marker(self, waypoint: PointStamped):
        m = Marker()
        m.header = waypoint.header
        m.ns, m.id = "mars_scout", 1
        m.type   = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position = waypoint.point
        m.scale = Vector3(x=0.3, y=0.3, z=0.3)
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.4, 0.0, 0.9
        self._pub_waypoint.publish(m)

    def _publish_plane_marker(self, plane):
        m = Marker()
        m.header.frame_id = self._cam_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id = "mars_scout", 2
        m.type   = Marker.CUBE
        m.action = Marker.ADD
        # Thin flat disc representing the ground plane estimate
        m.scale = Vector3(x=6.0, y=0.02, z=6.0)
        m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.8, 0.2, 0.3
        self._pub_plane.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = ProjectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
