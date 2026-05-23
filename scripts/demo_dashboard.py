#!/usr/bin/env python3
"""
Mars Scout — Live Dashboard (MJPEG web server + ROS2 subscriber)

Setup on Vast.ai instance:
  source /opt/ros/jazzy/setup.bash
  source /root/mars-rover-agent/ros2_ws/install/setup.bash
  pip install flask opencv-python numpy
  python3 scripts/demo_dashboard.py

Tunnel from Mac:
  ssh -p PORT -L 8765:localhost:8765 root@IP -N
  open http://localhost:8765
"""

import sys
import threading
import time
import datetime
import signal

import cv2
import numpy as np
from flask import Flask, Response

# ---------------------------------------------------------------------------
# Optional custom message type — gracefully degrade if not built yet
# ---------------------------------------------------------------------------
try:
    from mars_scout_msgs.msg import RoverState
    HAS_ROVER_STATE = True
except ImportError:
    HAS_ROVER_STATE = False
    print("[warn] mars_scout_msgs not found — rover state overlay disabled", file=sys.stderr)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped

# ---------------------------------------------------------------------------
# Shared state (written by ROS thread, read by Flask thread)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_state = {
    "frame_bgr": None,          # latest raw numpy frame
    "last_frame_ts": 0.0,       # time.monotonic() of last frame
    "rover_x": 0.0,
    "rover_y": 0.0,
    "rover_yaw_deg": 0.0,
    "linear_vel": 0.0,
    "target_x": None,
    "target_y": None,
    "last_target_ts": 0.0,      # time.monotonic() of last target msg
}

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX
DARK_BAR = (20, 20, 20)        # BGR
WHITE = (255, 255, 255)
GREEN = (0, 220, 80)
RED = (60, 60, 240)
BAR_ALPHA = 0.65
BAR_H = 36


def _overlay_bar(img, y0, y1, alpha=BAR_ALPHA):
    """Draw a semi-transparent dark horizontal bar in-place."""
    roi = img[y0:y1, :]
    bar = np.full_like(roi, DARK_BAR)
    cv2.addWeighted(bar, alpha, roi, 1 - alpha, 0, roi)
    img[y0:y1, :] = roi


def draw_overlay(frame, state_snap):
    img = frame.copy()
    h, w = img.shape[:2]

    now_utc = datetime.datetime.utcnow().strftime("%H:%M:%S UTC")

    # ---- top bar ----
    _overlay_bar(img, 0, BAR_H)

    cv2.putText(img, "MARS SCOUT  |  LIVE", (10, 25),
                FONT, 0.65, WHITE, 1, cv2.LINE_AA)

    (tw, _), _ = cv2.getTextSize(now_utc, FONT, 0.55, 1)
    cv2.putText(img, now_utc, (w - tw - 10, 24),
                FONT, 0.55, WHITE, 1, cv2.LINE_AA)

    # ---- status dot (top-right corner, below time) ----
    fresh = (time.monotonic() - state_snap["last_frame_ts"]) < 2.0
    dot_color = GREEN if fresh else RED
    cv2.circle(img, (w - 10, BAR_H + 12), 6, dot_color, -1, cv2.LINE_AA)

    # ---- bottom bar ----
    _overlay_bar(img, h - BAR_H, h)

    rx = state_snap["rover_x"]
    ry = state_snap["rover_y"]
    hdg = state_snap["rover_yaw_deg"] % 360
    vel = state_snap["linear_vel"]
    tx = state_snap["target_x"]
    ty = state_snap["target_y"]

    if tx is not None:
        tgt_str = f"Target: ({tx:.1f}, {ty:.1f})"
    else:
        tgt_str = "Target: --"

    bottom_text = (
        f"Rover  x={rx:.1f}m  y={ry:.1f}m  hdg={hdg:05.1f}°"
        f"  |  {tgt_str}"
        f"  |  Vel: {vel:.2f} m/s"
    )
    cv2.putText(img, bottom_text, (10, h - 10),
                FONT, 0.55, WHITE, 1, cv2.LINE_AA)

    # ---- target crosshair (centre of frame) ----
    target_age = time.monotonic() - state_snap["last_target_ts"]
    if target_age < 3.0 and tx is not None:
        cx, cy = w // 2, h // 2
        r = 28
        thickness = 2
        cv2.circle(img, (cx, cy), r, GREEN, thickness, cv2.LINE_AA)
        cv2.line(img, (cx - r - 8, cy), (cx + r + 8, cy), GREEN, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy - r - 8), (cx, cy + r + 8), GREEN, thickness, cv2.LINE_AA)

    return img


def encode_jpeg(img, quality=82):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return buf.tobytes()


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------
class DashboardNode(Node):
    def __init__(self):
        super().__init__("demo_dashboard")

        self.create_subscription(Image, "/rover/camera/image_raw",
                                 self._cb_image, 1)
        self.create_subscription(PointStamped, "/rover/geometry/terrain_target",
                                 self._cb_target, 10)

        if HAS_ROVER_STATE:
            self.create_subscription(RoverState, "/rover/state",
                                     self._cb_state, 10)

        self.get_logger().info("DashboardNode ready — subscribing to camera, state, target")

    # ROS Image → numpy BGR
    def _cb_image(self, msg: Image):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = arr.reshape((msg.height, msg.width, 3))
            if msg.encoding.lower() == "rgb8":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.get_logger().warn(f"image decode error: {e}")
            return

        with _lock:
            _state["frame_bgr"] = frame
            _state["last_frame_ts"] = time.monotonic()

    def _cb_state(self, msg):
        import math
        with _lock:
            _state["rover_x"] = float(msg.x)
            _state["rover_y"] = float(msg.y)
            _state["rover_yaw_deg"] = float(math.degrees(msg.yaw))
            _state["linear_vel"] = float(msg.linear_vel)

    def _cb_target(self, msg: PointStamped):
        with _lock:
            _state["target_x"] = float(msg.point.x)
            _state["target_y"] = float(msg.point.y)
            _state["last_target_ts"] = time.monotonic()


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Mars Scout — Live View</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0a0a0a;
      color: #00e05a;
      font-family: 'Courier New', Courier, monospace;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 12px 0 20px;
    }
    h1 {
      font-size: 1.1rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      margin-bottom: 12px;
      color: #00e05a;
      text-shadow: 0 0 8px #00e05a88;
    }
    .stream-wrap {
      width: 95vw;
      max-width: 1280px;
      border: 1px solid #00e05a44;
      box-shadow: 0 0 18px #00e05a22;
      background: #000;
      line-height: 0;
    }
    img {
      width: 100%;
      height: auto;
      display: block;
    }
    footer {
      margin-top: 10px;
      font-size: 0.7rem;
      color: #00e05a55;
      letter-spacing: 0.1em;
    }
  </style>
</head>
<body>
  <h1>Mars Scout &mdash; Live View</h1>
  <div class="stream-wrap">
    <img src="/stream" alt="live stream">
  </div>
  <footer>MJPEG &bull; /rover/camera/image_raw &bull; port 8765</footer>
</body>
</html>
"""

_BLANK_FRAME: bytes | None = None  # lazily created placeholder


def _get_blank_jpeg():
    global _BLANK_FRAME
    if _BLANK_FRAME is None:
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(blank, "Waiting for camera...", (480, 360),
                    FONT, 1.2, (0, 180, 60), 2, cv2.LINE_AA)
        _BLANK_FRAME = encode_jpeg(blank)
    return _BLANK_FRAME


def _mjpeg_generator():
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        with _lock:
            frame = _state["frame_bgr"]
            snap = dict(_state)  # shallow copy of scalars is fine

        if frame is None:
            jpg = _get_blank_jpeg()
        else:
            rendered = draw_overlay(frame, snap)
            jpg = encode_jpeg(rendered)

        if jpg:
            yield boundary + jpg + b"\r\n"

        time.sleep(1 / 30)  # cap at ~30 fps


@app.route("/")
def index():
    return HTML_PAGE, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/stream")
def stream():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    rclpy.init()
    node = DashboardNode()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    def _shutdown(sig, frame):
        print("\n[demo_dashboard] shutting down…")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("[demo_dashboard] Flask MJPEG server → http://0.0.0.0:8765/")
    app.run(host="0.0.0.0", port=8765, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
