#!/usr/bin/env python3
"""
camera_viewer.py
================
Lightweight browser dashboard: shows MastCam + Chase camera feeds live.
Runs on the Vast.ai server, accessible from your Mac browser.

Usage (on Vast.ai, after starting setup_scene.py):
  python3 scripts/camera_viewer.py

Then open on your Mac:
  http://<vast-ai-ip>:8888

Shows:
  Left panel  — MastCam (what the rover sees)
  Right panel — Chase camera (3rd-person follow view)
  Bottom bar  — rover odometry (position, speed)

No ROS2 required on the viewer side — subscribes directly via rclpy.
"""

import sys
import threading
import io
import base64
import time
import os

# ── ROS2 / rclpy ─────────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image
    from nav_msgs.msg import Odometry
    _ROS2 = True
except ImportError:
    _ROS2 = False
    print("[viewer] rclpy not found — running in demo mode (static placeholder images)")

# ── HTTP + image handling ─────────────────────────────────────────────────────
from http.server import BaseHTTPRequestHandler, HTTPServer
import struct
import zlib

# ── Global state (updated by ROS2 callbacks) ──────────────────────────────────
_lock = threading.Lock()
_state = {
    "mastcam_jpeg": None,    # bytes
    "chase_jpeg":   None,    # bytes
    "odom": {
        "x": 0.0, "y": 0.0, "z": 0.0,
        "vx": 0.0, "wz": 0.0,
        "stamp": 0.0,
    },
    "frame_count": 0,
}

PORT = int(os.environ.get("VIEWER_PORT", "8888"))

# ── Tiny PNG encoder (no Pillow required) ─────────────────────────────────────
def _encode_png_rgb(rgb_bytes: bytes, w: int, h: int) -> bytes:
    """Minimal PNG encoder for an RGB image — no external deps."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + rgb_bytes[y*w*3:(y+1)*w*3] for y in range(h))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _ros_image_to_jpeg(msg) -> bytes:
    """Convert sensor_msgs/Image to JPEG bytes (falls back to PNG if no cv2)."""
    w, h = msg.width, msg.height
    enc = msg.encoding.lower()

    # Get raw RGB bytes
    raw = bytes(msg.data)
    if enc in ("bgr8", "bgr"):
        # swap B and R channels
        rgb = bytearray(len(raw))
        for i in range(0, len(raw), 3):
            rgb[i]   = raw[i+2]
            rgb[i+1] = raw[i+1]
            rgb[i+2] = raw[i]
        raw = bytes(rgb)
    elif enc in ("rgb8", "rgb"):
        pass  # already RGB
    elif enc == "rgba8":
        rgb = bytearray(w * h * 3)
        for i in range(w * h):
            rgb[i*3:i*3+3] = raw[i*4:i*4+3]
        raw = bytes(rgb)

    # Try cv2 for JPEG compression (much smaller)
    try:
        import cv2
        import numpy as np
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
        _, buf = cv2.imencode(".jpg", arr[:, :, ::-1],
                              [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes()
    except Exception:
        pass

    # Fallback: PNG (lossless but larger)
    return _encode_png_rgb(raw, w, h)


# ── Placeholder image (grey box with text, pure Python) ───────────────────────
def _placeholder_jpeg(label: str, w: int = 640, h: int = 360) -> bytes:
    """Return a grey PNG placeholder with a label, no deps."""
    grey = bytes([80, 80, 80]) * (w * h)
    return _encode_png_rgb(grey, w, h)


# ── ROS2 subscriber node ───────────────────────────────────────────────────────
if _ROS2:
    _QOS = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )

    class _ViewerNode(Node):
        def __init__(self):
            super().__init__("ares_camera_viewer")
            self.create_subscription(Image, "/isaac_camera/rgb",   self._on_mast,  _QOS)
            self.create_subscription(Image, "/isaac_camera/chase", self._on_chase, _QOS)
            self.create_subscription(Odometry, "/odom",            self._on_odom,  _QOS)
            self.get_logger().info("ARES camera viewer node started")

        def _on_mast(self, msg):
            jpg = _ros_image_to_jpeg(msg)
            with _lock:
                _state["mastcam_jpeg"] = jpg
                _state["frame_count"] += 1

        def _on_chase(self, msg):
            jpg = _ros_image_to_jpeg(msg)
            with _lock:
                _state["chase_jpeg"] = jpg

        def _on_odom(self, msg):
            p = msg.pose.pose.position
            t = msg.twist.twist
            with _lock:
                _state["odom"].update({
                    "x": p.x, "y": p.y, "z": p.z,
                    "vx": t.linear.x, "wz": t.angular.z,
                    "stamp": time.time(),
                })


    def _spin_ros():
        rclpy.init()
        node = _ViewerNode()
        try:
            rclpy.spin(node)
        except Exception:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()

    _ros_thread = threading.Thread(target=_spin_ros, daemon=True)
    _ros_thread.start()


# ── HTTP handler ───────────────────────────────────────────────────────────────
_HTML = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>ARES Mission Control</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0a0a0a; color: #e0e0e0; font-family: 'Courier New', monospace; }}
    h1 {{ text-align: center; padding: 12px; color: #ff6b35;
          letter-spacing: 3px; font-size: 1.1em; }}
    .cameras {{ display: flex; gap: 8px; padding: 0 8px; }}
    .cam-box {{ flex: 1; background: #111; border: 1px solid #333; border-radius: 4px; }}
    .cam-label {{ font-size: 0.75em; color: #888; padding: 6px 10px; border-bottom: 1px solid #222; }}
    .cam-label span {{ color: #ff6b35; }}
    .cam-box img {{ width: 100%; display: block; }}
    .status {{ display: flex; gap: 20px; padding: 10px 16px; background: #111;
               margin: 8px; border-radius: 4px; border: 1px solid #222;
               font-size: 0.8em; }}
    .stat {{ display: flex; flex-direction: column; }}
    .stat .val {{ color: #4fc; font-size: 1.1em; }}
    .stat .lbl {{ color: #666; font-size: 0.85em; }}
    #fps {{ color: #888; font-size: 0.75em; text-align: right; padding: 4px 12px; }}
  </style>
</head>
<body>
  <h1>🔴 ARES · MISSION CONTROL · JEZERO CRATER</h1>
  <div class="cameras">
    <div class="cam-box">
      <div class="cam-label">📷 <span>MASTCAM-Z</span> — rover's eye view</div>
      <img id="mastcam" src="/img/mastcam" alt="MastCam">
    </div>
    <div class="cam-box">
      <div class="cam-label">🎥 <span>CHASE CAM</span> — 3rd person follow</div>
      <img id="chase" src="/img/chase" alt="Chase Cam">
    </div>
  </div>
  <div class="status">
    <div class="stat"><span class="val" id="pos">–</span><span class="lbl">Position (m)</span></div>
    <div class="stat"><span class="val" id="speed">–</span><span class="lbl">Speed (m/s)</span></div>
    <div class="stat"><span class="val" id="turn">–</span><span class="lbl">Turn (rad/s)</span></div>
    <div class="stat"><span class="val" id="frames">0</span><span class="lbl">Frames received</span></div>
  </div>
  <div id="fps">Refreshing…</div>
  <script>
    var t0 = Date.now();
    function refresh() {{
      var t = '?t=' + Date.now();
      document.getElementById('mastcam').src = '/img/mastcam' + t;
      document.getElementById('chase').src   = '/img/chase'   + t;
      fetch('/api/state').then(r => r.json()).then(d => {{
        document.getElementById('pos').textContent =
          d.x.toFixed(1) + ', ' + d.y.toFixed(1) + ', ' + d.z.toFixed(1);
        document.getElementById('speed').textContent = d.vx.toFixed(3);
        document.getElementById('turn').textContent  = d.wz.toFixed(3);
        document.getElementById('frames').textContent = d.frames;
        var fps = (d.frames / ((Date.now() - t0) / 1000)).toFixed(1);
        document.getElementById('fps').textContent = fps + ' fps avg';
      }}).catch(() => {{}});
    }}
    setInterval(refresh, 200);   // 5 Hz refresh
    refresh();
  </script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request logging

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path.startswith("/img/mastcam"):
            self._serve_image("mastcam_jpeg", "MastCam")
        elif self.path.startswith("/img/chase"):
            self._serve_image("chase_jpeg", "ChaseCamera")
        elif self.path.startswith("/api/state"):
            self._serve_state()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_html(self):
        body = _HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_image(self, key: str, label: str):
        with _lock:
            data = _state.get(key)
        if data is None:
            data = _placeholder_jpeg(label)
            ctype = "image/png"
        else:
            ctype = "image/jpeg" if data[:2] == b"\xff\xd8" else "image/png"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", len(data))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _serve_state(self):
        import json
        with _lock:
            o = _state["odom"]
            f = _state["frame_count"]
        body = json.dumps({
            "x": o["x"], "y": o["y"], "z": o["z"],
            "vx": o["vx"], "wz": o["wz"],
            "frames": f,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), _Handler)
    print(f"\n{'='*55}")
    print(f" ARES Mission Control Dashboard")
    print(f"")

    try:
        import subprocess
        ip = subprocess.check_output(
            ["curl", "-s", "--max-time", "3", "ifconfig.me"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        ip = "YOUR_VAST_AI_IP"

    print(f" Open in your Mac browser:")
    print(f"   http://{ip}:{PORT}")
    print(f"")
    print(f" Make sure port {PORT} is open in Vast.ai instance settings.")
    print(f" Left panel:  MastCam (rover eye)")
    print(f" Right panel: Chase cam (3rd person)")
    print(f"{'='*55}\n")

    if not _ROS2:
        print("[viewer] WARNING: rclpy not available — showing placeholders only")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[viewer] Stopped.")
