# A target region identified by the VLM in the current camera frame.

std_msgs/Header header   # frame_id = camera optical frame

# Which query this target answers
string query_id

# ── 2-D detection (image space) ──────────────────────────────────────────────
# Normalised [0,1] bounding box: (cx, cy) = centre, (w, h) = size
float32 bbox_cx
float32 bbox_cy
float32 bbox_w
float32 bbox_h

# ── 3-D back-projection (camera frame) ───────────────────────────────────────
geometry_msgs/Point    position_camera   # metres, camera optical frame
geometry_msgs/Vector3  position_sigma    # 1-σ uncertainty on each axis (metres)

# ── Ground waypoint (map frame) ──────────────────────────────────────────────
geometry_msgs/PointStamped waypoint      # frame_id = map; z ≈ 0 (ground plane)

# ── Confidence ────────────────────────────────────────────────────────────────
float32 vlm_confidence      # [0,1]  VLM self-reported match score
float32 depth_confidence    # [0,1]  quality of the depth estimate at target pixel
float32 overall_confidence  # [0,1]  fused score used by the planner

# ── VLM reasoning ─────────────────────────────────────────────────────────────
string vlm_description      # raw VLM answer for this detection
bool   target_found         # false → no match in current frame
