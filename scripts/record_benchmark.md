## Recording benchmark runs — draft workflow

We will convert this into a script once topic names are finalized.

### Goal

For each trial, record a bag plus a run manifest:

- `/mars/rgb`
- `/mars/depth` (or points)
- `/mars/camera_info`
- `/tf`, `/tf_static`
- `/odom`
- `/cmd_vel`
- `/mars/goal_command`
- `/mars/goal_hypothesis`
- `/mars/waypoint`

### Output

- `bags/<run_id>/` (rosbag2)
- `bags/<run_id>/run_manifest.json`

