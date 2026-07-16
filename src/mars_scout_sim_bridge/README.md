## `mars_scout_sim_bridge`

Simulation bridge placeholder.

This package will be the ONLY layer allowed to directly depend on Isaac Sim / Omniverse APIs.

Responsibilities:

- Publish sensor topics (`/mars/rgb`, `/mars/depth`, `/mars/camera_info`)
- Publish TF (`/tf`, `/tf_static`) and odometry
- Subscribe to `/cmd_vel` (or equivalent) to drive the simulated rover

