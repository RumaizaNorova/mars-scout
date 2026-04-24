## Mars Scout

Mars Scout is a student project to build a **language-conditioned navigation loop** for a rover in a **Mars-like digital twin** (Isaac Sim) using **ROS 2**.

### What it will do (v0)

- You type a goal like: “go to the rock that looks like a skull”.
- The system finds the best matching region in the camera image.
- It turns that 2D region into a 3D ground waypoint using depth / ray intersection.
- The rover drives to that waypoint, with basic safety limits.
- Every run is recorded to `ros2 bag` for replay + evaluation.

### Core idea

We separate the stack into two machines:

- **Your Mac**: development, tests, offline analysis, bag replay.
- **AWS GPU host**: Isaac Sim + ROS 2 runtime + sensor publishing + optional perception for lowest-latency closed loop.

This is required because Isaac Sim is not a first-class native workload on macOS.

### Repo map

- `docs/PROJECT_MASTER_PLAN.md`: single source of truth (architecture, metrics, roadmap)
- `src/`: ROS 2 packages (nodes and message definitions)
- `benchmark/`: phrase lists and scene manifests
- `scripts/`: GPU host setup + benchmark runner helpers

### Next steps

Start by reading `docs/PROJECT_MASTER_PLAN.md` and then follow the “bringup” guide we will add in `src/mars_scout_bringup/`.

