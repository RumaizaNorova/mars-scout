# Mars Scout 🚀

Autonomous Vision-Language Agent for Mars Terrain Navigation.

Type a natural language target → rover finds it in Isaac Sim.

## What it does

- You type a goal like: *"go to the rock that looks like a skull"*
- Moondream2 VLM finds the best matching region in the camera image
- It turns that 2D region into a 3D ground waypoint using depth / ray intersection
- The rover drives to that waypoint, with basic safety limits
- Every run is recorded to `ros2 bag` for replay + evaluation

## Stack
- **Simulation**: NVIDIA Isaac Sim (Mars digital twin)
- **Brain**: Moondream2 VLM
- **Control**: ROS2 Humble + Python

## Architecture

```
Isaac Sim (camera + depth feed)
        ↓
  /rover/camera/image_raw  [ROS2 topic]
        ↓
   mars_scout_perception (Moondream2)
        ↓
  /rover/vlm/target_bearing
        ↓
   mars_scout_control (agent state machine)
        ↓
  /rover/cmd_vel → rover moves
```

## Repo map
- `src/` — ROS2 packages
- `isaac_sim/` — scene setup scripts
- `benchmark/` — phrase lists and scene manifests
- `scripts/` — GPU host setup + run helpers

## Quick Start (on Vast.ai / GPU instance)

```bash
git clone https://github.com/RumaizaNorova/mars-scout
cd mars-scout
bash scripts/setup_instance.sh
bash scripts/run_demo.sh "skull-shaped rock"
```

## Week 1 Checklist
- [ ] Isaac Sim + Mars terrain loaded
- [ ] ROS2 bridge live
- [ ] Moondream2 reading camera feed
- [ ] Agent navigating to NL target
