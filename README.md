# Mars Rover Agent 🚀

Autonomous Vision-Language Agent for Mars Terrain Navigation.

Type a natural language target → rover finds it in Isaac Sim.

## Stack
- **Simulation**: NVIDIA Isaac Sim
- **Brain**: Moondream2 VLM
- **Control**: ROS2 Humble + Python

## Architecture

```
Isaac Sim (camera feed)
        ↓
  /rover/camera/image_raw  [ROS2 topic]
        ↓
   VLM Node (Moondream2)
        ↓
  /rover/vlm/target_bearing
        ↓
   Agent Node (state machine)
        ↓
  /rover/cmd_vel → rover moves
```

## Quick Start (on Vast.ai instance)

```bash
git clone https://github.com/YOUR_USERNAME/mars-rover-agent
cd mars-rover-agent
bash scripts/setup_instance.sh
bash scripts/run_demo.sh "skull-shaped rock"
```

## Week 1 Goal
- [ ] Isaac Sim + Mars terrain loaded
- [ ] ROS2 bridge live
- [ ] Moondream2 reading camera feed
- [ ] Agent navigating to NL target
