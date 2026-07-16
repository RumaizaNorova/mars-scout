# GPU Day Cheatsheet — Vast.ai + Isaac Sim

Copy-paste commands. Do them in order.

---

## Step 1 — Rent the instance (you do this)

1. Go to [vast.ai](https://vast.ai) → **Search**
2. Filters: **RTX 3090 or 4090**, Ubuntu 22.04, ≥ 40 GB disk, ≤ $0.55/hr
3. Click **Rent** → you get an SSH command like:
   ```
   ssh -p 12345 root@xxx.xxx.xxx.xxx
   ```
4. Save that command. You'll use it in Step 2.

---

## Step 2 — First SSH in, clone repo, run setup

```bash
# On your Mac terminal — paste the SSH command from Vast.ai
ssh -p 12345 root@xxx.xxx.xxx.xxx

# Inside the instance:
git clone https://github.com/RumaizaNorova/mars-scout.git /root/mars-rover-agent
cd /root/mars-rover-agent
bash scripts/setup_instance.sh
```

`setup_instance.sh` installs: ROS2 Humble, Python deps, builds the colcon workspace.
Takes ~5 minutes. You'll see ✓ lines as it goes.

---

## Step 3 — Install Isaac Sim

```bash
pip install --extra-index-url https://pypi.nvidia.com \
    isaacsim-rl \
    isaacsim-replicator \
    isaacsim-extscache-physics \
    isaacsim-extscache-kit \
    isaacsim-extscache-kit-sdk \
    isaacsim
```

This downloads ~15 GB. Start it, go do something else.
Check back with: `pip show isaacsim | grep Version`

---

## Step 4 — Start VNC so you can see the Isaac Sim GUI

On the **instance**:
```bash
Xvfb :1 -screen 0 1280x720x24 &
export DISPLAY=:1
x11vnc -display :1 -nopw -forever &
websockify --web /usr/share/novnc 6080 localhost:5900 &
```

On your **Mac browser**: `http://xxx.xxx.xxx.xxx:6080/vnc.html`
(replace with your instance's IP — shown on the Vast.ai dashboard)

---

## Step 5 — Launch Isaac Sim with the Mars scene

```bash
# New terminal on the instance (VNC or SSH)
cd /root/mars-rover-agent
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash

# Start Isaac Sim — scene loads, rocks appear, camera starts publishing
python3 isaac_sim/setup_scene.py
# Add --headless if you don't need the GUI (faster):
# python3 isaac_sim/setup_scene.py --headless
```

Wait for the line: `[setup_scene] Simulation running.`

---

## Step 6 — Discover what Isaac Sim is publishing

```bash
# New terminal:
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ros2 run mars_scout_sim_bridge topic_inspector
```

This scans all active ROS2 topics and prints which ones the bridge needs.
It also prints the exact `--ros-args` flags to pass if topic names differ.

---

## Step 7 — Launch the full Mars Scout stack

```bash
# New terminal:
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ros2 launch mars_scout_sim_bridge sim_bridge.launch.py
```

If topic names don't match defaults, add the flags from Step 6:
```bash
ros2 launch mars_scout_sim_bridge sim_bridge.launch.py \
    isaac_rgb_topic:=/your/actual/topic
```

You should see all four nodes start: `sim_bridge_node`, `perception_node`, `projection_node`, `agent_node`.

---

## Step 8 — Send a navigation goal

```bash
# New terminal:
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ros2 run mars_scout_goal_interface goal_cli "find the large boulder"
```

Or in the interactive REPL:
```bash
ros2 run mars_scout_goal_interface goal_cli
# then type: find skull-like rock
```

Watch the rover move in Isaac Sim, watch confidence climb in the terminal.

---

## Step 9 — Run the live benchmark

```bash
python3 benchmark/run_benchmark.py --live --timeout 90 --save
```

Results saved to `benchmark/results/benchmark_live_*.json`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Action server not available` | `agent_node` didn't start — check Step 7 output |
| No camera topics after Step 5 | Check VNC: Isaac Sim may have an error dialog. Re-run Step 5. |
| `confidence always 0.0` | `perception_node` using mock backend — ensure `vlm_backend:=moondream2` |
| Moondream2 OOM | RTX 3090 has 24 GB — should be fine. If not: `--half` flag in vlm_backend.py |
| Isaac Sim crashes on startup | Likely driver issue: `nvidia-smi` to check, reinstall if needed |

---

## Quick sanity checks

```bash
nvidia-smi                          # GPU visible
ros2 topic list                     # topics active  
ros2 topic echo /rover/camera/image_raw --once   # camera frame arrives
ros2 topic echo /rover/odom --once               # odometry arrives
ros2 action list                    # /navigate_to_target listed
```
