#!/bin/bash
# Usage: bash run_demo.sh "skull-shaped rock"
TARGET="${1:-large rock}"

echo "=== Launching Mars Rover Agent ==="
echo "Target: $TARGET"

source /opt/ros/humble/setup.bash
source /root/mars-rover-agent/ros2_ws/install/setup.bash

# Start Isaac Sim scene in background
echo "[1/3] Starting Isaac Sim..."
python3 /root/mars-rover-agent/isaac_sim/setup_scene.py &
ISAAC_PID=$!
sleep 15   # give Isaac Sim time to load

# Launch ROS2 nodes
echo "[2/3] Launching ROS2 agent nodes..."
ros2 launch mars_scout_bringup full.launch.py \
    target_description:="$TARGET" &
ROS_PID=$!

echo "[3/3] Running. Press Ctrl+C to stop."
echo "Watch topics with: ros2 topic echo /rover/vlm/description"

wait $ROS_PID
kill $ISAAC_PID 2>/dev/null
