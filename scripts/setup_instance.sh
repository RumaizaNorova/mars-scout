#!/bin/bash
# Run this once on a fresh Ubuntu 24.04 GPU instance (AWS g5, Vast.ai A10G, etc.)
set -e

echo "=== [1/5] System packages ==="
apt-get update -qq
apt-get install -y \
    git curl wget python3-pip \
    libgl1-mesa-dri libglib2.0-0 \
    x11vnc xvfb novnc websockify \
    --no-install-recommends

echo "=== [2/5] ROS2 Jazzy ==="
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        | gpg --dearmor \
        | tee /usr/share/keyrings/ros-archive-keyring.gpg > /dev/null
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        | tee /etc/apt/sources.list.d/ros2.list > /dev/null
    chmod 644 /etc/apt/sources.list.d/ros2.list /usr/share/keyrings/ros-archive-keyring.gpg
    apt-get update -qq
    apt-get install -y ros-jazzy-desktop ros-jazzy-cv-bridge python3-colcon-common-extensions
fi
source /opt/ros/jazzy/setup.bash

echo "=== [3/5] Python deps ==="
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
pip install -r "$REPO_DIR/requirements.txt" --break-system-packages

echo "=== [4/5] Build ROS2 workspace ==="
cd "$REPO_DIR/ros2_ws"
colcon build --symlink-install
source install/setup.bash

echo "=== [5/5] Isaac Sim ==="
echo "Download Isaac Sim via Omniverse Launcher or:"
echo "  pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics \\"
echo "      isaacsim-extscache-kit isaacsim-extscache-kit-sdk --extra-index-url https://pypi.nvidia.com"

echo ""
echo "Setup complete. Run: bash scripts/run_demo.sh 'large rock'"
