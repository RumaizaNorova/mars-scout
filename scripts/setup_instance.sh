#!/bin/bash
# Run this once on the fresh Vast.ai instance to install everything.
set -e

echo "=== [1/5] System packages ==="
apt-get update -qq
apt-get install -y \
    git curl wget python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    x11vnc xvfb novnc \
    --no-install-recommends

echo "=== [2/5] ROS2 Humble ==="
if [ ! -f /opt/ros/humble/setup.bash ]; then
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        > /etc/apt/sources.list.d/ros2.list
    apt-get update -qq
    apt-get install -y ros-humble-desktop ros-humble-cv-bridge python3-colcon-common-extensions
fi
source /opt/ros/humble/setup.bash

echo "=== [3/5] Python deps ==="
pip install -r /root/mars-rover-agent/requirements.txt

echo "=== [4/5] Build ROS2 workspace ==="
cd /root/mars-rover-agent/ros2_ws
colcon build --symlink-install
source install/setup.bash

echo "=== [5/5] Isaac Sim ==="
echo "Download Isaac Sim via Omniverse Launcher or:"
echo "  pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics \\"
echo "      isaacsim-extscache-kit isaacsim-extscache-kit-sdk --extra-index-url https://pypi.nvidia.com"

echo ""
echo "Setup complete. Run: bash scripts/run_demo.sh 'large rock'"
