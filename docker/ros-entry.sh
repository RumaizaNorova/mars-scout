#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
source /opt/ares/ros2_ws/install/setup.bash
exec "$@"
