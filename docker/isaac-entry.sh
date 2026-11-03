#!/bin/bash
# Point Isaac Sim at its internal ROS libraries, then run the script under its own Python.
set -e
if [ "${ACCEPT_EULA:-}" != "Y" ]; then
    echo "[isaac-entry] Read the Isaac Sim licence, then set ACCEPT_EULA=Y in .env." >&2
    exit 1
fi
lib=""
for ext in isaacsim.ros2.bridge isaacsim.ros2.core; do
    candidate="/isaac-sim/exts/${ext}/${ROS_DISTRO}/lib"
    if [ -d "$candidate" ]; then lib="$candidate"; break; fi
done
if [ -z "$lib" ]; then
    echo "[isaac-entry] No internal ROS libraries for ROS_DISTRO=${ROS_DISTRO} under /isaac-sim/exts" >&2
    exit 1
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$lib"
exec /isaac-sim/python.sh "$@"
