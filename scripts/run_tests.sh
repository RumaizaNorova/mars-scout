#!/bin/bash
# Unit tests that need neither a GPU nor Isaac Sim. Each package runs separately
# so their import paths cannot collide.
set -euo pipefail
cd "$(dirname "$0")/.."
for dir in isaac_sim ros2_ws/src/mars_scout_geometry ros2_ws/src/mars_scout_mock \
           ros2_ws/src/mars_scout_perception ros2_ws/src/mars_scout_control; do
    echo "=== $dir ==="
    (cd "$dir" && python3 -m pytest test -q --tb=short -p no:cacheprovider)
done
