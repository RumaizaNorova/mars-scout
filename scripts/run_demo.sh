#!/bin/bash
# Usage: bash scripts/run_demo.sh ["dark olivine boulder"]
# Starts Isaac Sim and the ROS stack, then sends the goal. Ctrl+C stops everything.
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET="${1:-large rock}"

bash scripts/check_host.sh
trap 'docker compose --profile sim down' EXIT
docker compose --profile sim up -d --build
docker compose --profile sim logs -f &
docker compose --profile sim exec ros ros2 run mars_scout_goal_interface goal_cli "$TARGET"
wait
