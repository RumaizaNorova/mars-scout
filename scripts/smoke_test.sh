#!/bin/bash
# End-to-end proof that a GPU host can run ARES. Run this first on any new rental.
#   bash scripts/smoke_test.sh               # real HiRISE terrain
#   bash scripts/smoke_test.sh procedural    # no DTM download needed
set -euo pipefail
cd "$(dirname "$0")/.."

TERRAIN="${1:-hirise}"
bash scripts/check_host.sh

set -a; source .env; set +a
ISAAC_HOME="${ISAAC_HOME:-$HOME/docker/isaac-sim}"
ISAAC_HOME="${ISAAC_HOME/#\~/$HOME}"
REPORT="$ISAAC_HOME/outputs/smoke_report.json"

if [ "$TERRAIN" = "hirise" ] && [ ! -s data/jezero_hirise.tif ]; then
    echo "[smoke] data/jezero_hirise.tif is missing. Run: bash scripts/download_hirise.sh" >&2
    exit 1
fi

docker compose --profile sim --profile smoke build

echo "=== [1/2] Scene checks inside Isaac Sim ==="
sudo rm -f "$REPORT"
docker compose --profile sim run --rm isaac isaac_sim/setup_scene.py --smoke-test --terrain "$TERRAIN"
# Trust the report, not the exit code: a crash before the checks leaves no report.
if [ ! -s "$REPORT" ] || ! grep -q '"passed": true' "$REPORT"; then
    echo "[smoke] FAIL: see $REPORT" >&2
    [ -s "$REPORT" ] && cat "$REPORT" >&2
    exit 1
fi
cat "$REPORT"

echo "=== [2/2] ROS messages between containers ==="
trap 'docker stop ares-isaac-smoke >/dev/null 2>&1 || true' EXIT
docker compose --profile sim run -d --rm --name ares-isaac-smoke isaac \
    isaac_sim/setup_scene.py --terrain "$TERRAIN" >/dev/null
docker compose --profile smoke run --rm ros-smoke

echo "[smoke] PASS"
