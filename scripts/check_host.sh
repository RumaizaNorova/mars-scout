#!/bin/bash
# Fail early if this machine cannot run the Isaac Sim container, and prepare its folders.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "[host] No .env. Run: cp .env.example .env, then fill it in." >&2; exit 1; }
set -a; source .env; set +a
if [ "${ACCEPT_EULA:-}" != "Y" ]; then
    echo "[host] Read the Isaac Sim licence, then set ACCEPT_EULA=Y in .env." >&2
    exit 1
fi
command -v docker >/dev/null || { echo "[host] Docker is not installed." >&2; exit 1; }
docker info >/dev/null 2>&1 || {
    echo "[host] Docker is not usable here. Container rentals usually block Docker: rent a VM or bare-metal host." >&2
    exit 1
}
command -v nvidia-smi >/dev/null || { echo "[host] No NVIDIA driver found." >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
docker run --rm --runtime=nvidia --gpus all ubuntu:24.04 nvidia-smi -L >/dev/null || {
    echo "[host] Containers cannot see the GPU. Install the NVIDIA Container Toolkit." >&2
    exit 1
}

ISAAC_HOME="${ISAAC_HOME:-$HOME/docker/isaac-sim}"
ISAAC_HOME="${ISAAC_HOME/#\~/$HOME}"
mkdir -p "$ISAAC_HOME"/cache/main/ov "$ISAAC_HOME"/cache/main/warp "$ISAAC_HOME"/cache/computecache \
         "$ISAAC_HOME"/config "$ISAAC_HOME"/data/documents "$ISAAC_HOME"/data/Kit \
         "$ISAAC_HOME"/logs "$ISAAC_HOME"/pkg "$ISAAC_HOME"/outputs
sudo chown -R 1234:1234 "$ISAAC_HOME"
sudo chmod -R a+rX "$ISAAC_HOME"
echo "[host] OK"
