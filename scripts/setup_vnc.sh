#!/usr/bin/env bash
# =============================================================================
# setup_vnc.sh
# Sets up a browser-accessible desktop on Vast.ai so you can see Isaac Sim
# in full 3D GUI mode — orbit the rover, watch suspension flex, etc.
#
# Run ONCE on the Vast.ai server:
#   bash scripts/setup_vnc.sh
#
# Then open in your browser:
#   http://<vast-ai-ip>:6080/vnc.html
#
# After setup, start Isaac Sim in GUI mode:
#   export DISPLAY=:1
#   DISPLAY=:1 vglrun python3 isaac_sim/setup_scene.py --headed
# =============================================================================

set -e
echo "[vnc] Setting up desktop environment with VirtualGL (GPU-backed Vulkan)..."

# Install packages
apt-get update -qq
apt-get install -y -qq \
    xfce4 xfce4-goodies \
    x11vnc \
    xvfb \
    novnc \
    websockify \
    supervisor \
    mesa-utils \
    2>/dev/null

echo "[vnc] Desktop packages installed."

# ── Install VirtualGL (makes Vulkan/OpenGL work on virtual display) ───────────
VGLVER="3.1.1"
VGLDEB="virtualgl_${VGLVER}_amd64.deb"
if ! command -v vglrun &>/dev/null; then
    echo "[vnc] Installing VirtualGL ${VGLVER}..."
    wget -q "https://github.com/VirtualGL/virtualgl/releases/download/${VGLVER}/${VGLDEB}" \
        -O /tmp/${VGLDEB}
    dpkg -i /tmp/${VGLDEB}
    rm -f /tmp/${VGLDEB}
    # Configure VirtualGL to use the NVIDIA GPU
    vglserver_config -config +s +f -t 2>/dev/null || true
    echo "[vnc] VirtualGL installed."
else
    echo "[vnc] VirtualGL already installed."
fi

# Create virtual display startup script
cat > /usr/local/bin/start_display.sh << 'DISPLAY_EOF'
#!/bin/bash
export DISPLAY=:1
# Start Xvfb with GLX extension (VirtualGL bridges GPU → Xvfb)
Xvfb :1 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!
sleep 2

# Start desktop environment
startxfce4 &
sleep 3

# Start VNC server (no password for easy access; add -passwd FILE for security)
x11vnc -display :1 -nopw -listen localhost -forever -shared &
sleep 1

# Start noVNC websocket proxy (browser access on port 6080)
websockify --web /usr/share/novnc/ 0.0.0.0:6080 localhost:5900 &

PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_VAST_AI_IP")
echo "[vnc] Desktop running."
echo "[vnc] Open: http://${PUBLIC_IP}:6080/vnc.html"
echo ""
echo "[vnc] To run Isaac Sim with GPU-backed GUI:"
echo "  export DISPLAY=:1"
echo "  cd ~/mars-rover-agent"
echo "  DISPLAY=:1 vglrun python3 isaac_sim/setup_scene.py --headed"

wait $XVFB_PID
DISPLAY_EOF
chmod +x /usr/local/bin/start_display.sh

# Start it now (in background)
echo "[vnc] Starting virtual display..."
pkill -f "Xvfb :1" 2>/dev/null || true
pkill -f x11vnc      2>/dev/null || true
pkill -f websockify  2>/dev/null || true
sleep 1

nohup /usr/local/bin/start_display.sh > /tmp/vnc.log 2>&1 &
sleep 6

echo ""
echo "============================================================"
echo " VNC desktop is running!"
echo ""
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_VAST_AI_IP")
echo " Browser:  http://${PUBLIC_IP}:6080/vnc.html"
echo ""
echo " Make sure port 6080 is open in Vast.ai instance settings."
echo ""
echo " Start Isaac Sim (GPU-accelerated GUI):"
echo "   export DISPLAY=:1"
echo "   cd ~/mars-rover-agent"
echo "   DISPLAY=:1 vglrun python3 isaac_sim/setup_scene.py --headed"
echo ""
echo " Or just run headless (cameras still publish to ROS2):"
echo "   python3 isaac_sim/setup_scene.py"
echo "============================================================"
