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
# After setup, start Isaac Sim in GUI mode (no --headless flag):
#   python3 isaac_sim/setup_scene.py
# =============================================================================

set -e
echo "[vnc] Setting up desktop environment..."

# Install packages
apt-get update -qq
apt-get install -y -qq \
    xfce4 xfce4-goodies \
    x11vnc \
    xvfb \
    novnc \
    websockify \
    supervisor \
    2>/dev/null

echo "[vnc] Packages installed."

# Create virtual display startup script
cat > /usr/local/bin/start_display.sh << 'DISPLAY_EOF'
#!/bin/bash
export DISPLAY=:1
Xvfb :1 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
sleep 2
startxfce4 &
sleep 3
x11vnc -display :1 -nopw -listen localhost -forever -shared &
sleep 1
websockify --web /usr/share/novnc/ 0.0.0.0:6080 localhost:5900 &
echo "[vnc] Desktop running. Open: http://$(curl -s ifconfig.me 2>/dev/null || echo YOUR_IP):6080/vnc.html"
wait
DISPLAY_EOF
chmod +x /usr/local/bin/start_display.sh

# Start it now
echo "[vnc] Starting virtual display..."
nohup /usr/local/bin/start_display.sh > /tmp/vnc.log 2>&1 &
sleep 5

echo ""
echo "============================================================"
echo "VNC desktop is running!"
echo ""
echo "Open in your browser:"
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_VAST_AI_IP")
echo "  http://${PUBLIC_IP}:6080/vnc.html"
echo ""
echo "Make sure port 6080 is open in your Vast.ai instance settings."
echo ""
echo "To run Isaac Sim with GUI (so you can see the 3D viewport):"
echo "  export DISPLAY=:1"
echo "  cd ~/mars-rover-agent"
echo "  python3 isaac_sim/setup_scene.py"
echo "============================================================"
