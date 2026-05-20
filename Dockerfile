# ── Mars Scout — GPU instance image ──────────────────────────────────────────
#
# Target:  Vast.ai / Lambda Labs instance with NVIDIA GPU
# Base:    Ubuntu 22.04 + CUDA 12.1 + cuDNN 8  (matches Isaac Sim 4.x reqs)
#
# Build:
#   docker build -t mars-scout:gpu .
#
# Run (on GPU host):
#   docker run --gpus all --rm -it \
#     --network host \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     -e DISPLAY=$DISPLAY \
#     mars-scout:gpu bash
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV ROS_DISTRO=humble

# ── 1. System packages ────────────────────────────────────────────────────────
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    # Python
    python3.10 python3-pip python3-dev \
    # ROS2 prereqs
    curl gnupg2 lsb-release software-properties-common \
    # OpenCV / image tools
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    # Remote desktop (for Isaac Sim GUI on headless host)
    x11vnc xvfb novnc websockify \
    # Dev tools
    git vim tmux wget \
    && rm -rf /var/lib/apt/lists/*

# ── 2. ROS2 Humble ───────────────────────────────────────────────────────────
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) \
      signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu \
      $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        ros-humble-desktop \
        ros-humble-cv-bridge \
        ros-humble-tf2-ros \
        ros-humble-tf2-geometry-msgs \
        ros-humble-nav2-msgs \
        python3-colcon-common-extensions \
        python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# ── 3. Python dependencies ────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ── 4. Isaac Sim Python packages (pip-installable subset) ────────────────────
#    Full Isaac Sim GUI must be installed separately via Omniverse Launcher.
#    These pip packages provide the Python API used by setup_scene.py.
RUN pip3 install --no-cache-dir \
    isaacsim-rl \
    isaacsim-replicator \
    isaacsim-extscache-physics \
    isaacsim-extscache-kit \
    isaacsim-extscache-kit-sdk \
    --extra-index-url https://pypi.nvidia.com \
    || echo "⚠  Isaac Sim pip packages unavailable — install via Omniverse Launcher"

# ── 5. Copy project and build ROS2 workspace ─────────────────────────────────
WORKDIR /root/mars-scout
COPY . .

RUN bash -c "\
    source /opt/ros/humble/setup.bash && \
    cd ros2_ws && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
    || echo '⚠  colcon build failed — check ROS2 package dependencies'"

# ── 6. Environment ────────────────────────────────────────────────────────────
RUN echo "source /opt/ros/humble/setup.bash"                       >> /root/.bashrc && \
    echo "source /root/mars-scout/ros2_ws/install/setup.bash 2>/dev/null || true" \
                                                                    >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp"              >> /root/.bashrc

# ── 7. VNC setup for Isaac Sim GUI ───────────────────────────────────────────
ENV DISPLAY=:1
RUN echo '#!/bin/bash\n\
Xvfb :1 -screen 0 1920x1080x24 &\n\
x11vnc -display :1 -forever -nopw -rfbport 5900 &\n\
websockify --web /usr/share/novnc 6080 localhost:5900 &\n\
echo "VNC:    vnc://localhost:5900"\n\
echo "Browser: http://localhost:6080/vnc.html"\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
