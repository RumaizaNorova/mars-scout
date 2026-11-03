# ROS 2 Jazzy stack: perception, geometry, control, mock pipeline and tests. No GPU needed.
# Isaac Sim runs in its own container, so its Python never has to match this one.
FROM ros:jazzy-ros-base-noble@sha256:8eed0a4669bbf43ee331878a4c39bd8ebddea8e3312a58b4fae8edb14526b617

ARG WITH_TELEMETRY=0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-opencv \
    python3-yaml \
    python3-pytest \
    python3-colcon-common-extensions \
    ros-jazzy-cv-bridge \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-visualization-msgs \
    ros-jazzy-rosidl-default-generators \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ares
COPY requirements/ requirements/
# Install beside apt's packages, not over them: pip cannot replace apt-owned files.
RUN pip3 install --no-cache-dir --break-system-packages --no-deps --target /opt/ares/pydeps -r requirements/ros.txt
RUN if [ "$WITH_TELEMETRY" = "1" ]; then       pip3 install --no-cache-dir --break-system-packages --no-deps --upgrade --target /opt/ares/pydeps -r requirements/telemetry.txt;     fi
# Fail the build now if the locked packages cannot load next to apt's.
RUN PYTHONPATH=/opt/ares/pydeps python3 -c "from google import genai; from google.genai import types; import PIL, numpy, scipy, cv2; print('deps ok', numpy.__version__)"

COPY . .
RUN bash -c "set -e; source /opt/ros/jazzy/setup.bash; cd ros2_ws; colcon build --symlink-install --event-handlers console_direct-"

ENV ROS_DISTRO=jazzy
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/opt/ares/docker/fastdds.xml
ENV PYTHONPATH=/opt/ares:/opt/ares/pydeps

ENTRYPOINT ["/opt/ares/docker/ros-entry.sh"]
CMD ["bash"]
