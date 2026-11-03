# Isaac Sim scene. To change version, update the tag and digest here and re-test with
# scripts/smoke_test.sh. NVIDIA's docs mark 5.1.0 as no longer supported.
FROM nvcr.io/nvidia/isaac-sim:5.1.0@sha256:f3563cb2ba0c18af0b2fb321360dcb73a917b899f879e3213623d6bee484fa54

ARG ACCEPT_EULA
USER root
WORKDIR /opt/ares

COPY requirements/isaac.txt requirements/isaac.txt
RUN ACCEPT_EULA="$ACCEPT_EULA" /isaac-sim/python.sh -m pip install --no-cache-dir --no-deps \
    --target /opt/ares/pydeps -r requirements/isaac.txt

COPY isaac_sim/ isaac_sim/
COPY shaders/ shaders/
COPY data/ data/
COPY docker/ docker/

ENV PYTHONPATH=/opt/ares/pydeps
# Fail the build now, not hours into a rental, if a scene dependency is missing.
RUN ACCEPT_EULA="$ACCEPT_EULA" /isaac-sim/python.sh -c "import numpy, scipy, rasterio; print('deps ok', numpy.__version__, scipy.__version__, rasterio.__version__)"

ENV ROS_DISTRO=jazzy
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/opt/ares/docker/fastdds.xml

USER 1234:1234
ENTRYPOINT ["/opt/ares/docker/isaac-entry.sh"]
CMD ["isaac_sim/setup_scene.py"]
