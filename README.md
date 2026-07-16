# ARES Mars Scout

**Autonomous Vision-Language Agent for Mars Terrain Navigation**

Type a natural language target. The rover finds it.

```
"find the dark olivine boulder"              →  rover navigates to it
"vesicular basalt near the ripple crest"     →  rover navigates to it
```

Real NASA Jezero Crater terrain. Physics-accurate Perseverance-class rover. Gemini Vision AI. Full research telemetry stack.

---

## What it does

ARES is an end-to-end autonomous science targeting system for Mars rovers. A finite state machine (SEARCHING -> APPROACHING -> VERIFYING -> COMPLETED) drives a ROS2 navigation stack, using Gemini 1.5 Flash to evaluate camera frames at each tick and decide whether the rover is looking at the right target.

Every mission writes live telemetry into three backends simultaneously:

- **MongoDB Atlas** -- spatial terrain feature catalog with 2dsphere geospatial indexes and 384-dimension vector search on sentence-transformer embeddings. TTL-indexed alerts collection fires when consecutive low-confidence ticks signal a problem.
- **Elasticsearch + Kibana** -- every observation indexed with dense vector fields for hybrid keyword and semantic search. Ground Control dashboard auto-refreshes every 5 seconds: FSM state distribution, VLM confidence over time, inference latency (P50/P95), false positive rates by query type.
- **Arize Phoenix** -- every VLM inference recorded as an OpenTelemetry span with OpenInference semantic conventions. Post-mission hallucination and relevance evaluations run automatically.

The system adapts: after enough missions it detects query types that are systematically overconfident and raises their minimum confidence threshold automatically, without human intervention.

---

## Architecture

```
Isaac Sim (hand-built Jezero terrain, 500x500 m)
        |
        |  camera + depth @ 30 Hz
        v
  /rover/camera/image_raw        [ROS2]
        |
        v
  mars_scout_perception
  (Gemini 1.5 Flash / Moondream2 / MockVLM)
        |
        |  TerrainTarget.msg  (confidence, bbox, description)
        v
  mars_scout_geometry
  (bbox -> 3D waypoint via depth + RANSAC ground plane)
        |
        v
  mars_scout_control
  (FSM: SEARCHING / APPROACHING / VERIFYING / COMPLETED)
        |
        |  /rover/cmd_vel
        v
  mars_scout_sim_bridge  -->  Isaac Sim 6-wheel drive
        |
        v
  ground_control  (MongoDB Atlas + Elasticsearch + Arize Phoenix)
```

---

## Mars terrain

The terrain scene is built entirely from scratch -- no pre-made assets:

- **Elevation**: real NASA HiRISE DTM of Jezero Crater (1 m/px). Procedural multi-octave fractal fallback when the file is not available.
- **Rocks**: Golombek (2003/2008) CFA size-frequency distribution. Icosphere base meshes with Voronoi fracture (angularity), Laplacian smoothing (weathering), ventifact cuts (aeolian erosion), and a per-vertex 3-channel dust/geology/rust color model calibrated to CRISM spectral reflectance data.
- **Aeolian ripples**: asymmetric sawtooth profile, wavelength 2.1 m, height 12.5 cm (Lapotre 2016 ground truth), transport toward 276 WSW (MEDA Jezero wind data).
- **Impact craters**: Gaussian bowl + raised rim + ejecta blanket (Melosh 1989), power-law radius distribution (Hartmann 2005).
- **Wheel tracks**: Bekker (1956) wheel-soil interaction model. Computed sinkage 2.4 cm matches Perseverance telemetry.
- **Lighting**: Hapke (2002) BRDF MDL shader. Dust tau=0.56, solar elevation 28 degrees (Bell et al. 2021 Mastcam-Z calibration).

---

## Rover

Perseverance-class, built programmatically in USD/PhysX:

- Rocker-bogie suspension with correct joint topology
- 6 drive wheels, differential bar (passive side)
- Wheel diameter 0.525 m, track gauge 2.78 m, mass 1025 kg
- Stereo MastCam-Z at 25.8 degree HFOV (Hayes 2021 calibration)
- Mars gravity 3.72 m/s^2

---

## Stack

| Layer | Technology |
|---|---|
| Simulation | NVIDIA Isaac Sim 4.5 |
| Control | ROS2 Humble |
| VLM | Gemini 1.5 Flash / Moondream2 / MockVLM |
| Terrain | HiRISE GeoTIFF + USD triangle mesh |
| Observation store | MongoDB Atlas (vector search, 2dsphere, TTL alerts) |
| Search and dashboards | Elasticsearch 8 + Kibana |
| LLM observability | Arize Phoenix (OTLP, OpenInference) |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (384-dim) |
| Infrastructure | AWS EC2, Docker Compose |

---

## Quick start (GPU instance with Isaac Sim)

```bash
git clone https://github.com/RumaizaNorova/mars-scout
cd mars-scout

# Install system deps, ROS2 Humble, Python packages, build workspace
bash scripts/setup_instance.sh

# Download real Jezero Crater terrain (1.76 GB HiRISE, ~50 MB CTX fallback)
bash scripts/download_hirise.sh

# Configure env vars (copy and fill in your values)
cp .env.local .env
source .env

# Run
bash scripts/run_demo.sh "dark olivine boulder"
```

---

## Run without GPU (batch simulation)

The batch runner simulates complete missions using a kinematic differential drive model and procedural images -- no Isaac Sim required.

```bash
pip install -r requirements.txt

source .env.local   # set MONGODB_URI, ELASTICSEARCH_URL, PHOENIX_COLLECTOR_ENDPOINT

python3 scripts/batch_mission_runner.py --n-missions 50
```

This produces ~6000 observations written to MongoDB, Elasticsearch, and Phoenix. Open Kibana Ground Control at `http://localhost:5601`.

---

## Local telemetry stack

```bash
docker compose up elasticsearch kibana phoenix -d
bash scripts/import_kibana_dashboard.sh
```

Dashboard at `http://localhost:5601` (auto-refreshes every 5 seconds).
Phoenix traces at `http://localhost:6006`.

---

## MongoDB Atlas setup

```bash
# One-time index creation after creating your Atlas M0 cluster
MONGODB_URI="mongodb+srv://..." python3 scripts/setup_atlas_indexes.py
```

Creates compound, 2dsphere, TTL, and vector search indexes. Prints manual steps for M0 free tier which does not support programmatic vector search index creation.

---

## Repo layout

```
isaac_sim/            Isaac Sim scene builders (terrain, rover, controller)
ros2_ws/src/
  mars_scout_control/       FSM, Pure Pursuit controller, ROS2 action server
  mars_scout_perception/    VLM backends (Gemini, Moondream2, Mock)
  mars_scout_geometry/      3D back-projection, RANSAC ground plane
  mars_scout_sim_bridge/    Isaac Sim <-> ROS2 topic bridge
  mars_scout_mock/          Pure-Python kinematic simulator (no Isaac Sim needed)
  mars_scout_bringup/       Launch files
ground_control/       MongoDB, Elasticsearch, Arize Phoenix integrations
scripts/              Batch runner, setup, texture generation, HiRISE download
shaders/              MDL shaders (Hapke regolith BRDF, triplanar rock)
data/                 Terrain textures, reference imagery
kibana/               Ground Control dashboard NDJSON
paper/                Research notes, ARES handbook
```

---

## References

- Bell et al. 2021, SSR 217:24 -- Mastcam-Z calibration
- Golombek et al. 2008, JGR 113:E00A11 -- CFA rock size-frequency distribution
- Hapke 2002, Icarus 157:523 -- regolith BRDF
- Lapotre et al. 2016, Science 353:55 -- aeolian ripple ground truth
- Melosh 1989, "Impact Cratering" -- crater morphology model
- Hayes et al. 2021, SSR 217:48 -- Mastcam-Z FOV calibration
- Horgan et al. 2020, Icarus 339:113526 -- Jezero CRISM spectral units
- Viudez-Moreiras et al. 2022, JGR -- MEDA wind at Jezero (sols 1-216)
- Bekker 1956 / Golombek 2018 -- wheel-soil sinkage model

---

## License

MIT -- see [LICENSE](LICENSE)
