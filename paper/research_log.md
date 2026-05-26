# ARES Research Log
## Autonomous Rover Exploration System — Running Notes for Paper

**Target venue**: ICRA 2026 or IROS 2026  
**Track**: Field Robotics / Space Robotics / Autonomous Systems  
**Format**: 8-page IEEE conference paper + supplementary video  

---

## Working Title Options
1. "ARES: Autonomous Geological Surveying on Mars Terrain Using Vision-Language Models"
2. "VLM-Guided Science Target Identification for Mars Rovers in Physically-Simulated Environments"
3. "Integrating Real Mars Terrain Data and Vision-Language Models for Autonomous Rover Science"

**Current favourite**: Option 1 — clearest, most searchable.

---

## Paper Structure (IEEE 8-page)

### 1. Introduction (~0.75 page)
- Mars exploration bottleneck: human-in-the-loop science targeting is slow
- Opportunity: onboard VLM can pre-select science targets autonomously
- Our contribution: full sim-to-real pipeline with real HiRISE terrain, Perseverance-class rover, Moondream2 VLM, quantitative vs AEGIS baseline
- Key claim: first open-source system combining (1) real NASA DTM, (2) physics-accurate rocker-bogie, (3) VLM science targeting in a single loop

### 2. Related Work (~1 page)
- AEGIS (NASA JPL onboard targeting, ChemCam): Kiri Wagstaff et al. 2008-2019
- abmoRobotics isaac_rover_2.0: RL locomotion, no science targeting
- Other Mars sim: REMS, Gazebo-based sims (no real terrain)
- VLMs in robotics: SayCan, RT-2, PaLM-E
- Key gap we fill: none combine real terrain + VLM science autonomy + quantitative eval

### 3. System Architecture (~1.5 pages)
- Diagram: Isaac Sim → ROS2 → Perception → Projection → Navigation → Loop
- 3.1 Terrain: HiRISE DTM pipeline (this paper's contribution)
- 3.2 Rover: Perseverance-class rocker-bogie (specs, USD construction)
- 3.3 Sensor suite: stereo NavCam, MastCam-Z, HazCam
- 3.4 Perception: Moondream2, query templates, confidence scoring
- 3.5 Navigation: D* Lite on traversability grid

### 4. Terrain Pipeline (~1 page)
- HiRISE DTM source + preprocessing (crop, resample, nodata fill)
- Mesh generation: triangle strip, per-vertex normals
- Geological rock placement: CFA formula (Golombek et al. 2008)
- PBR materials: calibrated to Mastcam-Z RGB
- Comparison: procedural vs real HiRISE (visual + elevation profile figure)

### 5. Rover Design (~0.75 page)
- Rocker-bogie kinematics in USD/PhysX
- Breaking kinematic loop at differential bar
- Wheel parameters vs real Perseverance
- Control: keyboard/gamepad + autonomous FSM

### 6. Experimental Evaluation (~2 pages)
- 6.1 Terrain fidelity: RMSE vs real HiRISE elevations
- 6.2 Science target detection: precision/recall vs Perseverance sol logs
- 6.3 Navigation efficiency: path length, obstacle events per sol
- 6.4 vs AEGIS baseline: target quality score comparison
- 6.5 Ablation: VLM query quality (rock formation vs geological detail)

### 7. Results and Discussion (~0.75 page)
- Key numbers (fill in when experiments run)
- Failure modes: VLM hallucination on dust-covered rocks
- Limitations: single-rover, no re-localisation, daytime only

### 8. Conclusion (~0.25 page)
- Open-source release URL
- Future: multi-rover, real hardware deployment, Gemini/GPT-4V swap-in

---

## Design Decisions Log
*Add an entry every time we make a non-obvious technical choice.*

### 2026-05-25 — Terrain pipeline
- **Decision**: Use rasterio + scipy zoom for HiRISE loading rather than GDAL
- **Reason**: rasterio is pip-installable, no system deps, works in Isaac Sim venv
- **Alternative considered**: GDAL (heavier, system install required)
- **Impact on paper**: cite rasterio in implementation section

### 2026-05-25 — Rock placement
- **Decision**: CFA formula q(k) = 1.79 + 0.152/k (Golombek et al. 2008)
- **Reason**: physically grounded Mars rock size-frequency distribution
- **Counts**: 8 boulders (2.5m), 18 cobbles (0.8m), 30 pebbles (0.25m)
- **Impact on paper**: cite Golombek in terrain section, compare to real Jezero rock density

### 2026-05-25 — Rock mesh type
- **Decision**: Jittered icosphere (20 faces) not sphere/box
- **Reason**: geological rocks are irregular; spheres look fake in close-up NavCam
- **Alternative**: import real rock scans (too heavy for 56 rocks in PhysX)
- **Impact on paper**: figure showing rock appearance close-up vs sphere baseline

### 2026-05-25 — Camera orientation formula
- **Decision**: rotateXYZ(75, 0, -90) to face +X with 15° nose-down
- **Reason**: Isaac Sim cameras default to -Z (straight down in Z-up world)
- **Derived**: rotX(75°) tilts -Z toward +Y; rotZ(-90°) swings +Y to +X

---

## Key Citations to Collect
- [ ] Golombek et al. 2008 — Mars rock size-frequency CFA formula
- [ ] Wagstaff et al. 2013 — AEGIS onboard science targeting (JAIR)
- [ ] Moondream2 paper / model card
- [ ] HiRISE instrument paper (McEwen et al. 2007, JGR)
- [ ] Perseverance rover overview (Farley et al. 2020, Space Sci Rev)
- [ ] D* Lite (Koenig & Likhachev 2002, AAAI)
- [ ] Isaac Sim / PhysX citation
- [ ] ROS2 citation
- [ ] abmoRobotics repo (for related work)

---

## Figures to Generate (as we build)
- [ ] Fig 1: System architecture diagram
- [ ] Fig 2: HiRISE terrain comparison (real vs procedural elevation profile)
- [ ] Fig 3: Rover USD model (side view + isometric)
- [ ] Fig 4: Rocker-bogie kinematics diagram
- [ ] Fig 5: NavCam image + VLM detection overlay
- [ ] Fig 6: Overhead mission map (path + science targets)
- [ ] Fig 7: Quantitative results table
- [ ] Fig 8: Failure cases (dust-covered rock, false positive)

---

## Metrics to Track During Experiments
*(fill these in as system runs)*

| Metric | Our System | AEGIS Baseline | Notes |
|--------|-----------|----------------|-------|
| Science target precision | TBD | ~0.78 (Wagstaff 2013) | |
| Science target recall | TBD | ~0.71 | |
| Avg distance to target (m) | TBD | — | |
| Path efficiency (actual/optimal) | TBD | — | |
| False positive rate | TBD | — | |
| Sols to survey 40x40m area | TBD | — | |

---

## Raw Notes (append anything here)
*Stream of consciousness — clean up later*

- Jezero Crater chosen: Perseverance actual landing site, abundant published sol logs for ground truth comparison
- RTX 3090: 24GB VRAM — Isaac Sim ~9GB, Moondream2 4-bit ~2.5GB, ~12GB headroom
- Simulation runs headless on Vast.ai (Ubuntu 22.04, Isaac Sim 4.5 pip install)
- ROS2 Humble — QoS RELIABLE on all publishers to avoid BEST_EFFORT mismatch
- camera_info fabricated in sim_bridge: fx=fy=640 from (18mm/36mm)*1280px
- Zero-timestamp fix: Isaac Sim publishes stamp=0 in headless mode → replace with wall clock
- Mars gravity set to 3.72 m/s² in PhysX scene (not default 9.81)
