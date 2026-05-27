# Mars Geology Research Log
# ARES Simulation — Jezero Crater Surface Modeling
# Persistent reference: updated whenever new findings are confirmed.
# DO NOT lose this file — it is the scientific backbone of the simulation.

---

## 1. SITE: Jezero Crater, 18.4°N 77.7°E

Landing ellipse centred on Máaz formation (crater floor basalt).
Perseverance touched down 18 Feb 2021.  Simulation covers a 40×40 m patch
of crater floor terrain representative of Sols 50–200 traverse.

---

## 2. GEOLOGICAL UNITS (Stack et al. 2020, Science 370, 643)

### 2.1 Máaz Formation (dominant, ~70% of floor)
- Crater-floor fractured (Cf-fr) unit
- Composition: pyroxene + plagioclase basalt, minor olivine
- CRISM spectral unit: moderate red slope, pyroxene absorption at 1000/2000 nm
- Surface: angular fresh-fractured rocks, dark basaltic regolith
- Albedo: I/F(750 nm) ≈ 0.09–0.14
- Grain size of bedrock: 0.5–3 mm (Bhartia et al. 2021 SHERLOC)
- Rock shapes: predominantly Very Angular to Angular (VA/A dominant)

### 2.2 Séítah Formation (~30% of floor, erosional windows)
- Olivine-rich basalt with fayalitic olivine (Fo40–50)
- Composition: olivine + pyroxene + altered phases
- CRISM: strong 1050 nm olivine feature, darker than Máaz
- Albedo: I/F(750 nm) ≈ 0.07–0.11 (darker)
- Rocks rounder (more reworked), SA–SR dominant
- Spatial distribution: patches in SW quadrant + scattered erosional windows
- Ref: Farley et al. 2022, Science 377 (Perseverance core science)

### 2.3 Delta Front (not in rover's immediate 40×40 m path — note for future)
- Western Jezero delta: fluvial conglomerate + sandstone
- Contains rounded pebbles (water-transported), imbricated clast fabric
- Should use separate Conglomerate material + rounded rock shapes

---

## 3. ROCK MORPHOLOGY

### 3.1 Powers Roundness Distribution
Source: Khan et al. 2022, JGR Planets 127 (Gale Crater analog; closest published
data to Jezero crater floor)
- Very Angular (VA): 35%
- Angular (A): 30%
- Sub-Angular (SA): 20%
- Sub-Rounded (SR): 10%
- Rounded (R): 5%
Note: Jezero likely similar or slightly more angular (less fluvial reworking
in the crater floor proper than Gale).

### 3.2 Aspect Ratios and Protrusion (Golombek et al. 2008, JGR 113, E00A11)
- Height = ~0.5 × diameter (median; range 0.3–0.8)
- Protrusion above surface: ~40% of height (60% buried)
- This is the CFA standard used by all Mars landing site engineers

### 3.3 CFA Size-Frequency Distribution (Golombek 2008)
q(k) = 1.79 × exp(−0.152 / k)
where k = rock diameter in metres, q = fraction of surface covered
For 40×40 m scene at moderate rockiness (~5–10% CFA):
- Boulders (≥2.5 m): 10–15
- Cobbles (≥0.8 m): 60–90
- Pebbles (≥0.25 m): 150–220
- Granules (5–25 mm): thousands (part of ground texture, not separate objects)

### 3.4 Ventifacts (Herkenhoff et al. 2023, JGR Planets)
- Paleowind direction: from WEST, 94° azimuth (east of north)
- Wind-sculpted features: flat/concave faces on windward (west) side
- Frequency: ~35% of rocks (Bridges et al. 2014, Aeolian Research)
- Feature: extra planar cuts on windward hemisphere in our model
- Keels and pits visible on some boulders at Gale and Spirit sites

### 3.5 Rock Surface Textures (NOT YET IMPLEMENTED)
- Vesicularity: basalt rocks have gas-bubble pits (1–20 mm diameter)
  High vesicle density = erupted near surface (volatile-rich). Documented
  at Jezero in Bhartia et al. 2021.
- Thermal spalling: daily ΔT ≈ 120°C on Mars → thin flakes spall off surfaces
  Creates stepped texture on top surfaces of boulders.
  Ref: Eppes et al. 2020, Nature Communications (Mars thermal fatigue)
- Desert varnish analog: Fe/Mn oxide coating darkens exposed surfaces over time
  Older rocks (>1 Myr exposure) are darker-coated.
- Dust mantling: 0.5–2 mm dust layer on lee side (east face) of rocks
  Perseverance images show clear asymmetric dust coating
- Conglomerate texture: rocks near delta show embedded pebble inclusions
  Visible as bumpy surface, rounded sub-clasts at cm scale
- Exfoliation/onion-skin: pressure-release fracturing produces parallel
  surface layers on larger boulders

### 3.6 Rock Fracture Patterns (NOT YET IMPLEMENTED)
Types present at Jezero:
1. Columnar jointing — basalt cooling → hexagonal column cross-sections
2. Impact fracturing — radial + concentric fracture sets from hypervelocity impacts
3. Thermal fatigue fracturing — sub-parallel fractures from daily heating cycles
   (120°C diurnal range on Mars vs 50°C Earth) — Eppes et al. 2020
4. Pressure release fracturing — exhumation causes rock to expand, spall

Current model only does random Voronoi fracture. Real fracture is structured.
Next step: bias cut normals to follow coherent fracture families.

---

## 4. GROUND SURFACE — CURRENT STATE: INADEQUATE

The ground is currently a procedural sine-noise heightmap with uniform color.
Every Perseverance image shows the ground is NOT like this. Critical missing features:

### 4.1 Aeolian Ripples — CONFIRMED MEASUREMENTS (HIGH PRIORITY)
# Updated from research agent: Chojnacki 2018, Bridges 2017 (Bagnold Dunes analogue)

TWO WIND SYSTEMS at Jezero — critical distinction:
  A) Modern active transport: azimuth 276° (toward WNW) — drives CURRENT RIPPLES
  B) Ancient paleowind: azimuth 94° (from west) — recorded in VENTIFACTS (nearly inactive)
  These are OPPOSITE directions. Implement separately!

SMALL ACTIVE RIPPLES (dominant floor texture):
- Wavelength: 3–4 m (Chojnacki 2018 HiRISE measurement, landing ellipse)
- Height: 12–20 cm (Bridges 2017 Bagnold Dunes analogue; similar grain sizes)
- Ripple Index (λ/h): ~15–23
- Profile: ASYMMETRIC SAWTOOTH (not sinusoidal)
  - Stoss face (upwind/east): gentle, ~5–10°, occupies ~75% of wavelength
  - Lee face (downwind/west): steep, ~29–33°, occupies ~25% of wavelength
  - Crest is sharp (not rounded)
- Transport direction: 276° (WNW) → crests run N–S (~6° azimuth)
- Migration rate: ~0.2 m/yr (very slow, nearly static in simulation)
- Grain armor on crests: 1–2 mm olivine grains (gray, coarse)
- Grain fill in troughs: <100 μm pyroxene dust (red, fine)
- Color: crests slightly lighter/greyer (coarse olivine); troughs slightly redder (dust)

MEGARIPPLES / TARs (sparse, near boulders):
- Wavelength: 5–11 m, height up to 2 m
- Sparsely distributed; concentrated near large obstacles
- Flat-topped crests (TAR morphology), steep lee slopes
- Ref: Chojnacki 2018 (PMC5859260); Bridges 2017 (PMC5815379); Chojnacki 2020 (PMC7583471)

CODE-READY VALUES:
  ripple_wavelength_m = 3.5
  ripple_height_m = 0.15
  stoss_fraction = 0.75     # 75% of wavelength is gentle stoss face
  lee_fraction = 0.25       # 25% is steep lee face
  transport_azimuth_deg = 276  # WNW (sand moves this direction)
  crest_azimuth_deg = 6     # crests run N-S (perpendicular to transport)

### 4.2 Polygonal Ground Cracking — CONFIRMED AT JEZERO (HIGH PRIORITY)
# Updated from research agent: Crumpler 2023, Horgan 2023, SHERLOC PMC12002120

CONFIRMED: Polygonal terrain present prominently in Máaz formation at Jezero.
Origin: MIXED — desiccation (lake dried up) + volcanic cooling + thermal contraction.
NOT permafrost/ice-wedge origin (no ground ice confirmed).

LARGE-SCALE POLYGONS (dominant Máaz floor feature):
- Diameter: 3–10 m (mean ~5 m)
- 5–7 sides per polygon (Voronoi-like)
- Crack/ridge relief: 2–5 cm above/below surrounding surface
- Morphology: TWO types visible at Jezero:
  a) NEGATIVE relief — slight depressions (open/sand-filled cracks)
  b) POSITIVE relief — raised double ridges with central crack (sulfate vein-filled)
- Crack fill: light-toned sulfate minerals (BRIGHTER than surroundings)
  → Cracks appear LIGHTER, not darker (confirmed by SHERLOC, Farley et al. 2022)

SMALL-SCALE FRACTURES (rock/bedrock surface):
- Fracture spacing: 1.4–2.2 cm mean (SHERLOC/WATSON measurements)
- Vein width: 0.1–4 mm
- These are too small for terrain mesh — affect rock surface texture instead

CODE-READY VALUES:
  polygon_diameter_m = 5.0       # mean; range 3–10m
  polygon_seeds_per_40m = 64     # one per ~25 m²
  crack_half_width_m = 0.3       # crack region half-width in voronoi distance metric
  crack_color_boost = 0.20       # cracks are ~20% brighter (sulfate fill)
  Ref: Crumpler 2023 (10.1029/2022JE007444); SHERLOC paper (PMC12002120)

### 4.3 Rock/Regolith Interface Zones (IMPORTANT — not just "rocks on ground")
- Rock-free zones: deflation surfaces where wind has removed all loose material
  Surface shows lag pavement (coarse pebbles, no fines)
- Rock-rich zones: rocky patches, 20–60% surface coverage
- Transition: gradual, driven by topography and wind exposure
- Key: ground surface changes character depending on proximity to rocks/ridges

### 4.4 Lag Pavement / Deflation Surface (MEDIUM PRIORITY)
- Wind removes fine material, leaves coarse cobbles/pebbles
- Creates "desert pavement" — interlocking mosaic of flat rocks
- Color: very dark (ventifacted, dust-free surface)
- Albedo: I/F ≈ 0.06–0.08 (darkest surface type)
- Found on flat exposed ridge tops and wind-swept areas

### 4.5 Dust/Sand Accumulation Zones (MEDIUM PRIORITY)
- Fine material (< 0.1 mm) settles in topographic lows
- Color: bright pinkish-tan (palagonite dust)
- Albedo: I/F ≈ 0.30–0.40
- Found in: rock shadow zones, polygon crack interiors, depression centers
- Shape: roughly triangular "wind tail" behind large boulders

### 4.6 Exposed Bedrock Slabs (MEDIUM PRIORITY)
- Large flat slabs of Máaz basalt exposed where regolith has been removed
- Surface: smooth-ish, dark, occasional percussion marks from impacts
- Orientation: sub-horizontal (flat-lying flows)
- Size: 0.5–5 m diameter
- Perseverance drove over these in the first 200 sols
- Ref: Farley et al. 2022 — "Crater floor is thinly veneered with regolith
  overlying competent basalt"

### 4.7 Fluvial Features (for delta-proximal area)
- Imbricated clasts: pebbles stacked like fallen dominoes, pointing upstream
- Rounded pebble trains: elongate patches of rounded, sorted cobbles
- Sedimentary layers: horizontal banding visible in outcrops
- Ref: Mangold et al. 2021, Science 374 (conglomerate at delta front)

### 4.8 Impact Craters (LOW PRIORITY but present)
- Small craters 0.5–10 m: scattered across terrain, ~1–5 per 40×40 m area
- Each has ejecta blanket: slight elevation rim, brighter (fresher) material
- Secondaries: small elongated pits in chains from larger impacts
- Fill: sand/dust accumulates inside craters over time

### 4.9 Ground Color Variation (CURRENTLY SIMPLIFIED)
What we have: slope → color blend (basalt ↔ oxide ↔ dust)
What's missing:
- Polygon crack network changes local color (crack interiors are darker/lighter)
- Rock shadow zones accumulate dust (lighter)
- Lag pavement zones are darker
- Ancient lake floor sediment patches are lighter tan
- Wetted mineral patches (perchlorate/hydrate deposits) are brighter white

---

## 5. LIGHTING AND ATMOSPHERE

### 5.1 Solar Parameters at Jezero
- Mars-Sun distance: 1.52 AU → solar flux = 590 W/m² (44% of Earth)
- Atmospheric dust opacity: τ ≈ 0.5 (typical clear season)
- Effective irradiance reaching surface: ~450 W/m²
- Sun elevation at local noon: ≈ 71° (summer); simulation uses afternoon ≈ 28°
- Sun angular diameter: 0.35° (smaller than Earth's 0.53° → sharper shadows)
- Sun color: near-white with slight warm tint (dust absorbs blue)
  NOT deep amber (common mistake)

### 5.2 Sky Properties
- Dust aerosols scatter red light → sky is pinkish-tan (NOT blue)
- Sky RGB (normalised from Perseverance NavCam): ≈ (0.58, 0.42, 0.32)
- Rayleigh scattering negligible (CO2 atmosphere, very thin)
- Horizon glow: brighter/more saturated toward sun direction
- Reference: Bell et al. 2021, Space Science Reviews (Mastcam-Z calibration)

### 5.3 Shadow Properties
- Crisp shadows (thin atmosphere, no Rayleigh fill)
- Shadow color: pinkish-grey (sky fill only, no blue)
- Strong contrast between lit and shadowed surfaces

---

## 6. CRISM SPECTRAL CALIBRATION (calibrate_materials.py)

Method: I/F spectra → XYZ (CIE 1931 2°) under Martian solar spectrum (tau=0.5, 28°
elevation) → linear sRGB via IEC 61966-2-1 matrix.
Generated by scripts/calibrate_materials.py

Material               linear-sRGB (R, G, B)        Source
─────────────────────────────────────────────────────────────────
MarsOxide             (0.1615, 0.0996, 0.0642)     Clark et al. 2017 [C17]
Basalt                (0.0977, 0.0643, 0.0417)     Horgan et al. 2020 [H20]
Sandstone/Delta       (0.2215, 0.1668, 0.1198)     Quantin-Nataf 2021 [Q21]
IronRich              (0.2125, 0.1330, 0.0734)     Horgan et al. 2020 [H20]
PaleDust              (0.3535, 0.2690, 0.1975)     Clark et al. 2017 [C17]

Key result: Jezero terrain is 3–5× DARKER than intuition.
Real floor albedo I/F(750nm) ≈ 0.07–0.18. Old guesses were 0.35–0.65.

---

## 7. WHAT THE CURRENT MODEL GETS RIGHT
# Updated: 2026-05-27 — all five realism layers now implemented

### ROCKS (all done)
- [x] CRISM-calibrated material colors (5 units: Basalt, IronRich, MarsOxide, Sandstone, PaleDust)
- [x] Voronoi fracture rock shapes (angularity)
- [x] Powers roundness distribution per geological unit (Khan et al. 2022)
- [x] Laplacian weathering (rounds edges, Priour 2020)
- [x] Ventifact erosion — ancient paleowind 94° (Herkenhoff 2023)
- [x] Grain-scale noise (micro-roughness, two octaves)
- [x] Golombek aspect ratios (h = 0.5× d)
- [x] 40% protrusion above surface (60% buried, Golombek 2008)
- [x] Two geological units (Máaz / Séítah, Stack et al. 2020)
- [x] Slope-weighted spatial clustering near scarps
- [x] LAYER 4: Per-vertex dust mantling (leeward=WNW dust-coated, windward=clean)
      World-space normals via yaw/pitch/roll rotation matrix
      Wind 276° WNW (Chojnacki 2018), exponent=0.7 (Bridges 2014)
      Validated: leeward/windward ratio 2.16× ✓

### GROUND (all done)
- [x] 5-layer terrain vertex color model (comprehensive, with rock_xz wind shadows)
      1. Slope → basalt/oxide blend
      2. Aeolian ripple grain sorting (crest=olivine grain, trough=red dust)
      3. Topographic dust accumulation (lows=bright PaleDust)
      4. Polygon crack brightness +50% (sulfate fill, Crumpler 2023)
      5. Wind shadow patches behind boulders (transport 276°, Chojnacki 2018)
- [x] LAYER 1 (to run on server): HiRISE DEM loader ready
      download_hirise.sh → ~/mars-rover-agent/data/jezero_hirise.tif
      Validated: auto-fallback to procedural if file missing
- [x] LAYER 2/3 combined: Aeolian ripples baked into elevation
      λ=3.5m, h=0.15m, asymmetric 75/25 stoss/lee, transport 276°
      + megaripples (λ=6–10m, h=0.2–0.35m) near boulders
- [x] Polygon crack network (Voronoi, ~5m diameter, sulfate-bright, Crumpler 2023)
- [x] LAYER 5: Pebble scatter via USD PointInstancer
      2500 clasts, 1–5 cm diameter, 3 size prototypes (60:30:10%)
      65% clustered near rocks (exponential decay λ=0.4m, Vaughan 2023)
      35% uniform background
      Icosphere subs=0 + Voronoi fracture + grain noise — angular clasts

### ATMOSPHERE / SKY
- [x] Correct sun angle (28° elev, 15° az), color (near-white, Bell 2021)
- [x] Angular diameter 0.35° (Mars-correct, smaller than Earth)
- [x] LAYER 3: Mars sky HDR texture (scripts/generate_mars_sky.py)
      Henyey-Greenstein phase function, g=0.68 (Madeleine 2012)
      Zenith: pinkish-tan (0.43, 0.28, 0.20) in linear RGB
      Horizon: brighter/redder (0.72, 0.51, 0.34) — air-mass brightening
      Sun halo: 25° corona, 0.35° disk
      Output: data/mars_sky.png (2048×1024) — auto-loaded by DomeLight
      Validated: horizon/zenith ratio 1.73× ✓  forward-scatter 2.99× ✓

---

## 8. WHAT IS MISSING (remaining after 5-layer plan)

### GROUND — Next priority after GPU is back

P2 — Small impact craters (0.5–5m, ejecta rim)
      ~2–5 per 40×40m scene; slight elevation rim; brighter ejecta blanket
P2 — Bedrock slab exposure (flat dark Máaz basalt where regolith removed)
      Farley 2022: "thinly veneered regolith over competent basalt"
P3 — Lag pavement / deflation surfaces (dark, interlocking flat pebbles)
P3 — Dust devil tracks (lighter circular/linear paths, rare but visible)

### ROCKS — Next priority

P2 — Vesicular surface texture (pitted basalt, gas-bubble voids)
      Bhartia et al. 2021 SHERLOC — confirmed at Jezero
      Strategy: random negative spherical indentations on mesh surface
P2 — Coherent fracture families (cooling joints / thermal fatigue sets)
      Currently: random Voronoi cuts. Real: aligned with stress directions.
      Eppes 2020 (thermal fatigue), columnar joints in basalt
P3 — Conglomerate texture (embedded rounded sub-clasts for delta rocks)
P3 — Desert varnish darkening (>1 Myr exposed surfaces darker)
P3 — Impact pitting on boulder upper surfaces (microcraters)

### ATMOSPHERE / RENDERING

P2 — Shadow colour accuracy: pinkish not blue
      With HG sky texture this should be automatic — verify after GPU comes back
P3 — Near-surface haze is imperceptible at 40m scale (Beer-Lambert effect
      only meaningful at >100m; τ=0.5 over 40m → 0.2% extinction)

---

## 9. IMPLEMENTED ALGORITHMS (for reference)

### Aeolian ripple mesh deformation (DONE):
z_ripple(x, y) = asymmetric sawtooth along transport direction 276°
  phase = (proj / λ) % 1.0  where proj = x·sin(276°) + y·cos(276°)
  stoss face (phase < 0.75): h × 0.5 × (1 − cos(π × phase / 0.75))
  lee face  (phase ≥ 0.75): h × (1 − (phase − 0.75) / 0.25)
  λ = 3.5 m, h = 0.15 m (Chojnacki 2018, Bridges 2017)
  + sparse megaripples (λ=6–10m, h=0.2–0.35m) near obstacles

### Polygon crack network (DONE):
1. n_polygons random Voronoi seeds (~1 per 25 m²)
2. scipy cKDTree: d1, d2 = distances to 2 nearest seeds
3. crack_proximity = max(0, 1 − (d2−d1)/crack_half_width)  [0–1]
4. Blend terrain → sulfate_color with weight crack_proximity × 0.50
   sulfate_color = (0.45, 0.38, 0.30)  [BRIGHTER, not darker]

### Pebble scatter layer (DONE — USD PointInstancer):
- 3 prototype meshes (icosphere + Voronoi + grain noise)
- 2500 instances: 65% near rocks (Exp(λ=0.4m)), 35% uniform
- Vaughan 2023 (10.1029/2022JE007437)

### Rock dust mantling (DONE — Layer 4):
1. Build world-space rotation matrix from rock yaw/pitch/roll
2. Rotate vertices to world space
3. Per-face normals (cross product of world-space edge vectors)
4. dot = fn · wind_3d  (transport az=276°)
5. mantle_f = clip(dot, 0, 1)^0.7  [exponent extends equatorial belt]
6. Average to per-vertex via np.add.at
7. vertex_color = lerp(base_color, PaleDust, vert_mantle)

### Mars sky HDR (DONE — generate_mars_sky.py):
1. Equirectangular grid → direction vectors
2. cos_theta = dot(pixel_dir, sun_dir)
3. HG(cos_theta, g=0.68) normalised to 0–1
4. Zenith→horizon color gradient (smoothstep)
5. HG modulation: brightness × (0.40 + 0.60 × HG_norm)
6. Corona blend within 25° of sun
7. Sun disk at 0.35° radius

### Vesicular texture (NOT YET — next step for rocks):
- After grain noise, apply random negative spherical indentations on rock surface
- Vesicle radius: 0.5–10mm (for meter-scale rock, close-range detail)
- Strategy: subtract Gaussian bumps at random surface points
- Only worth implementing for cobble/boulder scale at this point

---

## 10. KEY REFERENCES (full list)

Stack, K.M. et al. (2020). Science 370, 643 — Jezero geological units
Farley, K.A. et al. (2022). Science 377 — Perseverance core science, Jezero floor
Horgan, B.H.N. et al. (2020). Icarus 339, 113526 — Jezero spectral units
Quantin-Nataf, C. et al. (2021). Science 374, 697 — Jezero delta
Clark, R.N. et al. (2017). Icarus 282, 130 — Mars surface spectroscopy
Bell, J.F. et al. (2021). Space Science Reviews 217, 24 — Mastcam-Z calibration
Golombek, M.P. et al. (2008). JGR 113, E00A11 — CFA distribution, rock shapes
Khan, Z.A. et al. (2022). JGR Planets 127 — Powers roundness at Mars craters
Herkenhoff, K.E. et al. (2023). JGR Planets — Jezero paleowind 94° azimuth
Bridges, N.T. et al. (2014). Aeolian Research 9 — ventifacts at Gale crater
Priour, D.J. (2020). arXiv:2003.03476 — curvature-driven weathering
Raghavachary, S. (2002). SIGGRAPH — Voronoi fracture for geological shapes
Sullivan, R. et al. (2005). Nature — aeolian processes at Spirit/Opportunity
Diniega, S. et al. (2021). Aeolian Research — bedforms at Jezero
El-Maarry, M.R. et al. (2015). Icarus — polygon formation mechanisms on Mars
Eppes, M.C. et al. (2020). Nature Communications — thermal fatigue fracturing Mars
Bhartia, R. et al. (2021). Science 374 — SHERLOC grain size at Jezero
Mangold, N. et al. (2021). Science 374 — conglomerate at Jezero delta front
Siebach, K.L. & Grotzinger, J.P. (2014). JGR — desiccation at Gale crater
Golombek, M.P. et al. (2020). SSR — Jezero surface characterization, landing site
