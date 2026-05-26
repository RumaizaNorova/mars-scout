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

- [x] CRISM-calibrated material colors (5 units)
- [x] Voronoi fracture rock shapes (angularity)
- [x] Powers roundness distribution (Khan et al. 2022)
- [x] Laplacian weathering (rounds edges)
- [x] Ventifact erosion (Herkenhoff 2023 paleowind)
- [x] Grain-scale noise (micro-roughness)
- [x] Golombek aspect ratios (h = 0.5× d)
- [x] 40% protrusion (embedded rocks)
- [x] Two geological units (Máaz / Séítah)
- [x] Slope-weighted spatial clustering
- [x] Slope-driven terrain vertex colors (basic)
- [x] Correct sun angle and color
- [x] Correct sky color

---

## 8. WHAT IS MISSING (PRIORITIZED)

### GROUND — Critical (every camera frame shows this)

P1 — Aeolian ripples (sinusoidal height modulation, λ=0.5–1m, A=1–3cm)
P1 — Polygonal crack network (Voronoi pattern, 1–4m polygons, cracks 2–8cm wide)
P2 — Wind shadow dust patches behind boulders (triangular lighter zones)
P2 — Bedrock slab exposure (flat dark patches in high-slope areas)
P2 — Pebble/granule scatter layer (dense 0.5–5cm particles as ground surface)
P3 — Small impact craters (0.5–5m, slight ejecta rim)
P3 — Dust devil tracks (lighter circular/linear paths)

### ROCKS — Important (individually convincing, not just collectively)

P2 — Vesicular surface texture (pitted basalt — gas bubbles during eruption)
P2 — Asymmetric dust mantling (dust on east/lee face, clean windward face)
P2 — Thermal spalling texture (thin stepped flakes on top surfaces)
P2 — Coherent fracture families (current cuts are random — real fractures are
      aligned with cooling/stress directions)
P3 — Conglomerate texture for delta-proximal rocks (embedded rounded sub-clasts)
P3 — Desert varnish darkening (older surfaces → darker Fe/Mn coating)
P3 — Impact pitting (microcraters on exposed upper surfaces of boulders)

### ATMOSPHERE/RENDERING

P2 — Horizon atmospheric haze (blue/pink gradient toward horizon)
P2 — Shadow colour: pinkish not blue (sky fill from dusty sky)
P3 — Sun disk visible in sky (0.35° angular diameter, slightly orangeish)

---

## 9. MODELING STRATEGIES (not yet coded)

### Aeolian ripple mesh deformation:
z_ripple(x, y) = A × (sin(2π × (x cos θ + y sin θ) / λ))^2_asymmetric
where A=0.015m, λ=0.7m, θ=4° (perpendicular to 94° paleowind)
Asymmetric: use sharper crest (e.g. sawtooth-blended sine) for lee side

### Polygon crack network:
1. Generate random Voronoi seeds at ~1 polygon per 2–4 m²
2. For each edge of Voronoi diagram, slightly perturb vertices (not perfectly straight)
3. Slightly lower elevation along crack lines (0.5–3 cm)
4. Color cracks darker (shadow) or lighter (dust-filled)
5. Option: add thin sub-cracking within polygons (smaller scale repeat)

### Pebble scatter layer:
- Not full 3D meshes (too many) — use displaced instances of simplified discs
- ~5000–10000 pebbles in 40×40m, 0.5–5 cm diameter
- Each: random orientation, mostly flat-lying
- Color: Basalt/MarsOxide mix with random dark/light variation
- Spatial density varies: sparse on bedrock slabs, dense in ripple troughs

### Rock dust mantling:
- Add windward/leeward color variation per rock mesh
- Windward face (west): darker, clean basalt color
- Lee face (east): lighter, apply PaleDust mix
- Implement as per-face color blending based on face normal dot wind_dir

### Vesicular texture:
- After grain noise, apply random negative spherical indentations on rock surface
- Vesicle radius: 0.5–10mm (for meter-scale rock, this is very small detail)
- Most visible at close range — only worth implementing for cobble/boulder scale

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
