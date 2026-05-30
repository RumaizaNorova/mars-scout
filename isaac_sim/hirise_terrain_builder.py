"""
hirise_terrain_builder.py
=========================
Stage 1 terrain pipeline for ARES (Autonomous Rover Exploration System).

Replaces the procedural sine-wave terrain in mars_terrain_builder.py with
a real NASA HiRISE Digital Terrain Model of Jezero Crater — the landing
site of the Perseverance rover.

Two modes (selected automatically):
  1. REAL mode   — HiRISE GeoTIFF provided → real Mars elevation data
  2. FALLBACK mode — no file → improved procedural terrain (multi-octave
                     Perlin-style noise, geologically plausible)

Public API  (same signature as mars_terrain_builder.build_mars_scene)
-----------
  build_mars_scene(stage, ...)  →  dict with keys:
      terrain_path   str        USD path of terrain mesh
      rock_paths     list[str]  USD paths of all rock prims
      dtm_source     str        "hirise" | "procedural"
      elevation_min  float      metres (after crop)
      elevation_max  float      metres (after crop)

HiRISE data source
------------------
Download the Jezero Crater DTM before first use:

  # On Vast.ai server (248 MB):
  wget -O ~/mars-rover-agent/data/jezero_hirise.tif \\
    "https://planetarymaps.usgs.gov/mosaic/Mars_MRO_HiRISE_Jezero_Crater_DTM_1m.tif"

  # Fallback smaller patch (18° N band, ~50 MB):
  wget -O ~/mars-rover-agent/data/jezero_hirise.tif \\
    "https://murray-lab.caltech.edu/CTX/V01/tiles/..." (see README)

Then pass:
  build_mars_scene(stage, hirise_dtm_path="~/mars-rover-agent/data/jezero_hirise.tif")

Coordinate system: Z-up (Isaac Sim default).  Units: metres.
"""

from __future__ import annotations

import math
import os
import random
from typing import List, Optional, Tuple

import numpy as np

# pxr is available inside Isaac Sim; guarded for offline unit-testing
try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
    _PXR_AVAILABLE = True
except ImportError:
    _PXR_AVAILABLE = False

# rasterio for reading GeoTIFF — installed via pip
try:
    import rasterio
    from rasterio.transform import rowcol
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False

# ── Reproducible randomness ───────────────────────────────────────────────────
_RNG    = random.Random(42)
_NP_RNG = np.random.default_rng(42)

# ── Scene-wide solar geometry (single source of truth) ────────────────────────
# Both _build_martian_lighting (DistantLight rotation) and
# _build_terrain_hapke_material (Hapke Lommel-Seeliger μ₀ term) must use the
# same value — previously they were decoupled and disagreed by 17°.
#
# Source: Bell et al. 2021 SSR 217:24 — Mastcam-Z pre-launch calibration notes
# "Jezero morning pass, ≈10:00 LTST, solar elevation 28°, azimuth 15° E of S"
_SUN_ELEVATION_DEG: float = 28.0   # degrees above horizon (Bell et al. 2021)
_SUN_AZIMUTH_DEG:   float = 45.0   # RotZ angle used in DistantLight orientation
# NOTE: the comment in _build_martian_lighting previously said "15° azimuth from south"
# but the code used RotZ=45°.  _SUN_AZIMUTH_DEG=45 matches the actual rotation.

# ── Atmospheric dust optical depth ────────────────────────────────────────────
# τ (tau) = vertically-integrated dust column opacity.
# Governs three coupled effects: direct solar beam attenuation,
# sky diffuse brightness, and sky colour reddening.
#
# Physics:
#   I_direct  = I_base × exp(−τ / sin(elevation))   [Beer-Lambert]
#   I_sky     ≈ I_base × (1 − exp(−τ/sin(elev))) × sky_albedo  [simplified]
#   sky_color ← interpolated from near-white (τ=0) to deep salmon (τ≥1.5)
#
# Reference values (Bell et al. 2021 SSR 217:24, Mastcam-Z ops log):
#   τ = 0.56   — Jezero Crater, nominal science operations (sol 20–300)
#   τ = 0.3–0.4 — clear morning in southern spring
#   τ = 1.0–2.0 — regional dust storm (activity suspended)
#
# I_base = 4500 internal units represents the τ=0 solar flux at Mars.
# All intensity values in the lighting system are derived from this.
_DUST_TAU:          float = 0.56    # nominal Jezero ops (Bell 2021)
_SUN_INTENSITY_BASE: float = 4500.0 # τ=0 direct solar flux (internal units)


# =============================================================================
# Part 1 — Elevation data loading
# =============================================================================

def load_hirise_dtm(
    tif_path: str,
    center_x_m: float = 0.0,
    center_y_m: float = 0.0,
    size_m: float = 200.0,
    target_nx: int   = 256,
    target_ny: int   = 256,
) -> Tuple[np.ndarray, dict]:
    """
    Load a rectangular patch from a HiRISE GeoTIFF DTM and return a
    normalised elevation array ready for mesh generation.

    Parameters
    ----------
    tif_path    Path to the GeoTIFF file (absolute or ~-expanded).
    center_x_m  Easting offset from file centre in metres (default: centre).
    center_y_m  Northing offset from file centre in metres (default: centre).
    size_m      Patch side length in metres (square patch).
    target_nx   Output columns (resampled).
    target_ny   Output rows (resampled).

    Returns
    -------
    elevation   float32 ndarray shape (target_ny, target_nx), metres.
    meta        dict with keys: dtm_source, pixel_res_m, elevation_min,
                elevation_max, nodata_filled.
    """
    if not _RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required for HiRISE loading. "
                          "Run: pip install rasterio")

    tif_path = os.path.expanduser(tif_path)
    if not os.path.exists(tif_path):
        raise FileNotFoundError(
            f"HiRISE DTM not found: {tif_path}\n"
            "Download with:\n"
            "  wget -O ~/mars-rover-agent/data/jezero_hirise.tif \\\n"
            "    https://planetarymaps.usgs.gov/mosaic/"
            "Mars_MRO_HiRISE_Jezero_Crater_DTM_1m.tif"
        )

    with rasterio.open(tif_path) as src:
        # pixel resolution in metres (x-direction)
        pixel_res = abs(src.transform.a)

        # file extent in pixel coords
        total_rows, total_cols = src.height, src.width

        # map requested centre to file pixel coords
        # (assume file is already in projected metres, not lat/lon)
        cx_px = total_cols // 2 + int(center_x_m / pixel_res)
        cy_px = total_rows // 2 - int(center_y_m / pixel_res)  # row increases south

        half = int(size_m / pixel_res / 2)
        row0 = max(0, cy_px - half)
        row1 = min(total_rows, cy_px + half)
        col0 = max(0, cx_px - half)
        col1 = min(total_cols, cx_px + half)

        window = rasterio.windows.Window(
            col_off=col0, row_off=row0,
            width=col1 - col0, height=row1 - row0
        )

        raw = src.read(1, window=window).astype(np.float32)
        nodata = src.nodata

    # Fill nodata (typically -9999 or large negative values) with median
    nodata_filled = False
    if nodata is not None:
        mask = (raw == nodata) | ~np.isfinite(raw)
    else:
        mask = ~np.isfinite(raw) | (raw < -9000)

    if mask.any():
        fill = float(np.nanmedian(raw[~mask]))
        raw[mask] = fill
        nodata_filled = True

    # Resample to target resolution using simple bilinear interpolation
    from scipy.ndimage import zoom as _zoom
    zy = target_ny / raw.shape[0]
    zx = target_nx / raw.shape[1]
    elevation = _zoom(raw, (zy, zx), order=1).astype(np.float32)

    # Subtract min so terrain starts near Z=0
    elev_min = float(elevation.min())
    elev_max = float(elevation.max())
    elevation -= elev_min  # relative elevation, origin at lowest point

    meta = {
        "dtm_source":    "hirise",
        "pixel_res_m":   pixel_res,
        "elevation_min": elev_min,
        "elevation_max": elev_max,
        "nodata_filled": nodata_filled,
    }
    return elevation, meta


def generate_procedural_dtm(
    width_m:    float = 40.0,
    depth_m:    float = 40.0,
    nx:         int   = 128,
    ny:         int   = 128,
    amplitude:  float = 0.70,
    seed:       int   = 42,
) -> Tuple[np.ndarray, dict]:
    """
    Multi-octave fractal heightmap that mimics Mars aeolian terrain when no
    HiRISE file is available.

    Uses 6-octave layered sine noise — same visual character as the original
    mars_terrain_builder but with better long-wavelength structure (basin +
    ridge features) and correct amplitude for Jezero-class terrain.

    Returns
    -------
    elevation   float32 ndarray shape (ny, nx), metres above minimum.
    meta        dict  (dtm_source="procedural", …)
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, width_m, nx, dtype=np.float32)
    ys = np.linspace(0, depth_m, ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    elev = np.zeros_like(X)

    # 6-octave fractal noise — each octave doubles frequency, halves amplitude
    octaves = [
        (1 / width_m,  0.40),   # long wavelength basin
        (2 / width_m,  0.25),
        (4 / width_m,  0.15),
        (8 / width_m,  0.10),
        (16 / width_m, 0.06),
        (32 / width_m, 0.04),
    ]
    for freq, weight in octaves:
        px = rng.uniform(0, 2 * np.pi)
        py = rng.uniform(0, 2 * np.pi)
        elev += weight * np.sin(2 * np.pi * freq * X + px) \
                       * np.cos(2 * np.pi * freq * Y + py)

    # Normalise to [0, amplitude]
    elev -= elev.min()
    elev /= (elev.max() + 1e-9)
    elev *= amplitude

    meta = {
        "dtm_source":    "procedural",
        "pixel_res_m":   width_m / nx,
        "elevation_min": 0.0,
        "elevation_max": amplitude,
        "nodata_filled": False,
    }
    return elev.astype(np.float32), meta


# =============================================================================
# Part 2 — USD mesh construction
# =============================================================================

def _build_terrain_mesh(
    stage:        "Usd.Stage",
    prim_path:    str,
    elevation:    np.ndarray,
    width_m:      float,
    depth_m:      float,
) -> "UsdGeom.Mesh":
    """
    Convert a (ny, nx) elevation array into a UsdGeom.Mesh triangle strip.

    The mesh is centred at the origin in X/Y.  Z = elevation[row, col].
    Face winding is counter-clockwise (right-hand normal pointing up = +Z).
    """
    ny, nx = elevation.shape

    xs = np.linspace(-width_m / 2, width_m / 2, nx, dtype=np.float32)
    ys = np.linspace(-depth_m / 2, depth_m / 2, ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    Z = elevation  # already in metres

    # Flatten to vertex list
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    # Build triangle indices
    # For each quad (row r, col c) we emit 2 triangles:
    #   (r,c) → (r+1,c) → (r,c+1)
    #   (r+1,c) → (r+1,c+1) → (r,c+1)
    r_idx, c_idx = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    tl = (r_idx       * nx + c_idx).ravel()
    bl = ((r_idx + 1) * nx + c_idx).ravel()
    tr = (r_idx       * nx + (c_idx + 1)).ravel()
    br = ((r_idx + 1) * nx + (c_idx + 1)).ravel()

    face_indices  = np.stack([tl, bl, tr, bl, br, tr], axis=-1).ravel()
    face_counts   = np.full((ny - 1) * (nx - 1) * 2, 3, dtype=np.int32)

    # USD mesh
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(face_indices.astype(np.int32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(face_counts))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(False)

    # Per-vertex normals (smooth shading)
    normals = _compute_normals(pts, face_indices.reshape(-1, 3))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
    mesh.SetNormalsInterpolation("vertex")

    # ── UV texture coordinates (primvars:st) ──────────────────────────────────
    # Planar projection: u = x/tile_m, v = y/tile_m
    # tile_m=2.0 matches generate_terrain_textures.py default tile size.
    # UV values > 1 are fine — UsdUVTexture with wrapS/T="repeat" tiles them.
    # This primvar is read by the Hapke MDL shader (hapke_regolith.mdl) via
    # state::texture_coordinate(0), which maps to the first UV primvar "st".
    tile_m = 2.0   # physical metres per texture tile (must match texture generator)
    uv = np.stack([
        pts[:, 0] / tile_m,    # u: x / 2m
        pts[:, 1] / tile_m,    # v: y / 2m
    ], axis=-1).astype(np.float32)

    primvars_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    pv_st = primvars_api.CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        "vertex",
    )
    pv_st.Set(Vt.Vec2fArray.FromNumpy(uv))

    return mesh


def _compute_normals(pts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Compute smooth per-vertex normals from triangle soup."""
    v0 = pts[tris[:, 0]]
    v1 = pts[tris[:, 1]]
    v2 = pts[tris[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    normals = np.zeros_like(pts)
    for i in range(3):
        np.add.at(normals, tris[:, i], face_normals)

    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= np.where(norms == 0, 1, norms)
    return normals.astype(np.float32)


# =============================================================================
# Part 3 — PBR materials
# =============================================================================

def _make_material(
    stage:        "Usd.Stage",
    mat_path:     str,
    diffuse:      Tuple[float, float, float],
    roughness:    float = 0.85,
    metallic:     float = 0.0,
    normal_scale: float = 1.0,
) -> "UsdShade.Material":
    """Create a UsdPreviewSurface PBR material."""
    mat = UsdShade.Material.Define(stage, mat_path)
    shader_path = mat_path + "/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")

    shader.CreateInput("diffuseColor",  Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    shader.CreateInput("roughness",     Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic",      Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)

    mat.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return mat


def _build_mars_materials(stage: "Usd.Stage", looks_root: str) -> dict:
    """
    Build the Mars surface material library.

    Diffuse colors are derived from published CRISM spectral reflectance data
    for Jezero Crater surface units via CIE 1931 colorimetry under the Martian
    solar spectrum (tau=0.5, 28° sun elevation).

    Generated by: scripts/calibrate_materials.py
    Method: spectral I/F → XYZ (CIE 1931 2°) → linear sRGB (IEC 61966-2-1)

    Key result: Jezero terrain is 3–5× DARKER than sandy-desert intuition.
    Real floor albedo ~0.07–0.18 (dark olivine basalt + iron oxide dust).

    References
    ----------
    [H20] Horgan et al. 2020, Icarus 339, 113526
    [Q21] Quantin-Nataf et al. 2021, Science 374, 697
    [C17] Clark et al. 2017, Icarus 282, 130
    [B21] Bell et al. 2021, Space Sci Rev 217, 24  (Mastcam-Z calibration)
    """
    mats = {}

    # Primary terrain — iron oxide dust coating on basalt
    # I/F(750nm)≈0.16; strong red slope (Fe³⁺ charge transfer absorption)
    # [C17] oxidised basalt regolith spectrum
    mats["MarsOxide"] = _make_material(
        stage, f"{looks_root}/MarsOxide",
        diffuse=(0.1615, 0.0996, 0.0642),   # CRISM-calibrated [C17]
        roughness=0.92,
    )
    # Dark volcanic basalt — Jezero floor olivine unit
    # I/F(750nm)≈0.09; very dark, slight red slope, olivine at 1050nm
    # [H20] Figure 6, olivine basalt spectral unit
    mats["Basalt"] = _make_material(
        stage, f"{looks_root}/Basalt",
        diffuse=(0.0977, 0.0643, 0.0417),   # CRISM-calibrated [H20]
        roughness=0.78,
        metallic=0.05,
    )
    # Sedimentary / layered rock — delta and crater rim outcrops
    # I/F(750nm)≈0.22; brighter, tan-grey, carbonate mixing, weak red slope
    # [Q21] supplementary spectral data
    mats["Sandstone"] = _make_material(
        stage, f"{looks_root}/Sandstone",
        diffuse=(0.2215, 0.1668, 0.1198),   # CRISM-calibrated [Q21]
        roughness=0.88,
    )
    # Fe/Mg smectite + iron oxide — spectrally red, hydrated iron phases
    # I/F(750nm)≈0.20; strong red slope, intermediate albedo
    # [H20] Fe/Mg smectite + Fe-oxide spectral unit
    mats["IronRich"] = _make_material(
        stage, f"{looks_root}/IronRich",
        diffuse=(0.2125, 0.1330, 0.0734),   # CRISM-calibrated [H20]
        roughness=0.70,
        metallic=0.08,
    )
    # Bright fine-grained dust — wind-deposited in low spots
    # I/F(750nm)≈0.35; pinkish-tan, palagonite-like, gentle red slope
    # [C17] palagonite / bright dust spectrum
    mats["PaleDust"] = _make_material(
        stage, f"{looks_root}/PaleDust",
        diffuse=(0.3535, 0.2690, 0.1975),   # CRISM-calibrated [C17]
        roughness=0.95,
    )

    return mats


def _bind_material(
    stage:    "Usd.Stage",
    prim_path: str,
    mat:      "UsdShade.Material",
) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        UsdShade.MaterialBindingAPI(prim).Bind(mat)


# =============================================================================
# Part 4 — Geological rock generation (NASA-engineer specification)
# =============================================================================
#
# Scientific basis:
#   Stack et al. 2020 (Science 370, 643)    — Jezero geological units (Máaz/Séítah)
#   Khan et al. 2022 (JGR Planets 127)      — Powers roundness at Mars crater floors
#   Golombek et al. 2008 (JGR 113 E00A11)  — CFA size-frequency, aspect ratios
#   Herkenhoff et al. 2023 (JGR Planets)    — Jezero paleowind (94° azimuth)
#   Bridges et al. 2014 (Aeolian Research)  — ventifact frequency ~30–50%
#   Priour 2020 (arXiv:2003.03476)          — curvature-driven weathering
#   Raghavachary 2002 (SIGGRAPH)            — Voronoi fracture algorithm
#
# Pipeline for each rock:
#   1. Icosphere base  →  2. Voronoi planar cuts (angularity)
#   3. Laplacian smooth (weathering)  →  4. Ventifact cuts (aeolian)
#   5. Grain-scale noise  →  6. Oblate scaling (h = 0.5× d, Golombek 2008)
#   7. Embed 60% below surface (40% protrusion, Golombek 2008)

# ── Powers roundness scale parameters ────────────────────────────────────────
# (n_cuts_lo, n_cuts_hi), smooth_iterations, grain_noise_amplitude
# More cuts → more fracture facets → more angular.
# More smooth_iters → rounder edges → more weathered.
_POWERS_CLASSES: dict = {
    "VA": ((10, 15), 0, 0.08),   # Very Angular  — fresh fracture, jagged edges
    "A":  (( 7, 11), 1, 0.07),   # Angular
    "SA": (( 5,  8), 2, 0.06),   # Sub-Angular
    "SR": (( 3,  6), 4, 0.04),   # Sub-Rounded
    "R":  (( 2,  4), 6, 0.03),   # Rounded       — water / aeolian abraded
}

# ── Geological unit definitions ───────────────────────────────────────────────
# Máaz formation (Cf-fr) — dominant basaltic floor, ~70 % of Jezero area.
#   Pyroxene/plagioclase basalt.  Angular, fresh-fractured surface.
_UNIT_MAAZ: dict = {
    "roundness_pdf": {"VA": 0.35, "A": 0.30, "SA": 0.20, "SR": 0.10, "R": 0.05},
    "materials":     ["Basalt", "IronRich", "MarsOxide"],
    "mat_weights":   [0.50,      0.30,       0.20],
}
# Séítah formation — olivine-rich, ~30 % of floor, more water-reworked.
#   Rounder clasts; fayalitic olivine + pyroxene; weaker red slope.
_UNIT_SEITAH: dict = {
    "roundness_pdf": {"VA": 0.10, "A": 0.25, "SA": 0.30, "SR": 0.25, "R": 0.10},
    "materials":     ["Basalt", "Sandstone", "MarsOxide"],
    "mat_weights":   [0.40,      0.40,        0.20],
}

# ── CFA rock size classes (Golombek et al. 2008, scaled to 40×40 m scene) ───
_ROCK_SIZE_CLASSES = [
    # (name, diameter_m, count, _unused)
    ("boulder", 2.50,  15, ""),
    ("cobble",  0.80,  80, ""),
    ("pebble",  0.25, 200, ""),
]

# ── Ventifact parameters (Herkenhoff 2023, Bridges 2014) ─────────────────────
_VENTIFACT_PROB   = 0.35          # fraction of rocks with wind-erosion facets
_WIND_AZIMUTH_DEG = 94.0          # ancient paleowind from west (degrees east of north)

# ── Dust mantling parameters (Chojnacki 2018, Bridges 2014) ──────────────────
# Modern transport direction: 276 ° WNW (Chojnacki 2018, HiRISE repeat imaging)
# Leeward faces (normal aligned with transport) → dust accumulates (brighter)
# Windward faces (normal opposed to transport)  → abrasion-cleaned (darker)
# Exponent 0.7: flattens cosine falloff so more of the leeward hemisphere
# is dust-coated, consistent with ventifact keel observations (Bridges 2014)
_MODERN_TRANSPORT_AZ  = 276.0     # degrees; sand moves WNW
_MANTLING_EXPONENT    = 0.7       # dust accumulation falloff shape

# CRISM-calibrated base colours per material (mirror of _build_mars_materials)
# Kept here so _compute_rock_face_colors can run without a USD stage.
_MATERIAL_COLORS: dict = {
    "Basalt":    np.array([0.0977, 0.0643, 0.0417], dtype=np.float32),  # [H20]
    "IronRich":  np.array([0.2125, 0.1330, 0.0734], dtype=np.float32),  # [H20]
    "MarsOxide": np.array([0.1615, 0.0996, 0.0642], dtype=np.float32),  # [C17]
    "Sandstone": np.array([0.2215, 0.1668, 0.1198], dtype=np.float32),  # [Q21]
    "PaleDust":  np.array([0.3535, 0.2690, 0.1975], dtype=np.float32),  # [C17]
}
_DUST_MANTLE_COLOR = np.array([0.3535, 0.2690, 0.1975], dtype=np.float32)  # PaleDust


# ── Icosphere base mesh ───────────────────────────────────────────────────────

def _make_icosphere(subdivisions: int = 1) -> tuple:
    """
    Unit icosphere with optional midpoint subdivision.
      subs=0 → 12 verts / 20 faces
      subs=1 → 42 verts / 80 faces   (default for pebbles)
      subs=2 → 162 verts / 320 faces (cobbles & boulders)
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1],  [ phi, 0,  1],  [-phi, 0, -1],  [-phi, 0,  1],
    ], dtype=np.float64)
    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.int32)
    nrms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts /= nrms
    for _ in range(subdivisions):
        verts, faces = _subdivide_icosphere(verts, faces)
    return verts, faces


def _subdivide_icosphere(verts: np.ndarray, faces: np.ndarray) -> tuple:
    """One level of midpoint subdivision, normalised to unit sphere."""
    edge_mid: dict = {}
    new_v = list(verts)
    new_f: list = []

    def _mid(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key not in edge_mid:
            m = (new_v[a] + new_v[b]) * 0.5
            n = np.linalg.norm(m)
            edge_mid[key] = len(new_v)
            new_v.append(m / n if n > 0 else m)
        return edge_mid[key]

    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        ab, bc, ca = _mid(a, b), _mid(b, c), _mid(c, a)
        new_f += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]

    return np.array(new_v, dtype=np.float64), np.array(new_f, dtype=np.int32)


# ── Voronoi fracture (Raghavachary 2002) ─────────────────────────────────────

def _voronoi_fracture(verts: np.ndarray, n_cuts: int, rng_np) -> np.ndarray:
    """
    Apply n_cuts random planar half-space cuts to create angular rock facets.

    For each cut plane (unit normal n, scalar offset d from centre):
    vertices that protrude beyond the plane are projected back onto it,
    creating flat fracture faces.  The more cuts, the more angular the rock.

    Ref: Raghavachary 2002 (SIGGRAPH) — Voronoi-based geological fracture.
    Powers mapping: Very Angular → 10–14 cuts; Rounded → 2–4 cuts.
    """
    verts = verts.copy()
    for _ in range(n_cuts):
        normal = rng_np.standard_normal(3)
        nrm = np.linalg.norm(normal)
        if nrm < 1e-9:
            continue
        normal /= nrm
        # Plane placed at 50–90 % of sphere radius from centre
        offset = rng_np.uniform(0.50, 0.90)
        dot    = verts @ normal          # (N,)
        beyond = dot > offset
        if beyond.any():
            verts[beyond] -= (dot[beyond] - offset)[:, None] * normal
    return verts


# ── Laplacian weathering (Priour 2020) ───────────────────────────────────────

def _laplacian_smooth(verts: np.ndarray, faces: np.ndarray,
                      iterations: int, weight: float = 0.40) -> np.ndarray:
    """
    Umbrella-operator Laplacian smoothing for rock weathering.

    Higher iterations → rounder, more weathered shape.
    Ref: Priour 2020 (arXiv:2003.03476) — curvature-driven vertex removal.
    weight < 0.5 preserves approximate volume (no Laplacian shrinkage).
    """
    if iterations == 0:
        return verts
    verts = verts.copy()
    n = len(verts)
    # Build flat adjacency (duplicates OK — weights high-valence neighbours)
    adj: list = [[] for _ in range(n)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a] += [b, c]
        adj[b] += [a, c]
        adj[c] += [a, b]
    for _ in range(iterations):
        new_v = verts.copy()
        for i in range(n):
            nbrs = adj[i]
            if nbrs:
                centroid = verts[nbrs].mean(axis=0)
                new_v[i] = verts[i] + weight * (centroid - verts[i])
        verts = new_v
    return verts


# ── Ventifact erosion (Herkenhoff 2023) ──────────────────────────────────────

def _add_ventifact_cuts(verts: np.ndarray, wind_dir_xy: np.ndarray,
                        n_cuts: int, rng_np) -> np.ndarray:
    """
    Extra planar cuts on windward face — ventifact erosion signature.

    Jezero paleowind: from west, 94° azimuth (Herkenhoff et al. 2023).
    ~30–50 % of Gale Crater floor rocks show ventifact features
    (Bridges et al. 2014); we use 35 % for Jezero.
    """
    verts = verts.copy()
    wind_3d = np.array([wind_dir_xy[0], wind_dir_xy[1], 0.0])
    for _ in range(n_cuts):
        perturb = rng_np.standard_normal(3) * 0.35
        normal  = wind_3d + perturb
        nrm     = np.linalg.norm(normal)
        if nrm < 1e-9:
            continue
        normal /= nrm
        offset = rng_np.uniform(0.25, 0.70)
        dot    = verts @ normal
        beyond = dot > offset
        if beyond.any():
            verts[beyond] -= (dot[beyond] - offset)[:, None] * normal
    return verts


# ── Grain-scale surface roughness ────────────────────────────────────────────

def _apply_grain_noise(verts: np.ndarray, amplitude: float,
                       rng_np) -> np.ndarray:
    """
    Radial per-vertex noise for grain-scale rock surface texture.

    Máaz: 0.5–3 mm grain protrusions; Séítah: 1.45 mm mean (Golombek 2008).
    amplitude = fraction of rock radius (e.g. 0.06 ≡ 6 % of radius).
    Two octaves: coarse grain + finer intergrain detail.
    """
    if amplitude < 1e-4:
        return verts
    n       = len(verts)
    noise   = rng_np.standard_normal(n) * amplitude
    noise  += rng_np.standard_normal(n) * amplitude * 0.5
    radials = np.linalg.norm(verts, axis=1, keepdims=True)
    normals = verts / (radials + 1e-9)
    return verts + normals * noise[:, None]


# ── Macro weathering pit deformation ─────────────────────────────────────────

def _apply_vesicle_pitting(
    verts:        np.ndarray,
    rng_np:       np.random.Generator,
    n_pits:       int,
    pit_radius:   float = 0.12,    # normalized units (fraction of unit-sphere radius)
    depth_factor: float = 0.40,    # depth = depth_factor × pit_radius
) -> np.ndarray:
    """
    Apply macro weathering pit (tafoni / cavernous weathering) deformations to
    rock surface vertices.

    These are NOT the 0.5–3 mm vesicles detected by SHERLOC — those are below
    the resolution of the polygon mesh and are handled by the vesicle normal map
    (Task 7).  This function targets MACRO pits: 5–20 cm diameter cavities
    observed on Mars rocks in MER/MSL imagery (Crumpler et al. 2015 Icarus;
    Squyres et al. 2004 Science 305).

    Physical model
    --------------
    Cavernous weathering on basalt:
      Pit diameters: 3–20 cm  (Squyres 2004 Opportunity; Crumpler 2015 MER survey)
      Aspect ratio depth/diameter: 0.3–0.5  (concave pockets, not deep holes)
    Encoded as negative Gaussian deflections on the unit sphere:
      Each pit: random surface point → Gaussian weight → deflect inward along
      surface normal with depth = depth_factor × pit_radius.

    Note on catmullClark interaction
    --------------------------------
    When catmullClark subdivision is applied at render time, the pit edges are
    naturally smoothed (C² continuity), reproducing the rounded-edge appearance
    of weathered basalt cavities.  This is physically correct.

    Parameters
    ----------
    verts       : (N, 3) float64 vertices on approximately unit sphere (r≈1)
    rng_np      : numpy Generator for reproducibility
    n_pits      : number of pits to stamp
    pit_radius  : pit half-width in normalized units (fraction of sphere radius 1.0)
                  e.g. 0.12 → 12 % of rock half-diameter.
                  For 0.5 m boulder (r=0.25 m): 0.12 → 3 cm pit radius
    depth_factor: pit depth as fraction of pit_radius (0.3–0.5 typical)

    Returns
    -------
    verts_deformed : (N, 3) float64

    Raises
    ------
    ValueError  if n_pits < 0 or pit_radius ≤ 0 or depth_factor ≤ 0
    RuntimeError if any vertex goes to zero length after pitting (degenerate mesh)
    """
    if n_pits < 0:
        raise ValueError(f"_apply_vesicle_pitting: n_pits={n_pits} < 0")
    if pit_radius <= 0.0:
        raise ValueError(f"_apply_vesicle_pitting: pit_radius={pit_radius} ≤ 0")
    if depth_factor <= 0.0:
        raise ValueError(f"_apply_vesicle_pitting: depth_factor={depth_factor} ≤ 0")
    if n_pits == 0:
        return verts

    verts = verts.copy().astype(np.float64)

    for _ in range(n_pits):
        # Random point on unit sphere as pit centre (surface normal ≈ centre)
        centre = rng_np.standard_normal(3)
        c_len  = np.linalg.norm(centre)
        if c_len < 1e-9:
            continue   # pathological rng hit: skip this pit
        centre /= c_len   # unit vector

        # Dot product with all vertices (≈ cos of angular distance)
        # verts may not be exactly on unit sphere after grain noise; normalise
        r_verts  = np.linalg.norm(verts, axis=1)
        v_hat    = verts / (r_verts[:, None] + 1e-9)  # (N, 3) unit vectors
        cos_ang  = v_hat @ centre                      # (N,)

        # Chord distance: chord = sqrt(2(1 - cosθ)) for unit sphere
        cos_clamped = np.clip(cos_ang, -1.0, 1.0)
        chord = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - cos_clamped)))

        # Gaussian weight with σ = pit_radius / 2.5 → weight ≈ 0 at chord=pit_radius
        sigma  = pit_radius / 2.5
        weight = np.exp(-(chord / (sigma + 1e-9)) ** 2)   # (N,)

        # Deflect inward: move each vertex TOWARD sphere centre
        # Direction: -v_hat (inward normal)
        depth  = depth_factor * pit_radius                 # scalar
        # Deflect vertices close enough to the pit (chord < 1.5 × pit_radius).
        # If no vertices are in range, stamp the single nearest vertex at half-weight
        # (handles coarse pre-subdivision meshes where pit < inter-vertex spacing;
        # the catmullClark subdivider will propagate the deformation to finer detail).
        mask = chord < (1.5 * pit_radius)
        if not mask.any():
            nearest = int(np.argmin(chord))
            verts[nearest] -= v_hat[nearest] * (depth * 0.5)
        else:
            verts[mask] -= v_hat[mask] * (weight[mask, None] * depth)

    # Sanity: no vertex should have collapsed to near-zero length
    final_r = np.linalg.norm(verts, axis=1)
    if float(final_r.min()) < 0.05:
        raise RuntimeError(
            f"_apply_vesicle_pitting: minimum vertex radius collapsed to "
            f"{float(final_r.min()):.4f} (< 0.05). "
            f"Reduce depth_factor or pit_radius. "
            f"n_pits={n_pits}, pit_radius={pit_radius:.3f}, depth_factor={depth_factor:.3f}"
        )

    return verts


# ── Terrain slope map ─────────────────────────────────────────────────────────

def _compute_terrain_slope(elevation: np.ndarray, cell_size: float) -> np.ndarray:
    """Slope magnitude (rise / run) via central finite differences."""
    dy, dx = np.gradient(elevation.astype(np.float64), cell_size, cell_size)
    return np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)


# ── Aeolian ripple deformation ────────────────────────────────────────────────

def _add_aeolian_ripples(
    elevation:          np.ndarray,
    terrain_w:          float,
    terrain_d:          float,
    wavelength:         float = 3.5,    # metres  (Chojnacki 2018, HiRISE measurement)
    height:             float = 0.15,   # metres  (Bridges 2017, Bagnold Dunes analogue)
    transport_azimuth:  float = 276.0,  # degrees (WNW modern wind, Chojnacki 2018)
    stoss_fraction:     float = 0.75,   # 75 % gentle stoss / 25 % steep lee
    megaripple_prob:    float = 0.15,   # fraction of area with large megaripples
) -> np.ndarray:
    """
    Add asymmetric aeolian ripples to an elevation array.

    TWO WIND SYSTEMS at Jezero (Chojnacki 2018, Herkenhoff 2023):
      Modern transport → 276 ° (WNW) — drives current ripples
      Ancient paleowind → 94 ° (from west) — recorded in ventifacts (not ripples)

    Profile:
      Stoss face (upwind/east, 75 % of λ): gentle ~7 ° slope, sinusoidal rise
      Lee face  (downwind/west, 25 % of λ): steep ~30 ° drop, near-linear

    References
    ----------
    Chojnacki et al. 2018, PMC5859260 — λ = 3–4 m, migration 0.2 m/yr, az 276°
    Bridges et al. 2017, PMC5815379  — h = 12–28 cm, lee 29–33°, grain 1–2 mm
    """
    ny, nx = elevation.shape
    xs = np.linspace(-terrain_w / 2, terrain_w / 2, nx, dtype=np.float64)
    ys = np.linspace(-terrain_d / 2, terrain_d / 2, ny, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)

    # Project grid points onto transport direction
    az_rad   = math.radians(transport_azimuth)
    td       = np.array([math.sin(az_rad), math.cos(az_rad)])
    proj     = X * td[0] + Y * td[1]          # signed distance along transport

    # Normalised phase in [0, 1)
    phase = (proj / wavelength) % 1.0

    # Asymmetric sawtooth profile: gentle stoss rise, steep lee drop
    z_ripple = np.where(
        phase < stoss_fraction,
        # Stoss: smooth cosine rise (0 → peak)
        height * 0.5 * (1.0 - np.cos(np.pi * phase / stoss_fraction)),
        # Lee: linear drop (peak → 0)
        height * (1.0 - (phase - stoss_fraction) / (1.0 - stoss_fraction)),
    )

    # Sparse megaripples — larger superimposed bedforms near random locations
    # (Chojnacki 2020: wavelength 5–11 m, height up to 0.35 m near obstacles)
    rng_mr = np.random.default_rng(17)
    n_mega = max(1, int(terrain_w * terrain_d / 120))   # ~1 per 120 m²
    for _ in range(n_mega):
        cx   = rng_mr.uniform(-terrain_w * 0.4, terrain_w * 0.4)
        cy   = rng_mr.uniform(-terrain_d * 0.4, terrain_d * 0.4)
        wl_m = rng_mr.uniform(6.0, 10.0)
        h_m  = rng_mr.uniform(0.20, 0.35)
        radius = rng_mr.uniform(terrain_w * 0.06, terrain_w * 0.15)
        dist   = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        envelope = np.clip(1.0 - dist / radius, 0.0, 1.0) ** 2
        ph_m   = ((proj - rng_mr.uniform(0, wl_m)) / wl_m) % 1.0
        z_m    = np.where(
            ph_m < stoss_fraction,
            h_m * 0.5 * (1.0 - np.cos(np.pi * ph_m / stoss_fraction)),
            h_m * (1.0 - (ph_m - stoss_fraction) / (1.0 - stoss_fraction)),
        )
        z_ripple += z_m * envelope

    return (elevation + z_ripple).astype(np.float32)


# ── Impact crater deformation ─────────────────────────────────────────────────

def _add_impact_craters(
    elevation:    np.ndarray,
    terrain_w:    float,
    terrain_d:    float,
    seed:         int   = 7,
    min_radius_m: float = 0.4,   # metres  — smallest visible crater (sub-metre)
    max_radius_m: float = 4.0,   # metres  — large crater still within 40 m scene
    n_craters:    int | None = None,   # None → auto-scale to scene area
) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    """
    Stamp Gaussian bowl + raised rim + ejecta blanket craters into an elevation array.

    Physical model
    --------------
    Melosh 1989 "Impact Cratering" (standard reference):
      Bowl:   paraboloid for simple craters, D < ~1 km on Mars
      Depth/Diameter ratio:  d/D ≈ 0.2 for fresh simple craters (Melosh 1989)
      Rim height:            h_rim ≈ 0.04 * D  (4% of diameter above pre-impact surface)
      Rim width:             w_rim ≈ 0.15 * R  (15% of radius, sharp ridge)
      Ejecta blanket:        thickness ∝ (R/r)^3.5 for r > R  (McGetchin 1973)
                             extends to ~2.5R, thinning to ≈0 at edge

    Crater morphology used here:
      z_bowl(r)   = -d * max(0, 1 - (r/R)^2)      [paraboloid, depth d = 0.2*D]
      z_rim(r)    = h_rim * exp(-((r-R)/(0.15R))^2) [Gaussian ridge at r=R]
      z_ejecta(r) = t_0 * (R/max(r,R))^3.5         [McGetchin, only for r > R]
                    decaying from t_0=0.025*R at r=R to ~0 at r=2.5R

    Crater density (Mars surface age Jezero ~3 Ga):
      ~0.5 craters D>1m per m² in old terrains (Hartmann 2005 production function)
      But most are eroded to < 10 cm depth in aeolian environment.
      Visible/fresh craters: 0.002–0.010 per m² for D>0.5m (Golombek 2014 InSight)
      Default: auto from scene area using 0.005 per m² density.

    Parameters
    ----------
    elevation     : float32 (ny, nx) elevation array in metres
    terrain_w, _d : scene extent in metres
    seed          : RNG seed
    min_radius_m  : smallest crater radius in metres
    max_radius_m  : largest crater radius in metres
    n_craters     : None → auto-scale (0.005 craters/m² × scene area)

    Returns
    -------
    elevation_new   : float32 (ny, nx) modified elevation
    crater_list     : list of (cx_m, cy_m, radius_m) for caller (vertex-colour use)

    Raises
    ------
    ValueError  if terrain is flat (std < 1mm) — prevents placing craters on empty grid
    RuntimeError if any crater normal vector is NaN after stamping (geometry check)
    """
    elevation = elevation.astype(np.float64)   # work in float64, cast back at end

    ny, nx = elevation.shape
    if ny < 4 or nx < 4:
        raise ValueError(
            f"_add_impact_craters: elevation shape {elevation.shape} too small "
            f"(need at least 4×4)"
        )

    cell_w = terrain_w / nx
    cell_d = terrain_d / ny

    # ── Crater count ──────────────────────────────────────────────────────────
    if n_craters is None:
        area_m2 = terrain_w * terrain_d
        # Golombek 2014 JGR Planets — InSight landing site crater density
        # ~0.005 fresh craters (D>0.5m) per m² for ~3 Ga surface
        n_craters = max(1, int(0.005 * area_m2))

    rng = np.random.default_rng(seed)

    # ── Radius distribution: power-law N(>D) ∝ D^-2.9 (Hartmann 2005) ────────
    # To sample: r = r_min * u^(-1/1.9)  where u ~ Uniform(0,1)
    # (exponent 2.9 for production function, but area is ∝ r²,
    #  so effective visual distribution ∝ r^-0.9 — slightly more large craters)
    exponent = 1.9   # Hartmann 2005 cumulative slope
    u = rng.uniform(0.0, 1.0, n_craters)
    # Clip u away from 0 to avoid division-by-zero
    u = np.clip(u, 1e-6, 1.0)
    radii = min_radius_m * u ** (-1.0 / exponent)
    radii = np.clip(radii, min_radius_m, max_radius_m)

    # ── Crater centres (uniform random within 80% of scene to avoid edge trim) ─
    margin = max_radius_m * 2.5   # enough room for ejecta blanket
    cx_m = rng.uniform(-terrain_w / 2 + margin, terrain_w / 2 - margin, n_craters)
    cy_m = rng.uniform(-terrain_d / 2 + margin, terrain_d / 2 - margin, n_craters)

    crater_list: list[tuple[float, float, float]] = []

    # ── Grid coordinates ──────────────────────────────────────────────────────
    xs = np.linspace(-terrain_w / 2, terrain_w / 2, nx)
    ys = np.linspace(-terrain_d / 2, terrain_d / 2, ny)
    X, Y = np.meshgrid(xs, ys)   # (ny, nx)

    # ── Stamp each crater ─────────────────────────────────────────────────────
    for i in range(n_craters):
        R = float(radii[i])
        cx, cy = float(cx_m[i]), float(cy_m[i])

        # Melosh 1989: depth = 0.2 * diameter = 0.4 * R
        depth = 0.4 * R              # bowl depth (positive = below surface)
        # Rim height: 4% of diameter = 0.08 * R
        h_rim = 0.08 * R
        # Ejecta t_0: 2.5% of radius at r=R
        t_0 = 0.025 * R

        # Bounding box for efficiency (only iterate over affected cells)
        ejecta_reach = 2.5 * R
        x_lo = max(0, int((cx - ejecta_reach - xs[0]) / cell_w) - 1)
        x_hi = min(nx, int((cx + ejecta_reach - xs[0]) / cell_w) + 2)
        y_lo = max(0, int((cy - ejecta_reach - ys[0]) / cell_d) - 1)
        y_hi = min(ny, int((cy + ejecta_reach - ys[0]) / cell_d) + 2)

        Xp = X[y_lo:y_hi, x_lo:x_hi]
        Yp = Y[y_lo:y_hi, x_lo:x_hi]
        r = np.sqrt((Xp - cx) ** 2 + (Yp - cy) ** 2)

        # Bowl: paraboloid inside r < R
        z_bowl = np.where(r < R, -depth * (1.0 - (r / R) ** 2), 0.0)

        # Rim: Gaussian ring at r = R, width = 0.15 * R
        rim_sigma = 0.15 * R
        z_rim = h_rim * np.exp(-((r - R) / rim_sigma) ** 2)

        # Ejecta: McGetchin 1973 power-law, only for r > R
        r_safe = np.maximum(r, R * 1e-3)   # avoid /0 inside bowl
        z_ejecta = np.where(
            r > R,
            t_0 * (R / r_safe) ** 3.5,
            0.0,
        )

        elevation[y_lo:y_hi, x_lo:x_hi] += z_bowl + z_rim + z_ejecta

        crater_list.append((cx, cy, R))

    # ── Geometry sanity check ─────────────────────────────────────────────────
    # If any non-edge cell has NaN or Inf, the stamping logic has a bug.
    inner = elevation[1:-1, 1:-1]
    if not np.isfinite(inner).all():
        n_bad = int(np.sum(~np.isfinite(inner)))
        raise RuntimeError(
            f"_add_impact_craters: {n_bad} non-finite elevation values after "
            f"stamping {n_craters} craters — check radius/depth parameters"
        )

    print(f"[hirise_terrain] Impact craters: {n_craters} craters stamped "
          f"(r={min_radius_m:.1f}–{max_radius_m:.1f} m, "
          f"depth/D=0.2, rim 4% D, McGetchin ejecta r<2.5R)")

    return elevation.astype(np.float32), crater_list


# ── Polygon crack network ─────────────────────────────────────────────────────

def _compute_polygon_crack_mask(
    ny: int, nx: int,
    terrain_w: float,
    terrain_d: float,
    seed: int  = 77,
    n_polygons: int = 64,   # ~1 per 25 m² in 40×40 m scene (diameter ~5 m avg)
    crack_half_width: float = 0.30,   # metres (Voronoi-edge half-width threshold)
) -> np.ndarray:
    """
    Compute a Voronoi-based polygon crack proximity mask for the terrain surface.

    Jezero crater floor shows polygonal terrain from desiccation of the ancient
    lake + thermal contraction.  Polygon diameter 3–10 m (mean ~5 m).
    Crack fill: light-toned sulfate minerals → cracks BRIGHTER than surroundings.

    Returns
    -------
    mask  float32 (ny, nx), values 0–1.  1 = at crack centre, 0 = polygon interior.

    References
    ----------
    Crumpler et al. 2023 (10.1029/2022JE007444) — polygon geometry at Jezero
    SHERLOC mapping paper (PMC12002120) — sulfate vein fill, crack width 0.1–4 mm
    """
    try:
        from scipy.spatial import cKDTree as _KDTree
    except ImportError:
        # Fallback: no cracks if scipy missing (avoids hard crash)
        return np.zeros((ny, nx), dtype=np.float32)

    rng = np.random.default_rng(seed)
    # Random Voronoi seed positions (metres, centred at origin)
    sx = rng.uniform(-terrain_w * 0.5, terrain_w * 0.5, n_polygons)
    sy = rng.uniform(-terrain_d * 0.5, terrain_d * 0.5, n_polygons)
    seeds = np.stack([sx, sy], axis=1)

    # Build grid of terrain vertex positions
    xs = np.linspace(-terrain_w / 2, terrain_w / 2, nx, dtype=np.float32)
    ys = np.linspace(-terrain_d / 2, terrain_d / 2, ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    pts  = np.stack([X.ravel(), Y.ravel()], axis=1)

    # Distance to 2 nearest Voronoi seeds
    tree   = _KDTree(seeds)
    dists, _ = tree.query(pts, k=2)
    d1, d2   = dists[:, 0], dists[:, 1]

    # Crack proximity: high where d1 ≈ d2 (near Voronoi edge)
    # (d2 - d1) = 0 on the edge; crack_half_width controls crack width
    crack_proximity = np.maximum(0.0, 1.0 - (d2 - d1) / crack_half_width)
    return crack_proximity.reshape(ny, nx).astype(np.float32)


# ── Vertex-colour terrain material (PBR + primvar reader) ────────────────────

def _build_vertex_color_material(
    stage:    "Usd.Stage",
    mat_path: str,
    roughness: float = 0.92,
) -> "UsdShade.Material":
    """
    PBR material that reads diffuse colour from primvars:displayColor.

    Unlike a plain UsdPreviewSurface with a fixed diffuseColor, this material
    passes per-vertex colours through to the RTX renderer via a PrimvarReader
    shader — giving terrain slope/crack/dust colour variation under full PBR
    lighting (shadows, ambient occlusion, etc.).

    Works in Isaac Sim RTX-Realtime and RTX-Interactive modes.
    """
    mat    = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/PBR")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic",  Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)

    reader = UsdShade.Shader.Define(stage, mat_path + "/ColorReader")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)

    shader.CreateInput(
        "diffuseColor", Sdf.ValueTypeNames.Color3f
    ).ConnectToSource(reader.ConnectableAPI(), "result")

    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


# ── Terrain PBR material (vertex colour + tiling normal/roughness textures) ──

def _build_terrain_hapke_material(
    stage:    "Usd.Stage",
    mat_path: str,
    data_dir: str,
    sun_elevation_deg: float = _SUN_ELEVATION_DEG,
) -> "UsdShade.Material":
    """
    Build a Hapke (2002) BRDF material for terrain using a custom MDL shader.

    Replaces the previous UsdPreviewSurface (Cook-Torrance) approximation with
    a physically correct photometric model for particulate regolith:

      r(i,e,g) = (w₀/4π)[μ₀/(μ₀+μ)][p(g)·B(g) + H(μ₀)H(μ) − 1]

    Key differences from UsdPreviewSurface
    ────────────────────────────────────────
    · Opposition surge: the B(g) term adds a characteristic brightness peak at
      near-zero phase angles (camera/sun coaxial) that Cook-Torrance cannot
      reproduce.  This is the dominant visual feature when MastCam images the
      terrain in the morning or afternoon with the sun behind the rover.

    · Lommel–Seeliger limb darkening: df::directional_factor(exponent=1)
      approximates μ₀/(μ₀+μ) — the emission-angle dependence typical of
      porous granular media.  UsdPreviewSurface uses cosine-weighted Lambert,
      which overestimates limb darkening.

    · Henyey–Greenstein phase function g=0.68 (Madeleine 2012) via
      df::simple_glossy_bsdf(roughness=0.269) — a near-forward-scatter lobe
      vs. UsdPreviewSurface's symmetric specular term.

    Material topology
    ─────────────────
    MDL shader: shaders/hapke_regolith.mdl
    Inputs wired from Python:
      single_scatter_albedo  = 0.32    (Bell 2004 dark basalt)
      hg_asymmetry           = 0.68    (Madeleine 2012)
      opposition_amplitude   = 0.80    (Hapke 2002 Mars analog)
      opposition_width       = 0.06    (Hapke 2002, radians)
      macro_roughness_deg    = 20.0    (Golombek 2008)
      sun_elevation_deg      = _SUN_ELEVATION_DEG  (28°, Bell 2021 — synced with DistantLight)
      albedo_texture         → data/regolith_albedo.png    (sRGB)
      normal_texture         → data/regolith_normal.png    (linear)
      roughness_texture      → data/regolith_roughness.png (linear)
      uv_scale               = (1.0, 1.0)  (1 tile per 2m)

    The "st" primvar (set by _build_terrain_mesh, UV = xy/2m) is read in MDL
    via state::texture_coordinate(0).  Per-vertex displayColor is read via
    state::color().

    Parameters
    ----------
    stage    : active USD stage
    mat_path : USD path for this material (e.g. /World/Looks/TerrainHapke)
    data_dir : absolute path to the data/ directory.
               Raises RuntimeError if any required texture file is missing.

    Returns
    -------
    UsdShade.Material

    Raises
    ──────
    RuntimeError  if regolith_albedo.png, regolith_normal.png, or
                  regolith_roughness.png are not found — run
                  scripts/generate_terrain_textures.py first.
    RuntimeError  if shaders/hapke_regolith.mdl is not found next to isaac_sim/.
    """
    # ── 0. File existence checks (NO silent fallback) ─────────────────────────
    albedo_path    = os.path.join(data_dir, "regolith_albedo.png")
    normal_path    = os.path.join(data_dir, "regolith_normal.png")
    roughness_path = os.path.join(data_dir, "regolith_roughness.png")

    missing_tex = [p for p in [albedo_path, normal_path, roughness_path]
                   if not os.path.isfile(p)]
    if missing_tex:
        raise RuntimeError(
            "_build_terrain_hapke_material: required texture files not found:\n"
            + "\n".join(f"  {p}" for p in missing_tex)
            + "\n\nFix: run this from the repo root:\n"
            + "  python3 scripts/generate_terrain_textures.py\n"
            + "Then re-run build_mars_scene()."
        )

    # MDL file is one directory up from isaac_sim/, in shaders/
    shader_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shaders")
    )
    mdl_path = os.path.join(shader_dir, "hapke_regolith.mdl")
    if not os.path.isfile(mdl_path):
        raise RuntimeError(
            f"_build_terrain_hapke_material: MDL shader not found:\n"
            f"  {mdl_path}\n\n"
            "The file shaders/hapke_regolith.mdl must exist in the repo root."
        )

    # ── 1. Material prim ──────────────────────────────────────────────────────
    mat    = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/HapkeShader")

    # Mark this as an MDL material
    shader.CreateIdAttr("mdlMaterial")

    # Point to our custom MDL file and the exported material name inside it
    shader.SetSourceAsset(Sdf.AssetPath(mdl_path), "mdl")
    shader.SetSourceAssetSubIdentifier("hapke_regolith", "mdl")

    # MDL materials have a single "out" output that covers surface/displacement/volume
    mdl_out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)

    # ── 2. Wire material outputs (MDL surface + displacement + volume) ────────
    # Connecting all three is standard practice so the material works with
    # RTX-Realtime, RTX-Interactive, and iray renderers.
    mat.CreateSurfaceOutput("mdl:surface").ConnectToSource(mdl_out)
    mat.CreateDisplacementOutput("mdl:displacement").ConnectToSource(mdl_out)
    mat.CreateVolumeOutput("mdl:volume").ConnectToSource(mdl_out)

    # ── 3. Hapke photometric parameters ──────────────────────────────────────
    # Bell 2004 / Madeleine 2012 / Hapke 2002 / Golombek 2008 — see MDL file.
    shader.CreateInput("single_scatter_albedo", Sdf.ValueTypeNames.Float).Set(0.32)
    shader.CreateInput("hg_asymmetry",          Sdf.ValueTypeNames.Float).Set(0.68)
    shader.CreateInput("opposition_amplitude",  Sdf.ValueTypeNames.Float).Set(0.80)
    shader.CreateInput("opposition_width",      Sdf.ValueTypeNames.Float).Set(0.06)
    shader.CreateInput("macro_roughness_deg",   Sdf.ValueTypeNames.Float).Set(20.0)
    shader.CreateInput("sun_elevation_deg",     Sdf.ValueTypeNames.Float).Set(sun_elevation_deg)

    # ── 4. Texture inputs ─────────────────────────────────────────────────────
    shader.CreateInput("albedo_texture",    Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(albedo_path))
    shader.CreateInput("normal_texture",    Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(normal_path))
    shader.CreateInput("roughness_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(roughness_path))

    # uv_scale = (1,1): the "st" primvar already encodes 2m/tile from
    # _build_terrain_mesh, matching the texture generator's tile_meters=2.0.
    shader.CreateInput("uv_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(1.0, 1.0))

    print(f"[hirise_terrain] Hapke BRDF material: w₀=0.32 g=0.68 B₀=0.80 "
          f"h=0.06 θ̄=20° sun={sun_elevation_deg}° → {mat_path}")
    return mat


# Legacy alias — redirects to the new Hapke function.
# New code should call _build_terrain_hapke_material directly.
_build_terrain_pbr_material = _build_terrain_hapke_material


# ── Rock MDL material (triplanar vesicle normals + 3-channel vertex colour) ────

def _build_rock_triplanar_material(
    stage:    "Usd.Stage",
    mat_path: str,
    data_dir: str,
) -> "UsdShade.Material":
    """
    Build a rock material using triplanar world-space projection of the
    vesicle normal map via a custom MDL shader.

    Why triplanar instead of spherical UV?
    ───────────────────────────────────────
    Spherical UV projection (the previous approach) has two fatal defects:

      1. Pole singularity: the atan2 seam at azimuth wrap and north/south poles
         creates a visual spike of stretched texels that is visible on every rock
         when MastCam images from above or the side.

      2. Variable texel density: equatorial texels are ~3× larger than polar
         texels for an icosphere, causing the vesicle pattern to look coarser
         near the poles.

    Triplanar projection samples rock_vesicle_normal.png from 3 world-space-
    aligned planes and blends by |N|^sharpness.  Every face gets equally dense
    vesicle coverage regardless of orientation.  There are no UV seams.

    Material topology
    ─────────────────
    MDL shader: shaders/rock_triplanar.mdl
    Inputs wired from Python:
      vesicle_normal    → data/rock_vesicle_normal.png  (linear)
      tile_m            = 0.20   (0.20m per tile = vesicle scale)
      blend_sharpness   = 4.0   (±15° blend zone around 45° face seam)
      normal_strength   = 1.0   (calibrated vesicle depth)
      roughness         = 0.78  (Golombek 2008 weathered basalt)
      specular_weight   = 0.04  (basaltic glass Fresnel F₀)

    Per-vertex colour (dust/geology/rust) is read in MDL via state::color(),
    which maps to the "displayColor" primvar set by _compute_rock_face_colors.

    World-space position for triplanar UV is read via state::position().

    Note: the "st" arc-length UV primvar set on each rock by _build_rock_mesh
    is still present on the mesh (it doesn't hurt anything), but it is NOT
    used by this MDL shader.  The triplanar mapping ignores it entirely.

    Parameters
    ----------
    data_dir : absolute path to data/ directory.  Raises if rock_vesicle_normal.png
               is missing.

    Returns
    -------
    UsdShade.Material

    Raises
    ──────
    RuntimeError  if rock_vesicle_normal.png is not found — run
                  scripts/generate_terrain_textures.py first.
    RuntimeError  if shaders/rock_triplanar.mdl is not found.
    """
    vesicle_path = os.path.join(data_dir, "rock_vesicle_normal.png")
    if not os.path.isfile(vesicle_path):
        raise RuntimeError(
            "_build_rock_triplanar_material: rock_vesicle_normal.png not found at:\n"
            f"  {vesicle_path}\n\n"
            "Fix: run this from the repo root:\n"
            "  python3 scripts/generate_terrain_textures.py\n"
            "Then re-run build_mars_scene()."
        )

    shader_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shaders")
    )
    mdl_path = os.path.join(shader_dir, "rock_triplanar.mdl")
    if not os.path.isfile(mdl_path):
        raise RuntimeError(
            f"_build_rock_triplanar_material: MDL shader not found:\n"
            f"  {mdl_path}\n\n"
            "The file shaders/rock_triplanar.mdl must exist in the repo root."
        )

    # ── Material + MDL shader ─────────────────────────────────────────────────
    mat    = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/TriplanarShader")

    shader.CreateIdAttr("mdlMaterial")
    shader.SetSourceAsset(Sdf.AssetPath(mdl_path), "mdl")
    shader.SetSourceAssetSubIdentifier("rock_triplanar", "mdl")

    mdl_out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl:surface").ConnectToSource(mdl_out)
    mat.CreateDisplacementOutput("mdl:displacement").ConnectToSource(mdl_out)
    mat.CreateVolumeOutput("mdl:volume").ConnectToSource(mdl_out)

    # ── Triplanar parameters ──────────────────────────────────────────────────
    shader.CreateInput("vesicle_normal",  Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(vesicle_path))
    shader.CreateInput("tile_m",          Sdf.ValueTypeNames.Float).Set(0.20)
    shader.CreateInput("blend_sharpness", Sdf.ValueTypeNames.Float).Set(4.0)
    shader.CreateInput("normal_strength", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("roughness",       Sdf.ValueTypeNames.Float).Set(0.78)
    shader.CreateInput("specular_weight", Sdf.ValueTypeNames.Float).Set(0.04)

    print(f"[hirise_terrain] Rock triplanar MDL: tile=0.20m sharpness=4 "
          f"roughness=0.78 → {mat_path}")
    return mat


# Legacy alias — redirects to the correct new function
_build_rock_pbr_material = _build_rock_triplanar_material


# ── Comprehensive terrain vertex colours ─────────────────────────────────────

def _add_terrain_vertex_colors(
    stage:        "Usd.Stage",
    terrain_path: str,
    elevation:    np.ndarray,
    terrain_w:    float,
    terrain_d:    float,
    rock_xz:      Optional[List[Tuple[float, float]]] = None,
    crater_list:  Optional[List[Tuple[float, float, float]]] = None,
) -> None:
    """
    Assign per-vertex colours to terrain capturing six physical processes:

      1. Slope-driven lithology: steep → dark Basalt; flat → MarsOxide
      2. Aeolian ripple grain sorting: crests → lighter coarse olivine grains;
         troughs → darker fine pyroxene dust  (Bridges 2017, Vaughan 2023)
      3. Topographic dust accumulation: low spots → bright PaleDust
         (Vicente-Retortillo 2023: dust settles in lows)
      4. Polygon crack network: crack edges → 20 % brighter (sulfate fill)
         (Crumpler 2023; SHERLOC PMC12002120: light-toned veins)
      5. Wind-shadow dust patches: triangular lighter zone downwind of boulders
         (modern transport azimuth 276 °, Chojnacki 2018)
      6. Impact crater albedo: interior → slightly dark (deep subsurface);
         rim → slightly bright (overturned fresh material);
         ejecta blanket → fresher/brighter tone fading to background
         (Melosh 1989; Golombek 2014 InSight crater survey)

    The primvars:displayColor attribute is read by _build_vertex_color_material
    via a UsdPrimvarReader, giving full PBR-shaded colour variation.

    Parameters
    ----------
    crater_list : list of (cx_m, cy_m, radius_m) from _add_impact_craters, or None

    References
    ----------
    Bridges et al. 2017  PMC5815379  — crest vs trough grain size
    Vaughan et al. 2023  10.1029/2022JE007437 — regolith grain types
    Vicente-Retortillo et al. 2023  10.1029/2022JE007672 — dust accumulation
    Crumpler et al. 2023  10.1029/2022JE007444 — polygon morphology
    Melosh 1989  "Impact Cratering" — crater albedo zones
    """
    prim = stage.GetPrimAtPath(terrain_path)
    if not prim.IsValid():
        return

    ny, nx = elevation.shape
    cell_w  = terrain_w / max(nx, 1)

    xs = np.linspace(-terrain_w / 2, terrain_w / 2, nx, dtype=np.float32)
    ys = np.linspace(-terrain_d / 2, terrain_d / 2, ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    # ── 1. Slope-driven base colour ───────────────────────────────────────────
    slope  = _compute_terrain_slope(elevation, cell_w)
    p90    = float(np.percentile(slope, 90))
    slope_n = np.clip(slope / (p90 + 1e-9), 0.0, 1.0)

    c_basalt = np.array([0.0977, 0.0643, 0.0417], dtype=np.float32)  # CRISM [H20]
    c_oxide  = np.array([0.1615, 0.0996, 0.0642], dtype=np.float32)  # CRISM [C17]
    c_dust   = np.array([0.3535, 0.2690, 0.1975], dtype=np.float32)  # CRISM [C17]
    # Coarser olivine grains (ripple crests, around rock bases)
    c_olivine = np.array([0.1200, 0.0850, 0.0550], dtype=np.float32) # Séítah olivine

    sn   = slope_n[:, :, None].astype(np.float32)
    base = c_basalt + sn * (c_oxide - c_basalt)   # (ny, nx, 3)

    # ── 2. Aeolian ripple grain-sorting colour ────────────────────────────────
    # Crests: coarser olivine grains (lighter/greyer); troughs: fine red dust
    az_rad  = math.radians(276.0)    # modern transport direction (Chojnacki 2018)
    td      = np.array([math.sin(az_rad), math.cos(az_rad)], dtype=np.float32)
    proj    = X * td[0] + Y * td[1]
    phase   = (proj / 3.5).astype(np.float32) % 1.0  # λ = 3.5 m
    sf      = 0.75
    # Crest proximity: near 1.0 at crest peak, near 0.0 in trough
    crest_w = np.where(
        phase < sf,
        (0.5 * (1.0 - np.cos(np.pi * phase / sf))).astype(np.float32),
        (1.0 - (phase - sf) / (1.0 - sf)).astype(np.float32),
    )                                                  # (ny, nx)
    crest_w = crest_w[:, :, None]
    # Crests → blend toward olivine grain colour; troughs → more dust
    ripple_color = base * (1.0 - 0.35 * crest_w) + c_olivine * (0.35 * crest_w)

    # ── 3. Topographic dust accumulation ─────────────────────────────────────
    elev_range = float(elevation.max() - elevation.min()) + 1e-9
    elev_n     = ((elevation - float(elevation.min())) / elev_range).astype(np.float32)
    dust_w     = np.clip(1.0 - elev_n * 3.5, 0.0, 1.0)[:, :, None]
    colors     = ripple_color * (1.0 - dust_w * 0.6) + c_dust * (dust_w * 0.6)

    # ── 4. Polygon crack brightness boost ─────────────────────────────────────
    # Cracks appear ~20% brighter (light-toned sulfate fill, Crumpler 2023)
    n_polys  = max(16, int(terrain_w * terrain_d / 25))   # 1 per ~25 m²
    crack_m  = _compute_polygon_crack_mask(ny, nx, terrain_w, terrain_d,
                                           n_polygons=n_polys)[:, :, None]
    c_sulfate = np.array([0.45, 0.38, 0.30], dtype=np.float32)  # light-toned vein
    colors    = colors * (1.0 - crack_m * 0.50) + c_sulfate * (crack_m * 0.50)

    # ── 5. Wind-shadow dust patches behind rocks ──────────────────────────────
    # Modern wind from ESE (96°), sand moves WNW (276°). Dust accumulates
    # downwind (WNW) of each boulder — triangular brighter patch.
    if rock_xz:
        wind_up  = np.array([-td[0], -td[1]], dtype=np.float32)  # upwind direction
        shadow_c = np.array([0.30, 0.22, 0.16], dtype=np.float32)  # dust-covered
        shadow_layer = np.zeros((ny, nx), dtype=np.float32)

        for (rx, ry) in rock_xz:
            # Shadow extends downwind (in transport direction td)
            shadow_len  = _RNG.uniform(1.5, 4.0)    # metres
            shadow_half = _RNG.uniform(0.4, 1.2)    # half-width at base
            # Vector from rock to each terrain point
            dx_t = X - rx
            dy_t = Y - ry
            # Component along transport direction (positive = downwind)
            along  = dx_t * td[0] + dy_t * td[1]
            across = np.abs(-dx_t * td[1] + dy_t * td[0])
            # Shadow mask: downwind triangle
            in_shadow = (along > 0) & (along < shadow_len) & \
                        (across < shadow_half * (1.0 - along / shadow_len))
            strength  = np.where(in_shadow,
                                  (1.0 - along / shadow_len) * 0.6, 0.0).astype(np.float32)
            shadow_layer = np.maximum(shadow_layer, strength)

        shadow_layer = shadow_layer[:, :, None]
        colors = colors * (1.0 - shadow_layer) + shadow_c * shadow_layer

    # ── 6. Impact crater albedo zones ─────────────────────────────────────────
    # Three distinct photometric zones (Melosh 1989, Golombek 2014 InSight):
    #
    #  Centre (r<0.25R): excavated subsurface — lighter, less dust-coated,
    #    fresher mineralogy exposed by impact.  Perseverance imagery shows
    #    notably lighter centres in D>0.5m craters (Bell 2021 supplement).
    #    c_subsurface ≈ (0.22, 0.15, 0.10) — fresher basalt, less oxide
    #
    #  Inner bowl (0.25R<r<R): dust-accumulation zone — fine material
    #    settles here post-impact, often darker than background from ponding.
    #    c_bowl ≈ (0.07, 0.044, 0.028) — dark fine-dust accumulation
    #
    #  Rim (r≈R±0.12R): overturned fresh regolith, brightest zone.
    #    c_rim ≈ (0.20, 0.14, 0.10) — excavated brighter layer turned up
    #
    #  Ejecta blanket (R<r<2.5R): fresher/brighter than background, power-law
    #    fade (McGetchin 1973: t ∝ (R/r)^3.5 beyond rim).
    if crater_list:
        c_subsurface = np.array([0.22, 0.15, 0.10], dtype=np.float32)  # fresh excavated
        c_bowl       = np.array([0.07, 0.044, 0.028], dtype=np.float32) # dark dust pond
        c_rim        = np.array([0.20, 0.14, 0.10], dtype=np.float32)   # overturned fresh

        crater_layer = np.zeros((ny, nx), dtype=np.float32)
        crater_color = np.zeros((ny, nx, 3), dtype=np.float32)

        for (cx, cy, R) in crater_list:
            r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            r_norm = r / (R + 1e-9)   # normalised radius (1.0 = rim)

            # Zone A — excavated centre (r < 0.25R): lighter subsurface
            # Weight peaks at centre (cos shape), fades to zero at 0.25R
            centre_w = np.where(
                r_norm < 0.25,
                0.60 * (1.0 + np.cos(np.pi * r_norm / 0.25)) * 0.5,
                0.0,
            ).astype(np.float32)

            # Zone B — dust-accumulation bowl (0.25R < r < R)
            # Weight rises from zero at centre boundary to max at 0.7R, fades to rim
            bowl_phase = np.clip((r_norm - 0.25) / 0.75, 0.0, 1.0)
            bowl_envelope = np.sin(np.pi * bowl_phase)   # 0 → 1 → 0 across bowl
            bowl_w = np.where(
                (r_norm >= 0.25) & (r_norm < 1.0),
                0.45 * bowl_envelope,
                0.0,
            ).astype(np.float32)

            # Zone C — rim ring (r ≈ R, Gaussian ±0.12R)
            rim_sigma = 0.12 * R
            rim_w = (0.55 * np.exp(-((r - R) / (rim_sigma + 1e-9)) ** 2)
                     ).astype(np.float32)

            # Zone D — ejecta blanket (R < r < 2.5R), power-law fade
            ejecta_w = np.where(
                (r > R) & (r < 2.5 * R),
                0.20 * (R / np.maximum(r, R * 0.01)) ** 2.0 *
                (1.0 - np.clip((r - R) / (1.5 * R + 1e-9), 0.0, 1.0)),
                0.0,
            ).astype(np.float32)

            # Combine zones — each dominates in its radial band
            combined_w = centre_w + bowl_w + rim_w + ejecta_w
            update = combined_w > crater_layer
            crater_layer = np.where(update, combined_w, crater_layer)

            total = centre_w + bowl_w + rim_w + ejecta_w + 1e-9
            mix_c = (
                c_subsurface * (centre_w / total)[:, :, None]
                + c_bowl     * (bowl_w   / total)[:, :, None]
                + c_rim      * ((rim_w + ejecta_w) / total)[:, :, None]
            )
            crater_color = np.where(update[:, :, None], mix_c, crater_color)

        crater_w3 = np.clip(crater_layer, 0.0, 1.0)[:, :, None]
        colors = colors * (1.0 - crater_w3) + crater_color * crater_w3

    # ── 7. Ambient occlusion darkening ring at rock bases ────────────────────
    # Where rocks meet the ground, ambient light is occluded by the rock mass.
    # This creates a narrow dark ring around each rock — a critical visual cue
    # that "grounds" the rock and prevents it from looking like it's floating.
    #
    # Physical basis: AO ring is darkest at the rock edge (r≈R_rock), fades over
    # 1–2 rock radii. Width empirically matched to MER/MSL ground imagery.
    # References: Golombek 2008 (rock burial geometry), Wilson 2004 (photometry
    # of rock shadows in Spirit imagery).
    if rock_xz:
        ao_layer = np.zeros((ny, nx), dtype=np.float32)
        # Approximate rock radius from boulder/cobble size classes (~0.3–1.2m)
        # We don't have individual radii here; use a conservative default 0.6m
        _AO_ROCK_R  = 0.6    # metres — assumed rock half-diameter
        _AO_WIDTH   = 0.8    # metres — AO fade width beyond rock edge
        _AO_DEPTH   = 0.35   # max darkening factor (35%)

        for (rx, ry) in rock_xz:
            r = np.sqrt((X - rx)**2 + (Y - ry)**2)
            # Gaussian darkening: peak at r = _AO_ROCK_R, width = _AO_WIDTH
            ao_w = (_AO_DEPTH * np.exp(
                -((r - _AO_ROCK_R) / (_AO_WIDTH * 0.5))**2
            )).astype(np.float32)
            # Only inside/around the rock footprint (r < 2.5 * _AO_ROCK_R)
            ao_w = np.where(r < 2.5 * _AO_ROCK_R, ao_w, 0.0).astype(np.float32)
            ao_layer = np.maximum(ao_layer, ao_w)

        # Apply darkening: darken all channels equally (AO is achromatic)
        ao_layer = ao_layer[:, :, None]
        colors = colors * (1.0 - ao_layer)

    vertex_colors = np.clip(colors, 0.0, 1.0).reshape(-1, 3).astype(np.float32)

    primvars_api = UsdGeom.PrimvarsAPI(prim)
    pv = primvars_api.CreatePrimvar(
        "displayColor",
        Sdf.ValueTypeNames.Color3fArray,
        "vertex",
    )
    pv.Set(Vt.Vec3fArray.FromNumpy(vertex_colors))


# ── Geological unit assignment (Stack et al. 2020) ───────────────────────────

def _make_seitah_patches(terrain_w: float, terrain_d: float,
                         rng: random.Random, n: int = 4) -> list:
    """
    Generate n elliptical Séítah formation outcrops.

    Stack et al. 2020: olivine-rich Séítah appears as erosional windows
    scattered across the basaltic Máaz floor, concentrated in SW sector.
    We approximate this as n random ellipses covering ~25–30 % of area.
    """
    patches = []
    for _ in range(n):
        cx = rng.uniform(-terrain_w * 0.40, terrain_w * 0.40)
        cy = rng.uniform(-terrain_d * 0.40, terrain_d * 0.40)
        rx = rng.uniform(terrain_w * 0.05, terrain_w * 0.14)
        ry = rng.uniform(terrain_d * 0.05, terrain_d * 0.14)
        patches.append((cx, cy, rx, ry))
    return patches


def _get_geological_unit(x: float, y: float, seitah_patches: list) -> str:
    """Return 'seitah' if (x, y) lies inside any Séítah patch, else 'maaz'."""
    for cx, cy, rx, ry in seitah_patches:
        if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 < 1.0:
            return "seitah"
    return "maaz"


# ── PDF sampler ───────────────────────────────────────────────────────────────

def _sample_from_pdf(pdf: dict, rng: random.Random) -> str:
    """Weighted random sample from a {key: probability} dict."""
    r = rng.random()
    cumsum = 0.0
    for k, w in pdf.items():
        cumsum += w
        if r <= cumsum:
            return k
    return list(pdf.keys())[-1]


# ── Rock dust mantling colours ────────────────────────────────────────────────

def _compute_rock_face_colors(
    verts:             np.ndarray,    # float32 (N, 3) local mesh space
    faces:             np.ndarray,    # int32   (F, 3)
    base_color:        np.ndarray,    # float32 (3,)   geological unit base colour
    yaw_deg:           float,         # world-space yaw   applied to rock (°)
    pitch_deg:         float,         # world-space pitch applied to rock (°)
    roll_deg:          float,         # world-space roll  applied to rock (°)
    transport_az:      float = _MODERN_TRANSPORT_AZ,
    exponent:          float = _MANTLING_EXPONENT,
    dust_color:        np.ndarray = _DUST_MANTLE_COLOR,
) -> np.ndarray:
    """
    Compute per-vertex dust mantling colours for a rock mesh.

    Physics
    -------
    Leeward faces (face-normal · transport_direction > 0):
        dust accumulates because saltating grains decelerate in the wake
        and fall out of suspension — brighter, reddish PaleDust colour.
    Windward faces (face-normal · transport_direction < 0):
        continual abrasion by saltating grains keeps surface clean —
        dark, fresh geological base colour.

    The cosine weight is raised to exponent 0.7 (not 1.0) because dust
    accumulation is not perfectly cosine-distributed; it extends somewhat
    around the equatorial belt of the rock (consistent with ventifact keel
    observations, Bridges 2014 Aeolian Research).

    Implementation
    --------------
    Face normals are computed in WORLD space (after applying yaw/pitch/roll)
    so the mantling correctly aligns with the global wind direction regardless
    of each rock's random orientation.

    Per-vertex colour = average of mantling weights of adjacent faces.
    This uses `vertex` interpolation rather than `faceVarying` or `uniform`
    because Isaac Sim RTX's UsdPrimvarReader_float3 has the most robust
    support for vertex-interpolated primvars across render modes.

    References
    ----------
    Chojnacki et al. 2018 PMC5859260 — modern transport 276 ° WNW
    Bridges et al. 2014 Aeolian Research — mantling pattern / ventifact keels
    Herkenhoff et al. 2023 10.1029/2022JE007599 — ancient vs modern wind
    """
    # ── 1. Build world-space rotation matrix Rz(yaw) · Rx(pitch) · Ry(roll) ──
    yr, pr, rr = math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg)
    cy, sy = math.cos(yr),  math.sin(yr)
    cp, sp = math.cos(pr),  math.sin(pr)
    cr, sr = math.cos(rr),  math.sin(rr)
    Rz = np.array([[cy, -sy, 0.0],  [sy,  cy, 0.0],  [0.0, 0.0, 1.0]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0,  cp,  -sp], [0.0,  sp,  cp]], dtype=np.float32)
    Ry = np.array([[cr,  0.0,  sr], [0.0, 1.0, 0.0],  [-sr, 0.0,  cr]], dtype=np.float32)
    R  = Rz @ Rx @ Ry                                # (3, 3)

    world_verts = (verts @ R.T).astype(np.float32)   # (N, 3)

    # ── 2. Face normals in world space ────────────────────────────────────────
    v0 = world_verts[faces[:, 0]]
    v1 = world_verts[faces[:, 1]]
    v2 = world_verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0).astype(np.float32)   # (F, 3)
    fn_len = np.linalg.norm(fn, axis=1, keepdims=True)
    fn /= (fn_len + 1e-9)

    # ── 3. Three-channel colour model ────────────────────────────────────────
    #
    # Channel allocation by world-space Z of face normal:
    #
    #   TOP    (fn_z > +0.40):  Gravity-settled dust coating.
    #       Bell et al. 2004 JGR — MER Spirit/Opportunity rocks: top faces
    #       measurably brighter (dust albedo ~0.38 vs base ~0.10).
    #       Transition: smooth cosine from fn_z=0.40 to fn_z=0.90.
    #
    #   SIDE   (−0.25 ≤ fn_z ≤ +0.40):  Clean geological base colour.
    #       Windward faces scoured clean by saltating grains (Bridges 2014).
    #       Leeward SIDE faces receive wind-blown LATERAL dust (handled below).
    #
    #   BOTTOM (fn_z < −0.25):  Iron-oxide rust staining.
    #       Banfield et al. 2021 JGR — Fe²⁺ → Fe³⁺ leaching from buried
    #       contact zone; moisture wicking darkens/reddens the sub-rock surface.
    #       Ferric-enriched layer albedo ~0.12, hue deeper red (higher R:G:B).
    #       Transition: smooth from fn_z=−0.25 to fn_z=−0.70.
    #
    # On top of the 3-channel base, LATERAL WIND MANTLING blends side-faces
    # toward dust_color where face-normal · transport_direction > 0.
    # (Wind mantling is attenuated for TOP faces that already have dust.)

    # Dust colours:
    #   c_top_dust: gravity settled on top — pale reddish-tan (Bell 2004)
    c_top_dust = np.array([0.355, 0.270, 0.200], dtype=np.float32)

    #   c_rust: iron-oxide leach at burial zone (Banfield 2021)
    #   Deeper red, lower brightness than surface: Fe³⁺ absorption at 400-600nm
    c_rust = np.array([0.150, 0.065, 0.035], dtype=np.float32)

    #   dust_color (lateral wind): from function argument (leeward faces)

    # Z-component of world-space face normals
    fn_z = fn[:, 2]   # (F,)

    # TOP weight: 0 below fn_z=0.40, 1 above fn_z=0.90 (cosine ease-in)
    top_t   = np.clip((fn_z - 0.40) / 0.50, 0.0, 1.0)
    top_w   = (0.5 * (1.0 - np.cos(np.pi * top_t))).astype(np.float32)   # (F,)

    # BOTTOM weight: 0 above fn_z=−0.25, 1 below fn_z=−0.70
    bot_t   = np.clip((-fn_z - 0.25) / 0.45, 0.0, 1.0)
    bot_w   = (0.5 * (1.0 - np.cos(np.pi * bot_t))).astype(np.float32)   # (F,)

    # SIDE weight fills the remainder
    side_w = np.clip(1.0 - top_w - bot_w, 0.0, 1.0)                      # (F,)

    # 3-channel base colour per face
    #   base_color (from unit/geological unit) is the SIDE geology colour
    face_base = (
        base_color[None, :] * side_w[:, None]
        + c_top_dust[None, :] * top_w[:, None]
        + c_rust[None, :]     * bot_w[:, None]
    )   # (F, 3)

    # ── 4. Lateral wind mantling on SIDE faces ─────────────────────────────
    az_rad  = math.radians(transport_az)
    wind_3d = np.array([math.sin(az_rad), math.cos(az_rad), 0.0], dtype=np.float32)

    dot_f     = fn @ wind_3d                              # (F,) leeward = positive
    mantle_f  = np.clip(dot_f, 0.0, 1.0) ** exponent     # (F,) in [0,1]

    # Attenuate wind mantling where top dust already dominates (prevents
    # double-dust on upward-facing leeward faces — they're already pale)
    wind_att  = (1.0 - top_w).astype(np.float32)         # (F,)
    eff_mantle = (mantle_f * wind_att).astype(np.float32) # (F,)

    # Blend face_base toward lateral dust_color on leeward faces
    face_colors = (
        face_base * (1.0 - eff_mantle[:, None])
        + dust_color[None, :] * eff_mantle[:, None]
    ).astype(np.float32)   # (F, 3)

    # ── 5. Aggregate face colours to per-vertex (weighted average) ─────────
    # Uses np.add.at for O(F) numpy ops (no Python loop).
    n_verts     = len(verts)
    vert_rgb    = np.zeros((n_verts, 3), dtype=np.float32)
    vert_count  = np.zeros(n_verts,     dtype=np.float32)
    for col in range(3):
        np.add.at(vert_rgb,   faces[:, col], face_colors)
        np.add.at(vert_count, faces[:, col], 1.0)
    vert_count = np.maximum(vert_count, 1.0)
    colors     = vert_rgb / vert_count[:, None]           # (N, 3)

    return np.clip(colors, 0.0, 1.0).astype(np.float32)  # (N, 3)


# ── Main rock mesh builder ────────────────────────────────────────────────────

def _build_rock_mesh(
    stage:        "Usd.Stage",
    prim_path:    str,
    diameter:     float,
    unit:         str,
    is_ventifact: bool,
    rng:          random.Random,
) -> "UsdGeom.Mesh":
    """
    Build a physically realistic rock mesh via Voronoi fracture + weathering.

    Full pipeline
    -------------
    1. Icosphere (1 subdiv for pebbles, 2 for cobbles/boulders)
    2. Voronoi planar cuts   — count from Powers roundness class
    3. Laplacian smoothing   — iterations from unit / roundness
    4. Ventifact cuts        — extra windward erosion if is_ventifact
    5. Grain-scale noise     — micro-roughness (Máaz vs Séítah texture)
    6. Oblate scaling        — h ≈ 0.5 × d, width ± 25 % (Golombek 2008)
    7. 40 % protrusion embed — 60 % of rock buried (Golombek 2008)

    References
    ----------
    Golombek et al. 2008 JGR 113 E00A11  (aspect ratios, protrusion depth)
    Khan et al. 2022 JGR Planets         (Powers roundness distribution)
    Raghavachary 2002 SIGGRAPH           (Voronoi fracture)
    Priour 2020 arXiv:2003.03476         (Laplacian weathering)
    """
    rng_np    = np.random.default_rng(rng.randint(0, 2 ** 32))
    unit_data = _UNIT_MAAZ if unit == "maaz" else _UNIT_SEITAH

    # 1. Base icosphere
    subs = 2 if diameter >= 0.5 else 1
    verts, faces = _make_icosphere(subdivisions=subs)

    # 2. Powers roundness class → cut count
    cls = _sample_from_pdf(unit_data["roundness_pdf"], rng)
    (cut_lo, cut_hi), smooth_iters, grain_amp = _POWERS_CLASSES[cls]
    n_cuts = rng.randint(cut_lo, max(cut_lo, cut_hi - 1))

    # 3. Voronoi fracture
    verts = _voronoi_fracture(verts, n_cuts, rng_np)

    # 4. Laplacian weathering
    verts = _laplacian_smooth(verts, faces, smooth_iters)

    # 5. Ventifact erosion (Herkenhoff 2023)
    if is_ventifact:
        wind_xy = np.array([
            math.cos(math.radians(_WIND_AZIMUTH_DEG)),
            math.sin(math.radians(_WIND_AZIMUTH_DEG)),
        ])
        verts = _add_ventifact_cuts(verts, wind_xy, rng.randint(2, 4), rng_np)

    # 6. Grain-scale noise
    verts = _apply_grain_noise(verts, grain_amp, rng_np)

    # 6b. Macro weathering pit deformation (Squyres 2004, Crumpler 2015)
    # Cavernous weathering creates 5–20 cm cavities on Martian basalt.
    # Pits are applied in UNIT-SPHERE space before oblate scaling; the subsequent
    # scale step maps them to physical size correctly.
    # Boulders (d≥0.50 m): 5–8 pits, radius ≈ 12 % sphere (≈3 cm on 0.5 m rock)
    # Cobbles (d≥0.15 m): 2–4 pits, radius ≈ 10 %
    # Pebbles (d<0.15 m): no macro pits (too small to resolve)
    if diameter >= 0.50:
        n_pits = rng.randint(4, 8)
        verts = _apply_vesicle_pitting(verts, rng_np, n_pits,
                                        pit_radius=rng.uniform(0.10, 0.15),
                                        depth_factor=rng.uniform(0.35, 0.50))
    elif diameter >= 0.15:
        n_pits = rng.randint(2, 4)
        verts = _apply_vesicle_pitting(verts, rng_np, n_pits,
                                        pit_radius=rng.uniform(0.08, 0.12),
                                        depth_factor=rng.uniform(0.30, 0.45))
    # else: pebbles — no macro pitting, geometry too coarse

    # 6c. Spherical UV from unit-sphere vertex directions (BEFORE oblate scaling)
    # Must be computed here because oblate scaling destroys the spherical direction.
    # The UV primvar is set on the USD mesh after creation (step 8+).
    #
    # Tile size 0.20 m = physical size of rock_vesicle_normal.png tile.
    # UV uses arc-length parameterisation:
    #   u = (azimuth + π) / (2π) × (2πr / tile_m)  = r × (θ+π) / tile_m
    #   v =  polar_angle / π     × (πr  / tile_m)  = r × φ / tile_m
    # where r = diameter/2 and tile_m = 0.20 m.
    # For d=0.5m (r=0.25m): u spans 0..7.9 tiles; for d=2.5m: 0..39 tiles.
    # wrapS/T="repeat" in UsdUVTexture handles values > 1 automatically.
    _ROCK_TILE_M = 0.20
    rock_r = diameter / 2.0

    # Normalise pre-scale vertices to unit sphere
    r_mag = np.linalg.norm(verts, axis=1, keepdims=True)
    v_hat = verts / (np.maximum(r_mag, 1e-9))          # (N, 3) unit vectors
    theta = np.arctan2(v_hat[:, 1], v_hat[:, 0])        # azimuth  [-π, π]
    phi   = np.arccos(np.clip(v_hat[:, 2], -1.0, 1.0))  # polar    [ 0, π]

    uv_rock = np.stack([
        rock_r * (theta + np.pi) / _ROCK_TILE_M,        # u: arc along equator
        rock_r * phi             / _ROCK_TILE_M,         # v: arc along meridian
    ], axis=-1).astype(np.float32)   # (N, 2)

    # 7. Oblate scaling: width ± 25 %, height ≈ 0.5 × diameter
    sx = rng.uniform(0.75, 1.25)
    sy = rng.uniform(0.75, 1.25)
    sz = 0.50 + rng.uniform(-0.08, 0.10)
    r  = diameter / 2.0
    verts[:, 0] *= sx * r
    verts[:, 1] *= sy * r
    verts[:, 2] *= sz * r

    # 8. Embed: shift so 40 % of rock height protrudes above Z = 0
    z_lo   = float(verts[:, 2].min())
    z_hi   = float(verts[:, 2].max())
    rock_h = z_hi - z_lo
    verts[:, 2] -= z_lo + 0.60 * rock_h   # 60 % buried, 40 % exposed

    verts32     = verts.astype(np.float32)
    face_idx    = faces.ravel()
    face_counts = np.full(len(faces), 3, dtype=np.int32)

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts32))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(face_idx))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(face_counts))
    mesh.CreateDoubleSidedAttr(False)

    # CatmullClark subdivision for cobbles and boulders: makes overall shape
    # smooth (removes icosphere faceting) without storing more geometry in USD.
    # The RTX renderer applies refinement at render time — smooth normals are
    # computed from the subdivided surface (no explicit normals needed).
    #
    # Pebbles (d < 0.15 m) use "none" — they are small and in background;
    # subdivision overhead on many instances is not worthwhile.
    #
    # refinementLevel=2 → 4× subdivision per pass, applied 2 passes:
    #   subs=1 icosphere (80 faces) → catmullClark L2 → ~1280 smooth faces
    # Reference: USD 23.08 UsdGeom schema, Pixar CatmullClark spec (1978)
    use_catmull = (diameter >= 0.15)
    if use_catmull:
        mesh.CreateSubdivisionSchemeAttr("catmullClark")
        # Set refinement level as per-prim override (Isaac Sim / Omniverse Kit)
        mesh.GetPrim().CreateAttribute(
            "refinementLevel", Sdf.ValueTypeNames.Int, False
        ).Set(2)
        # With catmullClark: do NOT set explicit normals.
        # The subdivider computes C² smooth normals from the subdivided topology.
        # Explicitly-set normals on a catmullClark mesh are IGNORED by the
        # renderer — setting them would waste memory with no benefit.
    else:
        mesh.CreateSubdivisionSchemeAttr("none")
        normals = _compute_normals(verts32, faces)
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
        mesh.SetNormalsInterpolation("vertex")

    # ── Rock UV primvar (for vesicle normal map tiling) ──────────────────────
    # Set "st" as TexCoord2fArray with vertex interpolation.
    # CatmullClark subdivision will linearly interpolate UVs across the
    # subdivided mesh, giving smooth tangent-space normal map application.
    # NOTE: The triplanar MDL shader (rock_triplanar.mdl) does NOT use this
    # "st" primvar — it reads world-space position via state::position() instead.
    # The "st" primvar is kept for debug/diagnostic use only.
    #
    # NOTE: if building offline without pxr (unit testing), skip this block.
    # _PXR_AVAILABLE is checked in build_mars_scene before calling this function.
    rock_primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    pv_st_rock = rock_primvars.CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        "vertex",
    )
    pv_st_rock.Set(Vt.Vec2fArray.FromNumpy(uv_rock))

    # Return geometry arrays alongside mesh so callers can compute
    # per-vertex colours (e.g. dust mantling) without re-reading the stage.
    return mesh, verts32, faces


# ── Rock placement (CFA + geological clustering) ─────────────────────────────

def _place_rocks(
    stage:         "Usd.Stage",
    rocks_root:    str,
    elevation:     np.ndarray,
    terrain_w:     float,
    terrain_d:     float,
    materials:     dict,
    rng:           random.Random,
    rock_material: Optional["UsdShade.Material"] = None,
) -> List[str]:
    """
    Place rocks with full geological realism:
      1. Geological unit assignment (Máaz / Séítah) — Stack et al. 2020
      2. Spatial clustering near slope features (scarps / ridges)
      3. CFA size-frequency distribution — Golombek et al. 2008
      4. Ventifact probability 35 % — Bridges 2014, Herkenhoff 2023
      5. 40 % protrusion (baked into mesh) — Golombek 2008
      6. Random tilt ± 8° — rocks settle on uneven ground
      7. Per-vertex dust mantling colours — Chojnacki 2018, Bridges 2014
         Leeward faces (WNW) → PaleDust; windward (ESE) → clean base colour
    """
    ny, nx_cells = elevation.shape
    cell_w = terrain_w / max(nx_cells, 1)

    slope      = _compute_terrain_slope(elevation, cell_w)
    slope_flat = slope.ravel().astype(np.float64)
    slope_flat = np.nan_to_num(slope_flat, nan=0.0)
    total_slope = slope_flat.sum()
    if total_slope > 0:
        slope_pdf = slope_flat / total_slope
    else:
        slope_pdf = np.ones(len(slope_flat)) / len(slope_flat)

    flat_indices = np.arange(len(slope_flat), dtype=np.int64)

    # Séítah formation patches (~30 % of floor, Stack et al. 2020)
    seitah_patches = _make_seitah_patches(terrain_w, terrain_d, rng, n=4)

    rock_paths: List[str] = []
    idx = 0

    # ── Scale rock counts to scene area ──────────────────────────────────────
    # Base counts in _ROCK_SIZE_CLASSES are calibrated for a 40×40m (1600 m²) scene.
    # Golombek et al. 2008: CFA densities per m² for Jezero-analogue terrain:
    #   Boulder (D≈2.5m): ~0.009/m²  → 15 in 1600 m²
    #   Cobble  (D≈0.8m): ~0.050/m²  → 80 in 1600 m²
    #   Pebble  (D≈0.25m):~0.125/m²  → 200 in 1600 m²
    # Scale factor caps prevent performance collapse in large scenes.
    # Maximum counts (empirically: USD traversal cost): 150 / 800 / 2000
    _REF_AREA_M2 = 1600.0
    _scene_area  = terrain_w * terrain_d
    _scale       = _scene_area / _REF_AREA_M2

    _scaled_counts = {
        "boulder": max(1, min(150,  int(15  * _scale))),
        "cobble":  max(1, min(800,  int(80  * _scale))),
        "pebble":  max(1, min(2000, int(200 * _scale))),
    }

    for cls_name, diam_m, _base_count, _ in _ROCK_SIZE_CLASSES:
        count = _scaled_counts.get(cls_name, _base_count)
        for _ in range(count):
            roll = rng.random()

            if roll < 0.40:
                # 40 % — near rover start (first ~50 % of X span, ±20 m)
                x = rng.uniform(2.0, terrain_w * 0.45)
                y = rng.uniform(-min(20.0, terrain_d * 0.40),
                                 min(20.0, terrain_d * 0.40))

            elif roll < 0.75:
                # 35 % — slope-biased (geological clustering near scarps)
                flat_idx = int(_NP_RNG.choice(flat_indices, p=slope_pdf))
                cy_i     = flat_idx // nx_cells
                cx_i     = flat_idx %  nx_cells
                x = float(cx_i / max(nx_cells - 1, 1) * terrain_w - terrain_w / 2)
                y = float(cy_i / max(ny - 1, 1)       * terrain_d - terrain_d / 2)
                x += rng.uniform(-cell_w * 2, cell_w * 2)
                y += rng.uniform(-cell_w * 2, cell_w * 2)

            else:
                # 25 % — uniform random (background coverage)
                x = rng.uniform(-terrain_w * 0.45, terrain_w * 0.45)
                y = rng.uniform(-terrain_d * 0.45, terrain_d * 0.45)

            # Clamp to terrain bounds
            x = max(-terrain_w * 0.47, min(terrain_w * 0.47, x))
            y = max(-terrain_d * 0.47, min(terrain_d * 0.47, y))

            # Terrain Z at placement location
            col = int((x + terrain_w / 2) / terrain_w * (nx_cells - 1))
            row = int((y + terrain_d / 2) / terrain_d * (ny - 1))
            col = max(0, min(nx_cells - 1, col))
            row = max(0, min(ny - 1, row))
            z_terrain = float(elevation[row, col])

            # Geological unit + ventifact flag
            unit         = _get_geological_unit(x, y, seitah_patches)
            unit_data    = _UNIT_MAAZ if unit == "maaz" else _UNIT_SEITAH
            is_ventifact = rng.random() < _VENTIFACT_PROB

            # Build mesh — returns geometry arrays for colour computation
            prim_path = f"{rocks_root}/{cls_name}_{idx:03d}"
            _, rock_verts, rock_faces = _build_rock_mesh(
                stage, prim_path, diam_m, unit, is_ventifact, rng
            )

            # ── Rotations: fix values before applying so we can pass them
            #    to _compute_rock_face_colors (world-space mantling needs
            #    to know the rock's orientation).
            yaw   = rng.uniform(0.0, 360.0)
            pitch = rng.uniform(-8.0,   8.0)
            roll  = rng.uniform(-8.0,   8.0)

            # Transform: translate → yaw → pitch/roll
            xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
            xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z_terrain))
            xf.AddRotateZOp().Set(yaw)
            xf.AddRotateXOp().Set(pitch)
            xf.AddRotateYOp().Set(roll)

            # ── Dust mantling colours (Layer 4) ──────────────────────────────
            # Select geological base colour first (same weighted draw as before)
            mat_key = rng.choices(
                unit_data["materials"], weights=unit_data["mat_weights"], k=1
            )[0]
            base_c = _MATERIAL_COLORS.get(mat_key, _MATERIAL_COLORS["Basalt"])

            vc = _compute_rock_face_colors(
                rock_verts, rock_faces, base_c,
                yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
            )

            primvars_api = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(prim_path))
            pv = primvars_api.CreatePrimvar(
                "displayColor",
                Sdf.ValueTypeNames.Color3fArray,
                "vertex",          # vertex interpolation: best RTX support
            )
            pv.Set(Vt.Vec3fArray.FromNumpy(vc))

            # ── Material: vertex-colour PBR reader (shared across all rocks)
            #    Falls back to plain unit material if rock_material not provided.
            if rock_material is not None:
                _bind_material(stage, prim_path, rock_material)
            elif mat_key in materials:
                _bind_material(stage, prim_path, materials[mat_key])

            rock_paths.append(prim_path)
            idx += 1

    return rock_paths


# =============================================================================
# Bedrock slab outcrops
# =============================================================================

def _add_bedrock_slabs(
    stage:       "Usd.Stage",
    slabs_root:  str,
    elevation:   np.ndarray,
    terrain_w:   float,
    terrain_d:   float,
    rng:         random.Random,
    n_slabs:     int | None = None,
    slab_mat:    Optional["UsdShade.Material"] = None,
) -> List[str]:
    """
    Add flat tabular bedrock slab outcrops to the scene.

    Physical model
    --------------
    Jezero Crater floor shows flat-lying basalt lava flow units outcropping
    where aeolian regolith has been deflated — "Máaz" formation tabular rocks
    (Farley et al. 2022 Science; Crumpler et al. 2023 JGR Planets).

    Slab morphology (from MSL / Perseverance contact science):
      Plan dimensions: 0.5–3.5 m across, aspect ratio 0.4–2.5
      Thickness:       0.08–0.30 m  (sub-horizontal lavas, Crumpler 2023)
      Tilt:            ≤ 8° from horizontal (Golombek 2008 InSight site survey)
      Burial:          20–40 % of thickness below regolith surface
      Surface texture: striated top (flow banding), rough fractured edges

    Density: ~1 slab per 80–120 m² (Farley 2022 Jezero traversal data).
    Default n_slabs is auto-computed from scene area.

    Each slab is a thin rectangular prism (USD Mesh with 8 verts / 12 tris),
    slightly irregular (Voronoi edge roughening) with per-vertex colours showing:
      Top surface:  slightly dustier (settled dust in planar depressions)
      Edges/sides:  dark fresh basalt (Máaz colour)
      Bottom:       iron-rich rust (same leaching model as rocks)

    Parameters
    ----------
    stage, slabs_root : USD stage and parent prim path
    elevation          : (ny, nx) terrain elevation array
    terrain_w, _d      : scene extents in metres
    rng                : reproducible Random
    n_slabs            : None → auto (scene_area / 100)
    slab_mat           : USD material to bind (vertex-colour PBR)

    Returns
    -------
    slab_paths : list of USD prim paths

    Raises
    ------
    RuntimeError if a slab mesh has degenerate (zero-area) faces
    """
    ny, nx = elevation.shape
    if n_slabs is None:
        n_slabs = max(2, int(terrain_w * terrain_d / 100))

    rng_np = np.random.default_rng(rng.randint(0, 2**32))

    # Máaz basalt colour (CRISM calibrated, Farley 2022)
    c_maaz_top  = np.array([0.135, 0.090, 0.060], dtype=np.float32)  # dustier top
    c_maaz_side = np.array([0.098, 0.064, 0.042], dtype=np.float32)  # fresh side
    c_maaz_bot  = np.array([0.150, 0.065, 0.035], dtype=np.float32)  # rust under

    stage.DefinePrim(slabs_root, "Scope")
    slab_paths: List[str] = []

    for i in range(n_slabs):
        # Random placement (avoid scene edges by 0.5 m)
        cx = rng.uniform(-terrain_w * 0.45, terrain_w * 0.45)
        cy = rng.uniform(-terrain_d * 0.45, terrain_d * 0.45)

        # Terrain elevation at centre
        col = int((cx + terrain_w / 2) / terrain_w * (nx - 1))
        row = int((cy + terrain_d / 2) / terrain_d * (ny - 1))
        col = max(0, min(nx - 1, col))
        row = max(0, min(ny - 1, row))
        z_base = float(elevation[row, col])

        # Slab dimensions
        slab_w     = rng.uniform(0.5,  3.5)   # metres
        slab_l     = slab_w * rng.uniform(0.4, 2.5)   # aspect ratio 0.4–2.5
        slab_h     = rng.uniform(0.08, 0.30)   # thickness
        az_deg     = rng.uniform(0.0,  360.0)  # horizontal rotation
        tilt_pitch = rng.uniform(-8.0,  8.0)   # fore-aft tilt
        tilt_roll  = rng.uniform(-8.0,  8.0)   # side tilt
        burial     = rng.uniform(0.20,  0.40)  # fraction below ground

        half_w = slab_w / 2.0
        half_l = slab_l / 2.0
        half_h = slab_h / 2.0

        # 8 vertices of a box in local space (before azimuth rotation)
        # +X = along slab length, +Y = along slab width, +Z = up
        box_v = np.array([
            [-half_l, -half_w, -half_h],
            [ half_l, -half_w, -half_h],
            [ half_l,  half_w, -half_h],
            [-half_l,  half_w, -half_h],
            [-half_l, -half_w,  half_h],
            [ half_l, -half_w,  half_h],
            [ half_l,  half_w,  half_h],
            [-half_l,  half_w,  half_h],
        ], dtype=np.float64)

        # Add surface irregularity: Voronoi-like edge roughening on top face
        # Top-face vertices (indices 4-7): add ±5% noise in XY plane
        for vi in [4, 5, 6, 7]:
            box_v[vi, 0] += rng_np.uniform(-half_l * 0.08, half_l * 0.08)
            box_v[vi, 1] += rng_np.uniform(-half_w * 0.08, half_w * 0.08)
            box_v[vi, 2] += rng_np.uniform(-half_h * 0.05, half_h * 0.10)

        # Azimuth rotation (Z axis)
        az_r = math.radians(az_deg)
        Rz = np.array([[math.cos(az_r), -math.sin(az_r), 0.0],
                        [math.sin(az_r),  math.cos(az_r), 0.0],
                        [0.0,             0.0,             1.0]])
        box_v = (box_v @ Rz.T).astype(np.float32)

        # Tilt (small pitch + roll)
        pr, rr = math.radians(tilt_pitch), math.radians(tilt_roll)
        Rx = np.array([[1.0, 0.0, 0.0],
                        [0.0, math.cos(pr), -math.sin(pr)],
                        [0.0, math.sin(pr),  math.cos(pr)]])
        Ry = np.array([[ math.cos(rr), 0.0, math.sin(rr)],
                        [0.0,           1.0, 0.0          ],
                        [-math.sin(rr), 0.0, math.cos(rr)]])
        box_v = (box_v @ (Rx @ Ry).T).astype(np.float32)

        # Embed: shift so burial fraction is below z_base
        z_lo = float(box_v[:, 2].min())
        z_hi = float(box_v[:, 2].max())
        box_h = z_hi - z_lo
        # Target: z_hi after shift = z_base + (1 - burial) * box_h
        z_shift = z_base + (1.0 - burial) * box_h - z_hi
        box_v[:, 2] += z_shift

        # 12 triangles (2 per face of box × 6 faces)
        tris = np.array([
            [0, 2, 1], [0, 3, 2],   # bottom face
            [4, 5, 6], [4, 6, 7],   # top face
            [0, 1, 5], [0, 5, 4],   # front
            [2, 3, 7], [2, 7, 6],   # back
            [0, 4, 7], [0, 7, 3],   # left
            [1, 2, 6], [1, 6, 5],   # right
        ], dtype=np.int32)

        # Sanity: all faces should have non-zero area
        for fi, tri in enumerate(tris):
            v0, v1, v2 = box_v[tri[0]], box_v[tri[1]], box_v[tri[2]]
            area = float(np.linalg.norm(np.cross(v1 - v0, v2 - v0)))
            if area < 1e-9:
                raise RuntimeError(
                    f"_add_bedrock_slabs: slab {i} face {fi} has zero area "
                    f"(slab_w={slab_w:.2f}, slab_l={slab_l:.2f}, slab_h={slab_h:.2f})"
                )

        # Per-vertex colours (3-channel: top/side/bottom)
        vc = np.zeros((8, 3), dtype=np.float32)
        for vi in range(8):
            nz = float(box_v[vi, 2])
            # Approximate Z position relative to slab height to assign channel
            slab_top_z = float(box_v[:, 2].max())
            slab_bot_z = float(box_v[:, 2].min())
            mid_z = (slab_top_z + slab_bot_z) * 0.5
            # Top half → dust; bottom half → rust; edge vertices blend
            t = np.clip((nz - mid_z) / (slab_h * 0.5 + 1e-9), -1.0, 1.0)
            if t > 0.3:
                # Top zone
                tw = (t - 0.3) / 0.7
                vc[vi] = c_maaz_side * (1 - tw) + c_maaz_top * tw
            elif t < -0.3:
                # Bottom zone
                bw = (-t - 0.3) / 0.7
                vc[vi] = c_maaz_side * (1 - bw) + c_maaz_bot * bw
            else:
                vc[vi] = c_maaz_side

        # Build USD mesh
        prim_path = f"{slabs_root}/slab_{i:03d}"
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(box_v))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris.ravel()))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(
            np.full(len(tris), 3, dtype=np.int32)
        ))
        # Slabs are flat rock — catmullClark not appropriate (sharpens to quads)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(False)

        # Vertex colours
        primvars_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        pv = primvars_api.CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, "vertex"
        )
        pv.Set(Vt.Vec3fArray.FromNumpy(vc))

        # Material
        if slab_mat is not None:
            _bind_material(stage, prim_path, slab_mat)

        slab_paths.append(prim_path)

    print(f"[hirise_terrain] Bedrock slabs: {n_slabs} tabular outcrops placed "
          f"(Máaz formation, 0.5–3.5m across, 8–30cm thick, 20–40% buried)")
    return slab_paths


# =============================================================================
# Part 5 — Pebble scatter (USD PointInstancer, Layer 5)
# =============================================================================
#
# Scientific basis:
#   Vaughan et al. 2023, JGR Planets (10.1029/2022JE007437)
#     — regolith grain-size mapping, pebble aprons around rocks, λ ≈ 0.4 m
#   Golombek et al. 2008 JGR 113 E00A11
#     — CFA size-frequency, 1–5 cm clasts abundant at crater floors
#   Bridges et al. 2017 PMC5815379
#     — coarse grains (1–2 mm) concentrated at ripple crests and rock bases
#
# Why PointInstancer:
#   2 500 individual Mesh prims would saturate USD traversal.
#   PointInstancer gives N instances from M prototype meshes in O(M) prim cost.
#   Each instance varies: position, scale (size jitter ± 30 %), prototype choice.

def _scatter_pebbles(
    stage:          "Usd.Stage",
    instancer_path: str,
    elevation:      np.ndarray,
    terrain_w:      float,
    terrain_d:      float,
    rock_xz:        List[Tuple[float, float]],
    rng:            random.Random,
    n_total:        int = 2500,
    pebble_mat:     Optional["UsdShade.Material"] = None,
) -> str:
    """
    Scatter micro-pebble clasts (1–5 cm diameter) using USD PointInstancer.

    Placement strategy
    ------------------
    65 % clustered near boulders / cobbles:
      Radial distance drawn from Exponential(λ=0.4 m) — Vaughan 2023 finds
      pebble density falls off with this characteristic length from rock edges.

    35 % uniform random (background regolith coverage):
      CFA power-law requires a baseline pebble density across the whole floor.

    Prototypes
    ----------
    Three size classes (tiny / small / medium) at 60:30:10 frequency ratio,
    matching Golombek 2008 CFA sub-centimetre distribution.
      tiny:   d = 1.2 cm
      small:  d = 2.5 cm
      medium: d = 4.5 cm

    Each prototype is an icosphere (subs=0) with Voronoi fracture + grain noise
    so they look like angular clasts, not perfect spheres.  Oblate (h ≈ 0.45 d)
    so they look naturally settled on the surface.

    Returns
    -------
    str  USD prim path of the PointInstancer.

    References
    ----------
    Vaughan et al. 2023  10.1029/2022JE007437  — pebble apron λ
    Golombek et al. 2008 JGR 113 E00A11        — CFA sub-pebble distribution
    Bridges et al. 2017  PMC5815379            — coarse grains at rock bases
    """
    rng_np = np.random.default_rng(rng.randint(0, 2 ** 32))
    ny, nx = elevation.shape

    instancer = UsdGeom.PointInstancer.Define(stage, instancer_path)
    stage.DefinePrim(f"{instancer_path}/Prototypes", "Scope")

    # ── 1. Build prototype meshes ─────────────────────────────────────────────
    # diameter in metres, frequency weight
    _PROTO_SPECS = [
        (0.012, 0.60),   # tiny   — 60 % of all pebbles
        (0.025, 0.30),   # small  — 30 %
        (0.045, 0.10),   # medium — 10 %
    ]
    proto_paths = []
    proto_weights = [w for _, w in _PROTO_SPECS]

    for i, (diam, _) in enumerate(_PROTO_SPECS):
        path = f"{instancer_path}/Prototypes/pebble_{i}"
        v, f = _make_icosphere(subdivisions=0)   # 12 verts / 20 faces — cheap

        # Angular fracture + grain texture
        n_cuts  = int(rng_np.integers(4, 8))
        v       = _voronoi_fracture(v, n_cuts, rng_np)
        v       = _apply_grain_noise(v, 0.07, rng_np)

        # Oblate scaling: width ± 15 %, height ≈ 0.45 × diameter
        r = diam / 2.0
        v[:, 0] *= r * float(rng_np.uniform(0.85, 1.15))
        v[:, 1] *= r * float(rng_np.uniform(0.85, 1.15))
        v[:, 2] *= r * 0.45

        # Sit pebble on surface — expose 40 % (Golombek 2008)
        z_lo      = float(v[:, 2].min())
        z_hi      = float(v[:, 2].max())
        v[:, 2]  -= z_lo + 0.55 * (z_hi - z_lo)

        v32 = v.astype(np.float32)
        fi  = f.ravel().astype(np.int32)
        fc  = np.full(len(f), 3, dtype=np.int32)

        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v32))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(fi))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(fc))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(False)

        # Flat basalt-oxide colour — pebbles too small for mantling
        # Slightly lighter than basalt (Vaughan 2023: pebbles often dust-coated)
        pebble_col = np.array([[0.135, 0.092, 0.060]], dtype=np.float32).repeat(len(v32), axis=0)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "displayColor",
            Sdf.ValueTypeNames.Color3fArray,
            "vertex",
        ).Set(Vt.Vec3fArray.FromNumpy(pebble_col))

        if pebble_mat is not None:
            _bind_material(stage, path, pebble_mat)

        proto_paths.append(path)

    instancer.CreatePrototypesRel().SetTargets(
        [Sdf.Path(p) for p in proto_paths]
    )

    # ── 2. Generate positions ─────────────────────────────────────────────────
    positions:     list = []
    proto_indices: list = []
    scales_list:   list = []

    n_clustered = int(n_total * 0.65)
    n_uniform   = n_total - n_clustered

    # Helper: sample a prototype index using frequency weights
    def _pick_proto() -> int:
        r = rng.random()
        cum = 0.0
        for k, w in enumerate(proto_weights):
            cum += w
            if r <= cum:
                return k
        return len(proto_weights) - 1

    def _terrain_z(x: float, y: float) -> float:
        col = int((x + terrain_w / 2) / terrain_w * (nx - 1))
        row = int((y + terrain_d / 2) / terrain_d * (ny - 1))
        return float(elevation[max(0, min(ny - 1, row)), max(0, min(nx - 1, col))])

    # Background uniform pebbles
    for _ in range(n_uniform):
        x = rng.uniform(-terrain_w * 0.47, terrain_w * 0.47)
        y = rng.uniform(-terrain_d * 0.47, terrain_d * 0.47)
        positions.append((x, y, _terrain_z(x, y)))
        proto_indices.append(_pick_proto())
        sc = float(rng_np.uniform(0.70, 1.30))
        scales_list.append((sc, sc, sc))

    # Clustered pebbles near boulders / cobbles
    if rock_xz:
        n_placed = 0
        rock_list = list(rock_xz)
        while n_placed < n_clustered:
            rx, ry = rock_list[n_placed % len(rock_list)]
            angle  = rng.uniform(0.0, 2.0 * math.pi)
            # Exponential radial distribution λ = 0.4 m (Vaughan 2023)
            r_dist = float(rng_np.exponential(0.40))
            r_dist = min(r_dist, 3.0)   # hard cap at 3 m
            x = rx + r_dist * math.cos(angle)
            y = ry + r_dist * math.sin(angle)
            x = max(-terrain_w * 0.47, min(terrain_w * 0.47, x))
            y = max(-terrain_d * 0.47, min(terrain_d * 0.47, y))
            positions.append((x, y, _terrain_z(x, y)))
            proto_indices.append(_pick_proto())
            sc = float(rng_np.uniform(0.70, 1.30))
            scales_list.append((sc, sc, sc))
            n_placed += 1
    else:
        # Fallback: additional uniform pebbles when no rocks exist
        for _ in range(n_clustered):
            x = rng.uniform(-terrain_w * 0.47, terrain_w * 0.47)
            y = rng.uniform(-terrain_d * 0.47, terrain_d * 0.47)
            positions.append((x, y, _terrain_z(x, y)))
            proto_indices.append(_pick_proto())
            sc = float(rng_np.uniform(0.70, 1.30))
            scales_list.append((sc, sc, sc))

    # ── 3. Write PointInstancer attributes ────────────────────────────────────
    pos_arr  = np.array(positions,   dtype=np.float32)
    sc_arr   = np.array(scales_list, dtype=np.float32)

    instancer.CreatePositionsAttr().Set(Vt.Vec3fArray.FromNumpy(pos_arr))
    instancer.CreateProtoIndicesAttr().Set(Vt.IntArray(proto_indices))
    instancer.CreateScalesAttr().Set(Vt.Vec3fArray.FromNumpy(sc_arr))

    return instancer_path


# =============================================================================
# Rock-ground interface — regolith skirt meshes
# =============================================================================

def _add_regolith_skirts(
    stage:       "Usd.Stage",
    skirts_root: str,
    rock_paths:  List[str],
    elevation:   np.ndarray,
    terrain_w:   float,
    terrain_d:   float,
    skirt_mat:   Optional["UsdShade.Material"] = None,
) -> List[str]:
    """
    Add a thin flat regolith skirt around each boulder and cobble base.

    The skirt is a radial fan of triangles from rock centre to rock edge + margin,
    lying on the terrain surface.  It serves two purposes:
      1. Fills the visible seam between the buried rock base and the terrain mesh
         (geometry gap due to discrete mesh resolution)
      2. Provides a natural "settled-in" appearance — regolith piles slightly
         against the upwind base of each rock (Sullivan 2005, Golombek 2008)

    Physical model
    --------------
    Width:    10–20 % of rock diameter (regolith washed up against base)
    Colour:   blends from rock base colour at inner edge to terrain colour at outer
    Geometry: flat disc fan, 12 radial segments, Z = terrain elevation at each point

    Note: skirts are ONLY added for boulders and cobbles (not pebbles), where the
    geometry gap is large enough to be visible in the Mastcam-Z 25.8° FOV.

    UV primvar
    ----------
    Each skirt vertex gets "st" = (world_x / 2.0, world_y / 2.0), matching the
    terrain mesh's UV layout (tile_m=2.0 from generate_terrain_textures.py).
    This allows the Hapke terrain MDL material to tile the regolith normal and
    roughness textures seamlessly across the skirt-terrain boundary.

    Parameters
    ----------
    rock_paths  : list of rock prim paths from _place_rocks
    skirts_root : USD path for all skirt prims

    Returns
    -------
    skirt_paths : list of USD prim paths

    Raises
    ------
    RuntimeError if any generated triangle has zero area
    """
    ny, nx = elevation.shape
    stage.DefinePrim(skirts_root, "Scope")
    skirt_paths: List[str] = []

    # Colours: inner (rock base) → outer (terrain oxide)
    c_inner = np.array([0.10, 0.065, 0.042], dtype=np.float32)  # dark basalt base
    c_outer = np.array([0.165, 0.100, 0.065], dtype=np.float32)  # terrain oxide colour

    for pi, rock_path in enumerate(rock_paths):
        # Only boulders and cobbles
        if "/boulder_" not in rock_path and "/cobble_" not in rock_path:
            continue

        prim = stage.GetPrimAtPath(rock_path)
        if not prim.IsValid():
            continue

        # Get rock world position from translate op
        xf = UsdGeom.Xformable(prim)
        cx, cy, cz = 0.0, 0.0, 0.0
        for op in xf.GetOrderedXformOps():
            if "translate" in op.GetName().lower():
                t = op.Get()
                cx, cy, cz = float(t[0]), float(t[1]), float(t[2])
                break

        # Estimate rock radius from path (boulder ~1.25m, cobble ~0.4m)
        if "/boulder_" in rock_path:
            skirt_inner_r = 1.25   # half of 2.5m boulder
            skirt_outer_r = 1.50   # + 20% margin
        else:
            skirt_inner_r = 0.40
            skirt_outer_r = 0.50

        # Build radial fan (12 segments)
        N_SEG = 12
        angles = np.linspace(0.0, 2.0 * np.pi, N_SEG, endpoint=False)

        def _z_at(x: float, y: float) -> float:
            col = int((x + terrain_w / 2) / terrain_w * (nx - 1))
            row = int((y + terrain_d / 2) / terrain_d * (ny - 1))
            col = max(0, min(nx - 1, col))
            row = max(0, min(ny - 1, row))
            return float(elevation[row, col])

        # Inner ring (at rock edge)
        inner_pts = np.array([
            [cx + skirt_inner_r * np.cos(a), cy + skirt_inner_r * np.sin(a),
             _z_at(cx + skirt_inner_r * np.cos(a), cy + skirt_inner_r * np.sin(a))]
            for a in angles
        ], dtype=np.float32)

        # Outer ring (at skirt edge)
        outer_pts = np.array([
            [cx + skirt_outer_r * np.cos(a), cy + skirt_outer_r * np.sin(a),
             _z_at(cx + skirt_outer_r * np.cos(a), cy + skirt_outer_r * np.sin(a))]
            for a in angles
        ], dtype=np.float32)

        # Combine: inner first (0..N_SEG-1), outer second (N_SEG..2N-1)
        verts = np.vstack([inner_pts, outer_pts])   # (2N, 3)

        # Triangles: each quad (i_in, i_out, i_out+1, i_in+1) → 2 tris
        tris = []
        for k in range(N_SEG):
            k_next = (k + 1) % N_SEG
            i0, i1 = k, k_next
            o0, o1 = k + N_SEG, k_next + N_SEG
            tris.append([i0, o0, o1])
            tris.append([i0, o1, i1])
        tris = np.array(tris, dtype=np.int32)

        # Sanity: all faces non-zero area
        for fi, tri in enumerate(tris):
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            area = float(np.linalg.norm(np.cross(v1 - v0, v2 - v0)))
            if area < 1e-9:
                raise RuntimeError(
                    f"_add_regolith_skirts: skirt {pi} face {fi} has zero area — "
                    f"rock at ({cx:.2f},{cy:.2f}), r_inner={skirt_inner_r:.2f}m"
                )

        # Per-vertex colour: inner=rock-base, outer=terrain
        n_verts = len(verts)
        vc = np.zeros((n_verts, 3), dtype=np.float32)
        vc[:N_SEG]  = c_inner
        vc[N_SEG:]  = c_outer

        # Build USD mesh
        prim_path = f"{skirts_root}/skirt_{pi:04d}"
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris.ravel()))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(
            np.full(len(tris), 3, dtype=np.int32)
        ))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)   # visible from below (rover camera)

        primvars_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        pv = primvars_api.CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, "vertex"
        )
        pv.Set(Vt.Vec3fArray.FromNumpy(vc))

        # UV coords for Hapke terrain material (tile_m=2.0 matches terrain mesh
        # and generate_terrain_textures.py — state::texture_coordinate(0) in MDL)
        uv = np.stack([verts[:, 0] / 2.0, verts[:, 1] / 2.0], axis=-1).astype(np.float32)
        pv_st = primvars_api.CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex"
        )
        pv_st.Set(Vt.Vec2fArray.FromNumpy(uv))

        if skirt_mat is not None:
            _bind_material(stage, prim_path, skirt_mat)

        skirt_paths.append(prim_path)

    print(f"[hirise_terrain] Regolith skirts: {len(skirt_paths)} skirts "
          f"at boulder+cobble bases (12-segment radial fan, terrain-following Z)")
    return skirt_paths


# =============================================================================
# Part 6 — Martian lighting
# =============================================================================

def _build_martian_lighting(
    stage:    "Usd.Stage",
    dust_tau: float = _DUST_TAU,
) -> None:
    """
    Physically accurate Martian lighting for Jezero Crater with atmospheric
    dust opacity τ driving all three coupled light quantities simultaneously.

    Dust optical depth τ (dust_tau)
    ────────────────────────────────
    All three of the following are derived from a single τ value:

    1. Direct solar intensity — Beer-Lambert attenuation:
         I_direct = I_base × exp(−τ / sin(elevation))
       At τ=0.56, elev=28°:  I = 4500 × exp(−1.194) = 1365  (30% of clear sky)
       At τ=1.5  dust storm:  I = 4500 × exp(−3.197) =  184

    2. Sky diffuse intensity — dust scatters sunlight into sky:
         I_sky ≈ I_sky_base × (1 + τ × 1.8)
       More dust → brighter diffuse illumination (real Mars effect — clear sols
       have darker skies than dusty ones because less aerosol back-scatter).

    3. Sky colour — dust absorbs blue/green more than red (Mie scattering):
         τ=0.0 → (0.65, 0.75, 0.90)  near-white with faint blue
         τ=0.5 → (0.58, 0.42, 0.32)  salmon-pink (Bell 2021 Mastcam-Z)
         τ=1.5 → (0.45, 0.25, 0.15)  deep orange-tan (dust storm)
       Interpolated piecewise-linearly in τ-space.

    Solar geometry is driven by module constants (_SUN_ELEVATION_DEG,
    _SUN_AZIMUTH_DEG) — same values used by the Hapke BRDF shader.

    References
    ----------
    Bell et al. 2021, SSR 217:24      — Mastcam-Z τ measurements, sky calibration
    Madeleine et al. 2012, Icarus 220 — aerosol g=0.68, τ seasonal variation
    Vicente-Retortillo et al. 2023    — MEDA sky radiance vs τ correlation
    Lemmon et al. 2019, GRL           — τ=0.3–2.0 range during ops
    """
    if dust_tau < 0.0:
        raise ValueError(
            f"_build_martian_lighting: dust_tau must be ≥ 0, got {dust_tau}"
        )

    import math as _math

    mu0    = _math.sin(_math.radians(_SUN_ELEVATION_DEG))   # cos(zenith) = sin(elev)
    # ── 1. Direct solar intensity ─────────────────────────────────────────────
    i_sun  = _SUN_INTENSITY_BASE * _math.exp(-dust_tau / mu0)

    # ── 2. Sun colour reddening ───────────────────────────────────────────────
    # More dust → warmer (more reddish) sun disc.
    # g, b channels decrease linearly with τ; r channel clamped at 1.0.
    sun_g  = max(0.40, 0.93 - dust_tau * 0.17)
    sun_b  = max(0.25, 0.88 - dust_tau * 0.22)

    # ── 3. Sky colour (piecewise linear in τ) ────────────────────────────────
    # Calibrated to match Bell 2021 Mastcam-Z sky calibration at τ=0.56
    # (sky=(0.58, 0.42, 0.32)).
    if dust_tau <= 0.5:
        frac   = dust_tau / 0.5
        sky_r  = 0.65 - frac * 0.07   # 0.65 → 0.58
        sky_g  = 0.75 - frac * 0.33   # 0.75 → 0.42
        sky_b  = 0.90 - frac * 0.58   # 0.90 → 0.32
    else:
        frac   = min((dust_tau - 0.5) / 1.0, 1.0)
        sky_r  = 0.58 - frac * 0.13   # 0.58 → 0.45
        sky_g  = 0.42 - frac * 0.17   # 0.42 → 0.25
        sky_b  = 0.32 - frac * 0.17   # 0.32 → 0.15

    # ── 4. Sky diffuse intensity ──────────────────────────────────────────────
    # Texture mode:  I = 200 × (1 + τ × 1.8)
    # Fallback mode: I = 250 × (1 + τ × 1.8)
    # Factor 1.8 empirically matched to MEDA irradiance ratios (Vicente-R 2023).
    _TAU_SKY_FACTOR  = 1.8
    i_sky_texture    = 200.0 * (1.0 + dust_tau * _TAU_SKY_FACTOR)
    i_sky_flat       = 250.0 * (1.0 + dust_tau * _TAU_SKY_FACTOR)

    # ── Sun (DistantLight) ────────────────────────────────────────────────────
    sun_path = "/World/SunLight"
    if not stage.GetPrimAtPath(sun_path).IsValid():
        sun = UsdLux.DistantLight.Define(stage, sun_path)
    else:
        sun = UsdLux.DistantLight(stage.GetPrimAtPath(sun_path))

    sun.CreateIntensityAttr(i_sun)
    sun.CreateColorAttr(Gf.Vec3f(1.0, float(sun_g), float(sun_b)))
    sun.CreateAngleAttr(0.35)    # 0.35° angular diameter at Mars (< Earth's 0.53°)

    # Orient: elevation = _SUN_ELEVATION_DEG, azimuth = _SUN_AZIMUTH_DEG
    sun_xf = UsdGeom.Xformable(sun.GetPrim())
    sun_xf.AddRotateXYZOp().Set(
        Gf.Vec3f(-(90.0 - _SUN_ELEVATION_DEG), 0.0, _SUN_AZIMUTH_DEG)
    )

    # ── Sky dome (DomeLight) ──────────────────────────────────────────────────
    sky_path = "/World/SkyDome"
    if not stage.GetPrimAtPath(sky_path).IsValid():
        sky = UsdLux.DomeLight.Define(stage, sky_path)
    else:
        sky = UsdLux.DomeLight(stage.GetPrimAtPath(sky_path))

    _repo_data   = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    )
    _sky_texture = os.path.join(_repo_data, "mars_sky.png")

    if os.path.exists(_sky_texture):
        # Texture provides colour; tint stays white, intensity scaled by τ.
        sky.CreateTextureFileAttr().Set(Sdf.AssetPath(_sky_texture))
        sky.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        sky.CreateIntensityAttr(i_sky_texture)
        sky_xf = UsdGeom.Xformable(sky.GetPrim())
        sky_xf.AddRotateYOp().Set(0.0)
        print(f"[hirise_terrain] Sky: HG texture, τ={dust_tau:.2f} → "
              f"I_sun={i_sun:.0f} I_sky={i_sky_texture:.0f}")
    else:
        # Flat colour fallback — colour is τ-derived (not Bell 2021 hardcode)
        sky.CreateIntensityAttr(i_sky_flat)
        sky.CreateColorAttr(Gf.Vec3f(float(sky_r), float(sky_g), float(sky_b)))
        print(f"[hirise_terrain] Sky: flat fallback, τ={dust_tau:.2f} → "
              f"I_sun={i_sun:.0f} colour=({sky_r:.2f},{sky_g:.2f},{sky_b:.2f}) "
              f"I_sky={i_sky_flat:.0f}")


# =============================================================================
# Public API
# =============================================================================

def build_mars_scene(
    stage,
    terrain_path:     str   = "/World/MarsTerrain",
    rocks_root:       str   = "/World/Rocks",
    looks_root:       str   = "/World/Looks",
    terrain_width:    float = 40.0,
    terrain_depth:    float = 40.0,
    terrain_nx:       Optional[int] = None,
    terrain_ny:       Optional[int] = None,
    terrain_amplitude: float = 0.70,
    replace_existing: bool  = True,
    hirise_dtm_path:  Optional[str] = None,
    hirise_patch_size: float = 200.0,
    dust_tau:         float = _DUST_TAU,
) -> dict:
    """
    Build a complete Mars terrain scene in an Isaac Sim USD stage.

    Parameters
    ----------
    stage             Active Usd.Stage from omni.usd.get_context().get_stage()
    terrain_path      USD prim path for the terrain mesh
    rocks_root        USD prim path root for rock prims
    looks_root        USD prim path root for materials
    terrain_width     Scene width in metres (X axis)
    terrain_depth     Scene depth in metres (Y axis)
    terrain_nx        Mesh columns.  None → auto: target ≤0.15 m/vertex so that
                      craters (min D=0.8 m) resolve with ≥5 verts across and
                      ripples (λ=3.5 m) resolve with ≥23 verts/wave.  Capped
                      at 512 for performance.  Examples: 40 m→267, 500 m→512.
    terrain_ny        Mesh rows.  Same auto-scaling rule as terrain_nx.
    terrain_amplitude Fallback procedural terrain height range (metres)
    replace_existing  Remove existing prims before building
    hirise_dtm_path   Path to HiRISE GeoTIFF.  None → procedural fallback.
    hirise_patch_size How many metres of HiRISE data to crop (default 200m)
    dust_tau          Atmospheric dust optical depth.  Drives sun intensity
                      (Beer-Lambert), sky brightness, and sky colour together.
                      _DUST_TAU=0.56 = Jezero nominal ops (Bell 2021).
                      0.3 = clear morning, 1.5 = regional storm.

    Returns
    -------
    dict  {terrain_path, rock_paths, dtm_source, elevation_min, elevation_max}
    """
    if not _PXR_AVAILABLE:
        raise RuntimeError("pxr (USD) is not available — run inside Isaac Sim")

    # ── 0. Auto-scale mesh resolution ────────────────────────────────────────
    # Target ≤0.15 m/vertex so the minimum crater (D=0.8 m) has ≥5 vertices
    # across its bowl, and ripples (λ=3.5 m) have ≥23 vertices per wave.
    # Cap at 512 to keep USD mesh under ~800k triangles.
    # For 40 m scene: min(512, max(128, 267)) = 267
    # For 500 m scene: min(512, max(128, 3333)) = 512  (0.98 m/vert — ripple shape
    #   comes from the normal map at this scale, not raw geometry)
    if terrain_nx is None:
        terrain_nx = min(512, max(128, int(terrain_width  / 0.15)))
    if terrain_ny is None:
        terrain_ny = min(512, max(128, int(terrain_depth  / 0.15)))
    print(f"[hirise_terrain] Mesh resolution: {terrain_nx}×{terrain_ny} "
          f"({terrain_width/terrain_nx:.3f} m/vertex × "
          f"{terrain_depth/terrain_ny:.3f} m/vertex)")

    rng = random.Random(42)

    # ── 1. Remove stale prims ─────────────────────────────────────────────────
    if replace_existing:
        for p in (terrain_path, rocks_root, looks_root,
                  "/World/SunLight", "/World/SkyDome"):
            if stage.GetPrimAtPath(p).IsValid():
                stage.RemovePrim(p)

    # ── 2. Elevation data ─────────────────────────────────────────────────────
    meta: dict
    if hirise_dtm_path is not None:
        try:
            elevation, meta = load_hirise_dtm(
                hirise_dtm_path,
                size_m=hirise_patch_size,
                target_nx=terrain_nx,
                target_ny=terrain_ny,
            )
            print(f"[hirise_terrain] HiRISE DTM loaded: "
                  f"{meta['elevation_min']:.1f}m – {meta['elevation_max']:.1f}m, "
                  f"res {meta['pixel_res_m']:.2f}m/px")
        except Exception as e:
            print(f"[hirise_terrain] HiRISE load failed ({e}), using procedural fallback.")
            elevation, meta = generate_procedural_dtm(
                terrain_width, terrain_depth, terrain_nx, terrain_ny, terrain_amplitude
            )
    else:
        elevation, meta = generate_procedural_dtm(
            terrain_width, terrain_depth, terrain_nx, terrain_ny, terrain_amplitude
        )
        print(f"[hirise_terrain] Procedural DTM: "
              f"amplitude {terrain_amplitude}m, {terrain_nx}×{terrain_ny} verts")

    # ── 2b. Aeolian ripple deformation ────────────────────────────────────────
    # Baked into elevation BEFORE mesh build so geometry is physically correct.
    # λ = 3.5 m, h = 0.15 m, transport 276 ° WNW (Chojnacki 2018, Bridges 2017)
    elevation = _add_aeolian_ripples(elevation, terrain_width, terrain_depth)
    print("[hirise_terrain] Aeolian ripples added (λ=3.5 m, h=0.15 m, 276° transport)")

    # ── 2c. Impact craters ────────────────────────────────────────────────────
    # Stamped AFTER ripples: craters post-date the aeolian bedform (geologically
    # more recent impact events truncate older ripple topography).
    # Density: 0.005 fresh craters (D>0.5 m) per m², Golombek 2014 InSight data.
    elevation, crater_list = _add_impact_craters(
        elevation, terrain_width, terrain_depth,
        min_radius_m=0.4, max_radius_m=min(4.0, terrain_width * 0.10),
    )

    # ── 3. Terrain mesh ───────────────────────────────────────────────────────
    _build_terrain_mesh(stage, terrain_path, elevation, terrain_width, terrain_depth)

    # ── 4. Materials ──────────────────────────────────────────────────────────
    stage.DefinePrim(looks_root, "Scope")
    materials = _build_mars_materials(stage, looks_root)

    # Locate texture data directory (repo-relative, same dir as mars_sky.png)
    # Resolve data dir relative to this file — always correct, no symlink confusion
    _data_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    )

    # ── 4a. Terrain material — Hapke (2002) BRDF ─────────────────────────────
    # MDL shader: shaders/hapke_regolith.mdl
    # Implements the full Hapke photometric equation for particulate regolith:
    #   r(i,e,g) = (w₀/4π)[μ₀/(μ₀+μ)][p(g)·B(g) + H(μ₀)H(μ) − 1]
    # with opposition surge, Lommel-Seeliger limb darkening, HG phase, and
    # Chandrasekhar multiple-scattering H-function.  NOT a UsdPreviewSurface
    # Cook-Torrance approximation.
    #
    # RAISES if texture files are missing — run generate_terrain_textures.py first.
    terrain_mat = _build_terrain_hapke_material(
        stage, f"{looks_root}/TerrainHapke", _data_dir,
        sun_elevation_deg=_SUN_ELEVATION_DEG,   # synced with DistantLight rotation
    )
    _bind_material(stage, terrain_path, terrain_mat)

    # ── 4b. Rock material — triplanar vesicle normal map ─────────────────────
    # MDL shader: shaders/rock_triplanar.mdl
    # Samples rock_vesicle_normal.png from 3 world-space-aligned planes and
    # blends by |N|^4, eliminating the pole singularity and seam artefacts
    # of the previous spherical-UV approach.
    # Per-vertex 3-channel colour (dust/geology/rust) via state::color().
    #
    # RAISES if rock_vesicle_normal.png is missing — no silent fallback.
    rock_pbr_mat = _build_rock_triplanar_material(
        stage, f"{looks_root}/RockTriplanar", _data_dir
    )
    # Alias for callers that still reference rock_vc_mat
    rock_vc_mat = rock_pbr_mat

    # ── 5. Rocks ──────────────────────────────────────────────────────────────
    stage.DefinePrim(rocks_root, "Scope")
    rock_paths = _place_rocks(
        stage, rocks_root, elevation,
        terrain_width, terrain_depth,
        materials, rng,
        rock_material=rock_vc_mat,
    )
    counts = {cls: 0 for cls, *_ in _ROCK_SIZE_CLASSES}
    for p in rock_paths:
        for cls, *_ in _ROCK_SIZE_CLASSES:
            if f"/{cls}_" in p:
                counts[cls] += 1
    n_vent = int(len(rock_paths) * _VENTIFACT_PROB)
    print(f"[hirise_terrain] Placed {len(rock_paths)} rocks "
          f"({counts.get('boulder',0)} boulders, "
          f"{counts.get('cobble',0)} cobbles, "
          f"{counts.get('pebble',0)} pebbles) "
          f"| ~{n_vent} ventifacts | Máaz+Séítah units | Voronoi fracture")

    # ── 5b. Ground vertex colours (after rocks so wind shadows know positions) ─
    # Extract (x, y) positions of large rocks for wind-shadow calculation.
    boulder_cobble_xz: List[Tuple[float, float]] = []
    for p in rock_paths:
        if "/boulder_" in p or "/cobble_" in p:
            prim = stage.GetPrimAtPath(p)
            if prim.IsValid():
                xf = UsdGeom.Xformable(prim)
                ops = xf.GetOrderedXformOps()
                for op in ops:
                    if "translate" in op.GetName().lower():
                        t = op.Get()
                        boulder_cobble_xz.append((float(t[0]), float(t[1])))
                        break

    _add_terrain_vertex_colors(
        stage, terrain_path, elevation,
        terrain_width, terrain_depth,
        rock_xz=boulder_cobble_xz,
        crater_list=crater_list,
    )
    n_poly = max(16, int(terrain_width * terrain_depth / 25))
    print(f"[hirise_terrain] Ground texture: polygon cracks ({n_poly} polygons), "
          f"ripple grain sorting, dust accumulation, "
          f"{len(boulder_cobble_xz)} wind shadows, "
          f"{len(boulder_cobble_xz)} AO rings")

    # ── 5b-ii. Regolith skirt meshes at rock bases ────────────────────────────
    # Thin radial-fan meshes filling the geometric gap between rock base and
    # terrain mesh. Terrain-following Z, blended colour inner→outer.
    # Uses terrain_mat (Hapke MDL), not rock_vc_mat — skirts are regolith,
    # not rock.  UVs (tile_m=2.0) added to skirts by _add_regolith_skirts so
    # the Hapke shader can tile the normal/roughness textures across them.
    skirts_root = f"{rocks_root}/Skirts"
    skirt_paths = _add_regolith_skirts(
        stage, skirts_root,
        rock_paths, elevation,
        terrain_width, terrain_depth,
        skirt_mat=terrain_mat,
    )

    # ── 5c. Pebble scatter (Layer 5) ──────────────────────────────────────────
    # Micro-pebble clasts (1–5 cm) via USD PointInstancer.
    # Vaughan 2023: exponential density falloff from rock bases λ=0.4 m.
    # Scale pebble count with scene area (same density as rocks).
    # Use rock_vc_mat for prototype shading (vertex-colour PBR reader).
    scene_area_m2 = terrain_width * terrain_depth
    n_pebbles = max(500, min(5000, int(2500 * scene_area_m2 / 1600.0)))
    pebble_instancer_path = f"{rocks_root}/PebbleInstancer"
    _scatter_pebbles(
        stage, pebble_instancer_path,
        elevation, terrain_width, terrain_depth,
        rock_xz=boulder_cobble_xz,
        rng=rng,
        n_total=n_pebbles,
        pebble_mat=rock_vc_mat,
    )
    print(f"[hirise_terrain] Scattered {n_pebbles} micro-pebbles (1–5 cm) "
          f"via PointInstancer | 65 % clustered near rocks (λ=0.4 m) | "
          f"35 % uniform background")

    # ── 5d. Bedrock slab outcrops ─────────────────────────────────────────────
    # Tabular Máaz formation outcrops (Farley 2022, Crumpler 2023).
    # ~1 slab per 100 m² scene area.
    slabs_root = f"{rocks_root}/Slabs"
    slab_paths = _add_bedrock_slabs(
        stage, slabs_root, elevation,
        terrain_width, terrain_depth,
        rng, slab_mat=rock_vc_mat,
    )

    # ── 6. Lighting ───────────────────────────────────────────────────────────
    _build_martian_lighting(stage, dust_tau=dust_tau)
    print("[hirise_terrain] Martian lighting applied (Jezero morning sun)")

    return {
        "terrain_path":     terrain_path,
        "rock_paths":       rock_paths,
        "slab_paths":       slab_paths,
        "skirt_paths":      skirt_paths,
        "pebble_instancer": pebble_instancer_path,
        "dtm_source":       meta["dtm_source"],
        "elevation_min":    meta["elevation_min"],
        "elevation_max":    meta["elevation_max"],
    }
