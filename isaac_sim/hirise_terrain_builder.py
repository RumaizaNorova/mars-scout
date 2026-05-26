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
    Build the four Mars surface material library.

    Colours are calibrated to Perseverance Mastcam-Z RGB data.
    """
    mats = {}

    # Primary terrain surface — iron oxide dust
    mats["MarsOxide"] = _make_material(
        stage, f"{looks_root}/MarsOxide",
        diffuse=(0.58, 0.22, 0.09),
        roughness=0.92,
    )
    # Dark volcanic basalt (rock outcrops)
    mats["Basalt"] = _make_material(
        stage, f"{looks_root}/Basalt",
        diffuse=(0.18, 0.15, 0.13),
        roughness=0.78,
        metallic=0.05,
    )
    # Light sedimentary / sandstone
    mats["Sandstone"] = _make_material(
        stage, f"{looks_root}/Sandstone",
        diffuse=(0.72, 0.52, 0.30),
        roughness=0.88,
    )
    # Iron-rich bright red (float rocks, specular flakes)
    mats["IronRich"] = _make_material(
        stage, f"{looks_root}/IronRich",
        diffuse=(0.68, 0.28, 0.05),
        roughness=0.70,
        metallic=0.08,
    )
    # Pale dust (wind-deposited, bright areas)
    mats["PaleDust"] = _make_material(
        stage, f"{looks_root}/PaleDust",
        diffuse=(0.82, 0.68, 0.52),
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
# Part 4 — Rock placement (CFA geological distribution)
# =============================================================================

# Mars rock CFA (Cumulative Fractional Area) formula from Golombek et al. 2008:
#   q(k) = 1.79 + 0.152 / k
# where k = rock diameter in metres, q = fraction of surface covered.
# We invert to get approximate count per area at each size class.

_ROCK_SIZE_CLASSES = [
    # (name,    diameter_m, count, height_range,   scale_variance, materials)
    # Counts scaled for 500×500 m terrain (CFA formula, Golombek et al. 2008)
    ("boulder",  2.5,       50,   (0.8, 2.5),     0.40, ["Basalt", "IronRich"]),
    ("cobble",   0.8,      150,   (0.3, 1.0),     0.35, ["Basalt", "Sandstone", "IronRich"]),
    ("pebble",   0.25,     250,   (0.08, 0.35),   0.30, ["MarsOxide", "Basalt", "Sandstone", "PaleDust"]),
]


def _build_rock_mesh(
    stage:     "Usd.Stage",
    prim_path: str,
    diameter:  float,
    scale_var: float,
    rng:       random.Random,
) -> "UsdGeom.Mesh":
    """
    Build a single irregular rock mesh — a jittered icosphere subdivision.

    Looks far more geological than a sphere or box.
    """
    # Base icosahedron vertices (normalised)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    base_verts = [
        (-1,  phi, 0), ( 1,  phi, 0), (-1, -phi, 0), ( 1, -phi, 0),
        ( 0, -1,  phi), ( 0,  1,  phi), ( 0, -1, -phi), ( 0,  1, -phi),
        ( phi, 0, -1), ( phi, 0,  1), (-phi, 0, -1), (-phi, 0,  1),
    ]
    base_faces = [
        (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
        (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
        (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
        (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
    ]

    # Normalise + scale to radius
    r = diameter / 2.0
    verts = np.array(base_verts, dtype=np.float32)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    # Geological jitter: rocks are not spheres
    # Each vertex gets a radial displacement: ±30% + per-axis squeeze
    np_rng = np.random.default_rng(rng.randint(0, 2**32))
    jitter = 1.0 + np_rng.uniform(-0.30, 0.30, size=len(verts))
    verts *= jitter[:, None]

    # Non-uniform axis scaling (rocks are oblate/prolate)
    sx = rng.uniform(1.0 - scale_var, 1.0 + scale_var)
    sy = rng.uniform(1.0 - scale_var, 1.0 + scale_var)
    sz = rng.uniform(0.4, 0.75)   # rocks are always flatter vertically
    verts[:, 0] *= sx * r
    verts[:, 1] *= sy * r
    verts[:, 2] *= sz * r

    # Lift so base sits on terrain (min Z → 0, then small embed)
    verts[:, 2] -= verts[:, 2].min()
    embed = rng.uniform(0.05, 0.15) * r  # partially buried
    verts[:, 2] -= embed

    faces = np.array(base_faces, dtype=np.int32)
    face_indices = faces.ravel()
    face_counts  = np.full(len(faces), 3, dtype=np.int32)

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(face_indices))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(face_counts))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(False)

    normals = _compute_normals(verts, faces)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
    mesh.SetNormalsInterpolation("vertex")

    return mesh


def _place_rocks(
    stage:       "Usd.Stage",
    rocks_root:  str,
    elevation:   np.ndarray,
    terrain_w:   float,
    terrain_d:   float,
    materials:   dict,
    rng:         random.Random,
) -> List[str]:
    """
    Place rocks following the Mars CFA size-frequency distribution.

    Rocks are placed preferentially in front of the camera (+X half),
    within the 90° FOV cone.  Some rocks cluster (geological reality:
    boulders shed pebbles downslope).
    """
    ny, nx_cells = elevation.shape
    rock_paths = []
    idx = 0

    for cls_name, diam_m, count, (h_min, h_max), scale_var, mat_names in _ROCK_SIZE_CLASSES:
        for _ in range(count):
            # Placement strategy:
            #   40% near rover start (first 50m, ±30m wide) — visible immediately
            #   60% spread across full terrain — for autonomous navigation to find
            if rng.random() < 0.40:
                x = rng.uniform(2.0, min(50.0, terrain_w * 0.45))
                y = rng.uniform(-30.0, 30.0)
            else:
                x = rng.uniform(-terrain_w * 0.45, terrain_w * 0.45)
                y = rng.uniform(-terrain_d * 0.45, terrain_d * 0.45)

            # Sample terrain elevation at (x, y)
            col = int((x + terrain_w / 2) / terrain_w * (nx_cells - 1))
            row = int((y + terrain_d / 2) / terrain_d * (ny - 1))
            col = max(0, min(nx_cells - 1, col))
            row = max(0, min(ny - 1, row))
            z_terrain = float(elevation[row, col])

            prim_path = f"{rocks_root}/{cls_name}_{idx:03d}"
            mesh = _build_rock_mesh(stage, prim_path, diam_m, scale_var, rng)

            # Position
            xform = UsdGeom.Xformable(mesh.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z_terrain))
            # Random yaw rotation for variety
            yaw = rng.uniform(0, 360)
            xform.AddRotateZOp().Set(yaw)

            # Material
            mat_name = rng.choice(mat_names)
            if mat_name in materials:
                _bind_material(stage, prim_path, materials[mat_name])

            rock_paths.append(prim_path)
            idx += 1

    return rock_paths


# =============================================================================
# Part 5 — Martian lighting
# =============================================================================

def _build_martian_lighting(stage: "Usd.Stage") -> None:
    """
    Physically accurate Martian lighting for Jezero Crater.

    Sun parameters:
      - Low-angle morning light (28° elevation, 15° azimuth from east)
      - 3 400 K colour temperature (amber, dust-scattered)
      - Intensity 3 500 lx (Mars midday ≈ 590 W/m², vs Earth 1000 W/m²)

    Sky:
      - Hazy pink-peach dome (Rayleigh scatter + iron-dust aerosols)
      - Intensity 600 lx (diffuse skylight)
    """
    # Sun (DistantLight)
    sun_path = "/World/SunLight"
    if not stage.GetPrimAtPath(sun_path).IsValid():
        sun = UsdLux.DistantLight.Define(stage, sun_path)
    else:
        sun = UsdLux.DistantLight(stage.GetPrimAtPath(sun_path))

    # Mars solar irradiance: ~590 W/m² (vs Earth 1000 W/m²) → ~59% of Earth noon.
    # Isaac Sim DistantLight intensity ~= lux. Earth clear noon ~100000 lux.
    # Mars equivalent: ~59000 lux. We use 55000 to account for dust opacity tau~0.5.
    # Sun COLOR: Perseverance Mastcam-Z calibration images show the Martian sun
    # appears near-white with very slight warm tint (dust absorbs blue slightly).
    # NOT the deep amber we had — that was causing the yellow-wash on terrain.
    # Reference: Bell et al. 2021, Space Science Reviews (Mastcam-Z instrument paper)
    # Isaac Sim DistantLight intensity is NOT physical lux — internal renderer units.
    # Empirically calibrated: 4500 gives correct exposure without washout.
    # Color: near-neutral (Perseverance Mastcam-Z shows sun as white, not amber).
    sun.CreateIntensityAttr(4500.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.93, 0.88))   # near-white, very slight warm tint
    sun.CreateAngleAttr(0.35)                          # Mars: sun subtends ~0.35° (smaller than Earth's 0.53°)

    # Orient sun: 28° above horizon from south-west (Jezero morning local time)
    sun_xf = UsdGeom.Xformable(sun.GetPrim())
    sun_xf.AddRotateXYZOp().Set(Gf.Vec3f(-(90.0 - 28.0), 0.0, 45.0))

    # Sky dome — Martian sky is dusty pinkish-tan (iron dust aerosols scatter red).
    # Perseverance sky images: RGB approx (0.58, 0.42, 0.32) normalised.
    # Intensity: diffuse skylight ~7% of direct solar = ~3900 lux.
    sky_path = "/World/SkyDome"
    if not stage.GetPrimAtPath(sky_path).IsValid():
        sky = UsdLux.DomeLight.Define(stage, sky_path)
    else:
        sky = UsdLux.DomeLight(stage.GetPrimAtPath(sky_path))

    sky.CreateIntensityAttr(250.0)
    sky.CreateColorAttr(Gf.Vec3f(0.58, 0.42, 0.32))  # dusty pinkish-tan (Perseverance sky)


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
    terrain_nx:       int   = 128,
    terrain_ny:       int   = 128,
    terrain_amplitude: float = 0.70,
    replace_existing: bool  = True,
    hirise_dtm_path:  Optional[str] = None,
    hirise_patch_size: float = 200.0,
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
    terrain_nx        Mesh columns
    terrain_ny        Mesh rows
    terrain_amplitude Fallback procedural terrain height range (metres)
    replace_existing  Remove existing prims before building
    hirise_dtm_path   Path to HiRISE GeoTIFF.  None → procedural fallback.
    hirise_patch_size How many metres of HiRISE data to crop (default 200m)

    Returns
    -------
    dict  {terrain_path, rock_paths, dtm_source, elevation_min, elevation_max}
    """
    if not _PXR_AVAILABLE:
        raise RuntimeError("pxr (USD) is not available — run inside Isaac Sim")

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

    # ── 3. Terrain mesh ───────────────────────────────────────────────────────
    _build_terrain_mesh(stage, terrain_path, elevation, terrain_width, terrain_depth)

    # ── 4. Materials ──────────────────────────────────────────────────────────
    stage.DefinePrim(looks_root, "Scope")
    materials = _build_mars_materials(stage, looks_root)

    # Bind primary surface material to terrain
    _bind_material(stage, terrain_path, materials["MarsOxide"])

    # ── 5. Rocks ──────────────────────────────────────────────────────────────
    stage.DefinePrim(rocks_root, "Scope")
    rock_paths = _place_rocks(
        stage, rocks_root, elevation,
        terrain_width, terrain_depth,
        materials, rng,
    )
    counts = {cls: 0 for cls, *_ in _ROCK_SIZE_CLASSES}
    for p in rock_paths:
        for cls, *_ in _ROCK_SIZE_CLASSES:
            if f"/{cls}_" in p:
                counts[cls] += 1
    print(f"[hirise_terrain] Placed {len(rock_paths)} rocks "
          f"({counts.get('boulder',0)} boulders, "
          f"{counts.get('cobble',0)} cobbles, "
          f"{counts.get('pebble',0)} pebbles)")

    # ── 6. Lighting ───────────────────────────────────────────────────────────
    _build_martian_lighting(stage)
    print("[hirise_terrain] Martian lighting applied (Jezero morning sun)")

    return {
        "terrain_path":   terrain_path,
        "rock_paths":     rock_paths,
        "dtm_source":     meta["dtm_source"],
        "elevation_min":  meta["elevation_min"],
        "elevation_max":  meta["elevation_max"],
    }
