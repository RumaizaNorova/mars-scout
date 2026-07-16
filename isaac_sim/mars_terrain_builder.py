"""
mars_terrain_builder.py
=======================
Photorealistic Mars terrain builder for Isaac Sim 4.5 (pip install).

Call `build_mars_scene(stage)` after SimulationApp has booted and the USD
stage exists.  The function is intentionally self-contained: it uses only
  • pxr  (bundled with Isaac Sim)
  • numpy (pip, always available)
  • Python 3.10 stdlib

No remote assets, no Nucleus calls, no file I/O, no blocking operations.

Coordinate system: Z-up (Isaac Sim default).
Units: metres.

What gets built
---------------
1. Procedural heightmap mesh  — /World/MarsTerrain
2. Mars red-oxide material    — /World/Looks/MarsOxide
3. Three rock materials       — /World/Looks/Basalt
                                /World/Looks/Sandstone
                                /World/Looks/IronRich
4. Martian lighting           — /World/SunLight   (DistantLight, low angle)
                                /World/SkyDome    (DomeLight, pink sky)
5. 33 rocks (7 boulders, 14 medium, 12 pebbles) — /World/Rocks/*
   Each rock is an irregular cube-ish mesh (not a sphere) with random scale
   per axis, so they look hand-placed.  All positioned > 0 m in front of
   camera (positive-X half of the scene) and within the 90° FOV cone.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np

# pxr is always available inside Isaac Sim — no guard needed.
from pxr import (
    Gf,
    Sdf,
    Usd,
    UsdGeom,
    UsdLux,
    UsdShade,
    Vt,
)

# ── Reproducible randomness (change seed for a different layout) ──────────────
_RNG = random.Random(42)
_NP_RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _define_xform(stage: Usd.Stage, path: str) -> UsdGeom.Xform:
    """Create or overwrite an Xform prim and return the typed schema."""
    prim = stage.DefinePrim(path, "Xform")
    return UsdGeom.Xform(prim)


def _set_translate(xformable: UsdGeom.Xformable,
                   pos: Tuple[float, float, float]) -> None:
    ops = xformable.GetOrderedXformOps()
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*pos))
            return
    xformable.AddTranslateOp().Set(Gf.Vec3d(*pos))


def _set_scale(xformable: UsdGeom.Xformable,
               s: Tuple[float, float, float]) -> None:
    ops = xformable.GetOrderedXformOps()
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            op.Set(Gf.Vec3f(*s))
            return
    xformable.AddScaleOp().Set(Gf.Vec3f(*s))


def _set_rotate_xyz(xformable: UsdGeom.Xformable,
                    deg: Tuple[float, float, float]) -> None:
    ops = xformable.GetOrderedXformOps()
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            op.Set(Gf.Vec3f(*deg))
            return
    xformable.AddRotateXYZOp().Set(Gf.Vec3f(*deg))


def _bind_material(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI(prim).Bind(material)


# ---------------------------------------------------------------------------
# 1.  Procedural Mars heightmap mesh
# ---------------------------------------------------------------------------

def _fractal_heightmap(nx: int, ny: int,
                        amplitude: float = 0.25,
                        base_freq: float = 1.0,
                        octaves: int = 6,
                        rng: np.random.Generator = _NP_RNG) -> np.ndarray:
    """
    Return an (ny, nx) float32 heightmap built from layered sine/cosine noise.
    No scipy, no perlin libs — pure numpy.  Looks convincingly rocky.
    """
    h = np.zeros((ny, nx), dtype=np.float32)
    xs = np.linspace(0, 1, nx, endpoint=False)
    ys = np.linspace(0, 1, ny, endpoint=False)
    xx, yy = np.meshgrid(xs, ys)
    amp = amplitude
    freq = base_freq
    for _ in range(octaves):
        phase_x = rng.uniform(0, 2 * math.pi)
        phase_y = rng.uniform(0, 2 * math.pi)
        angle   = rng.uniform(0, 2 * math.pi)
        # rotate the frequency direction for variety
        u = xx * math.cos(angle) - yy * math.sin(angle)
        v = xx * math.sin(angle) + yy * math.cos(angle)
        h += amp * (
            0.5 * np.sin(2 * math.pi * freq * u + phase_x)
            + 0.5 * np.cos(2 * math.pi * freq * v + phase_y)
        )
        amp  *= 0.5
        freq *= 2.2
    # Normalise to [0, amplitude]
    h -= h.min()
    max_h = h.max()
    if max_h > 1e-6:
        h = h / max_h * amplitude
    return h


def build_terrain(stage: Usd.Stage,
                  prim_path: str = "/World/MarsTerrain",
                  width: float  = 20.0,
                  depth: float  = 20.0,
                  nx: int       = 64,
                  ny: int       = 64,
                  amplitude: float = 0.30) -> UsdGeom.Mesh:
    """
    Create a subdivided flat mesh displaced by a fractal heightmap.

    The mesh spans [-width/2 .. width/2] in X and [-depth/2 .. depth/2] in Y
    (Z-up), centred at the origin.  Height varies 0 .. amplitude metres.

    Returns the UsdGeom.Mesh typed schema.
    """
    heightmap = _fractal_heightmap(nx, ny, amplitude=amplitude)

    # Build vertex positions ─────────────────────────────────────────────────
    xs = np.linspace(-width / 2, width / 2,  nx)
    ys = np.linspace(-depth / 2, depth / 2,  ny)
    xx, yy = np.meshgrid(xs, ys)          # (ny, nx)
    zz = heightmap                         # Z = up

    points_np = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)  # (N,3)

    # Build face indices (quads → triangles) ──────────────────────────────────
    # Vertex index at grid (row=r, col=c): r*nx + c
    face_vert_counts: List[int] = []
    face_vert_indices: List[int] = []
    for r in range(ny - 1):
        for c in range(nx - 1):
            i00 = r       * nx + c
            i10 = (r + 1) * nx + c
            i01 = r       * nx + (c + 1)
            i11 = (r + 1) * nx + (c + 1)
            # Triangle 1
            face_vert_counts.append(3)
            face_vert_indices.extend([i00, i10, i11])
            # Triangle 2
            face_vert_counts.append(3)
            face_vert_indices.extend([i00, i11, i01])

    # Create the USD Mesh prim ─────────────────────────────────────────────────
    mesh_prim = stage.DefinePrim(prim_path, "Mesh")
    mesh      = UsdGeom.Mesh(mesh_prim)

    mesh.CreatePointsAttr(
        Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))
                       for p in points_np])
    )
    mesh.CreateFaceVertexCountsAttr(face_vert_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vert_indices)
    mesh.CreateSubdivisionSchemeAttr("none")   # flat shading, faster physics

    # Flat normals (let USD compute them) ────────────────────────────────────
    mesh.CreateNormalsAttr(
        Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * len(points_np))
    )
    mesh.SetNormalsInterpolation("vertex")

    # Orient up correctly ─────────────────────────────────────────────────────
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Assign collision approximation ─────────────────────────────────────────
    try:
        from pxr import PhysxSchema, UsdPhysics
        UsdPhysics.CollisionAPI.Apply(mesh_prim)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mesh_collision.CreateApproximationAttr("meshSimplification")
    except Exception:
        pass   # Physics not critical for visual validation

    print(f"[mars_terrain_builder] Terrain mesh: {(ny-1)*(nx-1)*2} triangles "
          f"at {prim_path}")
    return mesh


# ---------------------------------------------------------------------------
# 2.  USD Preview Surface materials (no MDL, no remote shaders)
# ---------------------------------------------------------------------------

def _make_preview_surface(stage:     Usd.Stage,
                           mat_path: str,
                           diffuse:  Tuple[float, float, float],
                           roughness: float = 0.85,
                           metallic:  float = 0.0) -> UsdShade.Material:
    """Create a UsdPreviewSurface material with given diffuse colour."""
    mat = UsdShade.Material.Define(stage, mat_path)
    shader_path = mat_path + "/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")

    shader.CreateInput("diffuseColor",  Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*diffuse))
    shader.CreateInput("roughness",     Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic",      Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("opacity",       Sdf.ValueTypeNames.Float).Set(1.0)

    mat.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface")
    return mat


def build_materials(stage: Usd.Stage) -> dict:
    """
    Create all four scene materials.  Returns a dict:
      {
        "mars_oxide":  UsdShade.Material,
        "basalt":      UsdShade.Material,
        "sandstone":   UsdShade.Material,
        "iron_rich":   UsdShade.Material,
      }
    """
    looks_path = "/World/Looks"
    stage.DefinePrim(looks_path, "Scope")

    # Mars regolith — dull red-oxide
    mars_oxide = _make_preview_surface(
        stage,
        f"{looks_path}/MarsOxide",
        diffuse   = (0.58, 0.22, 0.09),
        roughness = 0.92,
        metallic  = 0.02,
    )

    # Basalt — very dark grey, slight sheen
    basalt = _make_preview_surface(
        stage,
        f"{looks_path}/Basalt",
        diffuse   = (0.18, 0.15, 0.13),
        roughness = 0.80,
        metallic  = 0.05,
    )

    # Sandstone — warm tan
    sandstone = _make_preview_surface(
        stage,
        f"{looks_path}/Sandstone",
        diffuse   = (0.72, 0.52, 0.30),
        roughness = 0.95,
        metallic  = 0.0,
    )

    # Iron-rich — rusty orange with slight metallic sheen
    iron_rich = _make_preview_surface(
        stage,
        f"{looks_path}/IronRich",
        diffuse   = (0.68, 0.28, 0.05),
        roughness = 0.75,
        metallic  = 0.15,
    )

    print("[mars_terrain_builder] 4 materials created (UsdPreviewSurface, no MDL).")
    return {
        "mars_oxide": mars_oxide,
        "basalt":     basalt,
        "sandstone":  sandstone,
        "iron_rich":  iron_rich,
    }


# ---------------------------------------------------------------------------
# 3.  Irregular rock mesh  (not a sphere — looks hand-placed)
# ---------------------------------------------------------------------------

def _make_irregular_box_mesh(
    stage:     Usd.Stage,
    prim_path: str,
    half_extents: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    jitter:    float = 0.35,
    rng:       random.Random = _RNG,
) -> UsdGeom.Mesh:
    """
    Build a 6-face (12 tri) box whose vertices are randomly displaced inward/
    outward by `jitter * min(half_extents)` to break the brick-like regularity.

    half_extents controls the base shape before per-axis scaling is applied
    by the caller's xform.
    """
    hx, hy, hz = half_extents
    j = jitter * min(half_extents)

    def jv(v: Gf.Vec3f) -> Gf.Vec3f:
        return Gf.Vec3f(
            v[0] + rng.uniform(-j, j),
            v[1] + rng.uniform(-j, j),
            v[2] + rng.uniform(-j, j),
        )

    # 8 corners of a box, jittered
    pts = [
        jv(Gf.Vec3f(-hx, -hy, -hz)),  # 0 bottom
        jv(Gf.Vec3f( hx, -hy, -hz)),  # 1
        jv(Gf.Vec3f( hx,  hy, -hz)),  # 2
        jv(Gf.Vec3f(-hx,  hy, -hz)),  # 3
        jv(Gf.Vec3f(-hx, -hy,  hz)),  # 4 top
        jv(Gf.Vec3f( hx, -hy,  hz)),  # 5
        jv(Gf.Vec3f( hx,  hy,  hz)),  # 6
        jv(Gf.Vec3f(-hx,  hy,  hz)),  # 7
    ]

    # 12 triangles (2 per face × 6 faces)
    tris = [
        # bottom (-Z)
        (0, 2, 1), (0, 3, 2),
        # top (+Z)
        (4, 5, 6), (4, 6, 7),
        # front (-Y)
        (0, 1, 5), (0, 5, 4),
        # back (+Y)
        (2, 3, 7), (2, 7, 6),
        # left (-X)
        (0, 4, 7), (0, 7, 3),
        # right (+X)
        (1, 2, 6), (1, 6, 5),
    ]

    mesh_prim = stage.DefinePrim(prim_path, "Mesh")
    mesh      = UsdGeom.Mesh(mesh_prim)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr([3] * 12)
    mesh.CreateFaceVertexIndicesAttr([i for tri in tris for i in tri])
    mesh.CreateSubdivisionSchemeAttr("none")

    return mesh


# ---------------------------------------------------------------------------
# 4.  Place 33 rocks
# ---------------------------------------------------------------------------

# Camera sits at (0, 0, 0.3) facing +X with a 15° nose-down pitch.
# "Visible in front" means:
#   • X > 0.5  (in front of camera, not behind)
#   • |Y/X| < tan(45°) = 1.0  (within ±45° horizontal FOV — generous)
#   • Z height set so rock sits on (or slightly above) the nominal ground plane

_ROCK_MATERIAL_KEYS = ["basalt", "sandstone", "iron_rich"]

# (category, count, base_half_extent_range, scale_range_per_axis)
_ROCK_CATEGORIES = [
    # boulders: 7, large
    ("boulder",  7,  (0.30, 0.55), (0.6, 1.4)),
    # medium rocks: 14
    ("medium",  14,  (0.12, 0.28), (0.5, 1.6)),
    # pebbles: 12, small
    ("pebble",  12,  (0.04, 0.10), (0.4, 1.8)),
]


def _visible_position(
    x_range: Tuple[float, float],
    y_spread: float,
    rng: random.Random,
) -> Tuple[float, float]:
    """
    Return (x, y) such that the rock is inside the camera FOV cone.
    x_range : (min_x, max_x) distance in front of camera.
    y_spread: max |y| allowed at the sampled x.
    """
    x = rng.uniform(*x_range)
    # Keep within ±45° horizontal cone from camera
    max_y = min(y_spread, x * math.tan(math.radians(44)))
    y = rng.uniform(-max_y, max_y)
    return x, y


def build_rocks(stage: Usd.Stage,
                materials: dict,
                rocks_root: str = "/World/Rocks",
                rng: random.Random = _RNG) -> List[str]:
    """
    Create 33 irregular rock meshes with random materials and placements.
    All rocks are visible in front of the camera.

    Returns list of prim paths created.
    """
    stage.DefinePrim(rocks_root, "Scope")
    paths_created: List[str] = []
    rock_idx = 0

    mat_keys = _ROCK_MATERIAL_KEYS

    for category, count, he_range, scale_range in _ROCK_CATEGORIES:
        for _ in range(count):
            # ── position ────────────────────────────────────────────────────
            if category == "boulder":
                x, y = _visible_position((1.5, 7.0), 5.0, rng)
            elif category == "medium":
                x, y = _visible_position((0.8, 6.0), 5.0, rng)
            else:  # pebble
                x, y = _visible_position((0.5, 4.0), 3.5, rng)

            # ── size ────────────────────────────────────────────────────────
            base_he = rng.uniform(*he_range)
            sx = base_he * rng.uniform(*scale_range)
            sy = base_he * rng.uniform(*scale_range)
            sz = base_he * rng.uniform(*scale_range)
            # Rock sits on the ground: Z offset = half-height + tiny terrain bump
            z = sz + rng.uniform(0.0, 0.04)

            # ── irregular mesh ──────────────────────────────────────────────
            prim_path = f"{rocks_root}/Rock_{rock_idx:03d}_{category}"
            mesh = _make_irregular_box_mesh(
                stage,
                prim_path,
                half_extents=(0.5, 0.5, 0.5),  # unit box; xform does scaling
                jitter=0.30,
                rng=rng,
            )

            # ── xform ───────────────────────────────────────────────────────
            xformable = UsdGeom.Xformable(mesh.GetPrim())
            _set_translate(xformable, (x, y, z))
            _set_scale(xformable, (sx, sy, sz))
            # Random yaw rotation so flat faces are not all aligned
            yaw = rng.uniform(0, 360)
            roll = rng.uniform(-25, 25)
            _set_rotate_xyz(xformable, (roll, 0.0, yaw))

            # ── material ────────────────────────────────────────────────────
            mat_key = rng.choice(mat_keys)
            _bind_material(mesh.GetPrim(), materials[mat_key])

            paths_created.append(prim_path)
            rock_idx += 1

    boulder_count = _ROCK_CATEGORIES[0][1]
    medium_count  = _ROCK_CATEGORIES[1][1]
    pebble_count  = _ROCK_CATEGORIES[2][1]
    print(
        f"[mars_terrain_builder] {len(paths_created)} rocks placed: "
        f"{boulder_count} boulders, {medium_count} medium, {pebble_count} pebbles."
    )
    return paths_created


# ---------------------------------------------------------------------------
# 5.  Martian lighting
# ---------------------------------------------------------------------------

def build_lighting(stage: Usd.Stage) -> None:
    """
    Add two lights:
      • SunLight  — DistantLight at ~30° elevation from the south-west.
                    Warm amber colour (Mars Sun is dimmer and more orange).
      • SkyDome   — DomeLight with very low intensity and a pink/peach colour
                    to simulate the Martian dusty atmosphere scattering.

    Any existing lights at these paths are overwritten so reruns are safe.
    """
    # ── Remove stale prims if they exist ────────────────────────────────────
    for path in ("/World/SunLight", "/World/SkyDome"):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            stage.RemovePrim(path)

    # ── Distant sun ─────────────────────────────────────────────────────────
    sun_prim = stage.DefinePrim("/World/SunLight", "DistantLight")
    sun = UsdLux.DistantLight(sun_prim)
    sun.CreateIntensityAttr(3500.0)      # dimmer than Earth; Sol at Mars is ~43% Earth
    sun.CreateAngleAttr(0.35)            # angular diameter of Sol from Mars in degrees
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.82, 0.62))   # warm amber-orange

    # Elevation ~28° above horizon, azimuth from SW
    # In Z-up: rotateXYZ(-62, 0, -45) tilts a -Z-facing light 28° above horizon
    xf_sun = UsdGeom.Xformable(sun_prim)
    xf_sun.AddRotateXYZOp().Set(Gf.Vec3f(-62.0, 0.0, -45.0))

    # Enable shadows
    try:
        from pxr import UsdLux as _ul
        shadow_api = _ul.ShadowAPI.Apply(sun_prim)
        shadow_api.CreateShadowEnableAttr(True)
        shadow_api.CreateShadowColorAttr(Gf.Vec3f(0.15, 0.08, 0.06))
    except Exception:
        pass

    # ── Pink/dust sky dome ───────────────────────────────────────────────────
    dome_prim = stage.DefinePrim("/World/SkyDome", "DomeLight")
    dome = UsdLux.DomeLight(dome_prim)
    # No texture (no remote assets): solid colour via the colour attribute.
    dome.CreateIntensityAttr(600.0)
    dome.CreateColorAttr(Gf.Vec3f(0.82, 0.55, 0.42))   # dusty pink-peach sky
    # Exposure nudge
    try:
        dome.CreateExposureAttr(-0.5)
    except Exception:
        pass

    print("[mars_terrain_builder] Lighting: SunLight (DistantLight 3500, amber) + "
          "SkyDome (DomeLight 600, pink-peach).")


# ---------------------------------------------------------------------------
# 6.  Master entry point
# ---------------------------------------------------------------------------

def build_mars_scene(
    stage:         Usd.Stage,
    terrain_path:  str   = "/World/MarsTerrain",
    rocks_root:    str   = "/World/Rocks",
    terrain_width: float = 20.0,
    terrain_depth: float = 20.0,
    terrain_nx:    int   = 64,
    terrain_ny:    int   = 64,
    terrain_amplitude: float = 0.30,
    replace_existing:  bool  = True,
) -> dict:
    """
    One-call entry point.  Call this after SimulationApp has booted.

    Parameters
    ----------
    stage              : Usd.Stage — from omni.usd.get_context().get_stage()
    terrain_path       : USD path for the terrain mesh prim
    rocks_root         : USD scope to place all rock prims under
    terrain_width/depth: extent of the terrain in metres
    terrain_nx/ny      : mesh resolution (64×64 = 4k triangles — fast physics)
    terrain_amplitude  : peak height variation in metres
    replace_existing   : if True, remove pre-existing terrain/rock prims first

    Returns
    -------
    dict with keys:
      "terrain_mesh"  : UsdGeom.Mesh
      "materials"     : dict of UsdShade.Material
      "rock_paths"    : list[str]
    """
    # ── Optionally clear prior terrain / rocks ───────────────────────────────
    if replace_existing:
        for path in (terrain_path, rocks_root):
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                stage.RemovePrim(path)
                print(f"[mars_terrain_builder] Removed existing prim: {path}")

    # ── Build in order ───────────────────────────────────────────────────────
    print("[mars_terrain_builder] Building Mars terrain ...")

    terrain_mesh = build_terrain(
        stage,
        prim_path = terrain_path,
        width     = terrain_width,
        depth     = terrain_depth,
        nx        = terrain_nx,
        ny        = terrain_ny,
        amplitude = terrain_amplitude,
    )

    materials = build_materials(stage)

    # Bind terrain to mars-oxide material
    _bind_material(terrain_mesh.GetPrim(), materials["mars_oxide"])

    rock_paths = build_rocks(
        stage,
        materials,
        rocks_root = rocks_root,
    )

    build_lighting(stage)

    print("[mars_terrain_builder] Scene build complete.")
    return {
        "terrain_mesh": terrain_mesh,
        "materials":    materials,
        "rock_paths":   rock_paths,
    }
