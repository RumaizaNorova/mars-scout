"""
ares_rover_builder.py
=====================
Builds a Perseverance-class Mars rover from scratch in USD/Python.

No external assets, no Nucleus, no URDF imports.
Everything is constructed programmatically using pxr + PhysX APIs.

Rover specs (based on Perseverance / Mars 2020):
  Total mass:       1025 kg
  Chassis:          ~800 kg  (body + electronics)
  Each wheel:       ~20 kg
  Wheel diameter:   0.525 m  (52.5 cm)
  Wheel width:      0.200 m
  Wheelbase:        2.83 m   (front to rear)
  Track width:      2.78 m   (left to right)
  Ground clearance: 0.60 m
  Max speed:        0.14 m/s (Perseverance), 0.5 m/s (sim)
  Mars gravity:     3.72 m/s²

Rocker-bogie suspension:
  Each side has a rocker beam (pivots at chassis).
  Each rocker has a front wheel + bogie sub-assembly.
  The bogie pivots at the end of the rocker and carries
  the middle and rear wheels.
  A differential bar links left/right rockers so the
  chassis stays level — we break this loop by making one
  side passive (zero stiffness, free to rotate).

USD hierarchy:
  /World/Rover                    (Xform, ArticulationRoot)
  /World/Rover/Body               (Mesh, rigid body, ~800 kg)
  /World/Rover/Mast               (Xform)
  /World/Rover/Mast/MastBody      (Mesh, visual only)
  /World/Rover/Mast/MastCamL      (Camera — left NavCam)
  /World/Rover/Mast/MastCamR      (Camera — right NavCam)
  /World/Rover/Mast/ChaseCamera   (Camera — 3rd person follow)
  /World/Rover/RTG                (Mesh, visual only — power source)
  /World/Rover/Arm                (Mesh, visual only — robotic arm)
  /World/Rover/SuspLeft           (Xform)
  /World/Rover/SuspLeft/Rocker    (Mesh, revolute joint to Body)
  /World/Rover/SuspLeft/WheelF    (Mesh, revolute drive joint to Rocker)
  /World/Rover/SuspLeft/Bogie     (Mesh, revolute joint to Rocker)
  /World/Rover/SuspLeft/WheelM    (Mesh, revolute drive joint to Bogie)
  /World/Rover/SuspLeft/WheelR    (Mesh, revolute drive joint to Bogie)
  /World/Rover/SuspRight/...      (mirrors SuspLeft)

Camera topics (via OmniGraph, wired in setup_scene.py):
  /isaac_camera/rgb               MastCam — what rover sees
  /isaac_camera/depth             MastCam depth
  /isaac_camera/chase             Chase camera — 3rd person view

Usage (from setup_scene.py):
  from ares_rover_builder import build_ares_rover
  result = build_ares_rover(stage, root_path="/World/Rover",
                             spawn_pos=(0.0, 0.0, 0.8))
"""

from __future__ import annotations

import math
from typing import Tuple, Optional

import numpy as np

try:
    from pxr import (
        Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt,
        PhysxSchema,
    )
    _PXR_AVAILABLE = True
except ImportError:
    _PXR_AVAILABLE = False


# =============================================================================
# Rover dimensions (metres, kilograms)
# =============================================================================

class RoverDims:
    # Chassis
    BODY_L          = 2.50   # length (X)
    BODY_W          = 2.00   # width  (Y)
    BODY_H          = 0.70   # height (Z)
    BODY_MASS       = 800.0

    # Wheels
    WHEEL_R         = 0.2625  # radius = 52.5 cm diameter / 2
    WHEEL_W         = 0.200   # width
    WHEEL_MASS      = 20.0
    WHEEL_SEGMENTS  = 24      # polygon approximation

    # Suspension geometry
    TRACK_HALF      = 1.39    # half track width (centre to wheel centreline)
    WHEELBASE       = 2.83    # front-to-rear wheel spacing
    ROCKER_LEN      = 1.60    # rocker beam length
    BOGIE_LEN       = 0.90    # bogie beam length
    SUSP_HEIGHT     = 0.60    # pivot height above ground (ground clearance)
    ROCKER_MASS     = 15.0
    BOGIE_MASS      = 10.0

    # Wheel X positions (relative to chassis centre)
    WHEEL_X_FRONT   =  1.20
    WHEEL_X_MID     =  0.00
    WHEEL_X_REAR    = -1.20

    # Mast
    MAST_H          = 1.80    # mast height above chassis top
    MAST_HEAD_FWD   = 0.15    # mast head box centre (visual)
    MAST_CAM_FWD    = 0.50    # camera forward offset — must clear head box (0.35m wide → half=0.175m + margin)

    # Chase camera offset from chassis centre
    CHASE_X         = -4.00   # behind rover
    CHASE_Z         = +2.50   # above rover
    CHASE_PITCH_DEG = 20.0    # nose-down angle

    # RTG (Radioisotope Thermoelectric Generator) — back of rover
    RTG_X           = -1.40
    RTG_Z           = +0.20

    # Drive joint params
    WHEEL_MAX_VEL   = 10.0    # rad/s (~0.5 m/s at wheel radius)
    WHEEL_MAX_FORCE = 500.0   # N·m stall torque per wheel

    # Suspension joint params (passive — just carry load)
    SUSP_STIFFNESS  = 0.0     # free to rotate
    SUSP_DAMPING    = 50.0    # light damping so it doesn't flop
    SUSP_MAX_FORCE  = 5000.0


D = RoverDims  # short alias


# =============================================================================
# USD geometry helpers
# =============================================================================

def _box_mesh(stage, path: str, sx: float, sy: float, sz: float,
              color: Tuple[float, float, float] = (0.4, 0.35, 0.3)) -> UsdGeom.Mesh:
    """Create a simple box mesh centred at origin."""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    pts = np.array([
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
    ], dtype=np.float32)
    # 6 faces × 2 triangles = 12 triangles
    faces = np.array([
        [0,1,2],[0,2,3],  # bottom
        [4,6,5],[4,7,6],  # top
        [0,4,5],[0,5,1],  # front
        [2,6,7],[2,7,3],  # back
        [0,3,7],[0,7,4],  # left
        [1,5,6],[1,6,2],  # right
    ], dtype=np.int32)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(False)
    _apply_color(stage, path, color)
    return mesh


def _cylinder_mesh(stage, path: str, radius: float, height: float,
                   segments: int = 24,
                   color: Tuple[float, float, float] = (0.2, 0.2, 0.2),
                   axis: str = "Y") -> UsdGeom.Mesh:
    """
    Cylinder along the given axis, centred at origin.
    axis="Y" → wheel rolling axis (Y in Isaac Sim Z-up world).
    """
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    cos_a  = np.cos(angles).astype(np.float32)
    sin_a  = np.sin(angles).astype(np.float32)
    hh     = height / 2.0

    if axis == "Y":
        # circle in X-Z plane, height along Y
        top_ring    = np.stack([ cos_a * radius, np.full(segments,  hh), sin_a * radius], axis=-1)
        bot_ring    = np.stack([ cos_a * radius, np.full(segments, -hh), sin_a * radius], axis=-1)
        top_centre  = np.array([[0.0,  hh, 0.0]], dtype=np.float32)
        bot_centre  = np.array([[0.0, -hh, 0.0]], dtype=np.float32)
    else:  # Z
        top_ring    = np.stack([ cos_a * radius, sin_a * radius, np.full(segments,  hh)], axis=-1)
        bot_ring    = np.stack([ cos_a * radius, sin_a * radius, np.full(segments, -hh)], axis=-1)
        top_centre  = np.array([[0.0, 0.0,  hh]], dtype=np.float32)
        bot_centre  = np.array([[0.0, 0.0, -hh]], dtype=np.float32)

    pts = np.concatenate([top_ring, bot_ring, top_centre, bot_centre], axis=0).astype(np.float32)
    top_c_idx = 2 * segments
    bot_c_idx = 2 * segments + 1

    face_verts = []
    face_counts = []
    for i in range(segments):
        j = (i + 1) % segments
        # side quad → 2 triangles
        face_verts += [i, j, segments + j, segments + i]
        face_counts.append(4)
        # top cap
        face_verts += [top_c_idx, i, j]
        face_counts.append(3)
        # bottom cap
        face_verts += [bot_c_idx, segments + j, segments + i]
        face_counts.append(3)

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(face_verts))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(face_counts))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(False)
    _apply_color(stage, path, color)
    return mesh


def _apply_color(stage, path: str, rgb: Tuple[float, float, float]) -> None:
    """Bind a simple flat-colour UsdPreviewSurface material."""
    looks = "/World/Looks"
    if not stage.GetPrimAtPath(looks).IsValid():
        stage.DefinePrim(looks, "Scope")

    safe_name = path.replace("/", "_").strip("_")
    mat_path  = f"{looks}/Mat_{safe_name}"
    if stage.GetPrimAtPath(mat_path).IsValid():
        mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
    else:
        mat = UsdShade.Material.Define(stage, mat_path)
        sh  = UsdShade.Shader.Define(stage, mat_path + "/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.7)
        sh.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.1)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")

    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        UsdShade.MaterialBindingAPI(prim).Bind(mat)


def _xform_at(stage, path: str,
              pos: Tuple[float, float, float] = (0, 0, 0),
              rot_xyz: Tuple[float, float, float] = (0, 0, 0)) -> UsdGeom.Xform:
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if any(r != 0 for r in rot_xyz):
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rot_xyz))
    return xform


# =============================================================================
# Physics helpers
# =============================================================================

def _add_rigid_body(stage, path: str, mass: float,
                    com: Tuple[float, float, float] = (0, 0, 0)) -> None:
    """Apply rigid body + mass APIs to a prim."""
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*com))


def _add_collider_box(stage, path: str,
                      sx: float, sy: float, sz: float) -> None:
    """Apply box collision approximation."""
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.CollisionAPI.Apply(prim)
    PhysxSchema.PhysxCollisionAPI.Apply(prim)
    col = UsdPhysics.MeshCollisionAPI.Apply(prim)
    col.CreateApproximationAttr("boundingCube")


def _add_collider_sphere(stage, path: str, radius: float) -> None:
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.CollisionAPI.Apply(prim)
    PhysxSchema.PhysxCollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("boundingSphere")


def _revolute_joint(stage, path: str,
                    body0_path: str, body1_path: str,
                    axis: str,
                    pos0: Tuple[float, float, float],
                    pos1: Tuple[float, float, float],
                    drive: bool = False,
                    stiffness: float = 0.0,
                    damping: float = 50.0,
                    max_force: float = 500.0,
                    lower_limit: Optional[float] = None,
                    upper_limit: Optional[float] = None) -> UsdPhysics.RevoluteJoint:
    """Create a revolute joint between two bodies."""
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)

    # Body relationships
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])

    # Local frames
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*pos1))
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))

    # Axis
    joint.CreateAxisAttr(axis)

    # Limits
    if lower_limit is not None:
        joint.CreateLowerLimitAttr(float(lower_limit))
    if upper_limit is not None:
        joint.CreateUpperLimitAttr(float(upper_limit))

    # Drive (for wheel motors)
    if drive:
        drive_api = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
        drive_api.CreateTypeAttr("force")
        drive_api.CreateStiffnessAttr(stiffness)
        drive_api.CreateDampingAttr(damping)
        drive_api.CreateMaxForceAttr(max_force)
    else:
        # Passive suspension joint
        drive_api = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
        drive_api.CreateTypeAttr("force")
        drive_api.CreateStiffnessAttr(0.0)
        drive_api.CreateDampingAttr(damping)
        drive_api.CreateMaxForceAttr(max_force)

    return joint


# =============================================================================
# Rover sub-assembly builders
# =============================================================================

def _build_chassis(stage, root: str) -> None:
    """Main body box + articulation root."""
    # Articulation root on the top-level Xform
    rover_prim = stage.GetPrimAtPath(root)
    UsdPhysics.ArticulationRootAPI.Apply(rover_prim)

    # PhysX articulation settings
    physx_art = PhysxSchema.PhysxArticulationAPI.Apply(rover_prim)
    physx_art.CreateEnabledSelfCollisionsAttr(False)
    physx_art.CreateSolverPositionIterationCountAttr(8)   # TGS max safe = 8
    physx_art.CreateSolverVelocityIterationCountAttr(4)   # TGS max safe = 4

    # Body mesh
    body_path = f"{root}/Body"
    _box_mesh(stage, body_path, D.BODY_L, D.BODY_W, D.BODY_H,
              color=(0.45, 0.42, 0.38))  # warm grey — rover aluminium
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(body_path))
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, D.SUSP_HEIGHT + D.BODY_H / 2))
    _add_rigid_body(stage, body_path, D.BODY_MASS,
                    com=(0, 0, D.SUSP_HEIGHT + D.BODY_H / 2))
    _add_collider_box(stage, body_path, D.BODY_L, D.BODY_W, D.BODY_H)


def _build_mast(stage, root: str) -> None:
    """Camera mast with MastCam-Z stereo pair and chase camera."""
    mast_root = f"{root}/Mast"
    body_top_z = D.SUSP_HEIGHT + D.BODY_H

    # Mast base position (front-centre of chassis)
    _xform_at(stage, mast_root,
              pos=(D.BODY_L * 0.25, 0.0, body_top_z))

    # Mast pole
    mast_body = f"{mast_root}/Pole"
    _cylinder_mesh(stage, mast_body, radius=0.06,
                   height=D.MAST_H, axis="Z",
                   color=(0.55, 0.52, 0.48))
    UsdGeom.Xformable(stage.GetPrimAtPath(mast_body)).AddTranslateOp().Set(
        Gf.Vec3d(0, 0, D.MAST_H / 2))

    # Mast head (camera box)
    mast_head = f"{mast_root}/Head"
    _box_mesh(stage, mast_head, 0.35, 0.20, 0.20,
              color=(0.5, 0.48, 0.44))
    UsdGeom.Xformable(stage.GetPrimAtPath(mast_head)).AddTranslateOp().Set(
        Gf.Vec3d(D.MAST_HEAD_FWD, 0, D.MAST_H))

    # MastCam-Z LEFT — science camera, wide-angle setting
    # Spec: Hayes et al. 2021 SSR 217:48, Section 3.2 (Mastcam-Z optical model)
    #   Focal length:        26 mm  (wide end of 26–110 mm zoom)
    #   Sensor:              1648 × 1200 active pixels, 4.34 µm pitch
    #   Sensor width:        1648 × 4.34 µm = 7.153 mm  (Malin 2020 SSR 215:30)
    #   HFOV at 26 mm:       25.8°  (Hayes 2021 Table 1, confirmed Bell 2021 Table 2)
    #   Required aperture:   2 × 26 × tan(12.9°) = 11.91 mm
    #   Stereo baseline:     24.3 cm  (centre-to-centre of L+R lenses)
    #
    # CORRECTION from previous session:
    #   Was: horizontal_aperture_mm=9.37 → FOV=20.4°  (WRONG — off by 5.4°)
    #   Now: horizontal_aperture_mm=11.91 → FOV=25.8° (Hayes 2021 — correct)
    #   The 9.37 mm value had no verified primary-source derivation and is discarded.
    cam_l = f"{mast_root}/MastCamL"
    _define_camera(stage, cam_l,
                   pos=(D.MAST_CAM_FWD, +0.1215, D.MAST_H),
                   pitch_down_deg=15.0,
                   focal_mm=26.0,
                   horizontal_aperture_mm=11.91)

    # MastCam-Z RIGHT — stereo partner (identical optics, +24.3 cm baseline)
    cam_r = f"{mast_root}/MastCamR"
    _define_camera(stage, cam_r,
                   pos=(D.MAST_CAM_FWD, -0.1215, D.MAST_H),
                   pitch_down_deg=15.0,
                   focal_mm=26.0,
                   horizontal_aperture_mm=11.91)

    # ── Chase camera (3rd-person follow) ──────────────────────────────────────
    # Wider FOV (≈80°) for situational awareness; not used for science.
    # focal_mm=12.0, horizontalAperture left at USD default (20.955) → ~88° FOV
    # — appropriately wide for a cinematic 3rd-person view.
    chase_cam = f"{root}/ChaseCamera"
    _define_camera(stage, chase_cam,
                   pos=(D.CHASE_X, 0.0, D.CHASE_Z + D.SUSP_HEIGHT + D.BODY_H / 2),
                   pitch_down_deg=D.CHASE_PITCH_DEG,
                   focal_mm=12.0)   # no aperture override: default ~88° FOV is correct


def _define_camera(
    stage,
    path: str,
    pos: Tuple[float, float, float],
    pitch_down_deg: float,
    focal_mm: float,
    horizontal_aperture_mm: float | None = None,
) -> None:
    """
    Define a Camera prim facing +X with a nose-down pitch.
    Isaac Sim cameras default to -Z (straight down in Z-up world).
    Formula: rotX(90 - pitch_down_deg) then rotZ(-90) → faces +X, pitched down.

    USD camera FOV formula:
      FOV_h = 2 × atan(horizontalAperture / (2 × focalLength))
    Both attributes use the same internal unit (USD "tenths of scene unit" is
    the spec, but in practice Isaac Sim uses the raw ratio — only RATIO matters).

    Mastcam-Z specification (Hayes et al. 2021 SSR 217:48, Table 1):
      Sensor:              1648×1200 active pixels, 4.34 µm pitch
      Sensor width:        1648 × 4.34 µm = 7.153 mm  (Malin 2020 SSR 215:30)
      Focal length (wide): 26 mm
      HFOV at 26 mm:       25.8°  → aperture = 2×26×tan(12.9°) = 11.91 mm  ✓
      NOTE: the value 9.37 mm (→ FOV=20.4°) used in the previous session was
      incorrect (no verified primary source).  11.91 mm is the correct value.

    Parameters
    ----------
    focal_mm                : focal length in camera model units
    horizontal_aperture_mm  : sensor width in SAME units as focal_mm.
                              None → USD default (20.955) → yields FOV ≈ 60°
                              which is appropriate for NavCam / chase camera.
    """
    cam_prim = stage.DefinePrim(path, "Camera")
    xf = UsdGeom.Xformable(cam_prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    rx = 90.0 - pitch_down_deg   # 75° for 15° nose-down
    xf.AddRotateXYZOp().Set(Gf.Vec3f(rx, 0.0, -90.0))
    cam_prim.GetAttribute("focalLength").Set(focal_mm)
    cam_prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.01, 10000.0))

    if horizontal_aperture_mm is not None:
        # Validate: aperture must be positive and smaller than focal length ×2π
        # (max physically meaningful FOV is 180°)
        import math as _math
        if horizontal_aperture_mm <= 0.0:
            raise ValueError(
                f"_define_camera {path}: horizontal_aperture_mm must be positive, "
                f"got {horizontal_aperture_mm}"
            )
        fov_rad = 2.0 * _math.atan(horizontal_aperture_mm / (2.0 * focal_mm))
        if fov_rad >= _math.radians(160.0):
            raise ValueError(
                f"_define_camera {path}: aperture/focal ratio gives FOV {_math.degrees(fov_rad):.1f}° "
                f"(≥160°) — likely wrong unit; "
                f"horizontal_aperture_mm={horizontal_aperture_mm}, focal_mm={focal_mm}. "
                f"Mastcam-Z should be focal=26.0, aperture=11.91 → ~25.8° (Hayes 2021)."
            )
        cam_prim.GetAttribute("horizontalAperture").Set(horizontal_aperture_mm)
        import math as _m2
        print(f"[ares_rover]   {path.split('/')[-1]}: "
              f"focal={focal_mm}mm aperture={horizontal_aperture_mm}mm "
              f"→ FOV={_math.degrees(fov_rad):.1f}°")


def _build_rtg(stage, root: str) -> None:
    """RTG (power source) — visual only, cylindrical, mounted at rear."""
    rtg_path = f"{root}/RTG"
    _cylinder_mesh(stage, rtg_path, radius=0.22, height=0.65, axis="Z",
                   color=(0.3, 0.3, 0.3))
    body_top_z = D.SUSP_HEIGHT + D.BODY_H
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(rtg_path))
    xf.AddTranslateOp().Set(Gf.Vec3d(D.RTG_X, 0.0, body_top_z + D.RTG_Z))
    # Tilt RTG outward at 45° (like Perseverance)
    xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 45.0, 0.0))


def _build_arm(stage, root: str) -> None:
    """Robotic arm — visual only, stowed position (folded down front)."""
    arm_root = f"{root}/Arm"
    body_top_z = D.SUSP_HEIGHT + D.BODY_H
    _xform_at(stage, arm_root,
              pos=(D.BODY_L * 0.45, 0.0, body_top_z - 0.1))

    segments = [
        ("Shoulder", (0.0,  0.0,  0.00), 0.60, 0.05),
        ("Elbow",    (0.0,  0.0, -0.55), 0.50, 0.04),
        ("Wrist",    (0.0,  0.0, -1.00), 0.30, 0.035),
        ("Turret",   (0.0,  0.0, -1.25), 0.18, 0.06),
    ]
    for name, pos, length, r in segments:
        seg_path = f"{arm_root}/{name}"
        _cylinder_mesh(stage, seg_path, radius=r, height=length, axis="Z",
                       color=(0.45, 0.42, 0.38))
        UsdGeom.Xformable(stage.DefinePrim(seg_path)).AddTranslateOp().Set(
            Gf.Vec3d(*pos))


def _build_wheel(stage, path: str,
                 parent_path: str,
                 local_pos: Tuple[float, float, float],
                 side: str) -> None:
    """
    Build one wheel mesh + rigid body + revolute drive joint.

    local_pos is relative to the parent suspension link.
    side: "L" or "R" — mirrors Y offset.
    """
    _cylinder_mesh(stage, path, D.WHEEL_R, D.WHEEL_W,
                   segments=D.WHEEL_SEGMENTS,
                   color=(0.15, 0.14, 0.13))  # dark rubber
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xf.AddTranslateOp().Set(Gf.Vec3d(*local_pos))

    _add_rigid_body(stage, path, D.WHEEL_MASS)
    _add_collider_sphere(stage, path, D.WHEEL_R)

    # Drive joint — wheel spins around Y axis
    joint_path = path + "_DriveJoint"
    # pos0: joint location in parent frame = local_pos
    # pos1: joint location in wheel frame = origin
    _revolute_joint(
        stage, joint_path,
        body0_path=parent_path,
        body1_path=path,
        axis="Y",
        pos0=local_pos,
        pos1=(0.0, 0.0, 0.0),
        drive=True,
        stiffness=0.0,
        damping=D.WHEEL_MAX_FORCE,
        max_force=D.WHEEL_MAX_FORCE,
    )


def _build_suspension_side(stage, root: str, side: str) -> dict:
    """
    Build one side of the rocker-bogie suspension (left or right).

    side: "L" (y > 0) or "R" (y < 0)
    Returns dict of prim paths for joint wiring.
    """
    y_sign  = +1.0 if side == "L" else -1.0
    y_outer = y_sign * D.TRACK_HALF          # wheel Y position
    susp_z  = D.SUSP_HEIGHT                  # rocker pivot height
    body_z  = D.SUSP_HEIGHT + D.BODY_H / 2  # body CoM Z

    susp_root = f"{root}/Susp{side}"
    _xform_at(stage, susp_root)

    # ── Rocker beam ───────────────────────────────────────────────────────────
    rocker_path = f"{susp_root}/Rocker"
    _box_mesh(stage, rocker_path,
              sx=D.ROCKER_LEN, sy=0.08, sz=0.08,
              color=(0.35, 0.33, 0.30))
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(rocker_path))
    # Rocker pivot is at chassis body, centre of rocker at pivot
    rocker_world_x = D.WHEEL_X_FRONT / 2   # roughly midpoint
    xf.AddTranslateOp().Set(Gf.Vec3d(rocker_world_x, y_outer, susp_z))
    _add_rigid_body(stage, rocker_path, D.ROCKER_MASS)

    # Rocker ↔ Body revolute joint (passive, Y axis — pitching)
    rocker_joint = f"{susp_root}/RockerJoint"
    _revolute_joint(
        stage, rocker_joint,
        body0_path=f"{root}/Body",
        body1_path=rocker_path,
        axis="Y",
        pos0=(rocker_world_x, y_outer, 0.0),   # in body local frame
        pos1=(0.0, 0.0, 0.0),
        drive=False,
        damping=D.SUSP_DAMPING,
        max_force=D.SUSP_MAX_FORCE,
        lower_limit=-30.0,
        upper_limit=30.0,
    )

    # ── Front wheel (directly on rocker) ──────────────────────────────────────
    wf_path = f"{susp_root}/WheelF"
    wf_local = (D.ROCKER_LEN / 2, 0.0, -(susp_z - D.WHEEL_R))
    _build_wheel(stage, wf_path, rocker_path, wf_local, side)

    # ── Bogie sub-assembly ────────────────────────────────────────────────────
    bogie_path = f"{susp_root}/Bogie"
    _box_mesh(stage, bogie_path,
              sx=D.BOGIE_LEN, sy=0.07, sz=0.07,
              color=(0.35, 0.33, 0.30))
    # Bogie attaches at rear of rocker
    bogie_local_x = -D.ROCKER_LEN / 2
    xf_b = UsdGeom.Xformable(stage.GetPrimAtPath(bogie_path))
    xf_b.AddTranslateOp().Set(
        Gf.Vec3d(rocker_world_x + bogie_local_x, y_outer, susp_z))
    _add_rigid_body(stage, bogie_path, D.BOGIE_MASS)

    bogie_joint = f"{susp_root}/BogieJoint"
    _revolute_joint(
        stage, bogie_joint,
        body0_path=rocker_path,
        body1_path=bogie_path,
        axis="Y",
        pos0=(bogie_local_x, 0.0, 0.0),  # rear end of rocker
        pos1=(0.0, 0.0, 0.0),
        drive=False,
        damping=D.SUSP_DAMPING,
        max_force=D.SUSP_MAX_FORCE,
        lower_limit=-40.0,
        upper_limit=40.0,
    )

    # ── Mid wheel (front of bogie) ────────────────────────────────────────────
    wm_path = f"{susp_root}/WheelM"
    wm_local = (D.BOGIE_LEN / 2, 0.0, -(susp_z - D.WHEEL_R))
    _build_wheel(stage, wm_path, bogie_path, wm_local, side)

    # ── Rear wheel (rear of bogie) ────────────────────────────────────────────
    wr_path = f"{susp_root}/WheelR"
    wr_local = (-D.BOGIE_LEN / 2, 0.0, -(susp_z - D.WHEEL_R))
    _build_wheel(stage, wr_path, bogie_path, wr_local, side)

    return {
        "rocker": rocker_path,
        "bogie":  bogie_path,
        "wheels": [wf_path, wm_path, wr_path],
    }


# =============================================================================
# Main entry point
# =============================================================================

def build_ares_rover(
    stage,
    root_path:  str   = "/World/Rover",
    spawn_pos:  Tuple[float, float, float] = (0.0, 0.0, 0.0),
    spawn_yaw:  float = 0.0,
) -> dict:
    """
    Build the full ARES rover in the USD stage.

    Parameters
    ----------
    stage       Active Usd.Stage
    root_path   USD path for the rover root Xform
    spawn_pos   World position (x, y, z) — z=0 means terrain surface
    spawn_yaw   Heading in degrees (0 = facing +X)

    Returns
    -------
    dict with keys:
        root_path       str
        wheel_paths     list[str]  — all 6 wheel prim paths
        mastcam_L       str        — left MastCam path
        mastcam_R       str        — right MastCam path
        chase_cam       str        — chase camera path
    """
    if not _PXR_AVAILABLE:
        raise RuntimeError("pxr not available — run inside Isaac Sim")

    # Remove existing rover
    if stage.GetPrimAtPath(root_path).IsValid():
        stage.RemovePrim(root_path)

    # Root xform
    root_xf = UsdGeom.Xform.Define(stage, root_path)
    root_xf.AddTranslateOp().Set(Gf.Vec3d(*spawn_pos))
    if spawn_yaw != 0.0:
        root_xf.AddRotateZOp().Set(float(spawn_yaw))

    # Build all sub-assemblies
    _build_chassis(stage, root_path)
    _build_mast(stage, root_path)
    _build_rtg(stage, root_path)
    _build_arm(stage, root_path)

    susp_l = _build_suspension_side(stage, root_path, "L")
    susp_r = _build_suspension_side(stage, root_path, "R")

    all_wheels = susp_l["wheels"] + susp_r["wheels"]

    print(f"[ares_rover] Built Perseverance-class rover at {root_path}")
    print(f"[ares_rover]   6 wheels, rocker-bogie suspension, stereo MastCam")
    print(f"[ares_rover]   Chase camera at ({D.CHASE_X}m, 0, {D.CHASE_Z}m)")
    print(f"[ares_rover]   Spawn: {spawn_pos}, yaw: {spawn_yaw}°")

    return {
        "root_path":   root_path,
        "wheel_paths": all_wheels,
        "mastcam_L":   f"{root_path}/Mast/MastCamL",
        "mastcam_R":   f"{root_path}/Mast/MastCamR",
        "chase_cam":   f"{root_path}/ChaseCamera",
    }
