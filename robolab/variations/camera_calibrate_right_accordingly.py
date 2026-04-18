# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Camera configs and helpers.

``principal_ray_hit_y_plane`` and the overshoulder pose constants do not require Isaac Lab.
Run ``python -m robolab.variations.camera`` (or ``python robolab/variations/camera.py``) to print
where each camera's principal ray hits ``y=0`` without importing ``isaaclab``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

# --- OverShoulderLeftCameraCfg pose constants (single source of truth; w,x,y,z) ---------------
OVERSHOULDER_EXTERNAL_CAM_POS = (0.05, 0.57, 0.66)
OVERSHOULDER_EXTERNAL_CAM_ROT_WXYZ = (-0.393, -0.195, 0.399, 0.805)

OVERSHOULDER_RIGHT_CAM_POS = (0.05, -0.57, 0.66)


def rot_wxyz_mirror_rotation_across_xz(
    rot_wxyz_left: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Stereo partner rotation **R' = S R S** with **S = diag(1, -1, 1)** (reflect across XZ / swap ±Y).

    ``R`` and ``R'`` are proper rotations. The OpenGL forward **(-Z)** in the parent frame keeps the
    same **tilt relative to the XY plane** (Y-up: angle between the ray and the vertical plane
    spanned by X and Y) and the same **|dip|** from the horizontal XZ plane as the left camera,
    while flipping the parent-frame **Y** component of forward (appropriate for a camera on the
    opposite side). Pair with a position mirrored in **Y** (same **x**, **z**).
    """
    w, x, y, z = rot_wxyz_left
    r = ScipyRotation.from_quat([x, y, z, w])
    s = np.diag([1.0, -1.0, 1.0])
    r_mir = s @ r.as_matrix() @ s
    q = ScipyRotation.from_matrix(r_mir).as_quat()  # x, y, z, w
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def rot_wxyz_compose_opengl_line_of_sight_roll(
    rot_wxyz: tuple[float, float, float, float],
    roll_rad: float,
) -> tuple[float, float, float, float]:
    """Compose **R' = R · R_roll** with **R_roll** a rotation about OpenGL camera **−Z** (optical axis).

    Applied in the camera frame **before** mapping to the parent, so the principal ray
    (**−Z** → parent) is **unchanged**, but **+Y** (image up) rotates around the view axis.
    Use this to fix a stereo partner that matches the left camera’s aim but appears **upside down**
    (180° roll around line of sight).
    """
    w, x, y, z = rot_wxyz
    r = ScipyRotation.from_quat([x, y, z, w])
    axis = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    r_roll = ScipyRotation.from_rotvec(axis * float(roll_rad))
    q = (r * r_roll).as_quat()  # x, y, z, w
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def rot_wxyz_opengl_minus_z_toward(
    from_pos: tuple[float, float, float],
    target_point: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Isaac ``(w, x, y, z)`` rotation: OpenGL forward **-Z** points from ``from_pos`` toward ``target_point``.

    Roll around the view axis is chosen by ``scipy`` (minimum-norm rotation).
    """
    d = np.asarray(target_point, dtype=np.float64) - np.asarray(from_pos, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-12:
        raise ValueError("from_pos and target_point must differ")
    d /= n
    rot, _ = ScipyRotation.align_vectors(d.reshape(1, 3), np.array([[0.0, 0.0, -1.0]]))
    q = rot.as_quat()  # x, y, z, w
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def principal_ray_hit_y_plane(
    pos_parent: tuple[float, float, float],
    rot_wxyz: tuple[float, float, float, float],
    *,
    plane_y: float = 0.0,
) -> tuple[np.ndarray, float] | None:
    """Where the OpenGL camera center pixel ray meets the plane y == plane_y (the XZ plane when plane_y=0).

    Uses Isaac ``OffsetCfg.rot`` order **(w, x, y, z)** and OpenGL camera forward **-Z** in the camera frame.

    **Assumption:** ``pos``/``rot`` are already in the frame you care about (usually parent prim). If the
    camera’s parent is not world-aligned at the origin, compose that transform before calling this.
    """
    w, x, y, z = rot_wxyz
    rot = ScipyRotation.from_quat([x, y, z, w])
    forward_parent = rot.apply(np.array([0.0, 0.0, -1.0], dtype=np.float64))
    eye = np.array(pos_parent, dtype=np.float64)
    fy = float(forward_parent[1])
    if abs(fy) < 1e-9:
        return None
    t = (plane_y - eye[1]) / fy
    hit = eye + t * forward_parent
    return hit, float(t)


# π fixes mirror-only pose leaving the right image rolled 180° vs external (same forward, wrong “up”).
OVERSHOULDER_RIGHT_CAM_ROLL_ABOUT_LOS_RAD = float(np.pi)
OVERSHOULDER_RIGHT_CAM_ROT_WXYZ = rot_wxyz_compose_opengl_line_of_sight_roll(
    rot_wxyz_mirror_rotation_across_xz(OVERSHOULDER_EXTERNAL_CAM_ROT_WXYZ),
    OVERSHOULDER_RIGHT_CAM_ROLL_ABOUT_LOS_RAD,
)


if __name__ == "__main__":
    def _forward_parent(rot_wxyz: tuple[float, float, float, float]) -> np.ndarray:
        w, x, y, z = rot_wxyz
        return ScipyRotation.from_quat([x, y, z, w]).apply(np.array([0.0, 0.0, -1.0]))

    def _tilt_deg_xy_plane_y_up(forward: np.ndarray) -> float:
        # Y-up: XY is z=0. Signed angle in (−90°, 90°): + if forward has +Z, − if −Z.
        return float(np.degrees(np.arctan2(forward[2], np.hypot(forward[0], forward[1]))))

    demos = [
        ("external_cam (left / base)", OVERSHOULDER_EXTERNAL_CAM_POS, OVERSHOULDER_EXTERNAL_CAM_ROT_WXYZ),
        ("right_cam", OVERSHOULDER_RIGHT_CAM_POS, OVERSHOULDER_RIGHT_CAM_ROT_WXYZ),
    ]
    for label, pos, rot in demos:
        out = principal_ray_hit_y_plane(pos, rot, plane_y=0.0)
        f = _forward_parent(rot)
        print(f"{label}: pos={pos} rot(wxyz)={rot}")
        print(f"  signed tilt vs XY plane (Y-up, deg): {_tilt_deg_xy_plane_y_up(f):.4f}")
        if out is None:
            print("  principal ray parallel to y=0 — no single intersection")
        else:
            hit, t = out
            print(
                f"  principal ray ∩ y=0: ({hit[0]:.4f}, {hit[1]:.4f}, {hit[2]:.4f})  (t={t:.4f})"
            )
    raise SystemExit(0)


# --- Isaac Lab (optional for normal imports from sim / env factory) ---------------------------
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass


@configclass
class OverShoulderLeftCameraCfg:
    """Over-shoulder stereo pair + policy naming: ``external_cam`` is the primary base view (Droid
    ``exterior_image_1_left``); ``right_cam`` is the reflected second view for
    stacks that expect ``observation/left_image`` and ``observation/right_image``."""

    external_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=OVERSHOULDER_EXTERNAL_CAM_POS,
            rot=OVERSHOULDER_EXTERNAL_CAM_ROT_WXYZ,
            convention="opengl",
        ),
    )

    # Mirrored rotation S R S plus π roll about line of sight so RGB matches external_cam orientation.
    right_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/right_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=OVERSHOULDER_RIGHT_CAM_POS,
            rot=OVERSHOULDER_RIGHT_CAM_ROT_WXYZ,
            convention="opengl",
        ),
    )

################################################################################
# Egocentric cameras
################################################################################
@configclass
class EgocentricWideAngleCameraCfg:
    egocentric_wide_angle_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/egocentric_wide_angle_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.5), rot=(0.653, 0.271, -0.271, -0.653), convention="opengl"
        ),
    )


################################################################################
# Egocentric mirrored, means the camera is looking at the robot from the front,
# Assuming the robot is at origin.
################################################################################
@configclass
class EgocentricMirroredWideAngleHighCameraCfg:
   egocentric_mirrored_wide_angle_high_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/egocentric_mirrored_wide_angle_high_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.9, 0, 1), rot=(0.653, 0.271, 0.271, 0.653), convention="opengl"
        ),
    )

@configclass
class EgocentricMirroredWideAngleCameraCfg:
   egocentric_mirrored_wide_angle_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/egocentric_mirrored_wide_angle_camera",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.9, 0, 0.5), rot=(0.653, 0.271, 0.271, 0.653), convention="opengl"
        ),
    )

@configclass
class EgocentricMirroredCameraCfg:
   egocentric_mirrored_camera = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/egocentric_mirrored_camera",
    # height=720,
    # width=1280,
    height = 480,
    width = 864,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        vertical_aperture=15.29,
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(1.5, 0.0, 1.0),
        rot=(0.653, 0.271, 0.271, 0.653),
        convention="opengl"
    ),
)
