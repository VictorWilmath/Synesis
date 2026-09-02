"""Conversions between the three coordinate conventions in play.

===================  ==========  =====  =========
Space                Handedness  Up     Forward
===================  ==========  =====  =========
OpenCV camera        right       -y     +z
SteamVR play         right       +y     -z
Unity / VRChat       left        +y     +z
===================  ==========  =====  =========

Camera and play space are both right-handed, so moving between them is a plain
rigid transform with no sign games. Play and Unity differ only in the sign of
the forward axis, so that conversion is a single reflection.

Because SteamVR play space *is* VRChat's tracking space, everything this
pipeline produces is already correctly located: the only step before the wire
is the handedness flip. That is the payoff of anchoring on the HMD instead of
estimating a free-floating skeleton in camera space.
"""

from __future__ import annotations

import numpy as np

from .rotations import orthonormalize

# Play (right-handed, -z forward) to Unity (left-handed, +z forward).
# This reflection is its own inverse.
_FLIP_Z = np.diag([1.0, 1.0, -1.0])


def unity_position_from_play(position: np.ndarray) -> np.ndarray:
    """Convert a play-space point to Unity/VRChat space (metres)."""
    return np.asarray(position, dtype=np.float64) * np.array([1.0, 1.0, -1.0])


def unity_rotation_from_play(rotation: np.ndarray) -> np.ndarray:
    """Convert a play-space rotation to Unity/VRChat space.

    A change of basis by a reflection S is a similarity transform, ``S R S``,
    which preserves the determinant and therefore still yields a rotation.
    Naively negating a column or a row here instead is the classic way to end
    up with a mirrored avatar.
    """
    rot = np.asarray(rotation, dtype=np.float64)
    return _FLIP_Z @ rot @ _FLIP_Z


def play_from_steamvr_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an OpenVR ``HmdMatrix34_t`` into position and rotation.

    SteamVR's standing universe already *is* our play space, so this is a
    reshape rather than a conversion. Returns ``(position, rotation)``.
    """
    mat = np.asarray(matrix, dtype=np.float64).reshape(3, 4)
    return mat[:, 3].copy(), mat[:, :3].copy()


def play_from_camera(
    points_camera: np.ndarray,
    rotation_cw: np.ndarray,
    translation_cw: np.ndarray,
) -> np.ndarray:
    """Transform camera-space points into play space.

    `rotation_cw` and `translation_cw` are the world-to-camera extrinsics as
    produced by ``cv2.solvePnP`` (its rvec converted to a matrix, and its
    tvec), so that ``x_camera = R @ x_play + t``. This inverts that.

    Accepts a single ``(3,)`` point or an ``(N, 3)`` array.
    """
    pts = np.atleast_2d(np.asarray(points_camera, dtype=np.float64))
    rot = np.asarray(rotation_cw, dtype=np.float64)
    trans = np.asarray(translation_cw, dtype=np.float64).reshape(3)
    out = (pts - trans) @ rot
    return out.reshape(np.shape(points_camera))


def camera_from_play(
    points_play: np.ndarray,
    rotation_cw: np.ndarray,
    translation_cw: np.ndarray,
) -> np.ndarray:
    """Transform play-space points into camera space. Inverse of `play_from_camera`."""
    pts = np.atleast_2d(np.asarray(points_play, dtype=np.float64))
    rot = np.asarray(rotation_cw, dtype=np.float64)
    trans = np.asarray(translation_cw, dtype=np.float64).reshape(3)
    out = pts @ rot.T + trans
    return out.reshape(np.shape(points_play))


def project_points(
    points_camera: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Pinhole-project camera-space points to pixels.

    Distortion is deliberately not applied here; the calibrated path uses
    ``cv2.projectPoints`` with the full distortion model. This exists for the
    uncalibrated fallback and for tests, where a clean pinhole is what we want.
    """
    pts = np.atleast_2d(np.asarray(points_camera, dtype=np.float64))
    depth = np.clip(pts[:, 2:3], 1e-6, None)
    normalized = pts[:, :2] / depth
    mat = np.asarray(intrinsics, dtype=np.float64)
    pixels = normalized @ mat[:2, :2].T + mat[:2, 2]
    if np.ndim(points_camera) == 1:
        return pixels.reshape(2)
    return pixels


def look_at_extrinsics(
    camera_position: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build world-to-camera extrinsics for a camera at a point, facing a target.

    Only used to synthesise plausible camera placements for tests and for the
    OSC smoke tool. Returns ``(rotation_cw, translation_cw)`` matching the
    convention `play_from_camera` expects.
    """
    eye = np.asarray(camera_position, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    world_up = np.array([0.0, 1.0, 0.0]) if up is None else np.asarray(up, dtype=np.float64)

    forward = tgt - eye
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)

    # OpenCV camera space has +y pointing down.
    down = np.cross(forward, right)

    rotation_cw = orthonormalize(np.vstack((right, down, forward)))
    translation_cw = -rotation_cw @ eye
    return rotation_cw, translation_cw


def intrinsics_from_fov(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Pinhole intrinsics from a horizontal field of view.

    The fallback used before a checkerboard calibration exists. Most consumer
    webcams sit between 55 and 78 degrees horizontally.
    """
    focal = (width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
