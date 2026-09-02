"""Rotation representations and conversions.

Rotation matrices are 3x3, column-vector convention (``v' = R @ v``), and
right-handed unless a function name says otherwise.

The Euler functions implement **Unity's ZXY convention**, which is what VRChat
expects on ``/tracking/trackers/{n}/rotation``. Unity applies the three
rotations in the order Z, then X, then Y, which composes as::

    R = Ry(y) @ Rx(x) @ Rz(z)

Note that the algebraic form of a basic rotation matrix is the same in
left- and right-handed systems; the handedness lives in the basis, so the
conversion to Unity space is a change of basis (see `spaces.py`), not a
different set of trig formulas here.
"""

from __future__ import annotations

import numpy as np

# Below this, a rotation axis or basis vector is treated as degenerate.
_EPS = 1e-8
# cos(x) below this means the ZXY decomposition is in gimbal lock.
_GIMBAL_EPS = 1e-6


def orthonormalize(matrix: np.ndarray) -> np.ndarray:
    """Snap a nearly-orthonormal matrix back onto SO(3).

    Rotations accumulated from noisy keypoint geometry drift off the manifold;
    left unchecked that drift shows up as shear in the avatar. Uses the SVD
    projection, and flips the sign of the smallest singular direction if the
    input had negative determinant.
    """
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result.astype(np.float64)


def matrix_from_euler_zxy_degrees(x: float, y: float, z: float) -> np.ndarray:
    """Build a rotation matrix from Unity-convention ZXY Euler angles."""
    rx, ry, rz = np.radians([x, y, z])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    mat_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    mat_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    mat_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return mat_y @ mat_x @ mat_z


def euler_zxy_degrees_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Decompose a rotation matrix into Unity-convention ZXY Euler degrees.

    Returns ``(x, y, z)`` in degrees, the order VRChat's OSC endpoint expects.
    """
    mat = np.asarray(matrix, dtype=np.float64)

    # From R = Ry @ Rx @ Rz, element [1][2] is exactly -sin(x).
    sin_x = np.clip(-mat[1, 2], -1.0, 1.0)
    x = np.arcsin(sin_x)
    cos_x = np.cos(x)

    if abs(cos_x) < _GIMBAL_EPS:
        # Gimbal lock: x is +/-90 degrees, so y and z are no longer separable.
        # Fold the whole remaining rotation into y and pin z at zero.
        y = np.arctan2(-mat[2, 0], mat[0, 0])
        z = 0.0
    else:
        y = np.arctan2(mat[0, 2], mat[2, 2])
        z = np.arctan2(mat[1, 0], mat[1, 1])

    return np.degrees([x, y, z]).astype(np.float64)


def look_rotation(forward: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Build a rotation whose local +Z axis points along `forward`.

    Columns of the result are the local x, y, z axes expressed in the parent
    frame, so ``R @ (0, 0, 1) == normalized(forward)``.

    `up` is a hint only; it is re-orthogonalized against `forward`. If the two
    are parallel a fallback axis is chosen so the result stays well-defined.
    """
    fwd = np.asarray(forward, dtype=np.float64)
    norm = np.linalg.norm(fwd)
    if norm < _EPS:
        return np.eye(3)
    fwd = fwd / norm

    hint = np.array([0.0, 1.0, 0.0]) if up is None else np.asarray(up, dtype=np.float64)
    hint_norm = np.linalg.norm(hint)
    hint = hint / hint_norm if hint_norm > _EPS else np.array([0.0, 1.0, 0.0])

    right = np.cross(hint, fwd)
    if np.linalg.norm(right) < _EPS:
        # `up` is parallel to `forward`; any perpendicular axis will do.
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(fwd[0]) > 0.9:
            fallback = np.array([0.0, 0.0, 1.0])
        right = np.cross(fallback, fwd)
    right /= np.linalg.norm(right)

    true_up = np.cross(fwd, right)
    return np.column_stack((right, true_up, fwd))


def rotation_from_forward_up(forward: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Build a play-space rotation for a body part facing `forward`.

    Play space follows the SteamVR/OpenGL convention in which a local frame's
    **-Z** axis is its forward direction, whereas Unity uses +Z. Constructing
    the basis that way means the handedness flip in
    `spaces.unity_rotation_from_play` lands correctly with no further sign
    juggling: a body facing SteamVR-forward comes out as Unity identity.

    Call this rather than `look_rotation` anywhere a body part's facing
    direction is being turned into a tracker orientation.
    """
    return look_rotation(-np.asarray(forward, dtype=np.float64), up)


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a unit quaternion ``(w, x, y, z)``.

    Uses Shepperd's method: pick the branch with the largest denominator so the
    result stays numerically stable near 180-degree rotations.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(mat)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (mat[2, 1] - mat[1, 2]) / s
        y = (mat[0, 2] - mat[2, 0]) / s
        z = (mat[1, 0] - mat[0, 1]) / s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def matrix_from_quaternion(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion ``(w, x, y, z)`` to a rotation matrix."""
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp(rot_a: np.ndarray, rot_b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation between two rotation matrices.

    Used to smooth tracker orientations, where naive per-element interpolation
    of matrices or Euler angles produces visible wobble and can pass through
    non-rotations.
    """
    quat_a = quaternion_from_matrix(rot_a)
    quat_b = quaternion_from_matrix(rot_b)

    dot = float(np.dot(quat_a, quat_b))
    # q and -q are the same rotation; pick the near copy so we take the short way.
    if dot < 0.0:
        quat_b = -quat_b
        dot = -dot

    if dot > 0.9995:
        # Nearly parallel: lerp and renormalize, which avoids dividing by ~0.
        result = quat_a + t * (quat_b - quat_a)
        return matrix_from_quaternion(result / np.linalg.norm(result))

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    sin_theta_0 = np.sin(theta_0)
    scale_a = np.sin(theta_0 - theta) / sin_theta_0
    scale_b = np.sin(theta) / sin_theta_0
    return matrix_from_quaternion(scale_a * quat_a + scale_b * quat_b)


def geodesic_angle_degrees(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    """The single rotation angle separating two orientations.

    The right way to report orientation error: it is one number, it does not
    depend on a Euler convention, and it does not blow up near gimbal lock the
    way a per-axis angle difference does.
    """
    relative = np.asarray(rot_a, dtype=np.float64) @ np.asarray(rot_b, dtype=np.float64).T
    cosine = (np.trace(relative) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
