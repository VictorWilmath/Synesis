"""Deriving tracker orientations from joint positions.

Keypoint models predict points, but VRChat wants a full 6DoF pose per tracker.
The orientations therefore have to be reconstructed from the geometry of
neighbouring joints, which is why the Halpe26 layout with its six foot
keypoints was chosen: without toes and heels there is no way to recover foot
yaw or pitch at all.

Everything here works in play space, where the subject facing SteamVR-forward
(-z) with +y up has their right hand toward +x. Orientations are built with
`rotation_from_forward_up`, which handles the -z-forward convention so the
handedness flip to Unity lands correctly.

Every derivation degrades to the pelvis orientation when its own geometry is
degenerate, e.g. a fully straight leg gives no information about which way the
knee points. A slightly wrong orientation beats a NaN.
"""

from __future__ import annotations

import numpy as np

from ..geom import look_rotation, rotation_from_forward_up
from ..skeleton import (
    HIP,
    LEFT_ANKLE,
    LEFT_BIG_TOE,
    LEFT_ELBOW,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_SMALL_TOE,
    LEFT_WRIST,
    NECK,
    RIGHT_ANKLE,
    RIGHT_BIG_TOE,
    RIGHT_ELBOW,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_SMALL_TOE,
    RIGHT_WRIST,
    TrackerRole,
)

_WORLD_UP = np.array([0.0, 1.0, 0.0])
_EPS = 1e-6
# How close to upright a derived sole normal must be before foot roll is
# trusted over a flat-footed assumption.
_FOOT_ROLL_TRUST = 0.6


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < _EPS:
        return None
    return vector / norm


def _basis_from_right_up(right: np.ndarray, up: np.ndarray) -> np.ndarray | None:
    """Orientation from a body segment's right and up vectors.

    Forward is ``cross(up, right)``, which for a subject facing -z with +y up
    and +x to their right yields -z, as it should.
    """
    right_n = _normalize(right)
    up_n = _normalize(up)
    if right_n is None or up_n is None:
        return None
    forward = _normalize(np.cross(up_n, right_n))
    if forward is None:
        # right and up are parallel, so the segment gives no yaw information.
        return None
    return rotation_from_forward_up(forward, up_n)


def pelvis_rotation(xyz: np.ndarray) -> np.ndarray | None:
    """Hip tracker orientation, from the hip line and the spine."""
    right = xyz[RIGHT_HIP] - xyz[LEFT_HIP]
    up = xyz[NECK] - xyz[HIP]
    return _basis_from_right_up(right, up)


def chest_rotation(xyz: np.ndarray) -> np.ndarray | None:
    """Chest tracker orientation, from the shoulder line and the spine.

    Kept separate from the pelvis so that torso twist survives; collapsing the
    two is what makes an avatar's upper body feel welded to its hips.
    """
    right = xyz[RIGHT_SHOULDER] - xyz[LEFT_SHOULDER]
    up = xyz[NECK] - xyz[HIP]
    return _basis_from_right_up(right, up)


def foot_rotation(xyz: np.ndarray, left: bool) -> np.ndarray | None:
    """Foot orientation, from heel to toe with the toe line giving roll."""
    if left:
        heel, big_toe, small_toe, ankle = LEFT_HEEL, LEFT_BIG_TOE, LEFT_SMALL_TOE, LEFT_ANKLE
    else:
        heel, big_toe, small_toe, ankle = RIGHT_HEEL, RIGHT_BIG_TOE, RIGHT_SMALL_TOE, RIGHT_ANKLE

    forward = _normalize(xyz[big_toe] - xyz[heel])
    if forward is None:
        forward = _normalize(xyz[big_toe] - xyz[ankle])
    if forward is None:
        return None

    # The line across the toes spans the foot, so crossing it with forward
    # recovers the sole normal and therefore foot roll.
    across = xyz[small_toe] - xyz[big_toe]
    if not left:
        across = -across
    across_n = _normalize(across)

    up = _WORLD_UP
    if across_n is not None:
        candidate = _normalize(np.cross(forward, across_n))
        if candidate is not None and abs(float(np.dot(candidate, _WORLD_UP))) > _FOOT_ROLL_TRUST:
            # Only trust the derived sole normal when it comes out close to
            # upright. The toe keypoints span a few centimetres, so a couple of
            # centimetres of noise swings this normal wildly, and a bad cross
            # product rolls the foot right over. Falling back to world up keeps
            # the foot flat, which is where it is nearly all the time anyway.
            up = candidate if float(np.dot(candidate, _WORLD_UP)) > 0 else -candidate

    return rotation_from_forward_up(forward, up)


def _joint_bend_rotation(
    xyz: np.ndarray,
    proximal: int,
    joint: int,
    distal: int,
) -> np.ndarray | None:
    """Orientation of a hinge joint, pointing in the direction it bends toward.

    For a knee that is the direction the kneecap faces. The bisector of the two
    limb segments points backward through the joint, so its negation points the
    way the joint opens. A straight limb makes the bisector vanish, which is
    the degenerate case the caller falls back from.
    """
    upper = _normalize(xyz[proximal] - xyz[joint])
    lower = _normalize(xyz[distal] - xyz[joint])
    if upper is None or lower is None:
        return None

    bisector = upper + lower
    forward = _normalize(-bisector)
    if forward is None or float(np.linalg.norm(bisector)) < 0.15:
        # Limb is straight or nearly so; direction of bend is unrecoverable.
        return None

    # The limb axis is the natural up for this joint.
    up = _normalize(xyz[proximal] - xyz[distal])
    if up is None:
        return None
    return rotation_from_forward_up(forward, up)


def knee_rotation(xyz: np.ndarray, left: bool) -> np.ndarray | None:
    if left:
        return _joint_bend_rotation(xyz, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    return _joint_bend_rotation(xyz, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)


def elbow_rotation(xyz: np.ndarray, left: bool) -> np.ndarray | None:
    if left:
        return _joint_bend_rotation(xyz, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
    return _joint_bend_rotation(xyz, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)


def derive_rotations(xyz: np.ndarray) -> dict[TrackerRole, np.ndarray]:
    """Orientations for all eight tracker roles.

    Any derivation that fails falls back to the pelvis, and if even that is
    degenerate, to an upright identity facing SteamVR-forward.
    """
    pelvis = pelvis_rotation(xyz)
    if pelvis is None:
        pelvis = look_rotation(np.array([0.0, 0.0, 1.0]), _WORLD_UP)

    candidates: dict[TrackerRole, np.ndarray | None] = {
        TrackerRole.HIP: pelvis,
        TrackerRole.CHEST: chest_rotation(xyz),
        TrackerRole.LEFT_FOOT: foot_rotation(xyz, left=True),
        TrackerRole.RIGHT_FOOT: foot_rotation(xyz, left=False),
        TrackerRole.LEFT_KNEE: knee_rotation(xyz, left=True),
        TrackerRole.RIGHT_KNEE: knee_rotation(xyz, left=False),
        TrackerRole.LEFT_ELBOW: elbow_rotation(xyz, left=True),
        TrackerRole.RIGHT_ELBOW: elbow_rotation(xyz, left=False),
    }
    return {role: (rot if rot is not None else pelvis) for role, rot in candidates.items()}
