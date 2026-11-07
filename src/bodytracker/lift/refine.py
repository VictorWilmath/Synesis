"""Solving every joint's depth at once, for the anchored lifter.

The tree walk in `kinematics` settles each joint from a single bone, and for a
bone seen side-on that rests on ``sqrt(L^2 - l^2)``: a few percent of error in
the assumed length, the focal length or one keypoint becomes tens of
centimetres of depth. The previous frame then holds the wrong answer in
place, because it is the tie-break. On a 640x480 webcam that is enough to put
a foot half a metre from where it is for seconds at a time. Nothing in the
walk knows that feet rest on the floor, or that the hands are where the
controllers are.

Here the walk only provides the starting point. All depths are then solved
together as a small least-squares problem:

- bone lengths are soft, so a body that differs from the prior bends the
  answer a little instead of tipping whole limbs toward the camera
- the headset fixes the head's depth and the controllers the wrists'
- the floor stops the feet from sinking and pulls them gently down onto it
- the torso prefers to be upright, and the ankles to be under the hips

Every joint stays on its camera ray, so the 2D detections are still honoured
exactly; only depth moves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ..skeleton import (
    DEPTH_COPY_JOINTS,
    HEAD,
    HIP,
    LEFT_ANKLE,
    LEFT_BIG_TOE,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_SMALL_TOE,
    NECK,
    NUM_KEYPOINTS,
    RIGHT_ANKLE,
    RIGHT_BIG_TOE,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_SMALL_TOE,
)

_MIN_DEPTH = 0.2
_MAX_DEPTH = 12.0

_SOLVED = np.array([j for j in range(NUM_KEYPOINTS) if j not in DEPTH_COPY_JOINTS])
_COLUMN = {int(j): i for i, j in enumerate(_SOLVED)}

# Resting heights above the floor, as a fraction of standing height, for a
# foot that is planted. Keypoint conventions put the ankle at the malleolus.
_FOOT_REST_RATIO = {
    LEFT_ANKLE: 0.039,
    RIGHT_ANKLE: 0.039,
    LEFT_HEEL: 0.020,
    RIGHT_HEEL: 0.020,
    LEFT_BIG_TOE: 0.004,
    RIGHT_BIG_TOE: 0.004,
    LEFT_SMALL_TOE: 0.004,
    RIGHT_SMALL_TOE: 0.004,
}
_ANKLES = (LEFT_ANKLE, RIGHT_ANKLE)
_LEGS = ((LEFT_HIP, LEFT_ANKLE), (RIGHT_HIP, RIGHT_ANKLE))


@dataclass(slots=True)
class RefineWeights:
    """Standard deviations, in metres or as a fraction of bone length."""

    # How far a bone may differ from the prior: people and keypoint
    # conventions both vary by this much.
    bone_ratio: float = 0.08
    bone_floor_m: float = 0.01
    head_m: float = 0.01
    wrist_m: float = 0.05
    # Points below the floor by more than the slack are pushed back up hard.
    floor_m: float = 0.01
    floor_slack_m: float = 0.03
    # A gentle pull down onto the floor. Weak enough that a lifted foot, which
    # would need its leg stretched to reach the floor, stays lifted.
    ground_m: float = 0.12
    # How far the torso, and each ankle from under its hip, may lean toward or
    # away from the camera, the one direction the image cannot see. The leg
    # is measured hip to ankle rather than bone by bone, because a crouch
    # tips the thigh and shin opposite ways while keeping the foot under the
    # body. Beyond `upright_limit` of these the prior gives up, so a real lean
    # or a raised knee, which the foreshortening in the image does show, is
    # left alone.
    torso_upright_m: float = 0.10
    leg_upright_m: float = 0.08
    upright_limit: float = 2.0
    previous_m: float = 0.25


def refine_depths(
    rays: np.ndarray,
    depths: np.ndarray,
    lengths: dict[tuple[int, int], float],
    *,
    up_camera: np.ndarray,
    camera_height_m: float,
    height_m: float,
    anchor_depths: dict[int, float] | None = None,
    previous: np.ndarray | None = None,
    weights: RefineWeights | None = None,
    max_iterations: int = 30,
) -> np.ndarray:
    """Refine camera-space depths for a skeleton whose rays are fixed.

    `rays` are the back-projected keypoints with z = 1, so a joint at depth
    ``Z`` sits at ``Z * ray``. `depths` is the starting point, normally the
    tree walk's answer. `up_camera` is world up in camera space and
    `camera_height_m` the camera's height above the floor, which together say
    how high any point on any ray is. `anchor_depths` maps joints to depths
    measured by SteamVR; the head and wrists, in practice.

    Returns a (26,) array of depths. Face joints copy the head, as in the walk.
    """
    weights = weights or RefineWeights()
    rays = np.asarray(rays, dtype=np.float64)
    up = np.asarray(up_camera, dtype=np.float64)
    up = up / max(float(np.linalg.norm(up)), 1e-9)
    anchor_depths = anchor_depths or {}

    bones = [(a, b, float(length)) for (a, b), length in lengths.items() if length > 0]
    bone_a = np.array([_COLUMN[a] for a, _, _ in bones])
    bone_b = np.array([_COLUMN[b] for _, b, _ in bones])
    bone_length = np.array([length for _, _, length in bones])
    bone_sigma = weights.bone_ratio * bone_length + weights.bone_floor_m

    solved_rays = rays[_SOLVED]
    # Height above the floor is linear in depth along each ray.
    rise = solved_rays @ up

    feet = np.array([_COLUMN[j] for j in _FOOT_REST_RATIO])
    rest = np.array([_FOOT_REST_RATIO[j] * height_m for j in _FOOT_REST_RATIO])
    ankles = np.array([_COLUMN[j] for j in _ANKLES])
    ankle_rest = np.array([_FOOT_REST_RATIO[j] * height_m for j in _ANKLES])

    anchored = np.array([_COLUMN[j] for j in anchor_depths], dtype=int)
    anchor_target = np.array(list(anchor_depths.values()), dtype=np.float64)
    anchor_sigma = np.array(
        [weights.head_m if j == HEAD else weights.wrist_m for j in anchor_depths]
    )

    upright = [(NECK, HIP, weights.torso_upright_m)] + [
        (a, b, weights.leg_upright_m) for a, b in _LEGS
    ]
    upright_a = np.array([_COLUMN[a] for a, _, _ in upright])
    upright_b = np.array([_COLUMN[b] for _, b, _ in upright])
    upright_sigma = np.array([sigma for _, _, sigma in upright])
    flat = solved_rays - np.outer(solved_rays @ up, up)
    # Lean is only measured along the horizontal line of sight: sideways lean
    # is in the image already and is not the prior's business.
    sight = flat[upright_a] + flat[upright_b]
    sight /= np.maximum(np.linalg.norm(sight, axis=1), 1e-9)[:, None]
    upright_pa = np.einsum("ij,ij->i", flat[upright_a], sight) / upright_sigma
    upright_pb = np.einsum("ij,ij->i", flat[upright_b], sight) / upright_sigma
    limit = weights.upright_limit

    prior = None
    if previous is not None:
        prior = np.asarray(previous, dtype=np.float64)[_SOLVED]
        if not np.isfinite(prior).all():
            prior = None

    n_bones = len(bones)
    n_vars = len(_SOLVED)

    def residuals(z: np.ndarray) -> np.ndarray:
        points = solved_rays * z[:, None]
        gap = points[bone_a] - points[bone_b]
        bone_r = (np.linalg.norm(gap, axis=1) - bone_length) / bone_sigma

        heights = rise * z + camera_height_m
        floor_r = np.minimum(0.0, heights[feet] - (rest - weights.floor_slack_m)) / weights.floor_m
        ground_r = np.maximum(0.0, heights[ankles] - ankle_rest) / weights.ground_m

        anchor_r = (z[anchored] - anchor_target) / anchor_sigma
        lean = upright_pa * z[upright_a] - upright_pb * z[upright_b]
        upright_r = limit * np.tanh(lean / limit)

        parts = [bone_r, floor_r, ground_r, anchor_r, upright_r]
        if prior is not None:
            parts.append((z - prior) / weights.previous_m)
        return np.concatenate(parts)

    def jacobian(z: np.ndarray) -> np.ndarray:
        points = solved_rays * z[:, None]
        gap = points[bone_a] - points[bone_b]
        norm = np.maximum(np.linalg.norm(gap, axis=1), 1e-9)
        direction = gap / norm[:, None]

        rows = []
        bone_j = np.zeros((n_bones, n_vars))
        index = np.arange(n_bones)
        bone_j[index, bone_a] = np.einsum("ij,ij->i", solved_rays[bone_a], direction) / bone_sigma
        bone_j[index, bone_b] = -np.einsum("ij,ij->i", solved_rays[bone_b], direction) / bone_sigma
        rows.append(bone_j)

        heights = rise * z + camera_height_m
        floor_j = np.zeros((len(feet), n_vars))
        below = heights[feet] < rest - weights.floor_slack_m
        floor_j[np.arange(len(feet))[below], feet[below]] = rise[feet[below]] / weights.floor_m
        rows.append(floor_j)

        ground_j = np.zeros((len(ankles), n_vars))
        above = heights[ankles] > ankle_rest
        ground_j[np.arange(len(ankles))[above], ankles[above]] = (
            rise[ankles[above]] / weights.ground_m
        )
        rows.append(ground_j)

        anchor_j = np.zeros((len(anchored), n_vars))
        anchor_j[np.arange(len(anchored)), anchored] = 1.0 / anchor_sigma
        rows.append(anchor_j)

        lean = upright_pa * z[upright_a] - upright_pb * z[upright_b]
        damping = 1.0 / np.cosh(lean / limit) ** 2
        upright_j = np.zeros((len(upright), n_vars))
        index = np.arange(len(upright))
        upright_j[index, upright_a] = damping * upright_pa
        upright_j[index, upright_b] = -damping * upright_pb
        rows.append(upright_j)

        if prior is not None:
            rows.append(np.eye(n_vars) / weights.previous_m)
        return np.vstack(rows)

    start = np.clip(np.asarray(depths, dtype=np.float64)[_SOLVED], _MIN_DEPTH, _MAX_DEPTH)
    # Plain least squares with Levenberg-Marquardt. Robust losses were tried
    # and were five times slower for no measurable gain.
    solution = least_squares(residuals, start, jac=jacobian, method="lm", max_nfev=max_iterations)

    out = np.asarray(depths, dtype=np.float64).copy()
    out[_SOLVED] = np.clip(solution.x, _MIN_DEPTH, _MAX_DEPTH)
    for joint in DEPTH_COPY_JOINTS:
        out[joint] = out[HEAD]
    return out
