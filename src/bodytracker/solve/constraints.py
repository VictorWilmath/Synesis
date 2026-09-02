"""Physical plausibility constraints applied after lifting.

Cheap priors that a single camera cannot supply on its own: limbs do not
stretch, and feet do not sink through the floor. Both failure modes read as
obviously broken in VR even when the average joint error is small.
"""

from __future__ import annotations

import numpy as np

from ..lift.kinematics import enforce_bone_lengths
from ..skeleton import (
    LEFT_ANKLE,
    LEFT_BIG_TOE,
    LEFT_HEEL,
    LEFT_SMALL_TOE,
    RIGHT_ANKLE,
    RIGHT_BIG_TOE,
    RIGHT_HEEL,
    RIGHT_SMALL_TOE,
)

FOOT_JOINTS = (
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_HEEL,
    RIGHT_HEEL,
    LEFT_BIG_TOE,
    RIGHT_BIG_TOE,
    LEFT_SMALL_TOE,
    RIGHT_SMALL_TOE,
)

__all__ = ["FOOT_JOINTS", "clamp_to_floor", "enforce_bone_lengths", "floor_penetration"]


def floor_penetration(xyz: np.ndarray) -> float:
    """How far the lowest foot point sits below the floor, in metres.

    Zero when the subject is on or above the floor. Reported by the eval
    harness as a quality signal, since penetration correlates with depth error.
    """
    lowest = float(np.min(xyz[list(FOOT_JOINTS), 1]))
    return max(0.0, -lowest)


def clamp_to_floor(
    xyz: np.ndarray,
    epsilon: float = 0.03,
    lift_whole_body: bool = True,
) -> np.ndarray:
    """Keep the skeleton from sinking through the floor.

    Two separate corrections:

    A foot point *below* the floor is raised to it, which fixes the common case
    of a slightly over-estimated leg length pushing one foot under.

    If the whole body is below the floor, which happens when the root depth
    estimate is off, the entire skeleton is translated up rather than having
    each foot clamped independently; clamping alone would compress the legs.
    """
    out = np.array(xyz, dtype=np.float64, copy=True)
    foot_indices = list(FOOT_JOINTS)

    if lift_whole_body:
        lowest = float(np.min(out[foot_indices, 1]))
        if lowest < -epsilon:
            out[:, 1] -= lowest

    below = out[foot_indices, 1] < 0.0
    if below.any():
        indices = np.array(foot_indices)[below]
        out[indices, 1] = 0.0

    return out
