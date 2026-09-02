"""Turning a 3D skeleton into VRChat tracker targets."""

from __future__ import annotations

import numpy as np

from ..skeleton import ROLE_ANCHOR_KEYPOINT, TrackerRole
from ..types import Skeleton3D, TrackerTarget
from .rotations import derive_rotations


def build_targets(
    skeleton: Skeleton3D,
    roles: list[TrackerRole],
    *,
    min_score: float = 0.3,
) -> list[TrackerTarget]:
    """Build one target per requested role.

    Targets whose anchor keypoint is not confident enough are still returned,
    marked invalid, so the caller can decide between holding the previous value
    and dropping the tracker. Silently omitting them would let a slot go quiet
    mid-session, which VRChat handles worse than a stale pose.
    """
    xyz = np.asarray(skeleton.xyz, dtype=np.float64)
    rotations = derive_rotations(xyz)

    targets: list[TrackerTarget] = []
    for role in roles:
        anchor = ROLE_ANCHOR_KEYPOINT[role]
        targets.append(
            TrackerTarget(
                role=role,
                position=xyz[anchor].copy(),
                rotation=rotations[role],
                valid=bool(skeleton.scores[anchor] >= min_score),
            )
        )
    return targets
