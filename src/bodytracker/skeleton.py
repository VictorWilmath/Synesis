"""Keypoint layout and tracker-slot definitions.

The 2D front-end produces Halpe26 keypoints. That layout is chosen over
COCO-17 because it carries the six foot keypoints (big toe, small toe, heel per
side) needed to derive foot orientation, plus explicit ``head``, ``neck`` and
``hip`` joints that make the torso chain far easier to work with. It is chosen
over the 133-keypoint whole-body layouts because the hands come from the
controllers, so the extra 107 points are pure cost.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

# ---------------------------------------------------------------------------
# Halpe26 keypoint indices
# ---------------------------------------------------------------------------

NOSE: Final = 0
LEFT_EYE: Final = 1
RIGHT_EYE: Final = 2
LEFT_EAR: Final = 3
RIGHT_EAR: Final = 4
LEFT_SHOULDER: Final = 5
RIGHT_SHOULDER: Final = 6
LEFT_ELBOW: Final = 7
RIGHT_ELBOW: Final = 8
LEFT_WRIST: Final = 9
RIGHT_WRIST: Final = 10
LEFT_HIP: Final = 11
RIGHT_HIP: Final = 12
LEFT_KNEE: Final = 13
RIGHT_KNEE: Final = 14
LEFT_ANKLE: Final = 15
RIGHT_ANKLE: Final = 16
HEAD: Final = 17
NECK: Final = 18
HIP: Final = 19
LEFT_BIG_TOE: Final = 20
RIGHT_BIG_TOE: Final = 21
LEFT_SMALL_TOE: Final = 22
RIGHT_SMALL_TOE: Final = 23
LEFT_HEEL: Final = 24
RIGHT_HEEL: Final = 25

NUM_KEYPOINTS: Final = 26

KEYPOINT_NAMES: Final[tuple[str, ...]] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "head",
    "neck",
    "hip",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
)

KEYPOINT_INDEX: Final[dict[str, int]] = {n: i for i, n in enumerate(KEYPOINT_NAMES)}

# Left/right mirror map, used by the horizontal-flip augmentation. Flipping an
# image without also swapping these produces silently corrupt training data.
FLIP_PAIRS: Final[tuple[tuple[int, int], ...]] = (
    (LEFT_EYE, RIGHT_EYE),
    (LEFT_EAR, RIGHT_EAR),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_ELBOW, RIGHT_ELBOW),
    (LEFT_WRIST, RIGHT_WRIST),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_KNEE, RIGHT_KNEE),
    (LEFT_ANKLE, RIGHT_ANKLE),
    (LEFT_BIG_TOE, RIGHT_BIG_TOE),
    (LEFT_SMALL_TOE, RIGHT_SMALL_TOE),
    (LEFT_HEEL, RIGHT_HEEL),
)

# Rigid segments whose length is constant for a given person. The constraint
# solver uses these to reject depth solutions that stretch the body, and the
# body calibration estimates their lengths per user.
BONES: Final[tuple[tuple[int, int], ...]] = (
    (HIP, NECK),
    (NECK, HEAD),
    (NECK, LEFT_SHOULDER),
    (NECK, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (HIP, LEFT_HIP),
    (HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL),
    (LEFT_ANKLE, LEFT_BIG_TOE),
    (LEFT_BIG_TOE, LEFT_SMALL_TOE),
    (RIGHT_ANKLE, RIGHT_HEEL),
    (RIGHT_ANKLE, RIGHT_BIG_TOE),
    (RIGHT_BIG_TOE, RIGHT_SMALL_TOE),
)

# Fraction of total standing height, used as the prior before a per-user body
# calibration exists. Derived from standard anthropometric proportions; only
# needs to be close enough to seed the solver.
BONE_LENGTH_PRIOR_RATIO: Final[dict[tuple[int, int], float]] = {
    (HIP, NECK): 0.288,
    (NECK, HEAD): 0.081,
    (NECK, LEFT_SHOULDER): 0.101,
    (NECK, RIGHT_SHOULDER): 0.101,
    (LEFT_SHOULDER, LEFT_ELBOW): 0.186,
    (LEFT_ELBOW, LEFT_WRIST): 0.146,
    (RIGHT_SHOULDER, RIGHT_ELBOW): 0.186,
    (RIGHT_ELBOW, RIGHT_WRIST): 0.146,
    (HIP, LEFT_HIP): 0.058,
    (HIP, RIGHT_HIP): 0.058,
    (LEFT_HIP, LEFT_KNEE): 0.245,
    (LEFT_KNEE, LEFT_ANKLE): 0.246,
    (RIGHT_HIP, RIGHT_KNEE): 0.245,
    (RIGHT_KNEE, RIGHT_ANKLE): 0.246,
    (LEFT_ANKLE, LEFT_HEEL): 0.032,
    (LEFT_ANKLE, LEFT_BIG_TOE): 0.115,
    (LEFT_BIG_TOE, LEFT_SMALL_TOE): 0.038,
    (RIGHT_ANKLE, RIGHT_HEEL): 0.032,
    (RIGHT_ANKLE, RIGHT_BIG_TOE): 0.115,
    (RIGHT_BIG_TOE, RIGHT_SMALL_TOE): 0.038,
}

# Drawn by the debug overlay.
SKELETON_EDGES: Final[tuple[tuple[int, int], ...]] = BONES


# ---------------------------------------------------------------------------
# Kinematic tree
# ---------------------------------------------------------------------------

ROOT: Final = HIP

# Parent of each joint, indexed by joint. The root has no parent. Depth
# propagation walks this tree outward from the pelvis, which is the most
# reliably detected joint and the one least affected by occlusion.
KINEMATIC_PARENT: Final[tuple[int | None, ...]] = (
    HEAD,  # nose
    HEAD,  # left_eye
    HEAD,  # right_eye
    HEAD,  # left_ear
    HEAD,  # right_ear
    NECK,  # left_shoulder
    NECK,  # right_shoulder
    LEFT_SHOULDER,  # left_elbow
    RIGHT_SHOULDER,  # right_elbow
    LEFT_ELBOW,  # left_wrist
    RIGHT_ELBOW,  # right_wrist
    HIP,  # left_hip
    HIP,  # right_hip
    LEFT_HIP,  # left_knee
    RIGHT_HIP,  # right_knee
    LEFT_KNEE,  # left_ankle
    RIGHT_KNEE,  # right_ankle
    NECK,  # head
    HIP,  # neck
    None,  # hip (root)
    LEFT_ANKLE,  # left_big_toe
    RIGHT_ANKLE,  # right_big_toe
    LEFT_BIG_TOE,  # left_small_toe
    RIGHT_BIG_TOE,  # right_small_toe
    LEFT_ANKLE,  # left_heel
    RIGHT_ANKLE,  # right_heel
)

# Bones that hang along the gravity axis in any upright pose, whichever way the
# subject is facing. Their expected direction is therefore known without
# knowing the body's yaw, which makes gravity a usable tie-break when the depth
# quadratic is ambiguous. See `lift.kinematics.propagate_depths`.
#
# The lateral bones (collarbones, pelvis width, the line across the toes) are
# deliberately absent: their rest direction depends on which way the subject is
# turned, and recovering that turn is part of what we are trying to solve.
VERTICAL_BONES: Final[frozenset[tuple[int, int]]] = frozenset(
    {
        (HIP, NECK),
        (NECK, HEAD),
        (LEFT_SHOULDER, LEFT_ELBOW),
        (RIGHT_SHOULDER, RIGHT_ELBOW),
        (LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_ELBOW, RIGHT_WRIST),
        (LEFT_HIP, LEFT_KNEE),
        (RIGHT_HIP, RIGHT_KNEE),
        (LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_KNEE, RIGHT_ANKLE),
    }
)


# Left/right joints that hang off a common parent and must stay on opposite
# sides of it. Solving them one at a time lets a shoulder end up on the wrong
# side of the neck; solving each pair together rules that out. See
# `lift.kinematics.propagate_depths`.
LATERAL_PAIRS: Final[tuple[tuple[int, int, int], ...]] = (
    (NECK, LEFT_SHOULDER, RIGHT_SHOULDER),
    (HIP, LEFT_HIP, RIGHT_HIP),
)


# Face keypoints inherit their parent's depth rather than solving for it. The
# segments involved are only a few centimetres long, so the bone-length
# quadratic is dominated by keypoint noise and produces wild depths. Nothing
# downstream uses face depth anyway.
DEPTH_COPY_JOINTS: Final[frozenset[int]] = frozenset(
    {NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR}
)


def _topological_order() -> tuple[int, ...]:
    """Joints ordered so every parent precedes its children."""
    order: list[int] = []
    remaining = set(range(NUM_KEYPOINTS))
    placed: set[int] = set()
    while remaining:
        progressed = False
        for joint in sorted(remaining):
            parent = KINEMATIC_PARENT[joint]
            if parent is None or parent in placed:
                order.append(joint)
                placed.add(joint)
                remaining.discard(joint)
                progressed = True
        if not progressed:
            raise RuntimeError("kinematic tree contains a cycle")
    return tuple(order)


TOPOLOGICAL_ORDER: Final[tuple[int, ...]] = _topological_order()


# ---------------------------------------------------------------------------
# VRChat tracker slots
# ---------------------------------------------------------------------------


class TrackerRole(StrEnum):
    """The eight body points VRChat's OSC tracker system accepts.

    VRChat exposes numbered slots 1-8 with no inherent meaning; the role-to-slot
    assignment is ours, and must stay fixed for the whole session because the
    user calibrates against it.
    """

    HIP = "hip"
    CHEST = "chest"
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"


# The keypoint each role is anchored to. Orientation needs more than this one
# point; see `solve.rotations`.
ROLE_ANCHOR_KEYPOINT: Final[dict[TrackerRole, int]] = {
    TrackerRole.HIP: HIP,
    TrackerRole.CHEST: NECK,
    TrackerRole.LEFT_FOOT: LEFT_ANKLE,
    TrackerRole.RIGHT_FOOT: RIGHT_ANKLE,
    TrackerRole.LEFT_KNEE: LEFT_KNEE,
    TrackerRole.RIGHT_KNEE: RIGHT_KNEE,
    TrackerRole.LEFT_ELBOW: LEFT_ELBOW,
    TrackerRole.RIGHT_ELBOW: RIGHT_ELBOW,
}

MAX_TRACKERS: Final = 8


def assign_slots(roles: list[TrackerRole]) -> dict[TrackerRole, int]:
    """Map roles onto VRChat's 1-indexed tracker slots.

    Assignment is by list order and must not change mid-session: VRChat's
    full-body calibration binds each slot to a body point, so reshuffling
    silently swaps the user's limbs.
    """
    if len(roles) > MAX_TRACKERS:
        raise ValueError(f"VRChat supports at most {MAX_TRACKERS} trackers, got {len(roles)}")
    if len(set(roles)) != len(roles):
        raise ValueError(f"duplicate roles in tracker assignment: {roles}")
    return {role: i + 1 for i, role in enumerate(roles)}
