"""Mapping BEDLAM's joint layouts onto the Halpe26 skeleton this tracker uses.

BEDLAM never produced Halpe26. What it did produce depends on the version:

*BEDLAM 1.0* projected the raw SMPL-X joint set. Body joints occupy indices
0-24 of that set (pelvis through the eyes); toes, heels and ears are absent
and have to be inferred from what is there.

*BEDLAM 2.0* concatenated three layouts into ``gtkps``. The first 25 entries
are OpenPose BODY_25, which is almost Halpe26: it has the six foot keypoints
and the face, and is only missing the explicit ``head`` joint. That is the
layout this module prefers, because it is the one that actually contains the
toes the foot tracker is built on.

Getting this permutation wrong is silent and catastrophic. The lifter would
train with the left wrist where the right elbow should be, and every accuracy
number measured against the result would be meaningless.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from bodytracker.skeleton import (
    HEAD,
    HIP,
    LEFT_ANKLE,
    LEFT_BIG_TOE,
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_EYE,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_SMALL_TOE,
    LEFT_WRIST,
    NECK,
    NOSE,
    NUM_KEYPOINTS,
    RIGHT_ANKLE,
    RIGHT_BIG_TOE,
    RIGHT_EAR,
    RIGHT_ELBOW,
    RIGHT_EYE,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_SMALL_TOE,
    RIGHT_WRIST,
)

# OpenPose BODY_25, the first block of BEDLAM 2.0 ``gtkps``.
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
OP_NOSE: Final = 0
OP_NECK: Final = 1
OP_RIGHT_SHOULDER: Final = 2
OP_RIGHT_ELBOW: Final = 3
OP_RIGHT_WRIST: Final = 4
OP_LEFT_SHOULDER: Final = 5
OP_LEFT_ELBOW: Final = 6
OP_LEFT_WRIST: Final = 7
OP_MID_HIP: Final = 8
OP_RIGHT_HIP: Final = 9
OP_RIGHT_KNEE: Final = 10
OP_RIGHT_ANKLE: Final = 11
OP_LEFT_HIP: Final = 12
OP_LEFT_KNEE: Final = 13
OP_LEFT_ANKLE: Final = 14
OP_RIGHT_EYE: Final = 15
OP_LEFT_EYE: Final = 16
OP_RIGHT_EAR: Final = 17
OP_LEFT_EAR: Final = 18
OP_LEFT_BIG_TOE: Final = 19
OP_LEFT_SMALL_TOE: Final = 20
OP_LEFT_HEEL: Final = 21
OP_RIGHT_BIG_TOE: Final = 22
OP_RIGHT_SMALL_TOE: Final = 23
OP_RIGHT_HEEL: Final = 24
OPENPOSE_BODY25: Final = 25

# Direct correspondences. Halpe ``head`` has no OpenPose counterpart; it is
# filled in from the ears after the permutation.
OPENPOSE_TO_HALPE: Final[tuple[tuple[int, int], ...]] = (
    (OP_NOSE, NOSE),
    (OP_LEFT_EYE, LEFT_EYE),
    (OP_RIGHT_EYE, RIGHT_EYE),
    (OP_LEFT_EAR, LEFT_EAR),
    (OP_RIGHT_EAR, RIGHT_EAR),
    (OP_LEFT_SHOULDER, LEFT_SHOULDER),
    (OP_RIGHT_SHOULDER, RIGHT_SHOULDER),
    (OP_LEFT_ELBOW, LEFT_ELBOW),
    (OP_RIGHT_ELBOW, RIGHT_ELBOW),
    (OP_LEFT_WRIST, LEFT_WRIST),
    (OP_RIGHT_WRIST, RIGHT_WRIST),
    (OP_LEFT_HIP, LEFT_HIP),
    (OP_RIGHT_HIP, RIGHT_HIP),
    (OP_LEFT_KNEE, LEFT_KNEE),
    (OP_RIGHT_KNEE, RIGHT_KNEE),
    (OP_LEFT_ANKLE, LEFT_ANKLE),
    (OP_RIGHT_ANKLE, RIGHT_ANKLE),
    (OP_NECK, NECK),
    (OP_MID_HIP, HIP),
    (OP_LEFT_BIG_TOE, LEFT_BIG_TOE),
    (OP_RIGHT_BIG_TOE, RIGHT_BIG_TOE),
    (OP_LEFT_SMALL_TOE, LEFT_SMALL_TOE),
    (OP_RIGHT_SMALL_TOE, RIGHT_SMALL_TOE),
    (OP_LEFT_HEEL, LEFT_HEEL),
    (OP_RIGHT_HEEL, RIGHT_HEEL),
)

# SMPL-X body joints, the native output of ``smplx.SMPLX(...).joints``.
# Indices from the SMPL-X model card. The foot joints sit near the ball of
# the foot, which is closer to a big-toe marker than to a heel.
SMPLX_TO_HALPE: Final[tuple[tuple[int, int], ...]] = (
    (0, HIP),
    (1, LEFT_HIP),
    (2, RIGHT_HIP),
    (4, LEFT_KNEE),
    (5, RIGHT_KNEE),
    (7, LEFT_ANKLE),
    (8, RIGHT_ANKLE),
    (10, LEFT_BIG_TOE),
    (11, RIGHT_BIG_TOE),
    (12, NECK),
    (15, HEAD),
    (16, LEFT_SHOULDER),
    (17, RIGHT_SHOULDER),
    (18, LEFT_ELBOW),
    (19, RIGHT_ELBOW),
    (20, LEFT_WRIST),
    (21, RIGHT_WRIST),
    (23, LEFT_EYE),
    (24, RIGHT_EYE),
)


def from_openpose_body25(points: np.ndarray) -> np.ndarray:
    """Permute OpenPose BODY_25 joints into Halpe26, filling ``head`` from the ears.

    Accepts ``(25, D)`` or ``(N, 25, D)``. Trailing columns past 25 (the extra
    BEDLAM 2.0 blocks) are ignored, so passing the full ``gtkps`` row is fine.
    """
    array = np.asarray(points, dtype=np.float64)
    single = array.ndim == 2
    if single:
        array = array[None]
    if array.shape[1] < OPENPOSE_BODY25:
        raise ValueError(f"expected at least {OPENPOSE_BODY25} joints, got {array.shape[1]}")

    out = np.full((array.shape[0], NUM_KEYPOINTS, array.shape[2]), np.nan, dtype=np.float64)
    for src, dst in OPENPOSE_TO_HALPE:
        out[:, dst] = array[:, src]

    # Halpe's head is the crown, which OpenPose does not mark. The midpoint of
    # the ears is the closest thing it has, and sits a couple of centimetres
    # below; the offset is absorbed by the device-to-keypoint calibration.
    ears = np.stack([out[:, LEFT_EAR], out[:, RIGHT_EAR]], axis=1)
    out[:, HEAD] = np.nanmean(ears, axis=1)
    return out[0] if single else out


def from_smplx_body(points: np.ndarray) -> np.ndarray:
    """Lift the SMPL-X body joints that exist into Halpe26.

    Face keypoints other than the eyes, and the small toes and heels, are
    derived from neighbouring joints so the rest of the pipeline can assume a
    complete skeleton. Those derived points are worse than a real annotation,
    which is why BEDLAM 2.0's OpenPose block is preferred when it is present.
    """
    array = np.asarray(points, dtype=np.float64)
    single = array.ndim == 2
    if single:
        array = array[None]
    if array.shape[1] < 25:
        raise ValueError(f"expected at least 25 SMPL-X joints, got {array.shape[1]}")

    out = np.full((array.shape[0], NUM_KEYPOINTS, array.shape[2]), np.nan, dtype=np.float64)
    for src, dst in SMPLX_TO_HALPE:
        out[:, dst] = array[:, src]

    # Nose sits between the eyes, slightly forward of them. Without a face
    # regressor this is the honest approximation.
    out[:, NOSE] = np.nanmean(np.stack([out[:, LEFT_EYE], out[:, RIGHT_EYE]], axis=1), axis=1)

    # Ears are not in the body joint set. Put them out along the shoulder line
    # from the head so left/right at least have the right sign.
    across = out[:, RIGHT_SHOULDER] - out[:, LEFT_SHOULDER]
    out[:, LEFT_EAR] = out[:, HEAD] - 0.15 * across
    out[:, RIGHT_EAR] = out[:, HEAD] + 0.15 * across

    # Heel is behind the ankle along the foot; small toe is beside the big toe
    # along the same across direction the shoulders gave us.
    for ankle, big, heel, small, sign in (
        (LEFT_ANKLE, LEFT_BIG_TOE, LEFT_HEEL, LEFT_SMALL_TOE, -1.0),
        (RIGHT_ANKLE, RIGHT_BIG_TOE, RIGHT_HEEL, RIGHT_SMALL_TOE, 1.0),
    ):
        foot = out[:, big] - out[:, ankle]
        out[:, heel] = out[:, ankle] - 0.25 * foot
        out[:, small] = out[:, big] + sign * 0.15 * across
    return out[0] if single else out


def detect_layout(points: np.ndarray) -> str:
    """``'openpose25'`` if this looks like BEDLAM 2.0 gtkps, else ``'smplx'``.

    BEDLAM 2.0 concatenates OpenPose (25) + extra (19) + SMPL-X, so a row with
    44 or more joints is OpenPose-leading. Shorter rows are the BEDLAM 1.0
    native SMPL-X projection. This is a length check, not a content check: a
    mislabelled file would still be mapped wrong, and that is caught by the
    projection round-trip in the preprocessor, not here.
    """
    joints = np.asarray(points).shape[-2] if np.asarray(points).ndim >= 2 else 0
    if joints >= 44:
        return "openpose25"
    if joints >= OPENPOSE_BODY25:
        # Ambiguous: could be a truncated OpenPose block or a short SMPL-X
        # set. Prefer OpenPose because a 25-joint SMPL-X body is exactly the
        # same length, but BEDLAM 1.0's projected joints are the full set
        # (typically 127) and BEDLAM 2.0's gtkps is the long concatenation.
        return "openpose25" if joints == OPENPOSE_BODY25 else "smplx"
    raise ValueError(f"unrecognised joint layout with {joints} joints")


def to_halpe(points: np.ndarray, layout: str | None = None) -> np.ndarray:
    """Convert whatever BEDLAM handed us into Halpe26."""
    if layout is None:
        layout = detect_layout(points)
    if layout == "openpose25":
        return from_openpose_body25(points)
    if layout == "smplx":
        return from_smplx_body(points)
    raise ValueError(f"unknown layout {layout!r}")


def in_frame(
    xy: np.ndarray,
    width: int,
    height: int,
    margin_px: float = 0.0,
) -> np.ndarray:
    """Which 2D joints land inside the image, with an optional edge margin."""
    points = np.asarray(xy, dtype=np.float64)
    x, y = points[..., 0], points[..., 1]
    return (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= margin_px)
        & (y >= margin_px)
        & (x < width - margin_px)
        & (y < height - margin_px)
    )
