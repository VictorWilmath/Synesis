"""Per-user body proportions.

The lifter turns bone lengths into depth, so an error in the proportions
becomes an error in the pose. The generic anthropometric prior scaled by height
is a decent start, but two people of the same height can differ by several
centimetres in leg length, and that shows up directly as hip height.

Height comes from the headset, which is the one measurement available for free
and to within a couple of centimetres. Bone lengths are then measured from
tracked frames and blended toward the prior, so a few bad frames cannot deform
the skeleton.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..lift.kinematics import bone_lengths_from_height, measure_bone_lengths
from ..skeleton import BONES
from ..types import Skeleton3D, VRState
from ..vr.source import estimate_standing_height

log = logging.getLogger(__name__)

# Measured lengths are trusted this much against the height-scaled prior. Kept
# below 1 because a systematic depth error biases every measurement the same
# way, and fully trusting them would bake that bias into the skeleton.
_MEASUREMENT_WEIGHT = 0.6

# A measured bone this far from the prior is a bad frame, not a tall user.
_MAX_DEVIATION = 0.4


@dataclass(slots=True)
class BodyCalibration:
    height_m: float
    bone_lengths: dict[tuple[int, int], float]
    frames_used: int = 0

    @classmethod
    def from_height(cls, height_m: float) -> BodyCalibration:
        return cls(height_m=height_m, bone_lengths=bone_lengths_from_height(height_m))


def estimate_height(states: list[VRState], fallback: float = 1.75) -> float:
    height = estimate_standing_height(states)
    if height is None:
        log.warning("not enough headset samples to estimate height; using %.2f m", fallback)
        return fallback
    if not 1.2 <= height <= 2.3:
        log.warning("implausible height estimate %.2f m; using %.2f m", height, fallback)
        return fallback
    log.info("estimated standing height %.2f m from headset", height)
    return height


def refine_bone_lengths(
    skeletons: list[Skeleton3D],
    height_m: float,
    min_frames: int = 30,
) -> dict[tuple[int, int], float]:
    """Measure bone lengths across frames, blended toward the height prior."""
    prior = bone_lengths_from_height(height_m)
    if len(skeletons) < min_frames:
        log.warning(
            "only %d frames for bone measurement, need %d; keeping the prior",
            len(skeletons),
            min_frames,
        )
        return prior

    measurements: dict[tuple[int, int], list[float]] = {bone: [] for bone in BONES}
    for skeleton in skeletons:
        for bone, length in measure_bone_lengths(np.asarray(skeleton.xyz)).items():
            measurements[bone].append(length)

    refined: dict[tuple[int, int], float] = {}
    rejected = 0
    for bone, expected in prior.items():
        values = measurements.get(bone, [])
        if not values:
            refined[bone] = expected
            continue
        # Median rather than mean: single-frame lifting failures are wild, and
        # a mean would let one of them move the result.
        measured = float(np.median(values))
        if abs(measured - expected) > _MAX_DEVIATION * expected:
            refined[bone] = expected
            rejected += 1
            continue
        refined[bone] = _MEASUREMENT_WEIGHT * measured + (1.0 - _MEASUREMENT_WEIGHT) * expected

    if rejected:
        log.info("kept the prior for %d bones whose measurements were implausible", rejected)
    return refined


def calibrate_body(
    skeletons: list[Skeleton3D],
    states: list[VRState],
    fallback_height: float = 1.75,
) -> BodyCalibration:
    height = estimate_height(states, fallback_height)
    return BodyCalibration(
        height_m=height,
        bone_lengths=refine_bone_lengths(skeletons, height),
        frames_used=len(skeletons),
    )


def save_body(body: BodyCalibration, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "height_m": body.height_m,
                "frames_used": body.frames_used,
                # JSON keys must be strings, so bones are stored as "a-b".
                "bone_lengths": {f"{a}-{b}": v for (a, b), v in body.bone_lengths.items()},
            },
            indent=2,
        )
    )
    log.info("wrote body calibration to %s", path)


def load_body(path: Path | str) -> BodyCalibration | None:
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    bone_lengths = {}
    for key, value in data["bone_lengths"].items():
        a, b = key.split("-")
        bone_lengths[(int(a), int(b))] = float(value)
    return BodyCalibration(
        height_m=float(data["height_m"]),
        bone_lengths=bone_lengths,
        frames_used=int(data.get("frames_used", 0)),
    )
