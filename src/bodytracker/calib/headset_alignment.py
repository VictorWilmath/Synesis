"""A small, reliable bridge from webcam-model space to the SteamVR room.

The webcam lifter can estimate a body relative to its predicted head, but a
single camera cannot know which way that local coordinate frame faces in the
user's SteamVR room.  Room extrinsics solves that problem in theory, but a
noisy 2D PnP solve is not a good first-run requirement.

This module follows the useful part of SlimeVR's reset model instead: while
the user stands square to the camera and looks forward, save one yaw transform
between the estimated body and the HMD.  Live output is then always anchored
to the *current* HMD position, rotated by that fixed yaw, and sent directly in
SteamVR play space.  It is intentionally not a replacement for a good room
calibration; it is a dependable lower-friction starting point.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..skeleton import HIP, LEFT_SHOULDER, NECK, RIGHT_SHOULDER
from ..types import DevicePose, TrackerTarget

_HEADSET_TO_STATURE = 0.93


def _horizontal(vector: np.ndarray) -> np.ndarray | None:
    value = np.asarray(vector, dtype=np.float64).copy()
    value[1] = 0.0
    norm = float(np.linalg.norm(value))
    return None if norm < 1e-6 else value / norm


def source_forward_from_skeleton(xyz: np.ndarray) -> np.ndarray | None:
    """Infer horizontal body-forward from shoulders and the spine.

    Synesis's play/model convention is +y up and -z forward.  ``up x right``
    therefore gives the forward direction.  This uses broad body segments,
    rather than feet or face points, because they are much less noisy in a
    single webcam view.
    """
    points = np.asarray(xyz, dtype=np.float64)
    right = points[RIGHT_SHOULDER] - points[LEFT_SHOULDER]
    up = points[NECK] - points[HIP]
    return _horizontal(np.cross(up, right))


def headset_forward_from_rotation(rotation: np.ndarray) -> np.ndarray | None:
    """The HMD's neutral -z forward vector, flattened to the floor plane."""
    return _horizontal(np.asarray(rotation, dtype=np.float64) @ np.array([0.0, 0.0, -1.0]))


def yaw_rotation_from_to(source_forward: np.ndarray, target_forward: np.ndarray) -> np.ndarray:
    """Return the yaw-only rotation that maps ``source_forward`` to ``target_forward``."""
    source = _horizontal(source_forward)
    target = _horizontal(target_forward)
    if source is None or target is None:
        raise ValueError("both forward vectors must have a horizontal component")
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    sine = float(np.cross(source, target)[1])
    angle = math.atan2(sine, cosine)
    return yaw_rotation(angle)


def yaw_rotation(angle_rad: float) -> np.ndarray:
    """Active right-handed rotation about the SteamVR +y axis."""
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]], dtype=np.float64
    )


@dataclass(slots=True)
class HeadsetAlignment:
    """Persisted webcam-model-to-HMD yaw reference and observed stature."""

    rotation: np.ndarray
    height_m: float
    samples: int
    yaw_spread_deg: float

    @property
    def yaw_deg(self) -> float:
        return float(np.degrees(math.atan2(self.rotation[0, 2], self.rotation[0, 0])))


def fit_headset_alignment(
    source_forwards: list[np.ndarray],
    headset_forwards: list[np.ndarray],
    headset_heights: list[float],
) -> HeadsetAlignment:
    """Robustly fit a yaw offset from a short neutral-pose capture."""
    if not source_forwards or len(source_forwards) != len(headset_forwards):
        raise ValueError("need paired body and headset forward samples")
    if not headset_heights:
        raise ValueError("need at least one headset height")

    angles: list[float] = []
    for source, headset in zip(source_forwards, headset_forwards, strict=True):
        source_n = _horizontal(source)
        headset_n = _horizontal(headset)
        if source_n is None or headset_n is None:
            continue
        sine = float(np.cross(source_n, headset_n)[1])
        cosine = float(np.dot(source_n, headset_n))
        angles.append(math.atan2(sine, cosine))
    if len(angles) < 3:
        raise ValueError("need three usable horizontal orientation samples")

    mean_sine = float(np.mean(np.sin(angles)))
    mean_cosine = float(np.mean(np.cos(angles)))
    resultant = float(np.hypot(mean_sine, mean_cosine))
    yaw = math.atan2(mean_sine, mean_cosine)
    # Circular standard deviation, expressed in degrees, is more meaningful
    # than an ordinary spread across the -180/180 boundary.
    spread = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(max(resultant, 1e-9)))))

    eye_height = float(np.median(headset_heights))
    height_m = eye_height / _HEADSET_TO_STATURE
    if not 1.2 <= height_m <= 2.3:
        raise ValueError(f"implausible standing height inferred from HMD: {height_m:.2f} m")
    return HeadsetAlignment(yaw_rotation(yaw), height_m, len(angles), spread)


def align_targets_to_headset(
    targets: list[TrackerTarget],
    source_head: np.ndarray,
    headset: DevicePose,
    alignment: HeadsetAlignment,
) -> list[TrackerTarget]:
    """Map model-relative tracker poses directly into the SteamVR room.

    The head is re-anchored every frame because it is the one pose SteamVR
    measures precisely.  The *orientation* is deliberately fixed by the saved
    neutral pose, rather than following head yaw: looking around should not
    rotate a person's hips and feet around the room.
    """
    if not headset.valid:
        return targets
    head = np.asarray(source_head, dtype=np.float64)
    headset_position = np.asarray(headset.position, dtype=np.float64)
    rotation = np.asarray(alignment.rotation, dtype=np.float64)
    return [
        TrackerTarget(
            role=target.role,
            position=headset_position + rotation @ (np.asarray(target.position) - head),
            rotation=rotation @ np.asarray(target.rotation),
            valid=target.valid,
            stale=target.stale,
        )
        for target in targets
    ]


def save_headset_alignment(alignment: HeadsetAlignment, path: Path | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "rotation": np.asarray(alignment.rotation, dtype=np.float64).tolist(),
                "height_m": alignment.height_m,
                "samples": alignment.samples,
                "yaw_spread_deg": alignment.yaw_spread_deg,
            },
            indent=2,
        )
    )


def load_headset_alignment(path: Path | str) -> HeadsetAlignment | None:
    source = Path(path)
    if not source.is_file():
        return None
    data = json.loads(source.read_text())
    rotation = np.asarray(data["rotation"], dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"invalid headset-alignment rotation in {source}")
    return HeadsetAlignment(
        rotation=rotation,
        height_m=float(data["height_m"]),
        samples=int(data["samples"]),
        yaw_spread_deg=float(data.get("yaw_spread_deg", 0.0)),
    )
