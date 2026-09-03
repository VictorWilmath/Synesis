"""Inventing the headset and controllers BEDLAM never had.

The lifter is conditioned on SteamVR device poses. BEDLAM is a dataset of
people in ordinary clothes, so those poses have to be synthesised from the
body. That is less fake than it sounds: a real headset is a rigid offset from
the head keypoint, and a real controller is a rigid offset from the wrist, in
exactly the same model the runtime calibrator uses. Training against the
inverse of that model is what makes the two ends of the pipeline meet.

The orientations are derived from the skeleton rather than copied from a
single body yaw. A headset that only ever yaws with the torso cannot teach
the calibrator (or the lifter) to separate a nod from a camera-height error,
which is the same degeneracy the synthetic scene fixture had to grow head
pitch to break.
"""

from __future__ import annotations

import numpy as np

from bodytracker.calib.anchors import ANCHOR_BY_NAME
from bodytracker.geom import rotation_from_forward_up
from bodytracker.skeleton import (
    HEAD,
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_EYE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NECK,
    RIGHT_EAR,
    RIGHT_ELBOW,
    RIGHT_EYE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from bodytracker.solve.rotations import pelvis_rotation
from bodytracker.types import DevicePose, VRState

_WORLD_UP = np.array([0.0, 1.0, 0.0])
_EPS = 1e-8
_DEVICE_JOINT = {"head": HEAD, "left_hand": LEFT_WRIST, "right_hand": RIGHT_WRIST}


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < _EPS:
        return None
    return vector / norm


def head_rotation(xyz: np.ndarray) -> np.ndarray:
    """Which way the face is pointing, from the eyes and ears.

    Prefer the ear line for 'right' because the eyes can sit almost on top of
    each other in a profile view and then the cross product vanishes. Fall
    back to the shoulders, then to the pelvis, rather than inventing a yaw.
    """
    points = np.asarray(xyz, dtype=np.float64)
    up = _normalize(points[HEAD] - points[NECK])
    if up is None:
        up = _WORLD_UP

    for left, right in (
        (LEFT_EAR, RIGHT_EAR),
        (LEFT_EYE, RIGHT_EYE),
        (LEFT_SHOULDER, RIGHT_SHOULDER),
    ):
        across = _normalize(points[right] - points[left])
        if across is None:
            continue
        forward = _normalize(np.cross(up, across))
        if forward is not None:
            return rotation_from_forward_up(forward, up)

    pelvis = pelvis_rotation(points)
    return pelvis if pelvis is not None else rotation_from_forward_up(
        np.array([0.0, 0.0, -1.0]), _WORLD_UP
    )


def hand_rotation(xyz: np.ndarray, *, left: bool) -> np.ndarray:
    """Controller orientation, pointing along the forearm.

    A held controller's local +Z is roughly the pointing axis, which for a
    closed fist is the direction from elbow to wrist. Using the pelvis yaw
    instead would freeze the controllers to the torso, and the lifter would
    never see an outstretched arm whose tracker disagrees with the hips.
    """
    elbow, wrist = (LEFT_ELBOW, LEFT_WRIST) if left else (RIGHT_ELBOW, RIGHT_WRIST)
    points = np.asarray(xyz, dtype=np.float64)
    along = _normalize(points[wrist] - points[elbow])
    if along is None:
        pelvis = pelvis_rotation(points)
        return pelvis if pelvis is not None else np.eye(3)
    return rotation_from_forward_up(along, _WORLD_UP)


def device_pose(
    xyz: np.ndarray,
    name: str,
    rotation: np.ndarray,
    timestamp: float = 0.0,
    offsets: dict[str, np.ndarray] | None = None,
) -> DevicePose:
    """Place one SteamVR device so its calibrated offset lands on the keypoint."""
    anchor = ANCHOR_BY_NAME[name]
    offset = anchor.local_offset if offsets is None else offsets[name]
    keypoint = np.asarray(xyz[_DEVICE_JOINT[name]], dtype=np.float64)
    rot = np.asarray(rotation, dtype=np.float64)
    return DevicePose(
        position=keypoint - rot @ np.asarray(offset, dtype=np.float64),
        rotation=rot.copy(),
        valid=True,
        timestamp=timestamp,
    )


def from_skeleton(
    xyz: np.ndarray,
    timestamp: float = 0.0,
    offsets: dict[str, np.ndarray] | None = None,
) -> VRState:
    """The VRState a person with this skeleton would produce if they were in VR.

    `xyz` is Halpe26 in play space. The result is in play space too, ready to
    feed the same lifter the runtime uses.
    """
    points = np.asarray(xyz, dtype=np.float64)
    rotations = {
        "head": head_rotation(points),
        "left_hand": hand_rotation(points, left=True),
        "right_hand": hand_rotation(points, left=False),
    }
    poses = {
        name: device_pose(points, name, rotations[name], timestamp, offsets) for name in rotations
    }
    return VRState(
        head=poses["head"],
        left_hand=poses["left_hand"],
        right_hand=poses["right_hand"],
        timestamp=timestamp,
    )


def pack(vr: VRState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack the three devices into arrays the shard format stores.

    Returns ``(positions, rotations, valid)`` with the leading axis in
    ``(head, left_hand, right_hand)`` order, matching ``eval.session.DEVICE_NAMES``.
    """
    positions = np.full((3, 3), np.nan, dtype=np.float64)
    rotations = np.full((3, 3, 3), np.nan, dtype=np.float64)
    valid = np.zeros(3, dtype=np.bool_)
    for i, pose in enumerate((vr.head, vr.left_hand, vr.right_hand)):
        if pose is None or not pose.valid:
            continue
        positions[i] = pose.position
        rotations[i] = pose.rotation
        valid[i] = True
    return positions, rotations, valid


def unpack(
    positions: np.ndarray,
    rotations: np.ndarray,
    valid: np.ndarray,
    timestamp: float = 0.0,
) -> VRState:
    """Inverse of `pack`."""

    def pose(index: int) -> DevicePose | None:
        if not valid[index]:
            return None
        return DevicePose(
            position=np.asarray(positions[index], dtype=np.float64).copy(),
            rotation=np.asarray(rotations[index], dtype=np.float64).copy(),
            valid=True,
            timestamp=timestamp,
        )

    return VRState(
        head=pose(0), left_hand=pose(1), right_hand=pose(2), timestamp=timestamp
    )
