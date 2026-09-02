"""Synthetic body motion, used to exercise the output path without a camera.

This exists so the OSC wire can be proven before any computer vision is
involved. If the avatar follows this motion in VRChat, the transport, the slot
assignment, the handedness flip and the Euler convention are all correct, and
any later problem is upstream in the tracking.

Poses are in play space: origin on the floor at the centre of the play area,
+y up, and the subject facing SteamVR-forward, which is -z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..geom import rotation_from_forward_up
from ..skeleton import (
    BONE_LENGTH_PRIOR_RATIO,
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
    TrackerRole,
)
from ..types import TrackerTarget

PATTERNS = ("static", "bob", "walk", "sway", "spin")

# Joint heights as a fraction of standing height, from standard anthropometric
# proportions. Only needs to be plausible enough for VRChat's IK to accept.
_ANKLE_H = 0.039
_KNEE_H = 0.285
_HIP_H = 0.530
_ELBOW_H = 0.630
_CHEST_H = 0.720

_HIP_HALF_WIDTH = 0.055
_KNEE_HALF_WIDTH = 0.058
_FOOT_HALF_WIDTH = 0.060
_ELBOW_HALF_WIDTH = 0.150


def reference_skeleton(height_m: float = 1.75) -> np.ndarray:
    """A standing 26-joint skeleton whose bones match the anthropometric prior.

    Built so that every segment length equals
    ``BONE_LENGTH_PRIOR_RATIO * height_m`` exactly, which makes it a valid
    fixture for round-tripping the lifting math: project it, lift it back, and
    any discrepancy is the lifter's rather than the fixture's.

    Play space, feet on ``y = 0``, subject facing SteamVR-forward (-z).
    """
    h = height_m
    ratio = BONE_LENGTH_PRIOR_RATIO
    xyz = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)

    def length(a: int, b: int) -> float:
        return ratio[(a, b)] * h if (a, b) in ratio else ratio[(b, a)] * h

    down = np.array([0.0, -1.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])

    # Pelvis height follows from the leg chain, so the ankles land on the floor
    # at their natural height.
    ankle_height = 0.039 * h
    leg = length(LEFT_HIP, LEFT_KNEE) + length(LEFT_KNEE, LEFT_ANKLE)
    xyz[HIP] = [0.0, ankle_height + leg, 0.0]

    xyz[NECK] = xyz[HIP] + up * length(HIP, NECK)
    xyz[HEAD] = xyz[NECK] + up * length(NECK, HEAD)

    for side, shoulder, elbow, wrist, hip_j, knee, ankle, heel, big, small in (
        (
            -1.0,
            LEFT_SHOULDER,
            LEFT_ELBOW,
            LEFT_WRIST,
            LEFT_HIP,
            LEFT_KNEE,
            LEFT_ANKLE,
            LEFT_HEEL,
            LEFT_BIG_TOE,
            LEFT_SMALL_TOE,
        ),
        (
            1.0,
            RIGHT_SHOULDER,
            RIGHT_ELBOW,
            RIGHT_WRIST,
            RIGHT_HIP,
            RIGHT_KNEE,
            RIGHT_ANKLE,
            RIGHT_HEEL,
            RIGHT_BIG_TOE,
            RIGHT_SMALL_TOE,
        ),
    ):
        lateral = np.array([side, 0.0, 0.0])

        xyz[shoulder] = xyz[NECK] + lateral * length(NECK, shoulder)
        xyz[elbow] = xyz[shoulder] + down * length(shoulder, elbow)
        xyz[wrist] = xyz[elbow] + down * length(elbow, wrist)

        xyz[hip_j] = xyz[HIP] + lateral * length(HIP, hip_j)
        xyz[knee] = xyz[hip_j] + down * length(hip_j, knee)
        xyz[ankle] = xyz[knee] + down * length(knee, ankle)

        # Heel sits behind the ankle (+z is backward), toes ahead of it.
        heel_dir = np.array([0.0, -0.6, 0.8])
        xyz[heel] = xyz[ankle] + heel_dir / np.linalg.norm(heel_dir) * length(ankle, heel)
        toe_dir = np.array([0.0, -0.32, -0.95])
        xyz[big] = xyz[ankle] + toe_dir / np.linalg.norm(toe_dir) * length(ankle, big)
        xyz[small] = xyz[big] + lateral * length(big, small)

    # Face keypoints are not part of any bone and their depth is copied from
    # the head, so plausible placement is all that is required.
    forward = np.array([0.0, 0.0, -1.0])
    xyz[NOSE] = xyz[HEAD] + forward * 0.09 * h * 0.5
    xyz[LEFT_EYE] = xyz[HEAD] + forward * 0.035 * h + np.array([-0.018 * h, 0.012 * h, 0.0])
    xyz[RIGHT_EYE] = xyz[HEAD] + forward * 0.035 * h + np.array([0.018 * h, 0.012 * h, 0.0])
    xyz[LEFT_EAR] = xyz[HEAD] + np.array([-0.042 * h, 0.0, 0.012 * h])
    xyz[RIGHT_EAR] = xyz[HEAD] + np.array([0.042 * h, 0.0, 0.012 * h])

    # Rest the lowest point exactly on y = 0. A rigid translation, so every
    # bone length is preserved and the fixture stays exact.
    xyz[:, 1] -= float(np.min(xyz[:, 1]))
    return xyz


def _yaw_about_y(points: dict[TrackerRole, np.ndarray], angle: float) -> None:
    """Rotate every position in place about the vertical axis through the origin."""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    for role, pos in points.items():
        x, y, z = pos
        points[role] = np.array([cos_a * x + sin_a * z, y, -sin_a * x + cos_a * z])


@dataclass(slots=True)
class SyntheticBody:
    """Generates anatomically plausible tracker poses for a given pattern."""

    height_m: float = 1.75
    period_s: float = 2.0
    amplitude: float = 1.0

    def _rest_positions(self) -> dict[TrackerRole, np.ndarray]:
        h = self.height_m
        return {
            TrackerRole.HIP: np.array([0.0, _HIP_H * h, 0.0]),
            TrackerRole.CHEST: np.array([0.0, _CHEST_H * h, 0.0]),
            TrackerRole.LEFT_FOOT: np.array([-_FOOT_HALF_WIDTH, _ANKLE_H * h, 0.0]),
            TrackerRole.RIGHT_FOOT: np.array([_FOOT_HALF_WIDTH, _ANKLE_H * h, 0.0]),
            TrackerRole.LEFT_KNEE: np.array([-_KNEE_HALF_WIDTH, _KNEE_H * h, 0.0]),
            TrackerRole.RIGHT_KNEE: np.array([_KNEE_HALF_WIDTH, _KNEE_H * h, 0.0]),
            TrackerRole.LEFT_ELBOW: np.array([-_ELBOW_HALF_WIDTH, _ELBOW_H * h, 0.0]),
            TrackerRole.RIGHT_ELBOW: np.array([_ELBOW_HALF_WIDTH, _ELBOW_H * h, 0.0]),
        }

    def pose(self, t: float, pattern: str = "bob") -> list[TrackerTarget]:
        """Tracker poses at time `t` seconds."""
        if pattern not in PATTERNS:
            raise ValueError(f"unknown pattern '{pattern}', expected one of {PATTERNS}")

        positions = self._rest_positions()
        phase = 2.0 * math.pi * t / self.period_s
        amp = self.amplitude
        # Facing direction in play space; SteamVR-forward is -z.
        facing = np.array([0.0, 0.0, -1.0])
        foot_pitch = {TrackerRole.LEFT_FOOT: 0.0, TrackerRole.RIGHT_FOOT: 0.0}

        if pattern == "bob":
            # Squat: hips and chest drop, knees travel forward as they bend.
            drop = 0.18 * amp * (1.0 - math.cos(phase)) / 2.0
            positions[TrackerRole.HIP][1] -= drop
            positions[TrackerRole.CHEST][1] -= drop * 0.95
            positions[TrackerRole.LEFT_KNEE][1] -= drop * 0.35
            positions[TrackerRole.RIGHT_KNEE][1] -= drop * 0.35
            positions[TrackerRole.LEFT_KNEE][2] -= drop * 0.75
            positions[TrackerRole.RIGHT_KNEE][2] -= drop * 0.75
            positions[TrackerRole.LEFT_ELBOW][1] -= drop * 0.9
            positions[TrackerRole.RIGHT_ELBOW][1] -= drop * 0.9

        elif pattern == "walk":
            # March in place: legs in antiphase, arms opposing the same-side leg.
            for sign, foot, knee, elbow in (
                (1.0, TrackerRole.LEFT_FOOT, TrackerRole.LEFT_KNEE, TrackerRole.RIGHT_ELBOW),
                (-1.0, TrackerRole.RIGHT_FOOT, TrackerRole.RIGHT_KNEE, TrackerRole.LEFT_ELBOW),
            ):
                leg_phase = phase if sign > 0 else phase + math.pi
                swing = math.sin(leg_phase)
                lift = max(0.0, math.sin(leg_phase)) * 0.12 * amp

                positions[foot][2] -= swing * 0.22 * amp
                positions[foot][1] += lift
                positions[knee][2] -= swing * 0.12 * amp
                positions[knee][1] += lift * 0.45
                positions[elbow][2] += swing * 0.14 * amp
                foot_pitch[foot] = -swing * 0.35

            positions[TrackerRole.HIP][1] += abs(math.sin(phase)) * 0.02 * amp
            positions[TrackerRole.CHEST][1] += abs(math.sin(phase)) * 0.02 * amp

        elif pattern == "sway":
            shift = math.sin(phase) * 0.12 * amp
            positions[TrackerRole.HIP][0] += shift
            positions[TrackerRole.CHEST][0] += shift * 0.6
            positions[TrackerRole.LEFT_KNEE][0] += shift * 0.7
            positions[TrackerRole.RIGHT_KNEE][0] += shift * 0.7
            positions[TrackerRole.LEFT_ELBOW][0] += shift * 0.5
            positions[TrackerRole.RIGHT_ELBOW][0] += shift * 0.5

        elif pattern == "spin":
            # Exercises the rotation path, which position-only motion cannot.
            _yaw_about_y(positions, phase)
            # Carry the facing vector through the same rotation as the joints.
            cos_a, sin_a = math.cos(phase), math.sin(phase)
            facing = np.array([-sin_a, 0.0, -cos_a])

        body_rotation = rotation_from_forward_up(facing)

        targets: list[TrackerTarget] = []
        for role, position in positions.items():
            if role in foot_pitch and abs(foot_pitch[role]) > 1e-9:
                # Tilt the foot's forward vector to fake toe-off and heel-strike.
                pitch = foot_pitch[role]
                tilted = facing + np.array([0.0, math.sin(pitch), 0.0])
                rotation = rotation_from_forward_up(tilted)
            else:
                rotation = body_rotation
            targets.append(
                TrackerTarget(
                    role=role,
                    position=position.astype(np.float64),
                    rotation=rotation,
                    valid=True,
                )
            )
        return targets

    def head(self, t: float, pattern: str = "bob") -> np.ndarray:
        """Approximate head position, for the optional head alignment endpoint."""
        hip = next(x for x in self.pose(t, pattern) if x.role == TrackerRole.HIP)
        head = hip.position.copy()
        head[1] = self.height_m * 0.935 - (self.height_m * _HIP_H - hip.position[1])
        return head
