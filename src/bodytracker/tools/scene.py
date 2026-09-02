"""A synthetic camera-and-headset scene, for testing calibration and lifting.

Generates exactly what a real session produces, a stream of paired VR device
poses and 2D keypoints, but with the camera pose and the true 3D known. That
makes it possible to ask the one question that matters about the calibrator:
given perfect inputs, does it recover the camera we placed, and how fast does
that degrade as the inputs get worse?

Rendering images is deliberately skipped. The pose model's behaviour is not
what is under test here, and pixel noise is a better model of its error than a
synthetic render would be anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..calib.anchors import ANCHORS
from ..geom import camera_from_play, look_at_extrinsics, project_points, rotation_from_forward_up
from ..lift.kinematics import bone_lengths_from_height, enforce_bone_lengths
from ..skeleton import HEAD, LEFT_WRIST, NUM_KEYPOINTS, RIGHT_WRIST
from ..types import DevicePose, Keypoints2D, VRState
from .synthetic import reference_skeleton

_ANCHOR_KEYPOINT = {"head": HEAD, "left_hand": LEFT_WRIST, "right_hand": RIGHT_WRIST}


@dataclass(slots=True)
class ScenePose:
    """One frame: the truth, what SteamVR would report, what the camera sees."""

    play_xyz: np.ndarray  # (26, 3) ground-truth joints in play space
    vr: VRState
    keypoints: Keypoints2D


@dataclass(slots=True)
class Scene:
    frames: list[ScenePose]
    intrinsics: np.ndarray
    rotation_cw: np.ndarray
    translation_cw: np.ndarray
    camera_position: np.ndarray
    height_m: float

    def keypoint_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.stack([f.keypoints.xy for f in self.frames]),
            np.stack([f.play_xyz for f in self.frames]),
        )


def _yaw_matrix(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]])


def _pitch_matrix(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]])


def posed_skeleton(
    height_m: float,
    *,
    position: np.ndarray,
    yaw_deg: float = 0.0,
    crouch: float = 0.0,
) -> np.ndarray:
    """The reference skeleton placed in the room, turned, and optionally crouched.

    The crouch starts as a vertical squash, which is not a real articulation
    and would shorten every bone. Bone lengths are then restored by constraint
    projection, giving a bent-legged pose that is anatomically consistent.

    That consistency matters: a fixture whose bones do not match the prior
    would make a correct lifter look wrong, and every accuracy number measured
    against it would be meaningless.
    """
    xyz = reference_skeleton(height_m)
    if crouch:
        xyz = xyz * np.array([1.0, 1.0 - crouch, 1.0])
        xyz = enforce_bone_lengths(xyz, bone_lengths_from_height(height_m), iterations=200)
        xyz[:, 1] -= float(np.min(xyz[:, 1]))
    return xyz @ _yaw_matrix(yaw_deg).T + np.asarray(position, dtype=np.float64)


def vr_state_for(
    xyz: np.ndarray,
    yaw_deg: float,
    timestamp: float,
    offsets: dict[str, np.ndarray] | None = None,
    *,
    head_pitch_deg: float = 0.0,
    head_yaw_deg: float = 0.0,
) -> VRState:
    """The SteamVR poses that would produce these keypoints.

    Inverts the anchor model: given where a keypoint is and the device's
    orientation, place the device so its offset lands on the keypoint. Turning
    the head therefore swings the headset around the head centre, which is what
    really happens and what makes the offset observable at all.

    Head articulation matters more than it looks. If the headset only ever
    yaws, its local +y is always world +y, so raising the offset and lowering
    the camera produce identical images and the two cannot be separated. Pitch
    breaks that tie.
    """
    body_rotation = rotation_from_forward_up(_yaw_matrix(yaw_deg) @ np.array([0.0, 0.0, -1.0]))
    head_rotation = body_rotation @ _yaw_matrix(head_yaw_deg) @ _pitch_matrix(head_pitch_deg)
    rotations = {
        "head": head_rotation,
        "left_hand": body_rotation,
        "right_hand": body_rotation,
    }

    poses: dict[str, DevicePose] = {}
    for anchor in ANCHORS:
        offset = anchor.local_offset if offsets is None else offsets[anchor.name]
        keypoint = xyz[_ANCHOR_KEYPOINT[anchor.name]]
        rotation = rotations[anchor.name]
        poses[anchor.name] = DevicePose(
            position=keypoint - rotation @ offset,
            rotation=rotation.copy(),
            valid=True,
            timestamp=timestamp,
        )

    return VRState(
        head=poses["head"],
        left_hand=poses["left_hand"],
        right_hand=poses["right_hand"],
        timestamp=timestamp,
    )


def build_scene(
    *,
    intrinsics: np.ndarray,
    camera_position: np.ndarray = np.array([0.6, 1.15, 2.6]),
    look_at: np.ndarray = np.array([0.0, 0.95, 0.0]),
    height_m: float = 1.75,
    frames: int = 60,
    fps: float = 30.0,
    cycle_s: float = 8.0,
    noise_px: float = 0.0,
    dropout: float = 0.0,
    offsets: dict[str, np.ndarray] | None = None,
    head_motion: bool = True,
    seed: int = 0,
) -> Scene:
    """A session of someone moving around in front of a camera.

    `offsets` overrides the true device-to-keypoint offsets, which is how you
    test whether the calibrator can recover an offset that differs from its
    nominal starting value, as it always does in reality.

    `head_motion` adds the pitch and yaw of somebody looking around. Turning it
    off produces the degenerate case where the headset's offset cannot be told
    apart from the camera's height.

    `cycle_s` is how long one lap of the room takes, which together with `fps`
    sets how fast the subject moves. It matters more than it looks: anything
    measuring filter lag is really measuring speed, so a scene that whips the
    subject around in two seconds will condemn a perfectly good filter.
    """
    rng = np.random.default_rng(seed)
    rotation_cw, translation_cw = look_at_extrinsics(camera_position, look_at)

    poses: list[ScenePose] = []
    for i in range(frames):
        phase = (i / fps) / cycle_s
        # Cover the space: a lap around the room, turning, with some crouching.
        position = np.array(
            [
                0.9 * np.sin(2 * np.pi * phase),
                0.0,
                0.5 * np.cos(2 * np.pi * phase) - 0.3,
            ]
        )
        yaw = 40.0 * np.sin(2 * np.pi * phase + 0.7)
        crouch = 0.12 * (1.0 - np.cos(4 * np.pi * phase)) / 2.0

        head_pitch = 22.0 * np.sin(6 * np.pi * phase) if head_motion else 0.0
        head_yaw = 25.0 * np.sin(3 * np.pi * phase + 1.3) if head_motion else 0.0

        xyz = posed_skeleton(height_m, position=position, yaw_deg=yaw, crouch=crouch)
        camera_xyz = camera_from_play(xyz, rotation_cw, translation_cw)
        pixels = project_points(camera_xyz, intrinsics)

        if noise_px > 0:
            pixels = pixels + rng.normal(scale=noise_px, size=pixels.shape)

        scores = np.ones(NUM_KEYPOINTS, dtype=np.float32) * 0.9
        if dropout > 0:
            scores[rng.random(NUM_KEYPOINTS) < dropout] = 0.05

        timestamp = i / fps
        poses.append(
            ScenePose(
                play_xyz=xyz,
                vr=vr_state_for(
                    xyz,
                    yaw,
                    timestamp,
                    offsets,
                    head_pitch_deg=head_pitch,
                    head_yaw_deg=head_yaw,
                ),
                keypoints=Keypoints2D(
                    xy=pixels.astype(np.float32),
                    scores=scores,
                    timestamp=timestamp,
                ),
            )
        )

    return Scene(
        frames=poses,
        intrinsics=np.asarray(intrinsics, dtype=np.float64),
        rotation_cw=rotation_cw,
        translation_cw=translation_cw,
        camera_position=np.asarray(camera_position, dtype=np.float64),
        height_m=height_m,
    )
