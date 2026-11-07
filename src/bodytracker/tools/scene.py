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
from ..skeleton import (
    HEAD,
    LEFT_ANKLE,
    LEFT_BIG_TOE,
    LEFT_ELBOW,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_SMALL_TOE,
    LEFT_WRIST,
    NUM_KEYPOINTS,
    RIGHT_ANKLE,
    RIGHT_BIG_TOE,
    RIGHT_ELBOW,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_SMALL_TOE,
    RIGHT_WRIST,
)
from ..types import DevicePose, Keypoints2D, VRState
from .synthetic import reference_skeleton

_ANCHOR_KEYPOINT = {"head": HEAD, "left_hand": LEFT_WRIST, "right_hand": RIGHT_WRIST}
_LEGS = ((LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE))
_PLANTED = {
    LEFT_KNEE,
    LEFT_ANKLE,
    LEFT_HEEL,
    LEFT_BIG_TOE,
    LEFT_SMALL_TOE,
    RIGHT_KNEE,
    RIGHT_ANKLE,
    RIGHT_HEEL,
    RIGHT_BIG_TOE,
    RIGHT_SMALL_TOE,
}


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


def _yaw_matrix(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]])


def _pitch_matrix(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]])


def _bend(
    root: np.ndarray, end: np.ndarray, upper: float, lower: float, toward: np.ndarray
) -> np.ndarray:
    """Where the middle joint of a two-bone chain sits, bending toward `toward`."""
    span = end - root
    reach = float(np.linalg.norm(span))
    direction = span / reach
    along = (upper**2 - lower**2 + reach**2) / (2.0 * reach)
    radius = np.sqrt(max(upper**2 - along**2, 0.0))
    bend = toward - (toward @ direction) * direction
    bend /= np.linalg.norm(bend)
    return root + along * direction + radius * bend


def _crouch(xyz: np.ndarray, crouch: float) -> np.ndarray:
    out = xyz.copy()
    leg = float(
        np.linalg.norm(xyz[LEFT_KNEE] - xyz[LEFT_HIP])
        + np.linalg.norm(xyz[LEFT_ANKLE] - xyz[LEFT_KNEE])
    )
    lowered = [j for j in range(NUM_KEYPOINTS) if j not in _PLANTED]
    out[lowered, 1] -= crouch * leg
    forward = np.array([0.0, 0.0, -1.0])
    for hip, knee, ankle in _LEGS:
        thigh = float(np.linalg.norm(xyz[knee] - xyz[hip]))
        shin = float(np.linalg.norm(xyz[ankle] - xyz[knee]))
        out[knee] = _bend(out[hip], out[ankle], thigh, shin, forward)
    return out


def posed_skeleton(
    height_m: float,
    *,
    position: np.ndarray,
    yaw_deg: float = 0.0,
    crouch: float = 0.0,
) -> np.ndarray:
    """The reference skeleton placed in the room, turned, and optionally crouched.

    `crouch` is how far the hips drop, as a fraction of leg length. The feet
    stay planted and the knees bend forward to take it up, so every bone keeps
    its length and the ankles stay where a keypoint model puts them, a few
    centimetres off the floor.

    That consistency matters: a fixture whose bones or feet do not match the
    prior would make a correct lifter look wrong, and every accuracy number
    measured against it would be meaningless.
    """
    xyz = reference_skeleton(height_m)
    if crouch:
        xyz = _crouch(xyz, crouch)
    return xyz @ _yaw_matrix(yaw_deg).T + np.asarray(position, dtype=np.float64)


def reach_arms(xyz: np.ndarray, yaw_deg: float, phase: float) -> np.ndarray:
    """Swing both arms through their reach: forward, out to the sides, up and down.

    Only the elbows and wrists move. The elbow is placed by two-bone inverse
    kinematics, bending down and out, so both arm bones keep the lengths they
    had; a lifter that is scored on this fixture must not be fighting it.
    """
    out = xyz.copy()
    forward = _yaw_matrix(yaw_deg) @ np.array([0.0, 0.0, -1.0])
    up = np.array([0.0, 1.0, 0.0])
    t = 2 * np.pi * phase
    arms = (
        (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, 0.0),
        (RIGHT_SHOULDER, LEFT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, 1.9),
    )
    for shoulder, other, elbow, wrist, shift in arms:
        upper = float(np.linalg.norm(xyz[elbow] - xyz[shoulder]))
        lower = float(np.linalg.norm(xyz[wrist] - xyz[elbow]))
        lateral = xyz[shoulder] - xyz[other]
        lateral[1] = 0.0
        lateral /= np.linalg.norm(lateral)
        azimuth = np.radians(50.0 + 50.0 * np.sin(3 * t + shift))
        elevation = np.radians(5.0 + 65.0 * np.sin(2 * t + 2 * shift))
        reach = min(0.45 + 0.15 * np.sin(5 * t + shift), 0.97 * (upper + lower))
        direction = (
            np.cos(elevation) * (np.cos(azimuth) * forward + np.sin(azimuth) * lateral)
            + np.sin(elevation) * up
        )
        out[wrist] = xyz[shoulder] + reach * direction
        out[elbow] = _bend(xyz[shoulder], out[wrist], upper, lower, -up + 0.5 * lateral)
    return out


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
    motion: str = "walk",
    seed: int = 0,
) -> Scene:
    """A session of someone moving around in front of a camera.

    `motion` is "walk", a lap around the room, or "in_place": standing on one
    spot in a small play space, turning, crouching and reaching with both arms.

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
    if motion not in ("walk", "in_place"):
        raise ValueError(f"unknown motion '{motion}'")
    rng = np.random.default_rng(seed)
    rotation_cw, translation_cw = look_at_extrinsics(camera_position, look_at)

    poses: list[ScenePose] = []
    for i in range(frames):
        phase = (i / fps) / cycle_s
        if motion == "walk":
            # Cover the space: a lap around the room, turning, with some crouching.
            position = np.array(
                [
                    0.9 * np.sin(2 * np.pi * phase),
                    0.0,
                    0.5 * np.cos(2 * np.pi * phase) - 0.3,
                ]
            )
            yaw = 40.0 * np.sin(2 * np.pi * phase + 0.7)
            crouch = 0.08 * (1.0 - np.cos(4 * np.pi * phase)) / 2.0
        else:
            position = np.array(
                [
                    0.12 * np.sin(2 * np.pi * phase),
                    0.0,
                    0.08 * np.sin(4 * np.pi * phase + 0.4),
                ]
            )
            yaw = 45.0 * np.sin(2 * np.pi * phase + 0.7)
            crouch = 0.22 * (1.0 - np.cos(3 * np.pi * phase)) / 2.0

        head_pitch = 22.0 * np.sin(6 * np.pi * phase) if head_motion else 0.0
        head_yaw = 25.0 * np.sin(3 * np.pi * phase + 1.3) if head_motion else 0.0

        xyz = posed_skeleton(height_m, position=position, yaw_deg=yaw, crouch=crouch)
        if motion == "in_place":
            xyz = reach_arms(xyz, yaw, phase)
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
