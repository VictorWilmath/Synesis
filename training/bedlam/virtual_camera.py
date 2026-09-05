"""Turn real BEDLAM motion into webcam-like training sequences locally.

The released gendered motion files contain genuine human motion but no camera
metadata.  This module supplies controlled, static webcam placements instead
of pretending that missing BEDLAM scene data exists.  It is useful for the
first lifter because the target motion is real; later rendered-camera labels
can be added as an additional source of variation.
"""

from __future__ import annotations

import numpy as np

from bodytracker.geom import (
    camera_from_play,
    intrinsics_from_fov,
    look_at_extrinsics,
    project_points,
)
from bodytracker.skeleton import HIP, LEFT_ANKLE, RIGHT_ANKLE

from . import devices
from .joints import from_smplx_body, in_frame
from .samples import PoseSequence

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


def _floor_normalize(points: np.ndarray) -> np.ndarray:
    """Make an SMPL-X sequence a floor-origin play-space sequence.

    Motion capture starts in an arbitrary horizontal location.  Keeping that
    location would make every virtual camera see the same placement, while
    removing its first-frame root translation gives the sampler a clean room
    origin without changing any bone or motion.
    """
    xyz = np.asarray(points, dtype=np.float64).copy()
    if xyz.ndim != 3 or xyz.shape[1:] != (26, 3):
        raise ValueError(f"expected (frames, 26, 3) Halpe joints, got {xyz.shape}")
    root = xyz[0, HIP]
    floor = float(np.nanmin(xyz[:, [LEFT_ANKLE, RIGHT_ANKLE], 1]))
    xyz[..., 0] -= root[0]
    xyz[..., 2] -= root[2]
    xyz[..., 1] -= floor
    return xyz


def virtual_camera_sequence(
    smplx_joints: np.ndarray,
    *,
    fps: float = 30.0,
    seed: int = 0,
    source: str = "virtual-camera",
) -> PoseSequence:
    """Project true SMPL-X joints through one randomized, static webcam.

    The camera is deliberately static: that matches the physical webcam the
    application supports and avoids training the lifter to explain a moving
    viewpoint with a moving body.  The range covers typical desk and tripod
    placements at 720p while retaining all camera calibration inputs.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    raw = np.asarray(smplx_joints, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[-1] != 3 or raw.shape[1] < 25:
        raise ValueError(f"expected (frames, >=25, 3) SMPL-X joints, got {raw.shape}")

    xyz_play = _floor_normalize(from_smplx_body(raw))
    rng = np.random.default_rng(seed)
    target = np.nanmedian(xyz_play[:, HIP], axis=0)
    target[1] = float(np.clip(target[1] + 0.9, 0.85, 1.2))
    camera_position = np.array(
        [
            rng.uniform(-0.65, 0.65),
            rng.uniform(0.85, 1.65),
            rng.uniform(2.2, 3.3),
        ]
    )
    rotation_cw, translation_cw = look_at_extrinsics(camera_position, target)
    intrinsics = intrinsics_from_fov(IMAGE_WIDTH, IMAGE_HEIGHT, rng.uniform(58.0, 76.0))
    camera_xyz = camera_from_play(xyz_play, rotation_cw, translation_cw)
    xy = np.stack([project_points(frame, intrinsics) for frame in camera_xyz])
    visible = in_frame(xy, IMAGE_WIDTH, IMAGE_HEIGHT) & (camera_xyz[..., 2] > 0.1)
    scores = np.where(visible, 0.95, 0.05).astype(np.float64)

    positions, rotations, valid = [], [], []
    for frame, timestamp in zip(
        xyz_play, np.arange(len(xyz_play), dtype=np.float64) / fps, strict=True
    ):
        packed = devices.pack(devices.from_skeleton(frame, timestamp=timestamp))
        positions.append(packed[0])
        rotations.append(packed[1])
        valid.append(packed[2])
    return PoseSequence(
        xyz_play=xyz_play,
        xy=xy,
        scores=scores,
        device_pos=np.stack(positions),
        device_rot=np.stack(rotations),
        device_valid=np.stack(valid),
        timestamps=np.arange(len(xyz_play), dtype=np.float64) / fps,
        fps=float(fps),
        rotation_cw=rotation_cw,
        translation_cw=translation_cw,
        intrinsics=intrinsics,
        source=source,
    )
