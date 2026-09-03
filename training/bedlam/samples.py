"""Turning a pose sequence into the windows a temporal lifter trains on.

The lifter sees a short history of 2D keypoints plus the headset and
controllers, and has to emit metric 3D. A training sample is therefore a
window, not a frame: long enough to contain a step or a weight shift, short
enough to fit on an 8 GB card in a batch.

Two properties of the storage format are load-bearing.

The first is that ground-truth 3D lives in play space, the same frame the
runtime lifter is asked to produce. Training in camera space and converting
afterwards would hide a systematic tilt in the numbers until someone put a
headset on.

The second is that 2D observations are stored *after* the detector noise
model, not as clean projections. The clean projections stay in the sequence
so a noise model can be reapplied later; the shard the trainer loads is what
the model will actually see.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bodytracker.geom import camera_from_play, play_from_camera, project_points, slerp
from bodytracker.skeleton import LEFT_ANKLE, RIGHT_ANKLE
from bodytracker.types import VRState

from . import devices
from .coords import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    camera_height,
    play_space_transform,
    rotation_cw_from_extrinsics,
)
from .joints import in_frame, to_halpe
from .noise import KeypointNoise

TARGET_FPS = 30.0
DEFAULT_WINDOW_S = 0.9
DEFAULT_STRIDE_S = 0.45
SHARD_VERSION = 1


@dataclass(slots=True)
class PoseSequence:
    """One person's motion, already in the frames the lifter uses."""

    xyz_play: np.ndarray  # (F, 26, 3)
    xy: np.ndarray  # (F, 26, 2) clean projections
    scores: np.ndarray  # (F, 26)
    device_pos: np.ndarray  # (F, 3, 3)
    device_rot: np.ndarray  # (F, 3, 3, 3)
    device_valid: np.ndarray  # (F, 3) bool
    timestamps: np.ndarray  # (F,)
    fps: float
    rotation_cw: np.ndarray  # (3, 3)
    translation_cw: np.ndarray  # (3,)
    intrinsics: np.ndarray  # (3, 3)
    source: str = ""

    def __len__(self) -> int:
        return int(self.xyz_play.shape[0])


@dataclass(slots=True)
class SampleWindow:
    """One training example: a fixed-length clip at `TARGET_FPS`."""

    xy: np.ndarray  # (T, 26, 2)
    scores: np.ndarray  # (T, 26)
    xyz_play: np.ndarray  # (T, 26, 3)
    device_pos: np.ndarray  # (T, 3, 3)
    device_rot: np.ndarray  # (T, 3, 3, 3)
    device_valid: np.ndarray  # (T, 3)
    timestamps: np.ndarray  # (T,)
    rotation_cw: np.ndarray
    translation_cw: np.ndarray
    intrinsics: np.ndarray
    source: str = ""


def from_scene(scene, source: str = "synthetic") -> PoseSequence:
    """Pack a `tools.scene.Scene` into a PoseSequence.

    Exists so the training format can be exercised, and the lifter trained,
    before any BEDLAM bytes have been downloaded. The synthetic scene is the
    same fixture the eval harness already trusts.
    """
    frames = scene.frames
    xyz = np.stack([f.play_xyz for f in frames])
    xy = np.stack([f.keypoints.xy for f in frames]).astype(np.float64)
    scores = np.stack([f.keypoints.scores for f in frames]).astype(np.float64)
    packed = [devices.pack(f.vr) for f in frames]
    return PoseSequence(
        xyz_play=xyz.astype(np.float64),
        xy=xy,
        scores=scores,
        device_pos=np.stack([p[0] for p in packed]),
        device_rot=np.stack([p[1] for p in packed]),
        device_valid=np.stack([p[2] for p in packed]),
        timestamps=np.array([f.keypoints.timestamp for f in frames], dtype=np.float64),
        fps=(
            1.0 / max(frames[1].keypoints.timestamp - frames[0].keypoints.timestamp, 1e-9)
            if len(frames) >= 2
            else 30.0
        ),
        rotation_cw=np.asarray(scene.rotation_cw, dtype=np.float64),
        translation_cw=np.asarray(scene.translation_cw, dtype=np.float64),
        intrinsics=np.asarray(scene.intrinsics, dtype=np.float64),
        source=source,
    )


def from_bedlam_frame(
    xy_src: np.ndarray,
    xyz_cam: np.ndarray,
    cam_int: np.ndarray,
    cam_ext: np.ndarray,
    timestamp: float = 0.0,
    *,
    layout: str | None = None,
    offsets: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, VRState, np.ndarray, np.ndarray]:
    """One BEDLAM frame → Halpe 2D, play-space 3D, devices, extrinsics.

    `xyz_cam` is Halpe-or-source joints in OpenCV camera space, metres, already
    past the SMPL-X forward pass. Mapping to Halpe happens here so the caller
    can hand us either layout.
    """
    xyz_h = to_halpe(np.asarray(xyz_cam, dtype=np.float64), layout)
    xy_h = to_halpe(np.asarray(xy_src, dtype=np.float64)[..., :2], layout)

    feet = xyz_h[[LEFT_ANKLE, RIGHT_ANKLE]]
    height = camera_height(feet, cam_ext)
    rotation_cw, translation_cw = play_space_transform(cam_ext, height)
    xyz_play = play_from_camera(xyz_h, rotation_cw, translation_cw)
    vr = devices.from_skeleton(xyz_play, timestamp=timestamp, offsets=offsets)
    return xy_h, xyz_play, xyz_h, vr, rotation_cw, translation_cw


def resample(sequence: PoseSequence, fps: float = TARGET_FPS) -> PoseSequence:
    """Resample a sequence onto a uniform grid at `fps`.

    BEDLAM's released training images are mostly 6 fps; the tracker runs at
    30. Interpolating the ground-truth 3D is honest, because the body moved
    smoothly between those frames. Interpolating the 2D observations would
    invent a detector that is smoother than RTMPose, so those are re-projected
    from the resampled 3D instead. Detector noise is applied *after* this, at
    the target rate, which is what puts the high-frequency jitter back.
    """
    if len(sequence) < 2:
        return sequence
    src_t = np.asarray(sequence.timestamps, dtype=np.float64)
    duration = float(src_t[-1] - src_t[0])
    if duration <= 0:
        return sequence
    n = max(int(round(duration * fps)) + 1, 2)
    dst_t = src_t[0] + np.arange(n) / fps
    dst_t = dst_t[dst_t <= src_t[-1] + 1e-9]
    n = len(dst_t)

    def lerp(values: np.ndarray) -> np.ndarray:
        flat_src = values.reshape(len(src_t), -1)
        flat_dst = np.empty((n, flat_src.shape[1]), dtype=np.float64)
        for col in range(flat_src.shape[1]):
            flat_dst[:, col] = np.interp(dst_t, src_t, flat_src[:, col])
        return flat_dst.reshape(n, *values.shape[1:])

    xyz = lerp(sequence.xyz_play)
    camera_xyz = camera_from_play(xyz, sequence.rotation_cw, sequence.translation_cw)
    xy = np.stack([project_points(frame, sequence.intrinsics) for frame in camera_xyz])
    scores = np.where(
        in_frame(xy, IMAGE_WIDTH, IMAGE_HEIGHT),
        0.95,
        0.05,
    ).astype(np.float64)

    device_pos = lerp(sequence.device_pos)
    device_valid = lerp(sequence.device_valid.astype(np.float64)) > 0.5
    device_rot = np.empty((n, 3, 3, 3), dtype=np.float64)
    for d in range(3):
        for i, t in enumerate(dst_t):
            # Piecewise slerp between the surrounding source frames.
            idx = int(np.searchsorted(src_t, t, side="right") - 1)
            idx = min(max(idx, 0), len(src_t) - 2)
            span = src_t[idx + 1] - src_t[idx]
            alpha = 0.0 if span <= 0 else float((t - src_t[idx]) / span)
            device_rot[i, d] = slerp(
                sequence.device_rot[idx, d], sequence.device_rot[idx + 1, d], alpha
            )

    return PoseSequence(
        xyz_play=xyz,
        xy=xy,
        scores=scores,
        device_pos=device_pos,
        device_rot=device_rot,
        device_valid=device_valid,
        timestamps=dst_t,
        fps=fps,
        rotation_cw=sequence.rotation_cw,
        translation_cw=sequence.translation_cw,
        intrinsics=sequence.intrinsics,
        source=sequence.source,
    )


def window_starts(
    n_frames: int,
    fps: float,
    window_s: float = DEFAULT_WINDOW_S,
    stride_s: float = DEFAULT_STRIDE_S,
) -> list[int]:
    """Leading indices of every window that fits."""
    length = max(int(round(window_s * fps)), 2)
    stride = max(int(round(stride_s * fps)), 1)
    if n_frames < length:
        return []
    return list(range(0, n_frames - length + 1, stride))


def windows(
    sequence: PoseSequence,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    stride_s: float = DEFAULT_STRIDE_S,
    noise: KeypointNoise | None = None,
    rng: np.random.Generator | None = None,
) -> list[SampleWindow]:
    """Cut a sequence into overlapping training windows, with detector noise."""
    length = max(int(round(window_s * sequence.fps)), 2)
    starts = window_starts(len(sequence), sequence.fps, window_s, stride_s)
    out: list[SampleWindow] = []
    for start in starts:
        sl = slice(start, start + length)
        xy = sequence.xy[sl].copy()
        scores = sequence.scores[sl].copy()
        if noise is not None:
            xy, scores = noise(xy, rng=rng)
        out.append(
            SampleWindow(
                xy=xy,
                scores=scores,
                xyz_play=sequence.xyz_play[sl].copy(),
                device_pos=sequence.device_pos[sl].copy(),
                device_rot=sequence.device_rot[sl].copy(),
                device_valid=sequence.device_valid[sl].copy(),
                timestamps=sequence.timestamps[sl].copy(),
                rotation_cw=sequence.rotation_cw,
                translation_cw=sequence.translation_cw,
                intrinsics=sequence.intrinsics,
                source=sequence.source,
            )
        )
    return out


def save_shard(path: Path | str, samples: list[SampleWindow], *, notes: str = "") -> Path:
    """Write a list of windows as one compressed npz."""
    if not samples:
        raise ValueError("no samples to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    length = samples[0].xy.shape[0]
    if any(s.xy.shape[0] != length for s in samples):
        raise ValueError("all windows in a shard must have the same length")

    meta = {
        "version": SHARD_VERSION,
        "n": len(samples),
        "T": length,
        "fps": TARGET_FPS,
        "notes": notes,
        "sources": [s.source for s in samples],
    }
    np.savez_compressed(
        path,
        xy=np.stack([s.xy for s in samples]),
        scores=np.stack([s.scores for s in samples]),
        xyz_play=np.stack([s.xyz_play for s in samples]),
        device_pos=np.stack([s.device_pos for s in samples]),
        device_rot=np.stack([s.device_rot for s in samples]),
        device_valid=np.stack([s.device_valid for s in samples]),
        timestamps=np.stack([s.timestamps for s in samples]),
        rotation_cw=np.stack([s.rotation_cw for s in samples]),
        translation_cw=np.stack([s.translation_cw for s in samples]),
        intrinsics=np.stack([s.intrinsics for s in samples]),
        meta=np.asarray(json.dumps(meta)),
    )
    return path


def load_shard(path: Path | str) -> list[SampleWindow]:
    """Read a shard written by `save_shard`."""
    data = np.load(Path(path), allow_pickle=False)
    n = int(data["xy"].shape[0])
    meta = json.loads(str(data["meta"]))
    sources = meta.get("sources", [""] * n)
    samples = []
    for i in range(n):
        samples.append(
            SampleWindow(
                xy=data["xy"][i],
                scores=data["scores"][i],
                xyz_play=data["xyz_play"][i],
                device_pos=data["device_pos"][i],
                device_rot=data["device_rot"][i],
                device_valid=data["device_valid"][i],
                timestamps=data["timestamps"][i],
                rotation_cw=data["rotation_cw"][i],
                translation_cw=data["translation_cw"][i],
                intrinsics=data["intrinsics"][i],
                source=sources[i] if i < len(sources) else "",
            )
        )
    return samples


def load_bedlam_npz(
    path: Path | str,
    *,
    offsets: dict[str, np.ndarray] | None = None,
    layout: str | None = None,
    max_frames: int | None = None,
) -> PoseSequence:
    """Read one of BEDLAM's processed per-scene npz files.

    Expects the keys ``df_full_body.py`` writes: ``gtkps``, ``cam_int``,
    ``cam_ext``, and, if present, ``xyz_cam`` or ``joints3d`` for camera-space
    3D. Without 3D joints this cannot invent depth from the projections, and
    it will raise rather than silently training on a planar skeleton.

    Reconstructing 3D from ``pose_cam`` / ``shape`` / ``trans_cam`` needs
    ``smplx`` and the model files; that path is Phase 4b's training-box job.
    What this function guarantees is that once those joints exist, they land
    in play space with devices attached.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    if "gtkps" not in data:
        raise KeyError(f"{path} has no gtkps; is this a BEDLAM processed npz?")

    gtkps = np.asarray(data["gtkps"], dtype=np.float64)
    if gtkps.ndim != 3:
        raise ValueError(f"gtkps should be (N, J, 2 or 3), got {gtkps.shape}")

    xyz_key = next((k for k in ("xyz_cam", "joints3d", "xyz_play") if k in data), None)
    if xyz_key is None:
        raise KeyError(
            f"{path} has no camera-space 3D joints (xyz_cam / joints3d). "
            "Run the SMPL-X forward pass on the training box, or pass a file "
            "that already stores them."
        )
    xyz_src = np.asarray(data[xyz_key], dtype=np.float64)
    in_play = xyz_key == "xyz_play"

    cam_int = np.asarray(data["cam_int"], dtype=np.float64)
    cam_ext = np.asarray(data["cam_ext"], dtype=np.float64)
    n = len(gtkps) if max_frames is None else min(len(gtkps), max_frames)

    xy_all, xyz_all, pos_all, rot_all, valid_all = [], [], [], [], []
    rotation_cw = translation_cw = None
    for i in range(n):
        k = cam_int[i] if cam_int.ndim == 3 else cam_int
        e = cam_ext[i] if cam_ext.ndim == 3 else cam_ext
        if in_play:
            xyz_h = to_halpe(xyz_src[i], layout)
            xy_h = to_halpe(gtkps[i][..., :2], layout)
            xyz_play = xyz_h
            if rotation_cw is None:
                rotation_cw = rotation_cw_from_extrinsics(e)
                translation_cw = np.asarray(e, dtype=np.float64)[:3, 3]
            vr = devices.from_skeleton(xyz_play, timestamp=i / 30.0, offsets=offsets)
        else:
            xy_h, xyz_play, _, vr, rotation_cw, translation_cw = from_bedlam_frame(
                gtkps[i], xyz_src[i], k, e, timestamp=i / 30.0, layout=layout, offsets=offsets
            )
        xy_all.append(xy_h)
        xyz_all.append(xyz_play)
        pos, rot, valid = devices.pack(vr)
        pos_all.append(pos)
        rot_all.append(rot)
        valid_all.append(valid)

    xy = np.stack(xy_all)
    width = int(round(float((cam_int[0] if cam_int.ndim == 3 else cam_int)[0, 2] * 2)))
    height = int(round(float((cam_int[0] if cam_int.ndim == 3 else cam_int)[1, 2] * 2)))
    scores = np.where(in_frame(xy, width or IMAGE_WIDTH, height or IMAGE_HEIGHT), 0.95, 0.05)

    return PoseSequence(
        xyz_play=np.stack(xyz_all),
        xy=xy,
        scores=scores.astype(np.float64),
        device_pos=np.stack(pos_all),
        device_rot=np.stack(rot_all),
        device_valid=np.stack(valid_all),
        timestamps=np.arange(n, dtype=np.float64) / 30.0,
        fps=30.0,
        rotation_cw=np.asarray(rotation_cw, dtype=np.float64),
        translation_cw=np.asarray(translation_cw, dtype=np.float64),
        intrinsics=np.asarray(cam_int[0] if cam_int.ndim == 3 else cam_int, dtype=np.float64),
        source=path.stem,
    )
