"""Recorded tracking sessions.

Deliberately stores the pipeline's *inputs* — 2D keypoints and SteamVR device
poses — rather than only its outputs. Recording the outputs would let you check
that a session looked reasonable; recording the inputs lets you re-run the same
frames through a different lifter months later and compare the two fairly, on
identical data, with no camera and no headset attached. That is the only honest
way to answer "is the trained model actually better than the geometry", which
is the question the whole project turns on.

Solved output is stored alongside, so a session can be inspected without
replaying it, and so the recorded run's real timings survive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..skeleton import NUM_KEYPOINTS, TrackerRole
from ..types import DevicePose, FrameResult, Keypoints2D, Skeleton3D, TrackerTarget, VRState

FORMAT_VERSION = 1

DEVICE_NAMES = ("head", "left_hand", "right_hand")


@dataclass(slots=True)
class SessionMeta:
    """What the session was recorded with, so it can be interpreted later."""

    intrinsics: np.ndarray
    rotation_cw: np.ndarray | None = None
    translation_cw: np.ndarray | None = None
    height_m: float = 1.75
    roles: list[str] = field(default_factory=list)
    offsets: dict[str, np.ndarray] = field(default_factory=dict)
    label: str = ""
    notes: str = ""

    @property
    def calibrated(self) -> bool:
        return self.rotation_cw is not None and self.translation_cw is not None


@dataclass(slots=True)
class Session:
    """A recorded run, as parallel arrays over ``n`` frames."""

    meta: SessionMeta
    timestamps: np.ndarray  # (n,)
    keypoints_xy: np.ndarray  # (n, 26, 2), NaN where the frame produced nothing
    keypoints_scores: np.ndarray  # (n, 26)
    device_positions: np.ndarray  # (n, 3, 3) indexed by DEVICE_NAMES
    device_rotations: np.ndarray  # (n, 3, 3, 3)
    device_valid: np.ndarray  # (n, 3) bool
    skeleton_xyz: np.ndarray  # (n, 26, 3), NaN where nothing was solved
    target_positions: np.ndarray  # (n, roles, 3)
    target_rotations: np.ndarray  # (n, roles, 3, 3)
    target_valid: np.ndarray  # (n, roles) bool
    target_stale: np.ndarray  # (n, roles) bool
    latency_ms: np.ndarray  # (n,) capture to targets ready
    stage_ms: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def duration_s(self) -> float:
        if len(self) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def fps(self) -> float:
        return len(self) / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def roles(self) -> list[TrackerRole]:
        return [TrackerRole(r) for r in self.meta.roles]

    def tracked(self) -> np.ndarray:
        """(n,) mask of frames where a 3D pose was actually solved."""
        return ~np.isnan(self.skeleton_xyz[:, 0, 0])

    def has_keypoints(self) -> np.ndarray:
        return ~np.isnan(self.keypoints_xy[:, 0, 0])

    # -- reconstructing pipeline inputs ------------------------------------

    def keypoints_at(self, index: int) -> Keypoints2D | None:
        if not self.has_keypoints()[index]:
            return None
        return Keypoints2D(
            xy=self.keypoints_xy[index].astype(np.float32),
            scores=self.keypoints_scores[index].astype(np.float32),
            timestamp=float(self.timestamps[index]),
        )

    def vr_at(self, index: int) -> VRState | None:
        poses: dict[str, DevicePose | None] = {}
        for slot, name in enumerate(DEVICE_NAMES):
            if not self.device_valid[index, slot]:
                poses[name] = None
                continue
            poses[name] = DevicePose(
                position=self.device_positions[index, slot].copy(),
                rotation=self.device_rotations[index, slot].copy(),
                valid=True,
                timestamp=float(self.timestamps[index]),
            )
        if all(pose is None for pose in poses.values()):
            return None
        return VRState(timestamp=float(self.timestamps[index]), **poses)

    def device_track(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Positions and validity for one device across the session."""
        slot = DEVICE_NAMES.index(name)
        return self.device_positions[:, slot], self.device_valid[:, slot]

    def target_track(self, role: TrackerRole) -> tuple[np.ndarray, np.ndarray]:
        index = self.meta.roles.index(role.value)
        return self.target_positions[:, index], self.target_valid[:, index]

    # -- persistence --------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "format_version": FORMAT_VERSION,
            "timestamps": self.timestamps,
            "keypoints_xy": self.keypoints_xy,
            "keypoints_scores": self.keypoints_scores,
            "device_positions": self.device_positions,
            "device_rotations": self.device_rotations,
            "device_valid": self.device_valid,
            "skeleton_xyz": self.skeleton_xyz,
            "target_positions": self.target_positions,
            "target_rotations": self.target_rotations,
            "target_valid": self.target_valid,
            "target_stale": self.target_stale,
            "latency_ms": self.latency_ms,
            "meta_intrinsics": self.meta.intrinsics,
            "meta_height_m": self.meta.height_m,
            "meta_roles": np.array(self.meta.roles, dtype=object),
            "meta_label": self.meta.label,
            "meta_notes": self.meta.notes,
        }
        if self.meta.rotation_cw is not None:
            payload["meta_rotation_cw"] = self.meta.rotation_cw
            payload["meta_translation_cw"] = self.meta.translation_cw
        for name, offset in self.meta.offsets.items():
            payload[f"meta_offset_{name}"] = offset
        for name, values in self.stage_ms.items():
            payload[f"stage_{name}"] = values

        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: Path | str) -> Session:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        with np.load(path, allow_pickle=True) as data:
            version = int(data["format_version"])
            if version != FORMAT_VERSION:
                raise ValueError(
                    f"{path} is format version {version}, this build reads {FORMAT_VERSION}"
                )
            meta = SessionMeta(
                intrinsics=data["meta_intrinsics"],
                rotation_cw=data["meta_rotation_cw"] if "meta_rotation_cw" in data else None,
                translation_cw=(
                    data["meta_translation_cw"] if "meta_translation_cw" in data else None
                ),
                height_m=float(data["meta_height_m"]),
                roles=[str(r) for r in data["meta_roles"]],
                offsets={
                    key[len("meta_offset_") :]: data[key]
                    for key in data.files
                    if key.startswith("meta_offset_")
                },
                label=str(data["meta_label"]),
                notes=str(data["meta_notes"]),
            )
            return cls(
                meta=meta,
                timestamps=data["timestamps"],
                keypoints_xy=data["keypoints_xy"],
                keypoints_scores=data["keypoints_scores"],
                device_positions=data["device_positions"],
                device_rotations=data["device_rotations"],
                device_valid=data["device_valid"],
                skeleton_xyz=data["skeleton_xyz"],
                target_positions=data["target_positions"],
                target_rotations=data["target_rotations"],
                target_valid=data["target_valid"],
                target_stale=data["target_stale"],
                latency_ms=data["latency_ms"],
                stage_ms={
                    key[len("stage_") :]: data[key]
                    for key in data.files
                    if key.startswith("stage_")
                },
            )


class SessionRecorder:
    """Accumulates frames into a `Session`.

    Kept append-only and allocation-light so it can sit inside the live loop
    without perturbing the very latency it is there to measure.
    """

    def __init__(self, meta: SessionMeta) -> None:
        self.meta = meta
        self._roles = [TrackerRole(r) for r in meta.roles]
        self._timestamps: list[float] = []
        self._keypoints_xy: list[np.ndarray] = []
        self._keypoints_scores: list[np.ndarray] = []
        self._device_positions: list[np.ndarray] = []
        self._device_rotations: list[np.ndarray] = []
        self._device_valid: list[np.ndarray] = []
        self._skeleton: list[np.ndarray] = []
        self._target_positions: list[np.ndarray] = []
        self._target_rotations: list[np.ndarray] = []
        self._target_valid: list[np.ndarray] = []
        self._target_stale: list[np.ndarray] = []
        self._latency: list[float] = []
        self._stage: dict[str, list[float]] = {}

    def __len__(self) -> int:
        return len(self._timestamps)

    def add(self, result: FrameResult, *, latency_ms: float = float("nan")) -> None:
        self._timestamps.append(result.timestamp)
        self._add_keypoints(result.keypoints2d)
        self._add_vr(result.vr)
        self._add_skeleton(result.skeleton3d)
        self._add_targets(result.targets)
        self._latency.append(latency_ms)

        frame = len(self._timestamps) - 1
        for name, value in result.timings.items():
            series = self._stage.setdefault(name, [])
            series.extend([float("nan")] * (frame - len(series)))
            series.append(value)

    def _add_keypoints(self, keypoints: Keypoints2D | None) -> None:
        if keypoints is None:
            self._keypoints_xy.append(np.full((NUM_KEYPOINTS, 2), np.nan))
            self._keypoints_scores.append(np.zeros(NUM_KEYPOINTS))
            return
        self._keypoints_xy.append(np.asarray(keypoints.xy, dtype=np.float64))
        self._keypoints_scores.append(np.asarray(keypoints.scores, dtype=np.float64))

    def _add_vr(self, vr: VRState | None) -> None:
        positions = np.full((3, 3), np.nan)
        rotations = np.tile(np.eye(3), (3, 1, 1))
        valid = np.zeros(3, dtype=bool)
        if vr is not None:
            for slot, name in enumerate(DEVICE_NAMES):
                pose = getattr(vr, name)
                if pose is None or not pose.valid:
                    continue
                positions[slot] = pose.position
                rotations[slot] = pose.rotation
                valid[slot] = True
        self._device_positions.append(positions)
        self._device_rotations.append(rotations)
        self._device_valid.append(valid)

    def _add_skeleton(self, skeleton: Skeleton3D | None) -> None:
        if skeleton is None:
            self._skeleton.append(np.full((NUM_KEYPOINTS, 3), np.nan))
            return
        self._skeleton.append(np.asarray(skeleton.xyz, dtype=np.float64))

    def _add_targets(self, targets: list[TrackerTarget]) -> None:
        by_role = {t.role: t for t in targets}
        positions = np.full((len(self._roles), 3), np.nan)
        rotations = np.tile(np.eye(3), (len(self._roles), 1, 1))
        valid = np.zeros(len(self._roles), dtype=bool)
        stale = np.zeros(len(self._roles), dtype=bool)
        for index, role in enumerate(self._roles):
            target = by_role.get(role)
            if target is None or not target.valid:
                continue
            positions[index] = target.position
            rotations[index] = target.rotation
            valid[index] = True
            stale[index] = target.stale
        self._target_positions.append(positions)
        self._target_rotations.append(rotations)
        self._target_valid.append(valid)
        self._target_stale.append(stale)

    def build(self) -> Session:
        frames = len(self._timestamps)
        stage = {}
        for name, series in self._stage.items():
            padded = series + [float("nan")] * (frames - len(series))
            stage[name] = np.array(padded)

        return Session(
            meta=self.meta,
            timestamps=np.array(self._timestamps),
            keypoints_xy=np.array(self._keypoints_xy).reshape(frames, NUM_KEYPOINTS, 2),
            keypoints_scores=np.array(self._keypoints_scores).reshape(frames, NUM_KEYPOINTS),
            device_positions=np.array(self._device_positions).reshape(frames, 3, 3),
            device_rotations=np.array(self._device_rotations).reshape(frames, 3, 3, 3),
            device_valid=np.array(self._device_valid).reshape(frames, 3),
            skeleton_xyz=np.array(self._skeleton).reshape(frames, NUM_KEYPOINTS, 3),
            target_positions=np.array(self._target_positions).reshape(frames, len(self._roles), 3),
            target_rotations=np.array(self._target_rotations).reshape(
                frames, len(self._roles), 3, 3
            ),
            target_valid=np.array(self._target_valid).reshape(frames, len(self._roles)),
            target_stale=np.array(self._target_stale).reshape(frames, len(self._roles)),
            latency_ms=np.array(self._latency),
            stage_ms=stage,
        )
