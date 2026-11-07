"""Configuration loading.

`configs/default.toml` is the baseline and is version-controlled.
`configs/local.toml` is per-machine (camera index, exposure, tracker roles) and
is gitignored. The local file is overlaid recursively, so it only needs to
contain the keys it changes.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

from .skeleton import TrackerRole


@dataclass(slots=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    lock_exposure: bool = True
    exposure: float = -6.0
    gain: float = 0.0
    autofocus: bool = False
    fourcc: str = "MJPG"
    mirror: bool = False


@dataclass(slots=True)
class Pose2DConfig:
    mode: str = "balanced"
    backend: str = "onnxruntime"
    device: str = "cuda"
    detect_interval: int = 30
    bbox_padding: float = 0.25
    min_keypoint_score: float = 0.35
    min_valid_ratio: float = 0.5


@dataclass(slots=True)
class VRConfig:
    enabled: bool = True
    poll_hz: int = 90


@dataclass(slots=True)
class CalibrationConfig:
    """Automatic camera calibration from the skeleton and SteamVR poses."""

    path: str = "calibration/camera.json"
    # Lens assumed until the solve measures it.
    initial_fov_deg: float = 62.0
    # Frame delay assumed until the solve measures it, and the range searched.
    initial_latency_s: float = 0.05
    max_latency_s: float = 0.50
    latency_step_s: float = 0.02
    # Frames per second considered for new samples; most are then skipped as
    # repeats unless the user has moved.
    sample_rate_hz: float = 10.0
    min_sample_spacing_m: float = 0.04
    min_samples: int = 150
    max_samples: int = 900
    solve_interval_s: float = 3.0
    refine_interval_s: float = 30.0
    ransac_threshold_px: float = 10.0
    # Acceptance: how well the samples must fit and how well determined the
    # camera must be before anything is sent to VRChat.
    min_inlier_ratio: float = 0.6
    max_rms_px: float = 8.0
    max_position_std_m: float = 0.08
    max_focal_std: float = 0.04
    # A sustained reprojection error this large means the camera was moved.
    moved_error_px: float = 30.0
    monitor_frames: int = 40


@dataclass(slots=True)
class BodyConfig:
    height_m: float = 1.75
    path: str = "calibration/body.json"


@dataclass(slots=True)
class LiftConfig:
    # "anchored" pins the head to the HMD and falls back to "geometric" when
    # there is no headset or no calibration. "neural" is the trained lifter.
    method: str = "anchored"
    # The temporal model is inexpensive on CPU and must not compete with
    # SteamVR's compositor for the VR GPU.
    device: str = "cpu"
    model_path: str = "models/lifter.onnx"
    window: int = 27
    max_head_residual_m: float = 0.35
    # Anchored only: solve all joint depths together against the headset,
    # the controllers and the floor, rather than one bone at a time.
    refine: bool = True


@dataclass(slots=True)
class FilterConfig:
    min_cutoff: float = 1.0
    beta: float = 8.0
    d_cutoff: float = 1.0
    rotation_alpha: float = 0.35
    hold_last_valid_s: float = 0.5


@dataclass(slots=True)
class SolveConfig:
    enforce_bone_lengths: bool = True
    floor_clamp: bool = True
    floor_epsilon_m: float = 0.03


@dataclass(slots=True)
class OSCConfig:
    # "vrchat" sends VRChat's own OSC trackers. "vmt" drives virtual SteamVR
    # trackers through the Virtual Motion Tracker driver instead, which VRChat
    # then calibrates like Vive trackers. "auto" is "vmt" when VMT is
    # installed and "vrchat" otherwise.
    target: str = "vrchat"
    host: str = "127.0.0.1"
    port: int = 9000
    roles: list[str] = field(default_factory=lambda: ["hip", "left_foot", "right_foot"])
    # Synesis maps tracker positions into SteamVR play space itself. Sending
    # the OSC head endpoints would make VRChat continuously move that already
    # aligned source space again, so they are an advanced opt-in only.
    send_head: bool = False
    send_rate_hz: int = 60
    vmt_port: int = 39570
    # 7 is VMT's Vive tracker compatible mode, 1 its plain tracker.
    vmt_mode: int = 7

    def tracker_roles(self) -> list[TrackerRole]:
        return [TrackerRole(r) for r in self.roles]


@dataclass(slots=True)
class DebugConfig:
    overlay: bool = True
    print_stats_interval_s: float = 2.0


@dataclass(slots=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    pose2d: Pose2DConfig = field(default_factory=Pose2DConfig)
    vr: VRConfig = field(default_factory=VRConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    body: BodyConfig = field(default_factory=BodyConfig)
    lift: LiftConfig = field(default_factory=LiftConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    solve: SolveConfig = field(default_factory=SolveConfig)
    osc: OSCConfig = field(default_factory=OSCConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


T = TypeVar("T")


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _build(cls: type[T], data: dict[str, Any]) -> T:
    """Recursively construct a config dataclass from parsed TOML.

    Unknown keys are an error rather than a silent no-op, so a typo in a local
    config surfaces immediately instead of quietly leaving a default in place.
    """
    # `from __future__ import annotations` stringifies every field type, so the
    # nested-dataclass check needs the resolved hints.
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown config key '{key}' in [{cls.__name__}]")
        ftype = hints[key]
        if isinstance(value, dict) and is_dataclass(ftype):
            kwargs[key] = _build(ftype, value)  # type: ignore[arg-type]
        else:
            kwargs[key] = value
    return cls(**kwargs)  # type: ignore[call-arg]


DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load(path: Path | str | None = None, *, local: bool = True) -> Config:
    """Load `default.toml`, overlaid with `local.toml` when it exists."""
    config_dir = Path(path).parent if path else DEFAULT_CONFIG_DIR
    base_path = Path(path) if path else config_dir / "default.toml"

    data: dict[str, Any] = {}
    if base_path.exists():
        with base_path.open("rb") as fh:
            data = tomllib.load(fh)

    if local:
        local_path = config_dir / "local.toml"
        if local_path.exists():
            with local_path.open("rb") as fh:
                data = _merge(data, tomllib.load(fh))

    return _build(Config, data)
