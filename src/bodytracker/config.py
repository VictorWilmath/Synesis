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
class IntrinsicsConfig:
    fallback_fov_deg: float = 62.0
    path: str = "calibration/intrinsics.json"


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
    intrinsics: IntrinsicsConfig = field(default_factory=IntrinsicsConfig)


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
    path: str = "calibration/extrinsics.json"
    # High-RMS room solves are worse than no solve: use automatic headset
    # alignment instead of loading a calibration that cannot be trusted.
    max_accepted_rms_px: float = 12.0
    # Extrinsics needs varied, accurate samples rather than every video frame.
    # Limiting pose inference prevents it from starving SteamVR's compositor.
    sample_rate_hz: float = 2.0
    # A headset-only session produces one correspondence per position. Sixty
    # varied samples prevent a noisy webcam solve from choosing a mirror pose.
    min_samples: int = 60
    min_sample_spacing_m: float = 0.03
    ransac_reproj_threshold_px: float = 8.0
    online_refine: bool = True
    refine_interval_s: float = 5.0
    # With visible hand controllers, a 30 cm head span plus arm motion gives
    # enough 3D diversity for a stable solve in a compact room.
    min_coverage_m: float = 0.30


@dataclass(slots=True)
class HeadsetAlignmentConfig:
    """Low-friction model-space to SteamVR-space alignment.

    This is deliberately separate from camera extrinsics. It requires only a
    brief, neutral pose with the HMD on and is the normal first-run path.
    """

    path: str = "calibration/headset_alignment.json"
    capture_seconds: float = 5.0
    min_samples: int = 20


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


@dataclass(slots=True)
class FilterConfig:
    min_cutoff: float = 1.0
    beta: float = 2.0
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
    host: str = "127.0.0.1"
    port: int = 9000
    roles: list[str] = field(default_factory=lambda: ["hip", "left_foot", "right_foot"])
    # Synesis maps tracker positions into SteamVR play space itself. Sending
    # the OSC head endpoints would make VRChat continuously move that already
    # aligned source space again, so they are an advanced opt-in only.
    send_head: bool = False
    rebase_uncalibrated_to_head: bool = True
    send_rate_hz: int = 60

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
    alignment: HeadsetAlignmentConfig = field(default_factory=HeadsetAlignmentConfig)
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
