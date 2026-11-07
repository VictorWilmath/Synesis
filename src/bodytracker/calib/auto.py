"""Automatic camera calibration from the skeleton and SteamVR, with no setup steps.

The webcam already finds the head and both wrists, and SteamVR already knows
where the headset and controllers are, in metres. Every frame of ordinary use
is therefore a free calibration measurement. This module collects them in the
background while the user plays and solves, together:

* where the camera is and which way it points,
* the lens's focal length, so no checkerboard is needed,
* how late webcam frames arrive relative to SteamVR, which otherwise shows up
  as several centimetres of error whenever the user moves, and
* the device-to-keypoint offsets, which differ between headsets and hairstyles.

Nothing is sent to VRChat until a solve is both accurate and well determined:
its uncertainty, not a hand-tuned coverage rule, decides when there is enough
data. The result is saved and reused next time, and a camera that gets bumped
is noticed from the growing reprojection error and simply solved again.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import CalibrationConfig
from ..geom.rotations import orthonormalize
from ..types import Keypoints2D, VRState
from .anchors import ANCHORS

log = logging.getLogger(__name__)

FORMAT_VERSION = 1

# Below this many correspondences a solve is not worth attempting.
MIN_CORRESPONDENCES = 30
# Seeds for the focal-length search. Consumer webcams sit between about 55 and
# 90 degrees; the continuous refinement takes it from the best seed.
_SEED_FOV_DEG = (45.0, 55.0, 62.0, 70.0, 80.0, 90.0, 100.0)
_FOV_LIMITS_DEG = (25.0, 140.0)
# Weak priors that only matter when the data cannot decide: a lens near the
# configured field of view, and device offsets near their nominal values.
_FOCAL_PRIOR_SIGMA = 0.35  # natural-log units, roughly +-40 %
_OFFSET_PRIOR_SIGMA_M = 0.05
_OFFSET_BOUND_M = 0.15
_PRIOR_SCALE_PX = 4.0
_NOMINAL_OFFSETS = np.stack([a.local_offset for a in ANCHORS]).astype(np.float64)


def focal_from_fov(width: int, fov_deg: float) -> float:
    return (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def _rotation_from_rvec(rvec: np.ndarray) -> np.ndarray:
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    skew = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * skew @ skew


def _project(
    world: np.ndarray,
    rotation_cw: np.ndarray,
    translation_cw: np.ndarray,
    focal: float,
    centre: np.ndarray,
) -> np.ndarray:
    camera = np.atleast_2d(world) @ rotation_cw.T + translation_cw
    depth = np.maximum(camera[:, 2:3], 1e-3)
    return focal * camera[:, :2] / depth + centre


@dataclass(slots=True)
class CameraCalibration:
    """Everything needed to place webcam observations in the SteamVR room."""

    rotation_cw: np.ndarray  # world-to-camera, x_camera = R @ x_play + t
    translation_cw: np.ndarray
    focal_px: float
    width: int
    height: int
    latency_s: float
    offsets: dict[str, np.ndarray]
    rms_px: float
    inliers: int
    total: int
    position_std_m: float
    focal_std: float  # relative, one standard deviation
    mirror: bool = False

    @property
    def intrinsics(self) -> np.ndarray:
        return np.array(
            [
                [self.focal_px, 0.0, self.width / 2.0],
                [0.0, self.focal_px, self.height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )

    @property
    def camera_position(self) -> np.ndarray:
        return -self.rotation_cw.T @ self.translation_cw

    @property
    def fov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.focal_px)))

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / self.total if self.total else 0.0

    def project(self, world: np.ndarray) -> np.ndarray:
        centre = np.array([self.width / 2.0, self.height / 2.0])
        return _project(world, self.rotation_cw, self.translation_cw, self.focal_px, centre)


# -- samples ------------------------------------------------------------------


@dataclass(slots=True)
class SampleSet:
    """Correspondences as arrays, with device poses at several candidate latencies."""

    lags: np.ndarray  # (L,) seconds subtracted from the frame timestamp
    anchor: np.ndarray  # (N,) index into ANCHORS
    pixel: np.ndarray  # (N, 2)
    # Where this device would appear if the image were mirrored: the opposite
    # wrist's pixel, not yet flipped. NaN where that keypoint was not seen.
    mirror_pixel: np.ndarray  # (N, 2)
    position: np.ndarray  # (N, L, 3)
    rotation: np.ndarray  # (N, L, 3, 3)

    def __len__(self) -> int:
        return len(self.anchor)

    def poses_at(self, lag: float) -> tuple[np.ndarray, np.ndarray]:
        """Device poses at a latency between the sampled ones, by interpolation."""
        if len(self.lags) == 1:
            return self.position[:, 0], self.rotation[:, 0]
        position = float(np.interp(lag, self.lags, np.arange(len(self.lags))))
        index = min(int(position), len(self.lags) - 2)
        weight = position - index
        return (
            (1.0 - weight) * self.position[:, index] + weight * self.position[:, index + 1],
            (1.0 - weight) * self.rotation[:, index] + weight * self.rotation[:, index + 1],
        )

    def object_points(self, lag: float, offsets: np.ndarray) -> np.ndarray:
        position, rotation = self.poses_at(lag)
        return position + np.einsum("nij,nj->ni", rotation, offsets[self.anchor])

    def subset(self, rows: np.ndarray) -> SampleSet:
        return SampleSet(
            lags=self.lags,
            anchor=self.anchor[rows],
            pixel=self.pixel[rows],
            mirror_pixel=self.mirror_pixel[rows],
            position=self.position[rows],
            rotation=self.rotation[rows],
        )

    def mirrored(self, width: int) -> SampleSet:
        """The same samples under the hypothesis that the image is a mirror image."""
        keep = np.isfinite(self.mirror_pixel[:, 0])
        pixel = self.mirror_pixel[keep].copy()
        pixel[:, 0] = (width - 1) - pixel[:, 0]
        return SampleSet(
            lags=self.lags,
            anchor=self.anchor[keep],
            pixel=pixel,
            mirror_pixel=np.full_like(pixel, np.nan),
            position=self.position[keep],
            rotation=self.rotation[keep],
        )

    def save(self, path: Path | str, *, width: int, height: int, mirror: bool) -> None:
        """Write the samples a solve used, so a failed calibration can be replayed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            lags=self.lags,
            anchor=self.anchor,
            pixel=self.pixel,
            mirror_pixel=self.mirror_pixel,
            position=self.position,
            rotation=self.rotation,
            size=np.array([width, height]),
            mirror=np.array(mirror),
        )

    @classmethod
    def load(cls, path: Path | str) -> tuple[SampleSet, int, int, bool]:
        """Samples written by `save`, with the image size and mirror setting."""
        with np.load(path) as data:
            samples = cls(
                lags=data["lags"],
                anchor=data["anchor"],
                pixel=data["pixel"],
                mirror_pixel=data["mirror_pixel"],
                position=data["position"],
                rotation=data["rotation"],
            )
            width, height = (int(v) for v in data["size"])
            return samples, width, height, bool(data["mirror"])


def _constellation(state: VRState) -> np.ndarray:
    out = np.full((len(ANCHORS), 3), np.nan)
    for index, anchor in enumerate(ANCHORS):
        pose = getattr(state, anchor.name)
        if pose is not None and pose.valid:
            out[index] = pose.position
    return out


class SampleCollector:
    """Accumulates well-spread correspondences.

    A hundred frames of someone standing still are one correspondence repeated
    a hundred times, so a frame is only kept if some device has moved since
    every frame already stored. Oldest frames are dropped first once full.
    """

    def __init__(
        self,
        lags: Sequence[float],
        *,
        min_spacing_m: float = 0.04,
        min_score: float = 0.4,
        max_samples: int = 900,
    ) -> None:
        self.lags = np.asarray(lags, dtype=np.float64)
        self.min_spacing_m = min_spacing_m
        self.min_score = min_score
        self.max_samples = max_samples
        self.rejected_close = 0
        self.rejected_unconfident = 0
        self.clear()

    def clear(self) -> None:
        self._anchor: list[int] = []
        self._pixel: list[np.ndarray] = []
        self._mirror: list[np.ndarray] = []
        self._position: list[np.ndarray] = []
        self._rotation: list[np.ndarray] = []
        self._frame: list[int] = []
        self._constellations: list[np.ndarray] = []
        self._frame_ids: list[int] = []
        self._next_frame = 0

    def __len__(self) -> int:
        return len(self._anchor)

    @property
    def frames(self) -> int:
        return len(self._frame_ids)

    def counts(self) -> dict[str, int]:
        out = {anchor.name: 0 for anchor in ANCHORS}
        for index in self._anchor:
            out[ANCHORS[index].name] += 1
        return out

    def _novel(self, constellation: np.ndarray) -> bool:
        if not self._constellations:
            return True
        stored = np.stack(self._constellations)
        moved = np.linalg.norm(stored - constellation, axis=2)
        moved = np.where(np.isnan(moved), -np.inf, moved).max(axis=1)
        moved = np.where(np.isinf(moved), np.inf, moved)
        return bool(moved.min() >= self.min_spacing_m)

    def add(self, keypoints: Keypoints2D, states: Sequence[VRState | None]) -> int:
        """Record one frame's correspondences. `states` holds one VR state per lag."""
        if len(states) != len(self.lags) or any(state is None for state in states):
            return 0
        if any(state.head is None or not state.head.valid for state in states):
            return 0

        constellation = _constellation(states[0])
        if not self._novel(constellation):
            self.rejected_close += 1
            return 0

        frame = self._next_frame
        added = 0
        for index, anchor in enumerate(ANCHORS):
            poses = [getattr(state, anchor.name) for state in states]
            if any(pose is None or not pose.valid for pose in poses):
                continue
            if float(keypoints.scores[anchor.keypoint]) < self.min_score:
                self.rejected_unconfident += 1
                continue
            mirror = (
                np.asarray(keypoints.xy[anchor.mirrored_keypoint], dtype=np.float64)
                if float(keypoints.scores[anchor.mirrored_keypoint]) >= self.min_score
                else np.full(2, np.nan)
            )
            self._anchor.append(index)
            self._pixel.append(np.asarray(keypoints.xy[anchor.keypoint], dtype=np.float64))
            self._mirror.append(mirror)
            self._position.append(np.stack([pose.position for pose in poses]).astype(np.float64))
            self._rotation.append(np.stack([pose.rotation for pose in poses]).astype(np.float64))
            self._frame.append(frame)
            added += 1

        if added:
            self._next_frame += 1
            self._constellations.append(constellation)
            self._frame_ids.append(frame)
            self._evict()
        return added

    def flip(self, width: int) -> None:
        """Re-express the stored samples as if the image had been mirrored.

        Each device's pixel becomes the flipped pixel of its mirror keypoint,
        so samples gathered before the webcam setting was corrected still count.
        """

        def flipped(pixel: np.ndarray) -> np.ndarray:
            out = pixel.copy()
            out[0] = (width - 1) - out[0]
            return out

        keep = [i for i, mirror in enumerate(self._mirror) if np.isfinite(mirror[0])]
        pixel = [flipped(self._mirror[i]) for i in keep]
        mirror = [flipped(self._pixel[i]) for i in keep]
        self._pixel, self._mirror = pixel, mirror
        for name in ("_anchor", "_position", "_rotation", "_frame"):
            rows = getattr(self, name)
            setattr(self, name, [rows[i] for i in keep])

    def _evict(self) -> None:
        while len(self._anchor) > self.max_samples and self._frame_ids:
            oldest = self._frame_ids.pop(0)
            self._constellations.pop(0)
            drop = 0
            while drop < len(self._frame) and self._frame[drop] == oldest:
                drop += 1
            for rows in (
                self._anchor,
                self._pixel,
                self._mirror,
                self._position,
                self._rotation,
                self._frame,
            ):
                del rows[:drop]

    def snapshot(self) -> SampleSet:
        lags = len(self.lags)
        if not self._anchor:
            return SampleSet(
                lags=self.lags.copy(),
                anchor=np.zeros(0, dtype=np.int64),
                pixel=np.zeros((0, 2)),
                mirror_pixel=np.zeros((0, 2)),
                position=np.zeros((0, lags, 3)),
                rotation=np.zeros((0, lags, 3, 3)),
            )
        return SampleSet(
            lags=self.lags.copy(),
            anchor=np.asarray(self._anchor, dtype=np.int64),
            pixel=np.stack(self._pixel),
            mirror_pixel=np.stack(self._mirror),
            position=np.stack(self._position),
            rotation=np.stack(self._rotation),
        )


# -- the solve ----------------------------------------------------------------


def _msac(errors: np.ndarray, threshold: float) -> float:
    return float(np.sum(np.minimum(errors, threshold) ** 2))


def _errors(samples: SampleSet, x: np.ndarray, centre: np.ndarray) -> np.ndarray:
    rotation, translation, focal, offsets, lag = _unpack(x)
    projected = _project(samples.object_points(lag, offsets), rotation, translation, focal, centre)
    return np.linalg.norm(projected - samples.pixel, axis=1)


def _ransac(samples: SampleSet, lag: float, focal: float, centre: np.ndarray, threshold: float):
    import cv2

    points = samples.object_points(lag, _NOMINAL_OFFSETS)
    matrix = np.array([[focal, 0.0, centre[0]], [0.0, focal, centre[1]], [0.0, 0.0, 1.0]])
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            points,
            samples.pixel,
            matrix,
            None,
            flags=cv2.SOLVEPNP_SQPNP,
            reprojectionError=threshold,
            iterationsCount=300,
            confidence=0.999,
        )
    except cv2.error:
        return None
    if not ok or inliers is None or len(inliers) < 6:
        return None
    rotation = _rotation_from_rvec(rvec)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    points = samples.object_points(lag, _NOMINAL_OFFSETS)
    errors = np.linalg.norm(
        _project(points, rotation, translation, focal, centre) - samples.pixel, axis=1
    )
    return _msac(errors, threshold), np.asarray(rvec).reshape(3), translation


# Parameter vector: rvec (3), tvec (3), log focal (1), device offsets (9), latency (1).
def _unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    return (
        _rotation_from_rvec(x[:3]),
        x[3:6],
        math.exp(x[6]),
        x[7:16].reshape(3, 3),
        float(x[16]),
    )


def _refine(
    samples: SampleSet,
    x0: np.ndarray,
    *,
    centre: np.ndarray,
    width: int,
    prior_focal: float,
    threshold: float,
    loss: str = "soft_l1",
    max_nfev: int = 200,
):
    """Optimise camera pose, focal length, device offsets and latency together."""
    from scipy.optimize import least_squares

    log_prior = math.log(prior_focal)

    def residuals(x: np.ndarray) -> np.ndarray:
        rot_cw, trans_cw, focal, offsets, lag = _unpack(x)
        projected = _project(samples.object_points(lag, offsets), rot_cw, trans_cw, focal, centre)
        return np.concatenate(
            [
                (projected - samples.pixel).ravel(),
                [_PRIOR_SCALE_PX * (x[6] - log_prior) / _FOCAL_PRIOR_SIGMA],
                _PRIOR_SCALE_PX * (offsets - _NOMINAL_OFFSETS).ravel() / _OFFSET_PRIOR_SIGMA_M,
            ]
        )

    focal_lo = math.log(focal_from_fov(width, _FOV_LIMITS_DEG[1]))
    focal_hi = math.log(focal_from_fov(width, _FOV_LIMITS_DEG[0]))
    lower = np.concatenate(
        [
            np.full(6, -np.inf),
            [focal_lo],
            (_NOMINAL_OFFSETS - _OFFSET_BOUND_M).ravel(),
            [float(samples.lags[0])],
        ]
    )
    upper = np.concatenate(
        [
            np.full(6, np.inf),
            [focal_hi],
            (_NOMINAL_OFFSETS + _OFFSET_BOUND_M).ravel(),
            [float(samples.lags[-1]) + 1e-6],
        ]
    )
    start = np.clip(np.asarray(x0, dtype=np.float64), lower + 1e-9, upper - 1e-9)
    return least_squares(
        residuals,
        start,
        bounds=(lower, upper),
        method="trf",
        loss=loss,
        f_scale=threshold,
        x_scale="jac",
        max_nfev=max_nfev,
    )


def _position_std(x: np.ndarray, covariance: np.ndarray) -> float:
    """One-sigma camera position uncertainty, along its worst direction."""

    def position(params: np.ndarray) -> np.ndarray:
        return -_rotation_from_rvec(params[:3]).T @ params[3:6]

    base = position(x)
    gradient = np.zeros((3, len(x)))
    for index in range(6):
        step = np.zeros_like(x)
        step[index] = 1e-6
        gradient[:, index] = (position(x + step) - base) / 1e-6
    position_cov = gradient @ covariance @ gradient.T
    return float(math.sqrt(max(float(np.linalg.eigvalsh(position_cov).max()), 0.0)))


def solve_camera(
    samples: SampleSet,
    width: int,
    height: int,
    *,
    initial_fov_deg: float = 62.0,
    ransac_threshold_px: float = 10.0,
    latency_hint_s: float | None = None,
) -> CameraCalibration | None:
    """Recover camera pose, focal length and latency from correspondences.

    Seeds come from RANSAC PnP over a handful of focal lengths and then over
    the candidate latencies, which is robust to the frames where the pose model
    put a wrist on the wrong hand. A joint robust least-squares fit then takes
    it the rest of the way, and its Jacobian says how well determined the
    answer is.
    """
    if len(samples) < MIN_CORRESPONDENCES:
        return None
    threshold = ransac_threshold_px
    centre = np.array([width / 2.0, height / 2.0])
    prior_focal = focal_from_fov(width, initial_fov_deg)
    hint = 0.06 if latency_hint_s is None else latency_hint_s
    lag = float(samples.lags[int(np.argmin(np.abs(samples.lags - hint)))])

    best = None
    for fov in sorted({*_SEED_FOV_DEG, initial_fov_deg}):
        focal = focal_from_fov(width, fov)
        seeded = _ransac(samples, lag, focal, centre, threshold)
        if seeded is not None and (best is None or seeded[0] < best[0]):
            best = (seeded[0], focal, seeded[1], seeded[2])
    if best is None:
        return None
    cost, focal, rvec, tvec = best

    for candidate in samples.lags:
        seeded = _ransac(samples, float(candidate), focal, centre, threshold)
        if seeded is not None and seeded[0] < cost:
            cost, rvec, tvec = seeded
            lag = float(candidate)

    common = dict(centre=centre, width=width, prior_focal=prior_focal, threshold=threshold)
    x = np.concatenate([rvec, tvec, [math.log(focal)], _NOMINAL_OFFSETS.ravel(), [lag]])
    x = _refine(samples, x, **common).x

    errors = _errors(samples, x, centre)
    inliers = np.flatnonzero(errors <= threshold)
    if len(inliers) < MIN_CORRESPONDENCES:
        return None

    inlier_samples = samples.subset(inliers)
    final = _refine(inlier_samples, x, loss="linear", max_nfev=30, **common)
    x = final.x
    rot_cw, trans_cw, focal, offsets, lag = _unpack(x)
    inlier_errors = _errors(inlier_samples, x, centre)

    jacobian = final.jac
    dof = max(jacobian.shape[0] - jacobian.shape[1], 1)
    covariance = np.linalg.pinv(jacobian.T @ jacobian) * (2.0 * final.cost / dof)

    return CameraCalibration(
        rotation_cw=orthonormalize(rot_cw),
        translation_cw=np.asarray(trans_cw, dtype=np.float64).copy(),
        focal_px=float(focal),
        width=width,
        height=height,
        latency_s=lag,
        offsets={anchor.name: offsets[i].copy() for i, anchor in enumerate(ANCHORS)},
        rms_px=float(np.sqrt(np.mean(inlier_errors**2))),
        inliers=int(len(inliers)),
        total=len(samples),
        position_std_m=_position_std(x, covariance),
        focal_std=float(math.sqrt(max(float(covariance[6, 6]), 0.0))),
    )


def assess(calibration: CameraCalibration, config: CalibrationConfig) -> str | None:
    """Why a solve is not yet good enough to track with, or None if it is."""
    if calibration.inlier_ratio < config.min_inlier_ratio:
        return (
            f"only {calibration.inlier_ratio:.0%} of samples agree; "
            "keep your head and both hands in view"
        )
    if calibration.rms_px > config.max_rms_px:
        return f"fit error {calibration.rms_px:.1f} px; keep your head and both hands in view"
    if calibration.position_std_m > config.max_position_std_m:
        return (
            f"camera position still +-{calibration.position_std_m * 100:.0f} cm; "
            "reach high, low, out to the sides and toward the camera, and crouch"
        )
    if calibration.focal_std > config.max_focal_std:
        return (
            f"lens still +-{calibration.focal_std:.0%}; "
            "reach toward the camera and back, and turn side on"
        )
    return None


# -- the background calibrator ---------------------------------------------


def _mirror_fits_better(
    normal: CameraCalibration | None,
    mirrored: CameraCalibration | None,
    config: CalibrationConfig,
) -> bool:
    """Whether the samples are explained by a mirrored image rather than the image as it is.

    This decides between two hypotheses, so it asks only which one fits. How
    precisely the camera is known yet is a separate question, for `assess`.
    """
    if mirrored is None:
        return False
    if mirrored.inlier_ratio < config.min_inlier_ratio or mirrored.rms_px > config.max_rms_px:
        return False
    return normal is None or mirrored.inlier_ratio > normal.inlier_ratio + 0.2


def _describe_fit(calibration: CameraCalibration | None) -> str:
    if calibration is None:
        return "no fit"
    return f"{calibration.inlier_ratio:.0%} of samples fit ({calibration.rms_px:.1f} px)"


@dataclass(slots=True)
class CalibrationUpdate:
    kind: str  # "calibrated" | "refined" | "lost" | "mirrored"
    calibration: CameraCalibration | None = None


class AutoCalibrator:
    """Collects samples during tracking and solves the camera in the background.

    `history` is anything with ``at_many(timestamps)``, normally a `VRHistory`.
    Timestamps on the keypoints drive every interval here, so a recording plays
    back exactly as it ran live.
    """

    def __init__(
        self,
        config: CalibrationConfig,
        width: int,
        height: int,
        history,
        *,
        min_score: float = 0.4,
        current: CameraCalibration | None = None,
        mirror: bool = False,
        background: bool = True,
        samples_path: Path | str | None = None,
    ) -> None:
        self.config = config
        self.width = width
        self.height = height
        self.history = history
        self.min_score = min_score
        self.background = background
        self.samples_path = samples_path
        self.lags = np.arange(0.0, config.max_latency_s + 1e-9, config.latency_step_s)
        self.collector = SampleCollector(
            self.lags,
            min_spacing_m=config.min_sample_spacing_m,
            min_score=min_score,
            max_samples=config.max_samples,
        )
        self.current = current
        self.mirror = mirror
        self.last_attempt: CameraCalibration | None = None
        self.last_reason = ""

        self._now: float | None = None
        self._last_observe = -math.inf
        self._last_solve = -math.inf
        self._samples_at_last_solve = 0
        self._mirror_checked_at = 0
        self._failed_solves = 0
        self._monitor_errors: deque[float] = deque(maxlen=config.monitor_frames)
        self._generation = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._finished: tuple | None = None

    # -- input --------------------------------------------------------------

    def observe(self, keypoints: Keypoints2D | None) -> None:
        if keypoints is None:
            return
        timestamp = float(keypoints.timestamp)
        self._now = timestamp
        if timestamp - self._last_observe < 1.0 / self.config.sample_rate_hz:
            return
        self._last_observe = timestamp
        states = self.history.at_many([timestamp - lag for lag in self.lags])
        if self.current is not None:
            self._monitor(keypoints, states)
        self.collector.add(keypoints, states)

    def _monitor(self, keypoints: Keypoints2D, states: Sequence[VRState | None]) -> None:
        """Track how well the current calibration explains what the camera sees."""
        calibration = self.current
        state = states[int(np.argmin(np.abs(self.lags - calibration.latency_s)))]
        if state is None:
            return
        errors = []
        for anchor in ANCHORS:
            pose = getattr(state, anchor.name)
            if pose is None or not pose.valid:
                continue
            if float(keypoints.scores[anchor.keypoint]) < self.min_score:
                continue
            world = pose.position + pose.rotation @ calibration.offsets[anchor.name]
            pixel = calibration.project(world)[0]
            errors.append(float(np.linalg.norm(pixel - keypoints.xy[anchor.keypoint])))
        if errors:
            # The best-fitting device, so one misdetected wrist is not a moved camera.
            self._monitor_errors.append(min(errors))

    # -- output -------------------------------------------------------------

    def update(self) -> CalibrationUpdate | None:
        """Call once per frame. Returns a change the runtime has to act on."""
        finished = self._take_finished()
        if finished is not None:
            update = self._consider(*finished)
            if update is not None:
                return update

        if self.current is not None and self._camera_moved():
            log.warning(
                "the camera no longer matches its calibration (median %.0f px); "
                "it was probably moved. Recalibrating automatically.",
                float(np.median(self._monitor_errors)),
            )
            self._reset()
            return CalibrationUpdate("lost")

        self._maybe_start_solve()
        return None

    def _camera_moved(self) -> bool:
        if len(self._monitor_errors) < self._monitor_errors.maxlen:
            return False
        limit = max(self.config.moved_error_px, 4.0 * self.current.rms_px)
        return float(np.median(self._monitor_errors)) > limit

    def _reset(self) -> None:
        self.current = None
        self.last_attempt = None
        self.last_reason = ""
        self.collector.clear()
        self._monitor_errors.clear()
        self._samples_at_last_solve = 0
        self._mirror_checked_at = 0
        self._failed_solves = 0
        self._generation += 1

    def _flip(self) -> None:
        self.mirror = not self.mirror
        self.collector.flip(self.width)
        self.last_attempt = None
        self.last_reason = ""
        self._samples_at_last_solve = 0
        self._mirror_checked_at = len(self.collector)
        self._failed_solves = 0
        self._last_solve = -math.inf
        self._generation += 1

    def _busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _maybe_start_solve(self) -> None:
        if self._busy() or self._now is None:
            return
        interval = (
            self.config.solve_interval_s if self.current is None else self.config.refine_interval_s
        )
        if self._now - self._last_solve < interval:
            return
        count = len(self.collector)
        if count < self.config.min_samples or count == self._samples_at_last_solve:
            return
        self._last_solve = self._now
        self._samples_at_last_solve = count

        samples = self.collector.snapshot()
        check_mirror = (
            self.current is None
            and self._failed_solves > 0
            and count >= self._mirror_checked_at + self.config.min_samples
        )
        hint = self.current.latency_s if self.current is not None else None
        generation = self._generation
        dump = self.samples_path if self.current is None else None
        mirror = self.mirror

        def job() -> None:
            result = self._solve(samples, hint, check_mirror)
            if dump is not None:
                try:
                    samples.save(dump, width=self.width, height=self.height, mirror=mirror)
                except OSError:
                    log.debug("could not write calibration samples to %s", dump, exc_info=True)
            with self._lock:
                self._finished = (generation, *result)

        if self.background:
            self._thread = threading.Thread(target=job, name="auto-calibration", daemon=True)
            self._thread.start()
        else:
            job()

    def _solve(self, samples: SampleSet, hint: float | None, check_mirror: bool):
        options = dict(
            initial_fov_deg=self.config.initial_fov_deg,
            ransac_threshold_px=self.config.ransac_threshold_px,
            latency_hint_s=hint,
        )
        try:
            normal = solve_camera(samples, self.width, self.height, **options)
            mirrored = None
            if check_mirror and (normal is None or assess(normal, self.config) is not None):
                mirrored = solve_camera(
                    samples.mirrored(self.width), self.width, self.height, **options
                )
        except Exception:
            log.exception("automatic calibration solve failed")
            return None, None, check_mirror
        return normal, mirrored, check_mirror

    def _take_finished(self):
        with self._lock:
            finished, self._finished = self._finished, None
        if finished is None or finished[0] != self._generation:
            return None
        return finished[1:]

    def _consider(
        self,
        normal: CameraCalibration | None,
        mirrored: CameraCalibration | None,
        checked_mirror: bool,
    ) -> CalibrationUpdate | None:
        if checked_mirror:
            self._mirror_checked_at = len(self.collector)
            log.info(
                "mirror check: %s as the image is, %s flipped",
                _describe_fit(normal),
                _describe_fit(mirrored),
            )
        if _mirror_fits_better(normal, mirrored, self.config):
            log.warning(
                "the webcam image is mirrored relative to its setting; flipping it "
                "(%.0f%% of samples fit flipped, %s as it was)",
                mirrored.inlier_ratio * 100,
                f"{normal.inlier_ratio:.0%}" if normal is not None else "none",
            )
            self._flip()
            return CalibrationUpdate("mirrored")

        if normal is None:
            self._failed_solves += 1
            self.last_reason = "samples do not fit a single camera yet; keep moving in view"
            return None
        normal.mirror = self.mirror
        self.last_attempt = normal
        reason = assess(normal, self.config)
        self.last_reason = reason or ""
        if reason is not None:
            self._failed_solves += 1
            log.debug("calibration not accepted yet: %s", reason)
            return None

        if self.current is None:
            self.current = normal
            self._monitor_errors.clear()
            return CalibrationUpdate("calibrated", normal)
        if normal.position_std_m < self.current.position_std_m:
            self.current = normal
            return CalibrationUpdate("refined", normal)
        return None

    def status_lines(self) -> list[str]:
        if self.current is not None:
            c = self.current
            return [
                f"calibrated: {c.fov_deg:.0f} deg FOV, {c.latency_s * 1000:.0f} ms lag, "
                f"{c.rms_px:.1f} px"
            ]
        count = len(self.collector)
        if count < self.config.min_samples:
            lines = [
                f"calibrating {count}/{self.config.min_samples}: "
                "stay in view and reach around with both controllers"
            ]
        else:
            lines = ["calibrating: solving"]
        if self.last_reason:
            lines.append(self.last_reason)
        return lines

    def close(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


# -- persistence --------------------------------------------------------------


def save_calibration(calibration: CameraCalibration, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": FORMAT_VERSION,
                "rotation_cw": calibration.rotation_cw.tolist(),
                "translation_cw": calibration.translation_cw.tolist(),
                "focal_px": calibration.focal_px,
                "width": calibration.width,
                "height": calibration.height,
                "latency_s": calibration.latency_s,
                "offsets": {k: v.tolist() for k, v in calibration.offsets.items()},
                "rms_px": calibration.rms_px,
                "inliers": calibration.inliers,
                "total": calibration.total,
                "position_std_m": calibration.position_std_m,
                "focal_std": calibration.focal_std,
                "mirror": calibration.mirror,
                "camera_position": calibration.camera_position.tolist(),
                "fov_deg": calibration.fov_deg,
            },
            indent=2,
        )
    )
    log.info("wrote camera calibration to %s", path)


def load_calibration(path: Path | str, width: int, height: int) -> CameraCalibration | None:
    """Load a saved calibration for a camera now running at `width` x `height`.

    The same camera at another resolution with the same aspect ratio just
    scales the focal length; a different aspect ratio usually means the driver
    crops, which the saved lens cannot describe, so that is a fresh start.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        saved_width, saved_height = int(data["width"]), int(data["height"])
        calibration = CameraCalibration(
            rotation_cw=np.array(data["rotation_cw"], dtype=np.float64),
            translation_cw=np.array(data["translation_cw"], dtype=np.float64),
            focal_px=float(data["focal_px"]),
            width=saved_width,
            height=saved_height,
            latency_s=float(data["latency_s"]),
            offsets={k: np.array(v, dtype=np.float64) for k, v in data["offsets"].items()},
            rms_px=float(data["rms_px"]),
            inliers=int(data["inliers"]),
            total=int(data["total"]),
            position_std_m=float(data["position_std_m"]),
            focal_std=float(data["focal_std"]),
            mirror=bool(data.get("mirror", False)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("ignoring unreadable camera calibration at %s (%s)", path, exc)
        return None

    if (saved_width, saved_height) == (width, height):
        return calibration
    if abs(saved_width / saved_height - width / height) > 0.01:
        log.warning(
            "saved camera calibration is for %dx%d, camera now runs at %dx%d; recalibrating",
            saved_width,
            saved_height,
            width,
            height,
        )
        return None
    scale = width / saved_width
    calibration.focal_px *= scale
    calibration.rms_px *= scale
    calibration.width, calibration.height = width, height
    return calibration
