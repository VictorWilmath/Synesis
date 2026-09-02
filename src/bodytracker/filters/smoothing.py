"""Smoothing for whole skeletons and for rotations.

Rotations get SLERP rather than the One Euro filter. Interpolating Euler angles
componentwise wraps badly at +/-180 and passes through orientations that are
nowhere near either endpoint; interpolating matrices elementwise leaves the
rotation manifold entirely.
"""

from __future__ import annotations

import numpy as np

from ..config import FilterConfig
from ..geom import slerp
from ..skeleton import NUM_KEYPOINTS
from ..types import Skeleton3D
from .one_euro import OneEuroFilter


class SkeletonFilter:
    """One Euro over all joint positions, with per-joint gating on confidence.

    A joint whose keypoint confidence has collapsed is not fed into the filter;
    its filtered value is held instead. Feeding a garbage position in and
    smoothing it merely spreads the error over the following frames.
    """

    def __init__(self, config: FilterConfig, min_score: float = 0.3) -> None:
        self.config = config
        self.min_score = min_score
        self._filter = OneEuroFilter(
            min_cutoff=config.min_cutoff,
            beta=config.beta,
            d_cutoff=config.d_cutoff,
        )
        self._last: np.ndarray | None = None

    def reset(self) -> None:
        self._filter.reset()
        self._last = None

    def __call__(self, skeleton: Skeleton3D) -> Skeleton3D:
        xyz = np.asarray(skeleton.xyz, dtype=np.float64)

        if self._last is not None:
            # Substitute the previous accepted position for low-confidence
            # joints so they neither jump nor drag the derivative estimate.
            weak = skeleton.scores < self.min_score
            if weak.any():
                xyz = xyz.copy()
                xyz[weak] = self._last[weak]

        filtered = self._filter(xyz, skeleton.timestamp)
        self._last = filtered.copy()

        return Skeleton3D(
            xyz=filtered.astype(np.float32),
            scores=skeleton.scores.copy(),
            space=skeleton.space,
            timestamp=skeleton.timestamp,
        )


class RotationSmoother:
    """Exponential SLERP smoothing, framerate-compensated.

    `alpha` is expressed per 1/60 s so the perceived smoothing does not change
    when the camera framerate does.
    """

    _REFERENCE_DT = 1.0 / 60.0

    def __init__(self, alpha: float = 0.35) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._value: np.ndarray | None = None
        self._last_time: float | None = None

    def reset(self) -> None:
        self._value = None
        self._last_time = None

    def __call__(self, rotation: np.ndarray, timestamp: float) -> np.ndarray:
        rot = np.asarray(rotation, dtype=np.float64)

        if self._value is None or self._last_time is None:
            self._value = rot.copy()
            self._last_time = timestamp
            return self._value.copy()

        dt = max(1e-6, timestamp - self._last_time)
        self._last_time = timestamp
        # Compound the per-reference-frame retention over the elapsed time.
        effective = 1.0 - (1.0 - self.alpha) ** (dt / self._REFERENCE_DT)
        effective = float(np.clip(effective, 0.0, 1.0))

        self._value = slerp(self._value, rot, effective)
        return self._value.copy()


class HoldLastValid:
    """Keeps the most recent good value for each key, and ages it out.

    VRChat responds far better to a slightly stale tracker than to one that
    teleports to the origin, so a joint that drops out keeps reporting its last
    known pose. After `timeout` seconds without a fresh value the entry is
    dropped, since by then a wrong pose is worse than an absent one.
    """

    def __init__(self, timeout: float = 0.5) -> None:
        self.timeout = timeout
        self._values: dict[str, tuple[object, float]] = {}

    def update(self, key: str, value: object, timestamp: float) -> None:
        self._values[key] = (value, timestamp)

    def get(self, key: str, timestamp: float) -> tuple[object, bool] | None:
        """Return ``(value, stale)`` or None if missing or expired."""
        entry = self._values.get(key)
        if entry is None:
            return None
        value, when = entry
        age = timestamp - when
        if age > self.timeout:
            del self._values[key]
            return None
        return value, age > 1e-9

    def reset(self) -> None:
        self._values.clear()


def interpolate_missing(xyz: np.ndarray, scores: np.ndarray, min_score: float) -> np.ndarray:
    """Fill low-confidence joints from the skeleton's symmetric counterpart.

    A cheap fallback for the geometric baseline: when one ankle is occluded but
    the other is visible, mirroring the visible side about the pelvis is a
    better guess than leaving the joint at whatever the detector hallucinated.
    """
    from ..skeleton import FLIP_PAIRS, HIP

    out = np.array(xyz, dtype=np.float64, copy=True)
    weak = scores < min_score
    if not weak.any() or weak[HIP]:
        return out

    hip = out[HIP]
    for left, right in FLIP_PAIRS:
        if weak[left] and not weak[right]:
            mirrored = out[right] - hip
            mirrored[0] = -mirrored[0]
            out[left] = hip + mirrored
        elif weak[right] and not weak[left]:
            mirrored = out[left] - hip
            mirrored[0] = -mirrored[0]
            out[right] = hip + mirrored
    return out


def blend_skeletons(a: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
    """Linear blend between two (26, 3) skeletons."""
    if a.shape != (NUM_KEYPOINTS, 3) or b.shape != (NUM_KEYPOINTS, 3):
        raise ValueError("expected two Halpe26 skeletons")
    return (1.0 - weight) * a + weight * b
