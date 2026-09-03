"""Making clean projected keypoints look like they came from RTMPose.

The lifter never sees pixels. It sees 26 2D points with confidence scores.
Training it on BEDLAM's exact projections would teach it that wrists never
jitter and feet never vanish, which is the opposite of what a 720p webcam
produces.

The noise here is a model of the *detector*, not of the camera. Camera
degradation (the ``bodytracker.degrade`` package) is what you apply to images
before running RTMPose, when you have images. When you are training from
ground-truth projections alone, this is the stand-in: per-joint pixel noise
with temporal correlation, plus dropouts, with magnitudes that default to
what we measured against the geometric lifter (about 2.5 px on a 720p frame)
and that can be refit from a real RTMPose-vs-BEDLAM comparison later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bodytracker.skeleton import (
    HEAD,
    LEFT_ANKLE,
    LEFT_BIG_TOE,
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_HEEL,
    LEFT_KNEE,
    LEFT_SMALL_TOE,
    LEFT_WRIST,
    NUM_KEYPOINTS,
    RIGHT_ANKLE,
    RIGHT_BIG_TOE,
    RIGHT_EAR,
    RIGHT_ELBOW,
    RIGHT_HEEL,
    RIGHT_KNEE,
    RIGHT_SMALL_TOE,
    RIGHT_WRIST,
)

# Joints the detector is typically worse at. End effectors are small, often
# motion-blurred, and frequently occluded by the torso or the floor.
_NOISY = frozenset(
    {
        LEFT_WRIST,
        RIGHT_WRIST,
        LEFT_ANKLE,
        RIGHT_ANKLE,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_KNEE,
        RIGHT_KNEE,
        LEFT_BIG_TOE,
        RIGHT_BIG_TOE,
        LEFT_SMALL_TOE,
        RIGHT_SMALL_TOE,
        LEFT_HEEL,
        RIGHT_HEEL,
        LEFT_EAR,
        RIGHT_EAR,
    }
)


def default_sigma_px() -> np.ndarray:
    """Per-joint noise, in pixels, at 1280×720.

    2.5 px on the torso is the figure the One Euro defaults were tuned on.
    Wrists, ankles and feet get more because that is where RTMPose actually
    misses, and a model that was only trained for torso noise will trust a
    sliding foot.
    """
    sigma = np.full(NUM_KEYPOINTS, 2.5, dtype=np.float64)
    for joint in _NOISY:
        sigma[joint] = 4.5
    sigma[HEAD] = 3.0
    return sigma


@dataclass(slots=True)
class KeypointNoise:
    """Detector-like corruption of a 2D keypoint sequence.

    Temporal correlation is the load-bearing bit. Independent per-frame noise
    teaches a temporal model that the world twitches at the Nyquist frequency,
    which no real detector does: RTMPose's errors persist for several frames
    because they come from a blurred pose, not from pixel grain. An AR(1)
    process with ``rho`` around 0.8 is a cheap way to get that persistence
    without fitting a Kalman filter.
    """

    sigma_px: np.ndarray = field(default_factory=default_sigma_px)
    rho: float = 0.8
    dropout: float = 0.03
    # A dropped joint stays down this many extra frames on average, because
    # an occlusion lasts longer than one frame.
    dropout_persist: float = 0.7
    score_full: float = 0.92
    score_empty: float = 0.05

    def __call__(
        self,
        xy: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Corrupt a ``(T, 26, 2)`` (or ``(26, 2)``) sequence.

        Returns ``(noisy_xy, scores)``. Dropped joints keep their last visible
        position rather than leaping to the origin: that is what the runtime
        hold-last-valid path sees, and training against zeros would teach the
        lifter that people occasionally collapse to the top-left corner.
        """
        rng = rng or np.random.default_rng(seed)
        points = np.asarray(xy, dtype=np.float64)
        single = points.ndim == 2
        if single:
            points = points[None]
        if points.shape[1:] != (NUM_KEYPOINTS, 2):
            raise ValueError(f"expected (T, {NUM_KEYPOINTS}, 2), got {points.shape}")

        frames, _, _ = points.shape
        sigma = np.broadcast_to(self.sigma_px, (NUM_KEYPOINTS,))
        innovation = np.sqrt(max(1.0 - self.rho**2, 0.0))

        noise = np.zeros_like(points)
        noise[0] = rng.normal(0.0, 1.0, size=(NUM_KEYPOINTS, 2)) * sigma[:, None]
        for t in range(1, frames):
            noise[t] = self.rho * noise[t - 1] + innovation * rng.normal(
                0.0, 1.0, size=(NUM_KEYPOINTS, 2)
            ) * sigma[:, None]

        dropped = np.zeros((frames, NUM_KEYPOINTS), dtype=np.bool_)
        live = rng.random(NUM_KEYPOINTS) < self.dropout
        dropped[0] = live
        for t in range(1, frames):
            stay = live & (rng.random(NUM_KEYPOINTS) < self.dropout_persist)
            fresh = (~live) & (rng.random(NUM_KEYPOINTS) < self.dropout)
            live = stay | fresh
            dropped[t] = live

        noisy = points + noise
        # Hold last valid: walk forward filling dropped joints from the
        # previous visible frame. A joint dropped on frame 0 keeps the clean
        # (pre-noise) position, which is a missing detection rather than a lie.
        held = noisy.copy()
        previous = points[0].copy()
        for t in range(frames):
            held[t, dropped[t]] = previous[dropped[t]]
            previous = np.where(dropped[t, :, None], previous, held[t])

        scores = np.where(dropped, self.score_empty, self.score_full).astype(np.float64)
        if single:
            return held[0], scores[0]
        return held, scores


def fit_sigma(predicted_xy: np.ndarray, truth_xy: np.ndarray) -> np.ndarray:
    """Per-joint pixel RMSE of a detector against ground truth.

    `predicted_xy` and `truth_xy` are ``(N, 26, 2)``. Run this on a small
    BEDLAM image subset after RTMPose, then store the result as the noise
    model's ``sigma_px`` so training matches the detector you actually ship.
    """
    predicted = np.asarray(predicted_xy, dtype=np.float64)
    truth = np.asarray(truth_xy, dtype=np.float64)
    if predicted.shape != truth.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {truth.shape}")
    residual = predicted - truth
    return np.sqrt(np.nanmean(residual**2, axis=(0, 2)))
