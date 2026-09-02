"""Continuous extrinsics refinement while tracking.

The correspondences that made the initial calibration possible keep arriving
every frame, so there is no reason to treat calibration as a one-off. Knocking
the camera, or a tripod slowly sagging, would otherwise end the session.

Refinement is conservative: a new solve is only adopted if it fits the data
better than the current one. Otherwise a stretch of bad keypoints could walk
the camera pose somewhere wrong, and unlike a frame dropout that failure is
persistent.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..capture.intrinsics import Intrinsics
from ..types import Keypoints2D, VRState
from .anchors import CorrespondenceBuffer
from .extrinsics import MIN_COVERAGE_M, CameraExtrinsics, solve_extrinsics

log = logging.getLogger(__name__)


class OnlineExtrinsicsRefiner:
    """Keeps collecting correspondences and re-solves the camera pose."""

    def __init__(
        self,
        intrinsics: Intrinsics,
        *,
        interval_s: float = 5.0,
        min_samples: int = 120,
        ransac_threshold_px: float = 8.0,
        max_rms_increase: float = 1.25,
    ) -> None:
        self.intrinsics = intrinsics
        self.interval_s = interval_s
        self.min_samples = min_samples
        self.ransac_threshold_px = ransac_threshold_px
        self.max_rms_increase = max_rms_increase

        self.buffer = CorrespondenceBuffer(max_samples=900)
        self.current: CameraExtrinsics | None = None
        self._last_solve = 0.0
        self.solves_accepted = 0
        self.solves_rejected = 0

    def set_current(self, extrinsics: CameraExtrinsics | None) -> None:
        self.current = extrinsics

    def observe(self, vr: VRState | None, keypoints: Keypoints2D | None) -> None:
        if vr is None or keypoints is None:
            return
        self.buffer.add(vr, keypoints)

    def maybe_refine(self, now: float | None = None) -> CameraExtrinsics | None:
        """Re-solve if enough new, well-spread data has arrived.

        Returns the new extrinsics when one is adopted, otherwise None.
        """
        timestamp = time.monotonic() if now is None else now
        if timestamp - self._last_solve < self.interval_s:
            return None
        if len(self.buffer) < self.min_samples:
            return None
        if self.buffer.coverage() < MIN_COVERAGE_M:
            return None

        self._last_solve = timestamp
        candidate = solve_extrinsics(
            self.buffer.samples,
            self.intrinsics.matrix,
            self.intrinsics.distortion,
            ransac_threshold_px=self.ransac_threshold_px,
        )
        if candidate is None:
            self.solves_rejected += 1
            return None

        if self.current is not None:
            allowed = self.current.rms_error_px * self.max_rms_increase
            if candidate.rms_error_px > allowed:
                self.solves_rejected += 1
                log.debug(
                    "rejected refinement: %.2f px vs current %.2f px",
                    candidate.rms_error_px,
                    self.current.rms_error_px,
                )
                return None
            moved = float(np.linalg.norm(candidate.camera_position - self.current.camera_position))
            if moved > 0.02:
                log.info("camera pose moved %.0f mm during refinement", moved * 1000)

        self.current = candidate
        self.solves_accepted += 1
        return candidate

    def stats(self) -> dict[str, float]:
        return {
            "samples": len(self.buffer),
            "coverage_m": self.buffer.coverage(),
            "accepted": self.solves_accepted,
            "rejected": self.solves_rejected,
        }
