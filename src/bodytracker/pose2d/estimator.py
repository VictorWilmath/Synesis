"""2D keypoint estimation, wrapping rtmlib's ONNX runtime models.

rtmlib is used rather than full mmpose because it needs only numpy, opencv and
onnxruntime, which keeps the VR rig free of the mmcv toolchain.

The detector is treated as a recovery mechanism rather than a per-frame stage.
See `tracking.py` for why.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..config import Pose2DConfig
from ..skeleton import NUM_KEYPOINTS
from ..types import Keypoints2D
from .models import get_preset
from .tracking import (
    bbox_from_keypoints,
    expand_to_aspect,
    select_person,
    valid_ratio,
)

log = logging.getLogger(__name__)


class Pose2DEstimator:
    """Produces Halpe26 keypoints for the tracked subject."""

    def __init__(self, config: Pose2DConfig) -> None:
        self.config = config
        preset = get_preset(config.mode)
        self.preset = preset

        # Imported lazily so the module can be imported for its type
        # signatures on a machine with no onnxruntime.
        from rtmlib import YOLOX, RTMPose

        log.info("loading %s on %s (%s)", config.mode, config.device, preset.description)
        self._detector = YOLOX(
            preset.detector,
            model_input_size=preset.detector_input_size,
            backend=config.backend,
            device=config.device,
        )
        self._pose = RTMPose(
            preset.pose,
            model_input_size=preset.pose_input_size,
            backend=config.backend,
            device=config.device,
        )

        self._aspect = preset.pose_input_size[0] / preset.pose_input_size[1]
        self._bbox: np.ndarray | None = None
        self._frames_since_detect = 1 << 30
        self.detections_run = 0
        self.frames_processed = 0
        self.tracking_losses = 0
        self.last_timings: dict[str, float] = {}

    @property
    def has_track(self) -> bool:
        return self._bbox is not None

    def reset(self) -> None:
        """Force a fresh detection on the next frame."""
        self._bbox = None
        self._frames_since_detect = 1 << 30

    def _needs_detection(self) -> bool:
        return self._bbox is None or self._frames_since_detect >= self.config.detect_interval

    def __call__(self, frame: np.ndarray, timestamp: float = 0.0) -> Keypoints2D | None:
        """Estimate keypoints. Returns None when no subject could be tracked."""
        height, width = frame.shape[:2]
        timings: dict[str, float] = {}

        if self._needs_detection():
            start = time.perf_counter()
            detections = self._detector(frame)
            timings["detect_ms"] = (time.perf_counter() - start) * 1000.0
            self.detections_run += 1
            self._frames_since_detect = 0

            chosen = select_person(
                detections,
                frame_width=width,
                frame_height=height,
                previous=self._bbox,
            )
            if chosen is None:
                self._bbox = None
                self.last_timings = timings
                return None
            self._bbox = chosen

        assert self._bbox is not None
        crop = expand_to_aspect(self._bbox, self._aspect, width, height)

        start = time.perf_counter()
        keypoints, scores = self._pose(frame, bboxes=crop[None, :])
        timings["pose_ms"] = (time.perf_counter() - start) * 1000.0

        if keypoints is None or len(keypoints) == 0:
            self.reset()
            self.tracking_losses += 1
            self.last_timings = timings
            return None

        xy = np.asarray(keypoints[0], dtype=np.float64)
        conf = np.asarray(scores[0], dtype=np.float64)

        if xy.shape[0] != NUM_KEYPOINTS:
            raise RuntimeError(
                f"model returned {xy.shape[0]} keypoints, expected {NUM_KEYPOINTS}. "
                "The configured checkpoint is probably not a Halpe26 model."
            )

        self._frames_since_detect += 1
        self.frames_processed += 1
        self.last_timings = timings

        # A collapse in confidence means the crop has drifted off the subject;
        # a fresh detection next frame is cheaper than chasing it.
        if valid_ratio(conf, self.config.min_keypoint_score) < self.config.min_valid_ratio:
            self.reset()
            self.tracking_losses += 1
            return None

        next_bbox = bbox_from_keypoints(
            xy,
            conf,
            min_score=self.config.min_keypoint_score,
            padding=self.config.bbox_padding,
            frame_width=width,
            frame_height=height,
        )
        if next_bbox is None:
            self.reset()
        else:
            self._bbox = next_bbox

        return Keypoints2D(
            xy=xy.astype(np.float32),
            scores=conf.astype(np.float32),
            bbox=crop.astype(np.float32),
            timestamp=timestamp,
        )

    def stats(self) -> dict[str, float]:
        frames = max(1, self.frames_processed)
        return {
            "frames": self.frames_processed,
            "detections": self.detections_run,
            "detection_rate": self.detections_run / frames,
            "tracking_losses": self.tracking_losses,
        }
