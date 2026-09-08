"""Camera format negotiation must prefer compressed 30 FPS capture."""

from __future__ import annotations

import cv2

from bodytracker.capture.webcam import Webcam
from bodytracker.config import CameraConfig


class _Capture:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []

    def set(self, prop: int, value: float) -> None:
        self.calls.append((prop, value))


def test_format_negotiation_sets_mjpg_after_resolution_then_fps():
    camera = Webcam(CameraConfig(width=640, height=480, fps=30, fourcc="MJPG"))
    capture = _Capture()

    camera._configure_format(capture)

    assert [prop for prop, _ in capture.calls] == [
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FOURCC,
        cv2.CAP_PROP_FPS,
    ]
