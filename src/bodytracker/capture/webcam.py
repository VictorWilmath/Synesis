"""Webcam capture with the settings that actually matter for pose quality.

Three things about a consumer webcam degrade keypoints far more than its
resolution does, and all three are fixable here rather than in the model:

*Autoexposure.* Under artificial light it hunts, and every time it opens the
shutter to compensate it lengthens the exposure and smears the subject. A
locked, short exposure with the gain turned up is noisier but much sharper, and
noise is easier for the network than blur.

*Pixel format.* Most UVC webcams only reach 30fps at 720p over MJPEG; the
uncompressed YUYV path silently drops to 10fps. That costs more tracking
quality than any model choice.

*Buffer latency.* ``VideoCapture.read`` returns the oldest buffered frame. If
the pipeline runs slower than the camera even briefly, latency accumulates and
never recovers. The reader thread below always discards to the newest frame.
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from ..config import CameraConfig

log = logging.getLogger(__name__)

# OpenCV backends disagree on the value that means "manual exposure".
# DirectShow (Windows) wants 0.25, V4L2 wants 1. Try in order and keep the
# first that sticks.
_MANUAL_EXPOSURE_VALUES = (0.25, 1.0, 0.0)


class Webcam:
    """A camera that yields the most recent frame, never a stale buffered one."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: tuple[float, np.ndarray] | None = None
        self._frame_id = 0
        self._last_delivered = -1
        self._running = False
        self.dropped_frames = 0

    # -- lifecycle --------------------------------------------------------

    def open(self) -> None:
        cfg = self.config
        # CAP_DSHOW avoids the very slow MSMF startup path on Windows and is
        # the backend that actually honours manual exposure on most webcams.
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        capture = cv2.VideoCapture(cfg.index, backend)
        if not capture.isOpened():
            capture = cv2.VideoCapture(cfg.index)
        if not capture.isOpened():
            raise RuntimeError(f"could not open camera index {cfg.index}")

        if cfg.fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        capture.set(cv2.CAP_PROP_FPS, cfg.fps)
        # Ask the driver for the shallowest queue it will give us; the reader
        # thread handles the rest.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if cfg.lock_exposure:
            self._lock_exposure(capture)
        if not cfg.autofocus:
            capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        self._capture = capture
        self._log_actual_settings()

        self._running = True
        self._thread = threading.Thread(target=self._reader, name="webcam", daemon=True)
        self._thread.start()

    def _lock_exposure(self, capture: cv2.VideoCapture) -> None:
        for value in _MANUAL_EXPOSURE_VALUES:
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, value)
            if capture.get(cv2.CAP_PROP_AUTO_EXPOSURE) == value:
                log.info("manual exposure engaged (AUTO_EXPOSURE=%s)", value)
                break
        else:
            log.warning(
                "could not disable autoexposure; expect motion blur and "
                "brightness hunting under artificial light"
            )
        capture.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)
        if self.config.gain:
            capture.set(cv2.CAP_PROP_GAIN, self.config.gain)

    def _log_actual_settings(self) -> None:
        """Report what the driver actually granted, which is often not what we asked."""
        actual = self.properties()
        log.info(
            "camera %d: %dx%d @ %.0ffps, fourcc=%s, exposure=%.1f",
            self.config.index,
            actual["width"],
            actual["height"],
            actual["fps"],
            actual["fourcc"],
            actual["exposure"],
        )
        if actual["width"] != self.config.width or actual["height"] != self.config.height:
            log.warning(
                "requested %dx%d but got %dx%d",
                self.config.width,
                self.config.height,
                actual["width"],
                actual["height"],
            )
        if actual["fps"] and actual["fps"] < self.config.fps * 0.8:
            log.warning(
                "camera is running at %.0ffps, below the requested %d; "
                "check that the MJPG pixel format was accepted",
                actual["fps"],
                self.config.fps,
            )

    def properties(self) -> dict[str, float | int | str]:
        if self._capture is None:
            return {}
        fourcc_int = int(self._capture.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
        return {
            "width": int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self._capture.get(cv2.CAP_PROP_FPS),
            "fourcc": fourcc,
            "exposure": self._capture.get(cv2.CAP_PROP_EXPOSURE),
            "gain": self._capture.get(cv2.CAP_PROP_GAIN),
        }

    # -- reading ----------------------------------------------------------

    def _reader(self) -> None:
        assert self._capture is not None
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.005)
                continue
            if self.config.mirror:
                frame = cv2.flip(frame, 1)
            timestamp = time.monotonic()
            with self._lock:
                if self._latest is not None and self._frame_id != self._last_delivered:
                    # The pipeline never consumed the previous frame.
                    self.dropped_frames += 1
                self._latest = (timestamp, frame)
                self._frame_id += 1

    def read(self, timeout: float = 1.0) -> tuple[float, np.ndarray] | None:
        """Block for the next unseen frame. Returns ``(timestamp, bgr)`` or None."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None and self._frame_id != self._last_delivered:
                    self._last_delivered = self._frame_id
                    return self._latest
            time.sleep(0.001)
        return None

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> Webcam:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def list_cameras(max_index: int = 8) -> list[tuple[int, int, int]]:
    """``(index, width, height)`` for every camera that opens.

    Useful when the user has several, since a laptop's built-in camera is
    usually index 0 and usually not the one they want to track with.
    """
    found: list[tuple[int, int, int]] = []
    for index in range(max_index):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if capture.isOpened():
            found.append(
                (
                    index,
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
            )
        capture.release()
    return found
