"""Camera intrinsics: checkerboard calibration, plus a usable fallback.

Intrinsics matter here more than usual. The extrinsics solve is a PnP against
the headset and controllers, and PnP absorbs focal-length error as depth error.
A 10% focal error becomes roughly a 10% depth error on every joint, which is
the difference between an avatar that stands where you stand and one that
leans.

The fallback from a field-of-view guess is good enough to start tracking, but a
real calibration is worth the five minutes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..geom.spaces import intrinsics_from_fov

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Intrinsics:
    matrix: np.ndarray  # (3, 3)
    distortion: np.ndarray  # (5,) OpenCV plumb-bob coefficients
    width: int
    height: int
    # RMS reprojection error in pixels from the calibration that produced this,
    # or None for the field-of-view fallback.
    rms_error: float | None = None

    @property
    def calibrated(self) -> bool:
        return self.rms_error is not None

    @property
    def fov_x_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.width / (2.0 * self.matrix[0, 0]))))

    def scaled_to(self, width: int, height: int) -> Intrinsics:
        """Rescale for a different capture resolution.

        Calibrating at one resolution and running at another is a common and
        silent source of error.
        """
        if width == self.width and height == self.height:
            return self
        sx, sy = width / self.width, height / self.height
        matrix = self.matrix.copy()
        matrix[0, :] *= sx
        matrix[1, :] *= sy
        return Intrinsics(matrix, self.distortion.copy(), width, height, self.rms_error)

    @classmethod
    def from_fov(cls, width: int, height: int, fov_deg: float) -> Intrinsics:
        return cls(
            matrix=intrinsics_from_fov(width, height, fov_deg),
            distortion=np.zeros(5),
            width=width,
            height=height,
            rms_error=None,
        )


def save_intrinsics(intrinsics: Intrinsics, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "matrix": intrinsics.matrix.tolist(),
                "distortion": intrinsics.distortion.tolist(),
                "width": intrinsics.width,
                "height": intrinsics.height,
                "rms_error": intrinsics.rms_error,
            },
            indent=2,
        )
    )
    log.info("wrote intrinsics to %s", path)


def load_intrinsics(
    path: Path | str,
    *,
    width: int,
    height: int,
    fallback_fov_deg: float = 62.0,
) -> Intrinsics:
    """Load a calibration, or synthesise one from a FOV guess if absent."""
    path = Path(path)
    if not path.exists():
        log.warning(
            "no intrinsics at %s; falling back to a %.0f degree FOV guess. "
            "Depth accuracy will suffer until you run calibrate-intrinsics.",
            path,
            fallback_fov_deg,
        )
        return Intrinsics.from_fov(width, height, fallback_fov_deg)

    data = json.loads(path.read_text())
    intrinsics = Intrinsics(
        matrix=np.array(data["matrix"], dtype=np.float64),
        distortion=np.array(data["distortion"], dtype=np.float64),
        width=int(data["width"]),
        height=int(data["height"]),
        rms_error=data.get("rms_error"),
    )
    return intrinsics.scaled_to(width, height)


def calibrate_intrinsics(
    frames: list[np.ndarray],
    board_size: tuple[int, int] = (9, 6),
    square_size_m: float = 0.025,
) -> Intrinsics:
    """Standard checkerboard calibration.

    `board_size` counts *inner* corners, so the common 10x7-square board is
    (9, 6). `square_size_m` only sets the world scale; it does not affect the
    focal length in pixels, but getting it right keeps the reported translation
    meaningful.
    """
    if not frames:
        raise ValueError("no frames supplied")

    # Board corner coordinates in its own frame, z = 0.
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    objp *= square_size_m

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    height, width = frames[0].shape[:2]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        found, corners = cv2.findChessboardCorners(
            gray,
            board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
        )
        if not found:
            continue
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(objp)
        image_points.append(refined)

    if len(object_points) < 5:
        raise RuntimeError(
            f"only {len(object_points)} usable views; need at least 5. "
            "Cover the frame corners and vary the board's tilt."
        )

    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, (width, height), None, None
    )
    log.info("calibrated on %d views, RMS reprojection error %.3f px", len(object_points), rms)
    if rms > 1.0:
        log.warning("RMS error above 1px suggests a poor capture set; consider redoing it")

    return Intrinsics(
        matrix=np.asarray(matrix, dtype=np.float64),
        distortion=np.asarray(distortion, dtype=np.float64).reshape(-1)[:5],
        width=width,
        height=height,
        rms_error=float(rms),
    )
