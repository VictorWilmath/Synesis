"""Solving where the camera is, using the headset and controllers as targets.

No checkerboard: the user wears the headset, holds the controllers and moves
around while the camera watches. SteamVR supplies the 3D, the pose model
supplies the 2D, and PnP recovers the camera pose in play space.

The solve runs in three stages:

1. ``solvePnPRansac`` for a robust initial pose that ignores frames where the
   keypoint landed on the wrong thing.
2. ``solvePnPRefineLM`` on the inliers, for accuracy.
3. An optional joint refinement of the camera pose *and* the device-local
   offsets, since the nominal offset from a headset's tracking origin to where
   a pose model puts the "head" keypoint is a guess that varies by headset and
   by hairstyle.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..geom.rotations import orthonormalize
from .anchors import (
    ANCHORS,
    Correspondence,
    build_point_arrays,
    pack_offsets,
    unpack_offsets,
)

log = logging.getLogger(__name__)

# Below this many correspondences PnP is not worth attempting.
MIN_CORRESPONDENCES = 12
# Head positions must span at least this much for the solve to be conditioned.
MIN_COVERAGE_M = 0.35


@dataclass(slots=True)
class CameraExtrinsics:
    """World-to-camera transform: ``x_camera = R @ x_play + t``."""

    rotation_cw: np.ndarray
    translation_cw: np.ndarray
    rms_error_px: float
    inliers: int
    total: int
    offsets: dict[str, np.ndarray]

    @property
    def camera_position(self) -> np.ndarray:
        """Where the camera sits in play space."""
        return -self.rotation_cw.T @ self.translation_cw

    @property
    def forward(self) -> np.ndarray:
        """The camera's viewing direction in play space."""
        return self.rotation_cw[2, :]


def _rvec_tvec(rotation_cw: np.ndarray, translation_cw: np.ndarray):
    rvec, _ = cv2.Rodrigues(np.asarray(rotation_cw, dtype=np.float64))
    return rvec, np.asarray(translation_cw, dtype=np.float64).reshape(3, 1)


def _reprojection_errors(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, intrinsics, distortion)
    return np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)


def solve_extrinsics(
    samples: list[Correspondence],
    intrinsics: np.ndarray,
    distortion: np.ndarray | None = None,
    *,
    ransac_threshold_px: float = 8.0,
    refine_offsets: bool = True,
) -> CameraExtrinsics | None:
    """Recover the camera pose from device-to-keypoint correspondences."""
    if len(samples) < MIN_CORRESPONDENCES:
        log.warning("only %d correspondences, need at least %d", len(samples), MIN_CORRESPONDENCES)
        return None

    distortion = np.zeros(5) if distortion is None else np.asarray(distortion, dtype=np.float64)
    offsets = {a.name: a.local_offset.astype(np.float64).copy() for a in ANCHORS}
    object_points, image_points = build_point_arrays(samples, offsets)

    ok, rvec, tvec, inlier_indices = cv2.solvePnPRansac(
        object_points,
        image_points,
        np.asarray(intrinsics, dtype=np.float64),
        distortion,
        flags=cv2.SOLVEPNP_SQPNP,
        reprojectionError=ransac_threshold_px,
        iterationsCount=500,
        confidence=0.999,
    )
    if not ok or inlier_indices is None or len(inlier_indices) < MIN_CORRESPONDENCES:
        log.warning("PnP failed or found too few inliers")
        return None

    inliers = inlier_indices.reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(
        object_points[inliers],
        image_points[inliers],
        np.asarray(intrinsics, dtype=np.float64),
        distortion,
        rvec,
        tvec,
    )

    if refine_offsets:
        # Deliberately every sample, not just the RANSAC inliers. A wrong
        # offset makes that whole anchor reproject badly, so RANSAC throws the
        # lot away, and refining on what survives leaves the offset that caused
        # the problem completely unconstrained. Passing everything in with a
        # robust loss lets those samples pull the offset toward the truth while
        # still discounting genuine outliers.
        rvec, tvec, offsets = _refine_jointly(
            samples,
            intrinsics,
            distortion,
            rvec,
            tvec,
            offsets,
            outlier_scale_px=ransac_threshold_px,
        )
        object_points, image_points = build_point_arrays(samples, offsets)
        # Re-decide who the outliers are now the anchors have moved.
        all_errors = _reprojection_errors(
            object_points, image_points, rvec, tvec, intrinsics, distortion
        )
        refined_inliers = np.flatnonzero(all_errors <= ransac_threshold_px)
        if len(refined_inliers) >= MIN_CORRESPONDENCES:
            inliers = refined_inliers

    errors = _reprojection_errors(
        object_points[inliers], image_points[inliers], rvec, tvec, intrinsics, distortion
    )
    rms = float(np.sqrt(np.mean(errors**2)))

    rotation_cw = orthonormalize(cv2.Rodrigues(rvec)[0])
    extrinsics = CameraExtrinsics(
        rotation_cw=rotation_cw,
        translation_cw=np.asarray(tvec, dtype=np.float64).reshape(3),
        rms_error_px=rms,
        inliers=int(len(inliers)),
        total=len(samples),
        offsets=offsets,
    )

    log.info(
        "camera solved at %s looking %s, %.2f px RMS over %d/%d correspondences",
        np.round(extrinsics.camera_position, 2),
        np.round(extrinsics.forward, 2),
        rms,
        len(inliers),
        len(samples),
    )
    if rms > 12.0:
        log.warning(
            "reprojection error is high. Check the camera intrinsics and make "
            "sure the camera did not move during calibration."
        )
    return extrinsics


def _refine_jointly(
    samples: list[Correspondence],
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    offsets: dict[str, np.ndarray],
    outlier_scale_px: float = 8.0,
):
    """Optimise the camera pose and the device-local offsets together.

    The offset from a headset's tracking origin to where the pose model places
    the head keypoint depends on the headset, how it is worn and the user's
    hair. Solving for it rather than assuming it typically halves the residual.
    Offsets are bounded to a plausible range so the optimiser cannot explain
    away a genuinely bad camera pose by moving the anchors somewhere absurd.

    Uses a soft-L1 loss so that samples the initial guess reprojects badly,
    which is precisely the evidence that an offset is wrong, still influence
    the fit instead of being written off as noise.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:
        log.debug("scipy unavailable, skipping joint offset refinement")
        return rvec, tvec, offsets

    image_points = np.array([s.pixel for s in samples], dtype=np.float64)
    initial = np.concatenate(
        [np.asarray(rvec).reshape(3), np.asarray(tvec).reshape(3), pack_offsets(offsets)]
    )

    def residuals(params: np.ndarray) -> np.ndarray:
        current_offsets = unpack_offsets(params[6:])
        object_points = np.array(
            [s.world_point(current_offsets[s.anchor]) for s in samples], dtype=np.float64
        )
        projected, _ = cv2.projectPoints(
            object_points,
            params[:3].reshape(3, 1),
            params[3:6].reshape(3, 1),
            intrinsics,
            distortion,
        )
        return (projected.reshape(-1, 2) - image_points).ravel()

    lower = np.concatenate([np.full(6, -np.inf), pack_offsets(offsets) - 0.15])
    upper = np.concatenate([np.full(6, np.inf), pack_offsets(offsets) + 0.15])

    try:
        result = least_squares(
            residuals,
            initial,
            bounds=(lower, upper),
            method="trf",
            loss="soft_l1",
            f_scale=outlier_scale_px,
            max_nfev=400,
        )
    except Exception:
        log.exception("joint refinement failed; keeping the PnP result")
        return rvec, tvec, offsets

    refined = unpack_offsets(result.x[6:])
    for name, offset in refined.items():
        moved = float(np.linalg.norm(offset - offsets[name]))
        if moved > 0.005:
            log.info("refined %s offset by %.0f mm", name, moved * 1000)
    return result.x[:3].reshape(3, 1), result.x[3:6].reshape(3, 1), refined


def save_extrinsics(extrinsics: CameraExtrinsics, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "rotation_cw": extrinsics.rotation_cw.tolist(),
                "translation_cw": extrinsics.translation_cw.tolist(),
                "rms_error_px": extrinsics.rms_error_px,
                "inliers": extrinsics.inliers,
                "total": extrinsics.total,
                "offsets": {k: v.tolist() for k, v in extrinsics.offsets.items()},
                "camera_position": extrinsics.camera_position.tolist(),
            },
            indent=2,
        )
    )
    log.info("wrote camera extrinsics to %s", path)


def load_extrinsics(path: Path | str) -> CameraExtrinsics | None:
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return CameraExtrinsics(
        rotation_cw=np.array(data["rotation_cw"], dtype=np.float64),
        translation_cw=np.array(data["translation_cw"], dtype=np.float64),
        rms_error_px=float(data["rms_error_px"]),
        inliers=int(data["inliers"]),
        total=int(data["total"]),
        offsets={k: np.array(v, dtype=np.float64) for k, v in data.get("offsets", {}).items()},
    )
