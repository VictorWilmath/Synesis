"""Interactive calibration sessions.

Extrinsics calibration asks the user to move around the play space wearing the
headset while the camera watches. There is no checkerboard and nothing to
print: SteamVR provides the metric ground truth and the pose model provides the
pixels.

The UI is deliberately pushy about coverage, because the failure mode here is
silent. A user who stands still and presses enter gets a solve that looks fine,
reports a low reprojection error, and is wrong, since fitting a camera pose to a
cluster of nearly identical points is underdetermined. The session therefore
refuses to solve until the samples span enough of the room.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace

import cv2
import numpy as np

from ..calib.anchors import CorrespondenceBuffer
from ..calib.body import calibrate_body, save_body
from ..calib.extrinsics import (
    MIN_COVERAGE_M,
    save_extrinsics,
    solve_extrinsics,
)
from ..capture.intrinsics import calibrate_intrinsics, load_intrinsics, save_intrinsics
from ..capture.webcam import Webcam
from ..config import Config
from ..pose2d.estimator import Pose2DEstimator
from . import overlay

log = logging.getLogger(__name__)


def _open_vr(config: Config):
    from ..vr import OpenVRSource

    source = OpenVRSource()
    source.open()
    return source


def calibrate_extrinsics_session(config: Config) -> int:
    """Solve where the camera is, from headset and controller correspondences."""
    intrinsics = load_intrinsics(
        config.camera.intrinsics.path,
        width=config.camera.width,
        height=config.camera.height,
        fallback_fov_deg=config.camera.intrinsics.fallback_fov_deg,
    )
    if not intrinsics.calibrated:
        log.warning(
            "using a guessed focal length from fallback_fov_deg. The camera "
            "pose can only be as good as the intrinsics; run "
            "calibrate-intrinsics first for a real result."
        )

    try:
        vr = _open_vr(config)
    except Exception as exc:
        log.error("cannot calibrate without SteamVR: %s", exc)
        return 1

    camera = Webcam(config.camera)
    # SteamVR needs uninterrupted access to the GPU. Calibration tolerates a
    # lower 2D-pose rate, so always use the smallest model on CPU here.
    estimator = Pose2DEstimator(replace(config.pose2d, mode="lightweight", device="cpu"))
    sample_interval_s = 1.0 / config.calibration.sample_rate_hz
    buffer = CorrespondenceBuffer(
        min_spacing_m=config.calibration.min_sample_spacing_m,
        min_score=max(0.5, config.pose2d.min_keypoint_score),
        allowed_anchors=("head",),
    )

    print(
        "\nCamera extrinsics calibration\n"
        "  Stay in frame and move around: step side to side, forward and back,\n"
        "  crouch, and turn your head naturally while staying in view.\n"
        "  This first room solve deliberately uses the headset only.\n"
        "  Cover as much of your play space as you can.\n\n"
        "  Synesis saves automatically when it has enough valid movement.\n"
        "  [c] clear samples    [q] abort\n"
    )

    camera.open()
    started = time.monotonic()
    last_sample_at = float("-inf")
    latest_keypoints = None
    try:
        while True:
            frame_data = camera.read()
            if frame_data is None:
                continue
            timestamp, frame = frame_data

            # A calibration needs varied correspondences, not 30 expensive GPU
            # inferences per second. Throttle only this screen; the normal live
            # tracker remains free to use its full rate.
            if time.monotonic() - last_sample_at >= sample_interval_s:
                latest_keypoints = estimator(frame, timestamp=timestamp)
                state = vr.poll()
                last_sample_at = time.monotonic()
                if latest_keypoints is not None and state is not None:
                    buffer.add(state, latest_keypoints)

            ready_to_save = _meets_requirements(buffer, config)

            canvas = overlay.render(
                frame,
                _preview_result(timestamp, latest_keypoints),
                intrinsics=intrinsics.matrix,
                min_score=config.pose2d.min_keypoint_score,
                extra_lines=_calibration_lines(buffer, config, started),
            )
            cv2.imshow("calibrate-extrinsics", canvas)

            if ready_to_save:
                print("Enough room samples collected; solving and saving automatically.")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("aborted")
                return 1
            if key == ord("c"):
                buffer.clear()
                continue
    finally:
        camera.close()
        vr.close()
        cv2.destroyAllWindows()

    print("Solving from headset movement.")
    # A noisy intrinsics solve needs a correspondingly wider initial RANSAC
    # gate. The final RMS value still reports the real quality of the result.
    intrinsics_error = intrinsics.rms_error or 0.0
    ransac_threshold_px = max(
        config.calibration.ransac_reproj_threshold_px,
        6.0 + 4.0 * intrinsics_error,
    )
    extrinsics = solve_extrinsics(
        buffer.samples,
        intrinsics.matrix,
        intrinsics.distortion,
        ransac_threshold_px=ransac_threshold_px,
        refine_offsets=False,
    )
    if extrinsics is None:
        print("Calibration failed. Try again with more movement around the space.")
        return 1

    save_extrinsics(extrinsics, config.calibration.path)
    print(
        f"\nCamera is at {np.round(extrinsics.camera_position, 2)} in play space, "
        f"{extrinsics.rms_error_px:.1f} px RMS "
        f"over {extrinsics.inliers}/{extrinsics.total} correspondences."
    )
    if extrinsics.rms_error_px > 12.0:
        print(
            "That error is high. Check that the camera did not move, and "
            "calibrate the intrinsics if you have not."
        )
    return 0


def _preview_result(timestamp: float, keypoints):
    from ..types import FrameResult

    return FrameResult(timestamp=timestamp, keypoints2d=keypoints)


def _meets_requirements(buffer: CorrespondenceBuffer, config: Config) -> bool:
    return len(buffer) >= config.calibration.min_samples and buffer.coverage() >= max(
        config.calibration.min_coverage_m, MIN_COVERAGE_M
    )


def _calibration_lines(buffer: CorrespondenceBuffer, config: Config, started: float) -> list[str]:
    counts = buffer.counts()
    lines = [
        f"samples {len(buffer)}/{config.calibration.min_samples}  "
        f"coverage {buffer.coverage():.2f}/{config.calibration.min_coverage_m:.2f} m",
        f"headset samples {counts['head']}",
        "enough samples collected - saving automatically"
        if _meets_requirements(buffer, config)
        else "keep moving around your play space",
        f"sampling at {config.calibration.sample_rate_hz:.0f} Hz",
        f"{time.monotonic() - started:.0f}s",
    ]
    return lines


def calibrate_intrinsics_session(config: Config) -> int:
    """Classic checkerboard intrinsics calibration."""
    board = (9, 6)
    square_m = 0.025
    capture_interval_s = 5.0
    target_views = 15
    camera = Webcam(config.camera)
    frames: list[np.ndarray] = []
    last_capture_at = float("-inf")

    print(
        f"\nCamera intrinsics calibration\n"
        f"  Print a {board[0] + 1}x{board[1] + 1} checkerboard "
        f"({square_m * 1000:.0f} mm squares) and hold it up to the camera.\n"
        "  Capture 15-25 views: near and far, centred and in the corners,\n"
        "  and tilted. Corner and tilted views are what pin down distortion.\n\n"
        f"  A valid view is captured automatically every {capture_interval_s:.0f} seconds.\n"
        f"  It solves and saves automatically after {target_views} views.\n"
        "  Move the board between captures. [q] abort\n"
    )

    camera.open()
    try:
        while True:
            frame_data = camera.read()
            if frame_data is None:
                continue
            _, frame = frame_data

            canvas = frame.copy()
            found, corners = cv2.findChessboardCorners(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                board,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK,
            )
            if found:
                cv2.drawChessboardCorners(canvas, board, corners, found)
                now = time.monotonic()
                if now - last_capture_at >= capture_interval_s:
                    frames.append(frame.copy())
                    last_capture_at = now
                    print(f"captured {len(frames)}")
            else:
                now = time.monotonic()
            seconds_until_capture = max(0.0, capture_interval_s - (now - last_capture_at))
            overlay.draw_stats(
                canvas,
                [
                    f"captured {len(frames)}",
                    "board visible" if found else "no board detected",
                    (
                        f"next automatic capture in {seconds_until_capture:.1f}s"
                        if found
                        else "show the whole checkerboard to the camera"
                    ),
                    "[q] abort",
                ],
            )
            cv2.imshow("calibrate-intrinsics", canvas)

            if len(frames) >= target_views:
                print("Enough checkerboard views collected; solving and saving automatically.")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 1
    finally:
        camera.close()
        cv2.destroyAllWindows()

    try:
        intrinsics = calibrate_intrinsics(frames, board_size=board, square_size_m=square_m)
    except (RuntimeError, ValueError) as exc:
        print(f"Calibration failed: {exc}")
        return 1

    save_intrinsics(intrinsics, config.camera.intrinsics.path)
    print(
        f"\nfocal length {intrinsics.matrix[0, 0]:.0f}, "
        f"{intrinsics.matrix[1, 1]:.0f} px, "
        f"horizontal FOV {intrinsics.fov_x_deg:.1f} deg, "
        f"reprojection error {intrinsics.rms_error:.2f} px"
    )
    return 0


def calibrate_body_session(config: Config, duration_s: float = 20.0) -> int:
    """Measure the user's proportions from a short tracked session."""
    from .pipeline import Pipeline

    intrinsics = load_intrinsics(
        config.camera.intrinsics.path,
        width=config.camera.width,
        height=config.camera.height,
        fallback_fov_deg=config.camera.intrinsics.fallback_fov_deg,
    )
    try:
        vr = _open_vr(config)
    except Exception as exc:
        log.error("cannot calibrate the body without SteamVR: %s", exc)
        return 1

    camera = Webcam(config.camera)
    estimator = Pose2DEstimator(config.pose2d)
    pipeline = Pipeline(config, intrinsics=intrinsics.matrix)

    from ..calib.extrinsics import load_extrinsics

    extrinsics = load_extrinsics(config.calibration.path)
    if extrinsics is None:
        log.error("run calibrate-extrinsics first")
        return 1
    pipeline.set_extrinsics(extrinsics.rotation_cw, extrinsics.translation_cw)

    print(
        f"\nBody calibration\n"
        f"  Stand upright facing the camera, then move your arms and legs "
        f"slowly for {duration_s:.0f} seconds.\n"
    )

    skeletons = []
    states = []
    camera.open()
    started = time.monotonic()
    try:
        while time.monotonic() - started < duration_s:
            frame_data = camera.read()
            if frame_data is None:
                continue
            timestamp, frame = frame_data
            keypoints = estimator(frame, timestamp=timestamp)
            state = vr.poll()
            result = pipeline.process(keypoints, state, timestamp=timestamp)
            if result.skeleton3d is not None:
                skeletons.append(result.skeleton3d)
            if state is not None:
                states.append(state)

            canvas = overlay.render(
                frame,
                result,
                intrinsics=intrinsics.matrix,
                rotation_cw=extrinsics.rotation_cw,
                translation_cw=extrinsics.translation_cw,
                min_score=config.pose2d.min_keypoint_score,
                extra_lines=[
                    f"{duration_s - (time.monotonic() - started):.0f}s remaining",
                    f"{len(skeletons)} frames",
                ],
            )
            cv2.imshow("calibrate-body", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return 1
    finally:
        camera.close()
        vr.close()
        cv2.destroyAllWindows()

    body = calibrate_body(skeletons, states, fallback_height=config.body.height_m)
    save_body(body, config.body.path)
    print(f"\nheight {body.height_m:.2f} m from {body.frames_used} frames")
    return 0
