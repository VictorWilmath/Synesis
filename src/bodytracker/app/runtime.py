"""The live tracking loop: camera in, OSC out."""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ..capture.intrinsics import load_intrinsics
from ..capture.webcam import Webcam
from ..config import Config
from ..osc import VRChatOSCSender
from ..pose2d.estimator import Pose2DEstimator
from ..types import FrameResult, VRState
from . import overlay
from .pipeline import Pipeline

log = logging.getLogger(__name__)


class Runtime:
    """Owns the camera, the model, the pipeline and the sender."""

    def __init__(
        self,
        config: Config,
        *,
        vr_source: object | None = None,
        show_overlay: bool | None = None,
    ) -> None:
        self.config = config
        self.vr_source = vr_source
        self.show_overlay = config.debug.overlay if show_overlay is None else show_overlay

        self.intrinsics = load_intrinsics(
            config.camera.intrinsics.path,
            width=config.camera.width,
            height=config.camera.height,
            fallback_fov_deg=config.camera.intrinsics.fallback_fov_deg,
        )

        self.camera = Webcam(config.camera)
        self.estimator = Pose2DEstimator(config.pose2d)
        self.pipeline = Pipeline(config, intrinsics=self.intrinsics.matrix)
        self.sender = VRChatOSCSender(
            config.osc.tracker_roles(),
            host=config.osc.host,
            port=config.osc.port,
            send_head=config.osc.send_head,
            rate_hz=config.osc.send_rate_hz,
        )

        self.refiner = None
        self.recorder = None

        self._frame_times: deque[float] = deque(maxlen=120)
        self._latencies: deque[float] = deque(maxlen=120)
        self._last_stats_print = 0.0
        self._running = False

    # -- calibration --------------------------------------------------------

    def load_calibration(self) -> None:
        """Load extrinsics and body proportions if they have been solved."""
        from ..calib.body import load_body
        from ..calib.extrinsics import load_extrinsics
        from ..calib.online import OnlineExtrinsicsRefiner

        extrinsics = load_extrinsics(self.config.calibration.path)
        if extrinsics is not None:
            self._apply_extrinsics(extrinsics)
            log.info(
                "loaded camera extrinsics (%.1f px RMS, camera at %s)",
                extrinsics.rms_error_px,
                np.round(extrinsics.camera_position, 2),
            )
        else:
            log.warning(
                "no camera extrinsics at %s. Output will be camera-relative and "
                "will not line up with your play space. Run calibrate-extrinsics.",
                self.config.calibration.path,
            )

        body = load_body(self.config.body.path)
        if body is not None:
            self.pipeline.set_body(body.height_m, body.bone_lengths)
            log.info("loaded body calibration (height %.2f m)", body.height_m)

        if self.vr_source is not None and self.config.calibration.online_refine:
            self.refiner = OnlineExtrinsicsRefiner(
                self.intrinsics,
                interval_s=self.config.calibration.refine_interval_s,
                min_samples=self.config.calibration.min_samples,
            )
            self.refiner.set_current(extrinsics)

    def _apply_extrinsics(self, extrinsics) -> None:
        self.pipeline.set_extrinsics(extrinsics.rotation_cw, extrinsics.translation_cw)
        # The calibrator also solves where each device sits relative to its
        # keypoint, and the anchored lifter needs the same numbers to pin the
        # head in the right place.
        if extrinsics.offsets and hasattr(self.pipeline.lifter, "offsets"):
            self.pipeline.lifter.offsets = extrinsics.offsets

    # -- main loop ----------------------------------------------------------

    def _read_vr(self) -> VRState | None:
        if self.vr_source is None:
            return None
        try:
            return self.vr_source.poll()
        except Exception:
            log.exception("VR poll failed; continuing without an anchor this frame")
            return None

    def step(self) -> tuple[FrameResult, np.ndarray] | None:
        """Process one camera frame. Returns None if no frame arrived."""
        frame_data = self.camera.read()
        if frame_data is None:
            return None
        captured_at, frame = frame_data

        keypoints = self.estimator(frame, timestamp=captured_at)
        vr = self._read_vr()
        result = self.pipeline.process(keypoints, vr, timestamp=captured_at)
        result.timings.update(self.estimator.last_timings)

        if self.refiner is not None:
            self.refiner.observe(vr, keypoints)
            refined = self.refiner.maybe_refine()
            if refined is not None:
                self._apply_extrinsics(refined)

        head = vr.head if vr is not None else None
        self.sender.send(result.targets, head=head)

        # Latency measured from capture, so it includes everything the user
        # actually waits for, not just inference.
        latency_ms = (time.monotonic() - captured_at) * 1000.0
        if self.recorder is not None:
            self.recorder.add(result, latency_ms=latency_ms)
        self._latencies.append(latency_ms)
        self._frame_times.append(captured_at)
        return result, frame

    # -- recording ----------------------------------------------------------

    def start_recording(self, label: str = "", notes: str = "") -> None:
        """Capture this session's inputs and outputs for offline analysis."""
        from ..eval.session import SessionMeta, SessionRecorder

        self.recorder = SessionRecorder(
            SessionMeta(
                intrinsics=self.intrinsics.matrix,
                rotation_cw=self.pipeline.context.rotation_cw,
                translation_cw=self.pipeline.context.translation_cw,
                height_m=self.pipeline.context.height_m,
                roles=[role.value for role in self.pipeline.roles],
                offsets=getattr(self.pipeline.lifter, "offsets", None) or {},
                label=label,
                notes=notes,
            )
        )

    def save_recording(self, path: Path | str):
        if self.recorder is None or len(self.recorder) == 0:
            log.warning("nothing recorded, not writing a session")
            return None
        # The camera may have been calibrated part-way through, so take the
        # extrinsics as they ended up rather than as they started.
        self.recorder.meta.rotation_cw = self.pipeline.context.rotation_cw
        self.recorder.meta.translation_cw = self.pipeline.context.translation_cw
        written = self.recorder.build().save(path)
        log.info("wrote %d frames to %s", len(self.recorder), written)
        return written

    def run(self) -> None:
        self._running = True
        self.camera.open()
        self.load_calibration()
        log.info("tracking; press q in the overlay window or Ctrl-C to stop")

        try:
            while self._running:
                stepped = self.step()
                if stepped is None:
                    continue
                result, frame = stepped

                if self.show_overlay:
                    canvas = overlay.render(
                        frame,
                        result,
                        intrinsics=self.intrinsics.matrix,
                        rotation_cw=self.pipeline.context.rotation_cw,
                        translation_cw=self.pipeline.context.translation_cw,
                        min_score=self.config.pose2d.min_keypoint_score,
                        extra_lines=self._stat_lines(),
                    )
                    cv2.imshow("bodytracker", canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("r"):
                        log.info("resetting tracking state")
                        self.estimator.reset()
                        self.pipeline.reset()

                self._maybe_print_stats()
        except KeyboardInterrupt:
            log.info("interrupted")
        finally:
            self.close()

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self.camera.close()
        self.sender.close()
        if self.show_overlay:
            cv2.destroyAllWindows()
        stats = self.pipeline.stats()
        log.info(
            "%d frames tracked, %d dropped (%.0f%% tracked), %d detector runs",
            stats["tracked"],
            stats["dropped"],
            stats["track_rate"] * 100,
            self.estimator.detections_run,
        )

    # -- stats --------------------------------------------------------------

    def _fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def _stat_lines(self) -> list[str]:
        lines = [f"{self._fps():.0f} fps"]
        if self._latencies:
            lines[0] += f"  latency {np.median(self._latencies):.0f}ms"
        if not self.pipeline.context.calibrated:
            lines.append("UNCALIBRATED - run calibrate-extrinsics")
        if self.vr_source is None:
            lines.append("no HMD anchor")
            return lines

        diagnostics = self.pipeline.diagnostics()
        anchored = "anchored" if diagnostics.get("anchored") else "ANCHOR LOST"
        head = diagnostics.get("head_residual_m")
        wrist = diagnostics.get("wrist_residual_m")
        parts = [anchored]
        if head is not None:
            parts.append(f"head {head * 100:.0f}cm")
        if wrist is not None:
            parts.append(f"wrist {wrist * 100:.0f}cm")
        lines.append(" ".join(parts))
        return lines

    def _maybe_print_stats(self) -> None:
        interval = self.config.debug.print_stats_interval_s
        if interval <= 0:
            return
        now = time.monotonic()
        if now - self._last_stats_print < interval:
            return
        self._last_stats_print = now
        stats = self.pipeline.stats()
        log.info(
            "%.0f fps, latency %.0f ms, tracked %.0f%%, dropped camera frames %d",
            self._fps(),
            np.median(self._latencies) if self._latencies else 0.0,
            stats["track_rate"] * 100,
            self.camera.dropped_frames,
        )


def run(
    config: Config,
    *,
    use_vr: bool = True,
    show_overlay: bool | None = None,
    record_to: Path | str | None = None,
    record_label: str = "",
) -> None:
    """Entry point used by the CLI."""
    vr_source = None
    if use_vr and config.vr.enabled:
        from ..vr import OpenVRSource

        try:
            vr_source = OpenVRSource()
            vr_source.open()
            log.info("connected to SteamVR")
        except Exception as exc:
            log.warning(
                "could not connect to SteamVR (%s). Running without the HMD "
                "anchor, which is the whole advantage of this approach.",
                exc,
            )
            vr_source = None

    runtime = Runtime(config, vr_source=vr_source, show_overlay=show_overlay)
    if record_to is not None:
        runtime.start_recording(label=record_label or Path(record_to).stem)
    try:
        runtime.run()
    finally:
        if record_to is not None:
            runtime.save_recording(record_to)
        if vr_source is not None:
            vr_source.close()


def ensure_calibration_dir(config: Config) -> None:
    Path(config.calibration.path).parent.mkdir(parents=True, exist_ok=True)
