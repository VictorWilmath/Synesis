"""The live tracking loop: camera in, OSC out.

There is no setup step. With SteamVR running, the loop calibrates the camera
from the skeleton and the headset while the user moves around normally, and
only starts sending trackers to VRChat once that calibration is trustworthy.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ..calib.auto import (
    AutoCalibrator,
    CalibrationUpdate,
    CameraCalibration,
    load_calibration,
    save_calibration,
)
from ..calib.body import BodyCalibration, load_body, save_body
from ..capture.webcam import Webcam
from ..config import Config, OSCConfig
from ..geom import intrinsics_from_fov, slerp
from ..lift.kinematics import bone_lengths_from_height
from ..osc import VMTSender, VRChatOSCSender, device_name, find_vmt_driver
from ..pose2d.estimator import Pose2DEstimator
from ..types import DevicePose, FrameResult, TrackerTarget, VRState
from ..vr.history import VRHistory
from ..vr.source import standing_height_from_head_heights
from . import overlay
from .pipeline import Pipeline

log = logging.getLogger(__name__)

# Standing height is re-estimated from the headset this often, once there are
# enough samples, and only applied when it moves by more than the threshold.
_HEIGHT_UPDATE_INTERVAL_S = 10.0
_HEIGHT_MIN_SAMPLES = 300
_HEIGHT_CHANGE_M = 0.02

_MIN_EASE_S = 0.005
_MAX_EASE_S = 0.30


def _copy_target(target: TrackerTarget) -> TrackerTarget:
    return TrackerTarget(
        role=target.role,
        position=np.asarray(target.position, dtype=np.float64).copy(),
        rotation=np.asarray(target.rotation, dtype=np.float64).copy(),
        valid=target.valid,
        stale=target.stale,
    )


def _blend_targets(
    start: list[TrackerTarget] | None,
    end: list[TrackerTarget],
    fraction: float,
) -> list[TrackerTarget]:
    """`fraction` of the way from `start` to `end`, role by role."""
    out = [_copy_target(target) for target in end]
    if not start or fraction >= 1.0:
        return out
    fraction = max(0.0, fraction)
    start_by_role = {target.role: target for target in start}
    for target in out:
        origin = start_by_role.get(target.role)
        if origin is None:
            continue
        target.position = origin.position + fraction * (target.position - origin.position)
        target.rotation = slerp(origin.rotation, target.rotation, fraction)
    return out


def make_sender(config: OSCConfig, vr_source: object | None) -> VRChatOSCSender | VMTSender:
    target = config.target
    if target == "auto":
        target = "vmt" if find_vmt_driver() is not None else "vrchat"
        log.info(
            "output target auto: %s",
            "VMT is installed, using it"
            if target == "vmt"
            else "VMT is not installed, using VRChat's OSC trackers",
        )
    if target == "vmt":
        return VMTSender(
            config.tracker_roles(),
            host=config.host,
            port=config.vmt_port,
            mode=config.vmt_mode,
            rate_hz=config.send_rate_hz,
            standing_from_raw=(
                (lambda: getattr(vr_source, "standing_from_raw", None))
                if vr_source is not None
                else None
            ),
        )
    if target != "vrchat":
        raise ValueError(
            f"unknown output target '{config.target}', expected 'vrchat', 'vmt' or 'auto'"
        )
    return VRChatOSCSender(
        config.tracker_roles(),
        host=config.host,
        port=config.port,
        send_head=config.send_head,
        rate_hz=config.send_rate_hz,
    )


class _OSCOutputInterpolator:
    """Sends at the OSC output rate, easing from one webcam pose to the next.

    The camera gives 12 to 30 poses a second. Each new one is eased into over
    the time between camera frames, starting from wherever the output already
    is, so the trackers glide instead of stepping at the camera's rate. That
    costs one camera frame of delay. Predicting ahead instead was tried: from
    two noisy webcam poses it doubled the jitter and overshot every stop.
    """

    def __init__(self, sender: VRChatOSCSender | VMTSender, rate_hz: float) -> None:
        self.sender = sender
        self.interval_s = 1.0 / rate_hz if rate_hz > 0 else 0.0
        self._lock = threading.Lock()
        self._latest: tuple[list[TrackerTarget], DevicePose | None, float, float] | None = None
        self._start: list[TrackerTarget] | None = None
        self._span_s: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.interval_s <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="osc-output", daemon=True)
        self._thread.start()

    def publish(
        self,
        targets: list[TrackerTarget],
        head: DevicePose | None,
        timestamp: float,
        *,
        arrived_at: float | None = None,
    ) -> None:
        """Hand over the pose from the camera frame captured at `timestamp`."""
        arrived = time.monotonic() if arrived_at is None else arrived_at
        copied = [_copy_target(target) for target in targets]
        copied_head = (
            None
            if head is None
            else DevicePose(
                position=np.asarray(head.position, dtype=np.float64).copy(),
                rotation=np.asarray(head.rotation, dtype=np.float64).copy(),
                valid=head.valid,
                timestamp=head.timestamp,
            )
        )
        with self._lock:
            start = None
            span = None
            if self._latest is not None:
                span = timestamp - self._latest[2]
                start = self._current(self._latest, arrived)
            # After a gap there is nothing sensible to glide across.
            if span is None or not _MIN_EASE_S <= span <= _MAX_EASE_S:
                start = span = None
            self._start = start
            self._span_s = span
            self._latest = (copied, copied_head, timestamp, arrived)

    def clear(self) -> None:
        """Stop sending until the next publish."""
        with self._lock:
            self._latest = None
            self._start = None
            self._span_s = None

    def _current(
        self,
        latest: tuple[list[TrackerTarget], DevicePose | None, float, float],
        now: float,
    ) -> list[TrackerTarget]:
        targets, _, _, arrived = latest
        if self._span_s is None:
            return [_copy_target(target) for target in targets]
        return _blend_targets(self._start, targets, (now - arrived) / self._span_s)

    def _snapshot(self, now: float) -> tuple[list[TrackerTarget], DevicePose | None] | None:
        with self._lock:
            if self._latest is None:
                return None
            return self._current(self._latest, now), self._latest[1]

    def _run(self) -> None:
        next_send = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            snapshot = self._snapshot(now)
            if snapshot is not None:
                targets, head = snapshot
                self.sender.send(targets, head=head, now=now, force=True)
            next_send += self.interval_s
            self._stop.wait(max(0.0, next_send - time.monotonic()))
            if next_send < time.monotonic() - self.interval_s:
                next_send = time.monotonic()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


def describe_calibration(calibration: CameraCalibration) -> str:
    return (
        f"camera at {np.round(calibration.camera_position, 2)} m, "
        f"{calibration.fov_deg:.0f} deg FOV, "
        f"{calibration.latency_s * 1000:.0f} ms frame delay, "
        f"{calibration.rms_px:.1f} px fit, "
        f"+-{calibration.position_std_m * 100:.0f} cm"
    )


def describe_placement(targets: list[TrackerTarget], head: DevicePose | None) -> list[str]:
    """Where each tracker is relative to the headset: a sanity check on the 3D.

    Height is above the floor; ahead and right are along the direction the
    headset faces. Standing straight, the hip should be roughly under the head
    and the feet near the floor.
    """
    if head is None or not head.valid or not targets:
        return []
    forward = -np.asarray(head.rotation, dtype=np.float64)[:, 2]
    forward[1] = 0.0
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        return []
    forward /= norm
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    lines = []
    for target in targets:
        offset = np.asarray(target.position, dtype=np.float64) - head.position
        lines.append(
            f"{target.role.value} {float(target.position[1]):.2f}m up, "
            f"{float(offset @ forward):+.2f} ahead, {float(offset @ right):+.2f} right"
        )
    return lines


class Runtime:
    """Owns the camera, the model, the calibrator, the pipeline and the sender."""

    def __init__(
        self,
        config: Config,
        *,
        vr_source: object | None = None,
        show_overlay: bool | None = None,
        recalibrate: bool = False,
    ) -> None:
        self.config = config
        self.vr_source = vr_source
        self.show_overlay = config.debug.overlay if show_overlay is None else show_overlay
        self.recalibrate = recalibrate

        self.camera = Webcam(config.camera)
        self.estimator = Pose2DEstimator(config.pose2d)
        self.pipeline = Pipeline(
            config,
            intrinsics=intrinsics_from_fov(
                config.camera.width, config.camera.height, config.calibration.initial_fov_deg
            ),
        )
        self.sender = make_sender(config.osc, vr_source)
        self._output = _OSCOutputInterpolator(self.sender, config.osc.send_rate_hz)

        self.vr_history = (
            VRHistory(vr_source, poll_hz=config.vr.poll_hz) if vr_source is not None else None
        )
        self.calibrator: AutoCalibrator | None = None
        self.calibration: CameraCalibration | None = None
        self.latency_s = config.calibration.initial_latency_s
        self.recorder = None

        self._head_heights: deque[float] = deque(maxlen=20_000)
        self._last_height_update = -float("inf")
        self._frame_times: deque[float] = deque(maxlen=120)
        self._latencies: deque[float] = deque(maxlen=120)
        self._last_targets: list[TrackerTarget] = []
        self._last_head: DevicePose | None = None
        self._last_stats_print = 0.0
        self._running = False

    # -- calibration --------------------------------------------------------

    def _frame_size(self) -> tuple[int, int]:
        """The resolution the camera actually delivers, which is what keypoints use."""
        properties = self.camera.properties()
        width = int(properties.get("width") or 0) or self.config.camera.width
        height = int(properties.get("height") or 0) or self.config.camera.height
        return width, height

    def load_calibration(self) -> None:
        """Load body proportions and any saved camera calibration, then start calibrating."""
        width, height = self._frame_size()
        self.pipeline.set_intrinsics(
            intrinsics_from_fov(width, height, self.config.calibration.initial_fov_deg)
        )

        body = load_body(self.config.body.path)
        if body is not None:
            self.pipeline.set_body(body.height_m, body.bone_lengths)
            log.info("loaded body proportions (height %.2f m)", body.height_m)

        if self.vr_history is None:
            log.warning("no SteamVR: output is camera-relative and the camera cannot be calibrated")
            return

        saved = (
            None
            if self.recalibrate
            else load_calibration(self.config.calibration.path, width, height)
        )
        if saved is not None:
            if saved.mirror != self.config.camera.mirror:
                log.info(
                    "using the mirror setting found during calibration (mirror=%s)", saved.mirror
                )
                self.config.camera.mirror = saved.mirror
            self._apply_calibration(saved, reset=True)
            log.info("loaded camera calibration: %s", describe_calibration(saved))
        else:
            log.info(
                "no camera calibration yet. With the headset on, stand where the camera "
                "can see you from head to feet and reach around with both controllers: "
                "high, low, out to the sides and toward the camera, turning and crouching "
                "a little. There is no need to walk. Trackers start once the camera is "
                "solved, usually within 15-30 seconds."
            )

        self.calibrator = AutoCalibrator(
            self.config.calibration,
            width,
            height,
            self.vr_history,
            min_score=max(0.4, self.config.pose2d.min_keypoint_score),
            current=saved,
            mirror=self.config.camera.mirror,
            samples_path=Path(self.config.calibration.path).with_name("samples.npz"),
        )

    def _apply_calibration(self, calibration: CameraCalibration, *, reset: bool) -> None:
        self.calibration = calibration
        self.latency_s = calibration.latency_s
        self.pipeline.set_intrinsics(calibration.intrinsics)
        self.pipeline.set_extrinsics(calibration.rotation_cw, calibration.translation_cw)
        # The anchored lifter pins the head with the same device offsets the
        # calibration solved for.
        if hasattr(self.pipeline.lifter, "offsets"):
            self.pipeline.lifter.offsets = dict(calibration.offsets)
        if reset:
            self.pipeline.reset()

    def _handle(self, update: CalibrationUpdate | None) -> None:
        if update is None:
            return
        if update.kind in ("calibrated", "refined"):
            calibration = update.calibration
            first = update.kind == "calibrated"
            self._apply_calibration(calibration, reset=first)
            save_calibration(calibration, self.config.calibration.path)
            if first:
                log.info("camera calibrated: %s", describe_calibration(calibration))
                log.info("sending trackers to VRChat")
            else:
                log.info("camera calibration refined: %s", describe_calibration(calibration))
        elif update.kind == "lost":
            self.calibration = None
            self.pipeline.clear_extrinsics()
            self.pipeline.reset()
            self._output.clear()
        elif update.kind == "mirrored":
            self.config.camera.mirror = self.calibrator.mirror
            self.estimator.reset()
            self.pipeline.reset()

    @property
    def sending(self) -> bool:
        """Trackers go to VRChat only once they are in the SteamVR room."""
        return self.vr_history is None or self.calibration is not None

    # -- main loop ----------------------------------------------------------

    def _vr_at(self, captured_at: float) -> VRState | None:
        if self.vr_history is None:
            return None
        return self.vr_history.at(captured_at - self.latency_s)

    def _track_height(self, vr: VRState | None, now: float) -> None:
        if vr is None or vr.head is None or not vr.head.valid:
            return
        self._head_heights.append(float(vr.head.position[1]))
        if len(self._head_heights) < _HEIGHT_MIN_SAMPLES:
            return
        if now - self._last_height_update < _HEIGHT_UPDATE_INTERVAL_S:
            return
        self._last_height_update = now
        height = standing_height_from_head_heights(self._head_heights)
        if height is None or not 1.2 <= height <= 2.3:
            return
        if abs(height - self.pipeline.context.height_m) < _HEIGHT_CHANGE_M:
            return
        self.pipeline.set_body(height, bone_lengths_from_height(height))
        save_body(BodyCalibration.from_height(height), self.config.body.path)
        log.info("standing height %.2f m, measured from the headset", height)

    def step(self) -> tuple[FrameResult, np.ndarray] | None:
        """Process one camera frame. Returns None if no frame arrived."""
        frame_data = self.camera.read()
        if frame_data is None:
            return None
        captured_at, frame = frame_data

        keypoints = self.estimator(frame, timestamp=captured_at)
        if self.calibrator is not None:
            self.calibrator.observe(keypoints)
            self._handle(self.calibrator.update())

        vr = self._vr_at(captured_at)
        result = self.pipeline.process(keypoints, vr, timestamp=captured_at)
        result.timings.update(self.estimator.last_timings)
        self._track_height(vr, captured_at)
        self._last_targets = result.targets
        self._last_head = vr.head if vr is not None else None

        if self.sending:
            self._output.publish(result.targets, vr.head if vr is not None else None, captured_at)

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
                intrinsics=self.pipeline.context.intrinsics,
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
        # The camera is usually calibrated part-way through, so take the
        # calibration as it ended up rather than as it started.
        meta = self.recorder.meta
        meta.intrinsics = self.pipeline.context.intrinsics
        meta.rotation_cw = self.pipeline.context.rotation_cw
        meta.translation_cw = self.pipeline.context.translation_cw
        meta.height_m = self.pipeline.context.height_m
        meta.offsets = getattr(self.pipeline.lifter, "offsets", None) or {}
        written = self.recorder.build().save(path)
        log.info("wrote %d frames to %s", len(self.recorder), written)
        return written

    def run(self) -> None:
        self._running = True
        self.camera.open()
        if self.vr_history is not None:
            self.vr_history.start()
        self.load_calibration()
        self._output.start()
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
                        intrinsics=self.pipeline.context.intrinsics,
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
        self._output.stop()
        if self.calibrator is not None:
            self.calibrator.close()
        if self.vr_history is not None:
            self.vr_history.stop()
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
        if self.calibrator is None:
            lines.append("no SteamVR: camera-relative output")
            return lines

        lines.extend(self.calibrator.status_lines())
        if self.calibration is None:
            lines.append("not sending trackers yet")
            return lines

        diagnostics = self.pipeline.diagnostics()
        if "anchored" in diagnostics:
            parts = ["anchored" if diagnostics.get("anchored") else "ANCHOR LOST"]
            head = diagnostics.get("head_residual_m")
            wrist = diagnostics.get("wrist_residual_m")
            if head is not None:
                parts.append(f"head {head * 100:.0f}cm")
            if wrist is not None:
                parts.append(f"wrist {wrist * 100:.0f}cm")
            lines.append(" ".join(parts))
        lines.extend(describe_placement(self._last_targets, self._last_head))
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
        status = (
            "; ".join(self.calibrator.status_lines()) if self.calibrator is not None else "no VR"
        )
        log.info(
            "%.0f fps, latency %.0f ms, tracked %.0f%%, dropped camera frames %d, %s",
            self._fps(),
            np.median(self._latencies) if self._latencies else 0.0,
            stats["track_rate"] * 100,
            self.camera.dropped_frames,
            status,
        )
        if self.sending:
            placement = describe_placement(self._last_targets, self._last_head)
            if placement:
                log.info("trackers: %s", "; ".join(placement))


def _describe_vmt_output(config: OSCConfig, sender: VMTSender) -> None:
    devices = ", ".join(
        f"{role.value}={device_name(index)}" for role, index in sender.slots.items()
    )
    log.info(
        "SteamVR tracker output through Virtual Motion Tracker: %s:%d (%s)",
        config.host,
        config.vmt_port,
        devices or "no tracker roles configured",
    )
    driver = find_vmt_driver()
    if driver is None:
        log.warning(
            "Virtual Motion Tracker does not appear to be installed in SteamVR, so no "
            "trackers will appear. Install it from "
            "https://github.com/gpsnmeajp/VirtualMotionTracker/releases (run vmt_manager "
            'and press Install), restart SteamVR, or set target = "vrchat" under [osc].'
        )
    else:
        log.info("found the VMT driver at %s", driver)
    log.info(
        "The trackers show up in SteamVR once calibration finishes; Synesis sets VMT's "
        "room matrix itself, so there is no need to in vmt_manager. In SteamVR, give "
        "them roles under Settings > Controllers > Manage Trackers (%s). In VRChat, "
        "turn OSC trackers off and calibrate full body as with Vive trackers.",
        devices,
    )


def run(
    config: Config,
    *,
    use_vr: bool = True,
    show_overlay: bool | None = None,
    record_to: Path | str | None = None,
    record_label: str = "",
    recalibrate: bool = False,
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
            raise RuntimeError(
                "Cannot start Synesis with the VRChat profile because SteamVR is not "
                "available. Start SteamVR, wait until the headset is connected, then "
                "launch Synesis again."
            ) from exc

    runtime = Runtime(
        config, vr_source=vr_source, show_overlay=show_overlay, recalibrate=recalibrate
    )
    if isinstance(runtime.sender, VMTSender):
        _describe_vmt_output(config.osc, runtime.sender)
    else:
        slots = ", ".join(
            f"{role.value}=tracker {slot}" for role, slot in runtime.sender.slots.items()
        )
        log.info(
            "OSC output: %s:%d (%s)",
            config.osc.host,
            config.osc.port,
            slots or "no tracker roles configured",
        )
        log.info(
            "VRChat must have OSC enabled and its incoming OSC Debug view should show "
            "tracker position/rotation messages."
        )
    if record_to is not None:
        runtime.start_recording(label=record_label or Path(record_to).stem)
    try:
        runtime.run()
    finally:
        if record_to is not None:
            runtime.save_recording(record_to)
        if vr_source is not None:
            vr_source.close()
