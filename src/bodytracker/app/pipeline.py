"""Per-frame tracking logic, from 2D keypoints to VRChat tracker targets.

Deliberately takes keypoints rather than an image, so the whole chain can be
driven from recorded or synthetic data with no camera, no headset and no
onnxruntime. The camera and the model live in `runtime.py`.
"""

from __future__ import annotations

import time

import numpy as np

from ..config import Config
from ..filters import HoldLastValid, RotationSmoother, SkeletonFilter
from ..lift.anchored import AnchoredLifter
from ..lift.base import LiftContext
from ..lift.geometric import GeometricLifter
from ..lift.kinematics import bone_lengths_from_height
from ..skeleton import TrackerRole
from ..solve.constraints import clamp_to_floor, enforce_bone_lengths
from ..solve.targets import build_targets
from ..types import FrameResult, Keypoints2D, Skeleton3D, TrackerTarget, VRState


def _default_lifter(config: Config):
    """Anchored when a headset is expected, geometric otherwise.

    The anchored lifter falls back to the geometric path by itself when the
    headset or the calibration is missing, so choosing it costs nothing.
    """
    if config.lift.method == "geometric":
        return GeometricLifter(min_score=config.pose2d.min_keypoint_score)
    if config.lift.method == "neural":
        from ..lift.neural import NeuralLifter

        return NeuralLifter(config.lift.model_path, window=config.lift.window)
    return AnchoredLifter(min_score=config.pose2d.min_keypoint_score)


class Pipeline:
    """Turns a stream of 2D keypoints into smoothed tracker targets."""

    def __init__(
        self,
        config: Config,
        *,
        intrinsics: np.ndarray,
        lifter: object | None = None,
    ) -> None:
        self.config = config
        self.roles = config.osc.tracker_roles()
        self.lifter = lifter if lifter is not None else _default_lifter(config)

        self.context = LiftContext(
            intrinsics=intrinsics,
            bone_lengths=bone_lengths_from_height(config.body.height_m),
            height_m=config.body.height_m,
        )

        self._skeleton_filter = SkeletonFilter(
            config.filter, min_score=config.pose2d.min_keypoint_score
        )
        self._rotation_smoothers: dict[TrackerRole, RotationSmoother] = {
            role: RotationSmoother(config.filter.rotation_alpha) for role in self.roles
        }
        self._hold = HoldLastValid(timeout=config.filter.hold_last_valid_s)

        self.frames_tracked = 0
        self.frames_dropped = 0

    # -- calibration plumbing --------------------------------------------

    def set_extrinsics(self, rotation_cw: np.ndarray, translation_cw: np.ndarray) -> None:
        self.context.rotation_cw = rotation_cw
        self.context.translation_cw = translation_cw

    def set_body(self, height_m: float, bone_lengths: dict[tuple[int, int], float]) -> None:
        self.context.height_m = height_m
        self.context.bone_lengths = bone_lengths

    def reset(self) -> None:
        self.lifter.reset()
        self._skeleton_filter.reset()
        self._hold.reset()
        for smoother in self._rotation_smoothers.values():
            smoother.reset()

    # -- the frame ---------------------------------------------------------

    def process(
        self,
        keypoints: Keypoints2D | None,
        vr: VRState | None = None,
        *,
        timestamp: float | None = None,
    ) -> FrameResult:
        now = timestamp if timestamp is not None else time.monotonic()
        result = FrameResult(timestamp=now, keypoints2d=keypoints, vr=vr)

        if keypoints is None:
            self.frames_dropped += 1
            result.targets = self._held_targets(now)
            return result

        self.context.vr = vr

        start = time.perf_counter()
        skeleton = self.lifter(keypoints, self.context)
        result.timings["lift_ms"] = (time.perf_counter() - start) * 1000.0

        if skeleton is None:
            self.frames_dropped += 1
            result.targets = self._held_targets(now)
            return result

        start = time.perf_counter()
        skeleton = self._apply_constraints(skeleton)
        skeleton = self._skeleton_filter(skeleton)
        result.timings["solve_ms"] = (time.perf_counter() - start) * 1000.0
        result.skeleton3d = skeleton

        targets = build_targets(
            skeleton, self.roles, min_score=self.config.pose2d.min_keypoint_score
        )
        result.targets = self._smooth_and_hold(targets, now)
        result.diagnostics = self.diagnostics()
        self.frames_tracked += 1
        return result

    def diagnostics(self) -> dict[str, object]:
        getter = getattr(self.lifter, "diagnostics", None)
        return getter() if callable(getter) else {}

    def _apply_constraints(self, skeleton: Skeleton3D) -> Skeleton3D:
        xyz = np.asarray(skeleton.xyz, dtype=np.float64)
        if self.config.solve.enforce_bone_lengths:
            xyz = enforce_bone_lengths(xyz, self.context.bone_lengths)
        if self.config.solve.floor_clamp:
            xyz = clamp_to_floor(xyz, epsilon=self.config.solve.floor_epsilon_m)
        return Skeleton3D(
            xyz=xyz.astype(np.float32),
            scores=skeleton.scores,
            space=skeleton.space,
            timestamp=skeleton.timestamp,
        )

    def _smooth_and_hold(self, targets: list[TrackerTarget], now: float) -> list[TrackerTarget]:
        """Smooth valid targets and substitute held values for invalid ones."""
        out: list[TrackerTarget] = []
        for target in targets:
            if target.valid:
                target.rotation = self._rotation_smoothers[target.role](target.rotation, now)
                self._hold.update(
                    target.role.value, (target.position.copy(), target.rotation.copy()), now
                )
                out.append(target)
                continue

            held = self._hold.get(target.role.value, now)
            if held is None:
                # Nothing to fall back on, so pass the low-confidence estimate
                # through rather than letting the slot go silent.
                out.append(target)
                continue

            (position, rotation), stale = held
            out.append(
                TrackerTarget(
                    role=target.role,
                    position=position.copy(),
                    rotation=rotation.copy(),
                    valid=True,
                    stale=stale,
                )
            )
        return out

    def _held_targets(self, now: float) -> list[TrackerTarget]:
        """Everything we can still report when the frame produced nothing."""
        out: list[TrackerTarget] = []
        for role in self.roles:
            held = self._hold.get(role.value, now)
            if held is None:
                continue
            (position, rotation), stale = held
            out.append(
                TrackerTarget(
                    role=role,
                    position=position.copy(),
                    rotation=rotation.copy(),
                    valid=True,
                    stale=stale,
                )
            )
        return out

    def stats(self) -> dict[str, float]:
        total = self.frames_tracked + self.frames_dropped
        return {
            "tracked": self.frames_tracked,
            "dropped": self.frames_dropped,
            "track_rate": self.frames_tracked / total if total else 0.0,
        }
