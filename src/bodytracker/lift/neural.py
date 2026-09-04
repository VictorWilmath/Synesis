"""ONNX runtime adapter for the trained temporal lifter."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from ..onnx import configure_cuda_dlls
from ..types import Keypoints2D, Skeleton3D, VRState
from .base import LiftContext


class NeuralLifter:
    """Buffer live observations and run a fixed-window ONNX temporal model."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        window: int = 27,
        providers: list[str] | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"neural lifter model not found: {path}")
        if window < 1 or window % 2 == 0:
            raise ValueError("neural lifter window must be a positive odd number")
        try:
            configure_cuda_dlls()
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("install bodytracker[runtime] to use the neural lifter") from exc
        selected = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=selected)
        self.window = window
        self._frames: deque[tuple[np.ndarray, ...]] = deque(maxlen=window)

    def reset(self) -> None:
        self._frames.clear()

    @staticmethod
    def _devices(vr: VRState | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.zeros((3, 3), dtype=np.float32)
        rotations = np.zeros((3, 3, 3), dtype=np.float32)
        valid = np.zeros(3, dtype=np.float32)
        if vr is None:
            return positions, rotations, valid
        for index, pose in enumerate((vr.head, vr.left_hand, vr.right_hand)):
            if pose is None or not pose.valid:
                continue
            positions[index] = np.asarray(pose.position, dtype=np.float32)
            rotations[index] = np.asarray(pose.rotation, dtype=np.float32)
            valid[index] = 1.0
        return positions, rotations, valid

    def __call__(self, keypoints: Keypoints2D, context: LiftContext) -> Skeleton3D | None:
        position, rotation, valid = self._devices(context.vr)
        self._frames.append(
            (
                np.asarray(keypoints.xy, dtype=np.float32),
                np.asarray(keypoints.scores, dtype=np.float32),
                position,
                rotation,
                valid,
            )
        )
        frames = list(self._frames)
        frames = [frames[0]] * (self.window - len(frames)) + frames
        names = ("xy", "scores", "device_pos", "device_rot", "device_valid")
        feeds = {
            name: np.stack([frame[index] for frame in frames])[None]
            for index, name in enumerate(names)
        }
        feeds["intrinsics"] = np.asarray(context.intrinsics, dtype=np.float32)[None]
        xyz = self.session.run(["xyz_play"], feeds)[0][0]
        return Skeleton3D(
            xyz=np.asarray(xyz, dtype=np.float32),
            scores=np.asarray(keypoints.scores, dtype=np.float32).copy(),
            timestamp=keypoints.timestamp,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "neural_window_frames": len(self._frames),
            "neural_providers": self.session.get_providers(),
        }
