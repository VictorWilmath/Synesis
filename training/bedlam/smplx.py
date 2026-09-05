"""Reconstruct BEDLAM's world-space joints from released SMPL-X motion files.

The gendered BEDLAM archive contains animation parameters rather than an
already-evaluated skeleton.  Keeping this forward pass here makes the costly
and licensed model dependency a training-machine concern; the shipped tracker
will only ever consume its compact ONNX lifter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SMPLX_POSE_SIZE = 165
DEFAULT_BATCH_SIZE = 32


def split_pose(pose: np.ndarray) -> dict[str, np.ndarray]:
    """Split BEDLAM's 165-axis-angle SMPL-X vector into model inputs.

    The ordering is documented by the official BEDLAM processing script:
    root, body, jaw, left eye, right eye, left hand, then right hand.
    """
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] != SMPLX_POSE_SIZE:
        raise ValueError(
            f"expected poses shaped (frames, {SMPLX_POSE_SIZE}), got {pose.shape}"
        )
    return {
        "global_orient": pose[:, :3],
        "body_pose": pose[:, 3:66],
        "jaw_pose": pose[:, 66:69],
        "leye_pose": pose[:, 69:72],
        "reye_pose": pose[:, 72:75],
        "left_hand_pose": pose[:, 75:120],
        "right_hand_pose": pose[:, 120:165],
    }


def _model_directory(path: Path) -> Path:
    """Return the parent passed to ``smplx.create`` for common archive layouts."""
    # smplx.create(parent, model_type="smplx") itself appends ``smplx``.
    # The official archive extracts as ``<root>/models/smplx/*.npz``.
    candidates = (path, path / "models", path.parent)
    for candidate in candidates:
        if (candidate / "smplx" / "SMPLX_NEUTRAL.npz").is_file():
            return candidate
    raise FileNotFoundError(
        f"could not find smplx/SMPLX_NEUTRAL.npz near {path}; pass the extracted SMPL-X root"
    )


def _text_scalar(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar, got {array.shape}")
    return str(array.reshape(-1)[0]).strip().lower()


def joints_from_motion(
    path: Path | str,
    model_root: Path | str,
    *,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_frames: int | None = None,
    model_cache: dict[str, Any] | None = None,
) -> np.ndarray:
    """Evaluate one BEDLAM ``motion_seq.npz`` into ``(frames, joints, 3)``.

    Coordinates remain in the motion file's world frame.  Applying the scene
    camera transformation is intentionally a separate step: the motion archive
    has no camera metadata, and guessing one would create invalid training
    pairs.  ``device='cuda'`` is useful on the RTX 2070; CPU is the portable
    default for tests and inspection.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    required = {"poses", "betas", "trans", "gender"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"{path} is missing required fields: {', '.join(sorted(missing))}")

    poses = np.asarray(data["poses"], dtype=np.float32)
    parts = split_pose(poses)
    frames = len(poses) if max_frames is None else min(len(poses), max_frames)
    if frames < 1:
        raise ValueError(f"{path} has no motion frames")
    translation = np.asarray(data["trans"], dtype=np.float32)
    if translation.shape != (len(poses), 3):
        raise ValueError(f"expected trans shaped {(len(poses), 3)}, got {translation.shape}")
    betas = np.asarray(data["betas"], dtype=np.float32).reshape(-1)
    if not len(betas):
        raise ValueError(f"{path} has no shape coefficients")
    gender = _text_scalar(data["gender"], "gender")
    if gender not in {"female", "male", "neutral"}:
        raise ValueError(f"unsupported SMPL-X gender {gender!r} in {path}")

    # Delayed imports keep the data-format helpers usable without torch.
    import smplx
    import torch

    model = model_cache.get(gender) if model_cache is not None else None
    if model is None:
        model = smplx.create(
            _model_directory(Path(model_root)),
            model_type="smplx",
            gender=gender,
            ext="npz",
            num_betas=len(betas),
            flat_hand_mean=True,
            use_pca=False,
        ).to(device)
        model.eval()
        if model_cache is not None:
            model_cache[gender] = model
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, frames, batch_size):
            stop = min(start + batch_size, frames)
            inputs = {
                name: torch.as_tensor(values[start:stop], device=device)
                for name, values in parts.items()
            }
            inputs["betas"] = torch.as_tensor(
                np.broadcast_to(betas, (stop - start, len(betas))).copy(), device=device
            )
            # BEDLAM's 165-value body pose has no FLAME expression channel.
            # smplx defaults this to a batch of one, which fails for B > 1.
            inputs["expression"] = torch.zeros(
                (stop - start, model.num_expression_coeffs), device=device
            )
            inputs["transl"] = torch.as_tensor(translation[start:stop], device=device)
            output.append(model(**inputs).joints.detach().cpu().numpy())
    return np.concatenate(output, axis=0)
