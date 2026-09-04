"""Small temporal neural lifter and shard-backed training dataset.

The model deliberately consumes pose data rather than images. RTMPose remains
the image front-end; this model learns the ambiguous depth and occlusion cases
from a short history of noisy Halpe26 points and the tracked VR devices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from bodytracker.skeleton import BONES, NUM_KEYPOINTS


@dataclass(frozen=True, slots=True)
class Batch:
    """A torch batch with the same names and shapes as a training shard."""

    xy: Tensor
    scores: Tensor
    xyz_play: Tensor
    device_pos: Tensor
    device_rot: Tensor
    device_valid: Tensor
    intrinsics: Tensor

    def to(self, device: torch.device | str) -> Batch:
        return Batch(**{name: getattr(self, name).to(device) for name in self.__dataclass_fields__})


class ShardDataset(Dataset[dict[str, Tensor]]):
    """Eagerly load a small set of compressed shards into training examples.

    Shards are deliberately small (64 windows by default), so loading their
    arrays once makes batches deterministic and avoids holding open many npz
    handles while a Windows training run is checkpointing.
    """

    REQUIRED = (
        "xy",
        "scores",
        "xyz_play",
        "device_pos",
        "device_rot",
        "device_valid",
        "intrinsics",
    )

    def __init__(self, paths: list[str | Path]) -> None:
        if not paths:
            raise ValueError("at least one shard is required")
        arrays: dict[str, list[np.ndarray]] = {name: [] for name in self.REQUIRED}
        for path in (Path(p) for p in paths):
            with np.load(path, allow_pickle=False) as loaded:
                missing = [name for name in self.REQUIRED if name not in loaded]
                if missing:
                    raise ValueError(f"{path} is not a Synesis training shard; missing {missing}")
                for name in self.REQUIRED:
                    arrays[name].append(np.asarray(loaded[name], dtype=np.float32))
        self.arrays = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
        count = self.arrays["xy"].shape[0]
        if count == 0 or any(array.shape[0] != count for array in self.arrays.values()):
            raise ValueError("shards have inconsistent or empty sample counts")

    def __len__(self) -> int:
        return int(self.arrays["xy"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {name: torch.from_numpy(values[index]) for name, values in self.arrays.items()}


def collate(items: list[dict[str, Tensor]]) -> Batch:
    values = {name: torch.stack([item[name] for item in items]) for name in ShardDataset.REQUIRED}
    return Batch(**values)


def _origin(batch: Batch) -> Tensor:
    """Centre a sequence on the current HMD, with a robust zero fallback."""
    centre = batch.xy.shape[1] // 2
    head = batch.device_pos[:, centre, 0]
    valid = batch.device_valid[:, centre, 0].bool().unsqueeze(-1)
    return torch.where(valid, torch.nan_to_num(head), torch.zeros_like(head))


def features(batch: Batch) -> tuple[Tensor, Tensor]:
    """Encode a temporal window and return ``(features, HMD-origin)``.

    Pixel coordinates become normalized camera rays. Device locations and the
    target are centred on the HMD so the model learns body geometry rather than
    memorising where a SteamVR room origin happened to be.
    """
    intrinsics = batch.intrinsics
    if intrinsics.ndim == 3:
        intrinsics = intrinsics[:, None]
    focal = intrinsics[..., (0, 1), (0, 1)].unsqueeze(-2)
    principal = intrinsics[..., :2, 2].unsqueeze(-2)
    rays = (batch.xy - principal) / focal.clamp_min(1e-6)
    origin = _origin(batch)
    devices = torch.nan_to_num(batch.device_pos - origin[:, None, None, :])
    rotations = torch.nan_to_num(batch.device_rot).flatten(start_dim=2)
    valid = batch.device_valid.float()
    encoded = torch.cat(
        (
            rays.flatten(start_dim=2),
            batch.scores.unsqueeze(-1).flatten(start_dim=2),
            devices.flatten(start_dim=2),
            rotations,
            valid,
        ),
        dim=-1,
    )
    return encoded, origin


class TemporalLifter(nn.Module):
    """A compact bidirectional GRU that predicts the centre frame in metres."""

    def __init__(self, hidden_size: int = 192, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        # 26*(x, y, score) + 3 devices*(position + 3x3 rotation + valid)
        input_size = NUM_KEYPOINTS * 3 + 3 * (3 + 9 + 1)
        self.encoder = nn.GRU(
            input_size,
            hidden_size,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, NUM_KEYPOINTS * 3),
        )

    def forward(self, batch: Batch) -> Tensor:
        encoded, origin = features(batch)
        sequence, _ = self.encoder(encoded)
        centre = sequence[:, sequence.shape[1] // 2]
        return self.head(centre).reshape(-1, NUM_KEYPOINTS, 3) + origin[:, None, :]


def centre_target(batch: Batch) -> Tensor:
    return batch.xyz_play[:, batch.xyz_play.shape[1] // 2]


def loss(
    prediction: Tensor,
    target: Tensor,
    *,
    bone_weight: float = 0.1,
    floor_weight: float = 0.05,
) -> Tensor:
    """Metric joint loss plus small anatomy and floor regularizers."""
    joint_weight = torch.ones(NUM_KEYPOINTS, device=prediction.device)
    # Legs and feet are the visible failures that matter most in VR.
    joint_weight[11:17] = 1.5
    joint_weight[20:26] = 2.0
    position_loss = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    position = (position_loss * joint_weight[None, :, None]).mean()
    pairs = torch.as_tensor(BONES, device=prediction.device)
    pred_length = torch.linalg.vector_norm(
        prediction[:, pairs[:, 0]] - prediction[:, pairs[:, 1]], dim=-1
    )
    true_length = torch.linalg.vector_norm(
        target[:, pairs[:, 0]] - target[:, pairs[:, 1]], dim=-1
    )
    bone = torch.nn.functional.smooth_l1_loss(pred_length, true_length)
    floor = torch.relu(-prediction[..., 1]).mean()
    return position + bone_weight * bone + floor_weight * floor
