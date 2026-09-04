"""Export a trained temporal lifter checkpoint to portable ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn

from .lift import Batch, TemporalLifter


class ExportableLifter(nn.Module):
    """Tensor-only wrapper: ONNX cannot accept the ``Batch`` dataclass directly."""

    def __init__(self, model: TemporalLifter) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        xy: Tensor,
        scores: Tensor,
        device_pos: Tensor,
        device_rot: Tensor,
        device_valid: Tensor,
        intrinsics: Tensor,
    ) -> Tensor:
        targets = torch.empty_like(xy[..., :3])
        return self.model(
            Batch(xy, scores, targets, device_pos, device_rot, device_valid, intrinsics)
        )


def export(checkpoint: Path, output: Path, *, window: int = 27) -> None:
    """Write an ONNX model with a dynamic batch axis and a fixed time window."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = TemporalLifter(hidden_size=int(saved["hidden_size"]))
    model.load_state_dict(saved["model"])
    wrapper = ExportableLifter(model.eval())
    inputs = (
        torch.zeros(1, window, 26, 2),
        torch.zeros(1, window, 26),
        torch.zeros(1, window, 3, 3),
        torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(1, window, 3, 1, 1),
        torch.ones(1, window, 3),
        torch.eye(3).unsqueeze(0),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    names = ("xy", "scores", "device_pos", "device_rot", "device_valid", "intrinsics")
    torch.onnx.export(
        wrapper,
        inputs,
        output,
        input_names=names,
        output_names=("xyz_play",),
        dynamic_axes={name: {0: "batch"} for name in (*names, "xyz_play")},
        opset_version=17,
        dynamo=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window", type=int, default=27)
    args = parser.parse_args(argv)
    export(args.checkpoint, args.output, window=args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
