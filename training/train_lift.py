"""Train Synesis's temporal 2D-to-3D lifter from prepared training shards.

Example smoke run:
    python -m training.bedlam.prep --synthetic --out data/synthetic
    python -m training.train_lift --train data/synthetic --epochs 80 --overfit
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .lift import ShardDataset, TemporalLifter, centre_target, collate, loss


def _paths(value: str) -> list[Path]:
    path = Path(value)
    paths = [path] if path.is_file() else sorted(path.glob("*.npz"))
    if not paths:
        raise ValueError(f"no shards found under {path}")
    return paths


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _run_epoch(model, loader, optimizer, device, scaler) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            value = loss(model(batch), centre_target(batch))
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(value).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        total += float(value.detach()) * batch.xy.shape[0]
        count += batch.xy.shape[0]
    return total / max(count, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="shard directory or one npz shard")
    parser.add_argument("--val", help="optional held-out shard directory")
    parser.add_argument("--out", default="checkpoints/lifter", help="checkpoint directory")
    parser.add_argument("--log-dir", default="runs/lifter", help="TensorBoard log directory")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overfit", action="store_true", help="use only the first batch as a smoke test"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but PyTorch cannot see a CUDA device")
    train = ShardDataset(_paths(args.train))
    if args.overfit:
        train.arrays = {
            name: values[: min(args.batch_size, len(train))]
            for name, values in train.arrays.items()
        }
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = None
    if args.val:
        val_loader = DataLoader(
            ShardDataset(_paths(args.val)), batch_size=args.batch_size, collate_fn=collate
        )

    model = TemporalLifter(hidden_size=args.hidden_size).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    best = float("inf")
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = _run_epoch(model, train_loader, optimizer, device, scaler)
            with torch.no_grad():
                val_loss = (
                    _run_epoch(model, val_loader, None, device, scaler)
                    if val_loader
                    else train_loss
                )
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/validation", val_loss, epoch)
            print(f"epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f}")
            if val_loss < best:
                best = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "hidden_size": args.hidden_size,
                        "epoch": epoch,
                        "loss": best,
                    },
                    out / "best.pt",
                )
    finally:
        writer.close()
    (out / "metrics.json").write_text(
        json.dumps({"best_loss": best, "epochs": args.epochs}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
