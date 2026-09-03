"""Build training shards from a synthetic scene or a BEDLAM processed npz.

The synthetic path exists so the rest of the training stack can be written
and tested before anyone downloads a terabyte. It uses the same scene fixture
the eval harness already trusts, so a lifter that overfits it will at least
overfit something we understand.

    python -m training.bedlam.prep --synthetic --out data/shards
    python -m training.bedlam.prep --npz path/to/scene.npz --out data/shards
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .noise import KeypointNoise
from .samples import (
    from_scene,
    load_bedlam_npz,
    resample,
    save_shard,
    windows,
)

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="build shards from the eval scene")
    source.add_argument("--npz", help="path to a BEDLAM processed npz (or a directory of them)")
    parser.add_argument("--out", default="data/shards", help="directory to write shards into")
    parser.add_argument("--window", type=float, default=0.9, help="window length in seconds")
    parser.add_argument("--stride", type=float, default=0.45, help="window stride in seconds")
    parser.add_argument("--shard-size", type=int, default=64, help="windows per npz shard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-noise", action="store_true", help="leave projections clean")
    parser.add_argument("--frames", type=int, default=240, help="synthetic scene length")
    parser.add_argument("--fps", type=float, default=30.0, help="synthetic scene rate")
    return parser


def _iter_npz(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.npz"))


def _flush(out_dir: Path, batch: list, shard_index: int, notes: str) -> int:
    if not batch:
        return shard_index
    path = out_dir / f"shard_{shard_index:04d}.npz"
    save_shard(path, batch, notes=notes)
    log.info("wrote %d windows to %s", len(batch), path)
    return shard_index + 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    noise = None if args.no_noise else KeypointNoise()
    rng = __import__("numpy").random.default_rng(args.seed)
    batch: list = []
    shard_index = 0
    total = 0

    if args.synthetic:
        from bodytracker.geom import intrinsics_from_fov
        from bodytracker.tools.scene import build_scene

        scene = build_scene(
            intrinsics=intrinsics_from_fov(1280, 720, 70.0),
            frames=args.frames,
            fps=args.fps,
            seed=args.seed,
        )
        sequences = [from_scene(scene, source=f"synthetic-seed{args.seed}")]
    else:
        paths = _iter_npz(Path(args.npz))
        if not paths:
            log.error("no npz files in %s", args.npz)
            return 1
        sequences = []
        for path in paths:
            try:
                sequences.append(load_bedlam_npz(path))
            except (KeyError, ValueError) as exc:
                log.error("%s: %s", path, exc)
                return 1

    for sequence in sequences:
        resampled = resample(sequence)
        clips = windows(
            resampled,
            window_s=args.window,
            stride_s=args.stride,
            noise=noise,
            rng=rng,
        )
        log.info("%s: %d frames -> %d windows", sequence.source, len(resampled), len(clips))
        for clip in clips:
            batch.append(clip)
            if len(batch) >= args.shard_size:
                shard_index = _flush(out_dir, batch, shard_index, notes=sequence.source)
                total += len(batch)
                batch = []

    shard_index = _flush(out_dir, batch, shard_index, notes="mixed")
    total += len(batch)
    log.info("done: %d windows in %d shards under %s", total, shard_index, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
