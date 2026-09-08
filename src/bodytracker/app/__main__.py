"""Command line entry point: ``python -m bodytracker.app``."""

from __future__ import annotations

import argparse
import logging
import sys

from .. import __version__
from ..config import load as load_config


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bodytracker",
        description="HMD-anchored single-webcam full-body tracking for VRChat",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", help="path to a config TOML (default: configs/default.toml)")
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start tracking and stream to VRChat")
    run.add_argument("--no-vr", action="store_true", help="run without the HMD anchor")
    run.add_argument("--no-overlay", action="store_true", help="skip the debug window")
    run.add_argument("--record", metavar="PATH", help="also save the session for offline analysis")
    run.add_argument("--label", default="", help="name for the recorded session")

    sub.add_parser("calibrate-intrinsics", help="solve the camera's focal length and distortion")
    sub.add_parser("calibrate-extrinsics", help="solve where the camera is, using the headset")

    body = sub.add_parser("calibrate-body", help="measure your proportions")
    body.add_argument("--seconds", type=float, default=20.0)

    smoke = sub.add_parser("osc-smoke", help="stream synthetic trackers to check the OSC wire")
    smoke.add_argument("--pattern", default="bob")
    smoke.add_argument("--seconds", type=float, default=30.0)
    smoke.add_argument("--dry-run", action="store_true")

    sub.add_parser("cameras", help="list attached cameras")

    evaluate = sub.add_parser("eval", help="score a recorded session")
    evaluate.add_argument("session")
    evaluate.add_argument(
        "--held-out",
        action="store_true",
        help="also hide each SteamVR device in turn and measure the prediction error",
    )

    compare = sub.add_parser("compare", help="diff two recorded sessions")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--held-out", action="store_true")

    lifters = sub.add_parser(
        "compare-lifters",
        help="replay one session through two lifters, which controls for everything else",
    )
    lifters.add_argument("session")
    lifters.add_argument("--baseline", default="geometric")
    lifters.add_argument("--candidate", default="anchored")

    profile = sub.add_parser(
        "profile-camera",
        help="measure this camera's noise and drift, to aim the training augmentation at it",
    )
    profile.add_argument("--frames", type=int, default=60)
    profile.add_argument("--name", default="", help="label for the profile")
    profile.add_argument(
        "--out", default="configs/camera_profile.json", help="where to write the profile"
    )

    degrade = sub.add_parser(
        "degrade-preview",
        help="write degraded copies of an image so the augmentation can be eyeballed",
    )
    degrade.add_argument("image")
    degrade.add_argument("--out", default="out/degraded")
    degrade.add_argument("--count", type=int, default=6)
    degrade.add_argument("--profile", default="configs/camera_profile.json")

    return parser


def _lifter_factory(name: str, config):
    from ..lift.anchored import AnchoredLifter
    from ..lift.geometric import GeometricLifter

    min_score = config.pose2d.min_keypoint_score
    if name == "geometric":
        return lambda: GeometricLifter(min_score=min_score)
    if name == "anchored":
        return lambda: AnchoredLifter(
            min_score=min_score, max_head_residual_m=config.lift.max_head_residual_m
        )
    if name == "neural":
        from ..lift.neural import NeuralLifter

        providers = (
            ["CPUExecutionProvider"]
            if config.lift.device == "cpu"
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        return lambda: NeuralLifter(
            config.lift.model_path,
            window=config.lift.window,
            providers=providers,
        )
    raise SystemExit(f"unknown lifter '{name}' (expected 'geometric', 'anchored', or 'neural')")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    config = load_config(args.config)

    if args.command == "run":
        from .runtime import run

        run(
            config,
            use_vr=not args.no_vr,
            show_overlay=False if args.no_overlay else None,
            record_to=args.record,
            record_label=args.label,
        )
        return 0

    if args.command == "calibrate-intrinsics":
        from .calibrate import calibrate_intrinsics_session

        return calibrate_intrinsics_session(config)

    if args.command == "calibrate-extrinsics":
        from .calibrate import calibrate_extrinsics_session

        return calibrate_extrinsics_session(config)

    if args.command == "calibrate-body":
        from .calibrate import calibrate_body_session

        return calibrate_body_session(config, duration_s=args.seconds)

    if args.command == "osc-smoke":
        from ..tools.osc_smoke import main as smoke_main

        argv = ["--pattern", args.pattern, "--seconds", str(args.seconds)]
        if args.dry_run:
            argv.append("--dry-run")
        return smoke_main(argv)

    if args.command == "cameras":
        from ..capture.webcam import list_cameras

        found = list_cameras()
        if not found:
            print("no cameras found")
            return 1
        for index, width, height in found:
            print(f"  [{index}] {width}x{height}")
        return 0

    if args.command == "eval":
        from ..eval import Session, compute, held_out_errors

        session = Session.load(args.session)
        print(compute(session).report())
        if args.held_out:
            if not session.meta.calibrated:
                print("\nheld-out error needs camera extrinsics; this session has none")
                return 1
            print("\nheld-out anchor error (device hidden from the lifter, then predicted):")
            for error in held_out_errors(session, config).values():
                print(f"  {error}")
        return 0

    if args.command == "compare":
        from ..eval import Session, compare_report, compute, held_out_errors

        baseline = Session.load(args.baseline)
        candidate = Session.load(args.candidate)
        print(
            compare_report(
                compute(baseline),
                compute(candidate),
                baseline_held_out=held_out_errors(baseline, config) if args.held_out else None,
                candidate_held_out=held_out_errors(candidate, config) if args.held_out else None,
            )
        )
        return 0

    if args.command == "compare-lifters":
        from ..eval import Session, compare_report, compute, held_out_errors, replay

        session = Session.load(args.session)
        results = []
        for name in (args.baseline, args.candidate):
            factory = _lifter_factory(name, config)
            replayed = replay(session, config, lifter=factory(), label=name)
            results.append(
                (
                    compute(replayed),
                    held_out_errors(session, config, lifter_factory=factory)
                    if session.meta.calibrated
                    else None,
                )
            )
        print(
            compare_report(
                results[0][0],
                results[1][0],
                baseline_held_out=results[0][1],
                candidate_held_out=results[1][1],
            )
        )
        return 0

    if args.command == "profile-camera":
        from ..tools.profile_camera import format_profile, profile_camera

        print("Point the camera at a still scene with both a shadow and a bright")
        print("area, stay out of frame, and do not touch anything.")
        measured = profile_camera(
            config.camera, frames=args.frames, name=args.name, output=args.out
        )
        print()
        print(format_profile(measured))
        print(f"\nsaved to {args.out}")
        return 0

    if args.command == "degrade-preview":
        from ..degrade.profile import CameraProfile
        from ..tools.profile_camera import preview

        loaded = CameraProfile.load(args.profile)
        if loaded is None:
            print(f"no profile at {args.profile}; using defaults (run profile-camera first)")
            loaded = CameraProfile()
        for path in preview(loaded, args.image, args.out, count=args.count):
            print(f"  {path}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
