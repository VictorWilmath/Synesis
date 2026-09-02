"""Phase 0: prove the OSC wire before writing any computer vision.

Streams canned, anatomically plausible tracker motion to VRChat. If the avatar
follows it, the transport, slot assignment, handedness flip and Euler
convention are all correct.

    bodytracker-osc-smoke --pattern bob

In VRChat: Options -> OSC -> Enable, then run full-body calibration. Stand in
the calibration pose and the synthetic trackers will be picked up as if they
were hardware.

Deliberately depends on nothing but numpy and python-osc, so a failure here
implicates the wire and not the rest of the pipeline.
"""

from __future__ import annotations

import argparse
import sys
import time

from ..osc import VRChatOSCSender
from ..skeleton import MAX_TRACKERS, TrackerRole
from ..types import DevicePose
from .synthetic import PATTERNS, SyntheticBody

ALL_ROLES = [
    TrackerRole.HIP,
    TrackerRole.CHEST,
    TrackerRole.LEFT_FOOT,
    TrackerRole.RIGHT_FOOT,
    TrackerRole.LEFT_KNEE,
    TrackerRole.RIGHT_KNEE,
    TrackerRole.LEFT_ELBOW,
    TrackerRole.RIGHT_ELBOW,
]

MINIMAL_ROLES = [TrackerRole.HIP, TrackerRole.LEFT_FOOT, TrackerRole.RIGHT_FOOT]


class _PrintClient:
    """Stands in for the UDP client under --dry-run."""

    def send_message(self, address: str, value: list[float]) -> None:
        formatted = ", ".join(f"{v:8.3f}" for v in value)
        print(f"{address:42s} [{formatted}]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bodytracker-osc-smoke",
        description="Stream synthetic tracker data to VRChat to validate the OSC path.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="VRChat host (default: %(default)s)")
    parser.add_argument("--port", type=int, default=9000, help="OSC port (default: %(default)s)")
    parser.add_argument(
        "--pattern",
        default="bob",
        choices=PATTERNS,
        help="motion to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--roles",
        default="minimal",
        choices=("minimal", "all"),
        help=(
            "'minimal' sends hip and both feet, which VRChat's IK generally "
            "handles better; 'all' sends all eight (default: %(default)s)"
        ),
    )
    parser.add_argument("--height", type=float, default=1.75, help="subject height in metres")
    parser.add_argument("--period", type=float, default=2.0, help="seconds per motion cycle")
    parser.add_argument("--amplitude", type=float, default=1.0, help="motion scale")
    parser.add_argument("--rate", type=float, default=60.0, help="send rate in Hz")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds, 0 runs until Ctrl-C")
    parser.add_argument(
        "--send-head",
        action="store_true",
        help="also send the optional /tracking/trackers/head endpoints",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the messages instead of sending them",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    roles = ALL_ROLES if args.roles == "all" else MINIMAL_ROLES
    body = SyntheticBody(height_m=args.height, period_s=args.period, amplitude=args.amplitude)
    sender = VRChatOSCSender(
        roles,
        host=args.host,
        port=args.port,
        send_head=args.send_head,
        rate_hz=args.rate,
        client=_PrintClient() if args.dry_run else None,
    )

    slot_summary = ", ".join(f"{slot}={role.value}" for role, slot in sender.slots.items())
    target = "dry run" if args.dry_run else f"{args.host}:{args.port}"
    print(f"Sending '{args.pattern}' to {target} at {args.rate:g} Hz")
    print(f"Slots: {slot_summary}")
    if len(roles) < MAX_TRACKERS:
        print(f"({len(roles)} of {MAX_TRACKERS} slots in use)")
    print("Enable OSC in VRChat, then run full-body calibration. Ctrl-C to stop.\n")

    start = time.monotonic()
    interval = 1.0 / args.rate if args.rate > 0 else 0.0
    try:
        while True:
            now = time.monotonic()
            elapsed = now - start
            if args.duration > 0 and elapsed >= args.duration:
                break

            targets = body.pose(elapsed, args.pattern)
            head = None
            if args.send_head:
                head = DevicePose(
                    position=body.head(elapsed, args.pattern),
                    rotation=targets[0].rotation,
                    valid=True,
                )
            sender.send(targets, head=head, now=now, force=True)

            if interval:
                time.sleep(max(0.0, interval - (time.monotonic() - now)))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        sender.close()

    print(f"{sender.frames_sent} frames, {sender.messages_sent} OSC messages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
