"""Re-run a recorded session through the pipeline.

Two jobs. The obvious one is comparing lifters on identical frames, which is
how the trained model will eventually be judged against the geometry.

The subtler one is the held-out anchor test, and it is the only way to get a
true metric error without a mocap suit. Hide the headset from the lifter, let
it predict where the head is from the camera alone, then compare that against
what SteamVR reported for the very same frame. SteamVR's pose is accurate to
about a millimetre, so the difference is real error, measured on the user's own
hardware, in their own room, under their own lighting, for free.

The controllers give the same test for the wrists. Between them they bracket
the body: if the head and both wrists land in the right place, the torso
between them is unlikely to be far off.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from ..app.pipeline import Pipeline
from ..config import Config
from ..skeleton import HEAD, LEFT_WRIST, RIGHT_WRIST
from ..types import VRState
from .session import DEVICE_NAMES, Session, SessionMeta, SessionRecorder

# Which joint each device is a ground-truth measurement of.
DEVICE_JOINT = {"head": HEAD, "left_hand": LEFT_WRIST, "right_hand": RIGHT_WRIST}


def withhold_devices(vr: VRState | None, names: tuple[str, ...]) -> VRState | None:
    """A copy of `vr` with some devices hidden, as if they were not tracking."""
    if vr is None or not names:
        return vr
    hidden = VRState(
        head=vr.head,
        left_hand=vr.left_hand,
        right_hand=vr.right_hand,
        timestamp=vr.timestamp,
    )
    for name in names:
        setattr(hidden, name, None)
    return hidden


def config_for(session: Session, base: Config | None = None) -> Config:
    """A config matching how the session was recorded."""
    config = deepcopy(base) if base is not None else Config()
    config.body.height_m = session.meta.height_m
    config.osc.roles = list(session.meta.roles)
    return config


def replay(
    session: Session,
    config: Config | None = None,
    *,
    lifter=None,
    withhold: tuple[str, ...] = (),
    label: str | None = None,
) -> Session:
    """Feed a recorded session's inputs back through a fresh pipeline.

    `withhold` hides the named SteamVR devices from the lifter, which is what
    turns them into held-out ground truth.
    """
    config = config_for(session, config)
    pipeline = Pipeline(config, intrinsics=session.meta.intrinsics, lifter=lifter)
    if session.meta.calibrated:
        pipeline.set_extrinsics(session.meta.rotation_cw, session.meta.translation_cw)
    if session.meta.offsets and hasattr(pipeline.lifter, "offsets"):
        pipeline.lifter.offsets = session.meta.offsets

    meta = SessionMeta(
        intrinsics=session.meta.intrinsics,
        rotation_cw=session.meta.rotation_cw,
        translation_cw=session.meta.translation_cw,
        height_m=session.meta.height_m,
        roles=list(session.meta.roles),
        offsets=dict(session.meta.offsets),
        label=label if label is not None else f"{session.meta.label} (replay)",
        notes=f"withheld: {', '.join(withhold)}" if withhold else "",
    )
    recorder = SessionRecorder(meta)

    for index in range(len(session)):
        result = pipeline.process(
            session.keypoints_at(index),
            withhold_devices(session.vr_at(index), withhold),
            timestamp=float(session.timestamps[index]),
        )
        recorder.add(result)

    return recorder.build()


@dataclass(slots=True)
class HeldOutError:
    """How far a predicted joint sits from the device that really measured it."""

    device: str
    joint: int
    frames: int
    median_m: float
    mean_m: float
    p95_m: float

    def __str__(self) -> str:
        return (
            f"{self.device:11s} median {self.median_m * 100:5.1f} cm  "
            f"mean {self.mean_m * 100:5.1f} cm  p95 {self.p95_m * 100:5.1f} cm  "
            f"({self.frames} frames)"
        )


def held_out_errors(
    session: Session,
    config: Config | None = None,
    *,
    devices: tuple[str, ...] = DEVICE_NAMES,
    lifter_factory=None,
) -> dict[str, HeldOutError]:
    """Metric error for each device, measured by hiding it from the lifter.

    Each device is withheld on its own rather than all at once. Hiding
    everything would measure a tracker nobody runs; hiding one at a time
    measures the configuration the user actually has, minus the anchor being
    scored, which is the comparison that means something.
    """
    if not session.meta.calibrated:
        raise ValueError("held-out error needs camera extrinsics; the session has none")

    out: dict[str, HeldOutError] = {}
    for device in devices:
        joint = DEVICE_JOINT[device]
        lifter = lifter_factory() if lifter_factory is not None else None
        replayed = replay(session, config, lifter=lifter, withhold=(device,))

        truth, valid = session.device_track(device)
        predicted = replayed.skeleton_xyz[:, joint]
        usable = valid & ~np.isnan(predicted[:, 0])
        if not np.any(usable):
            continue

        distances = np.linalg.norm(predicted[usable] - truth[usable], axis=1)
        out[device] = HeldOutError(
            device=device,
            joint=joint,
            frames=int(usable.sum()),
            median_m=float(np.median(distances)),
            mean_m=float(np.mean(distances)),
            p95_m=float(np.percentile(distances, 95)),
        )
    return out
