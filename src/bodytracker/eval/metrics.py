"""Quality metrics for a recorded session.

Average joint error is a bad summary of how a tracker feels. A tracker can have
excellent mean error and still be unusable because it shivers when you stand
still, because your planted foot slides across the floor as you turn, or
because it is a fifth of a second behind you. Those are separate failures with
separate fixes, so they get separate numbers.

Everything here works on positions alone and needs no ground truth, so it can
be run on any recorded session. The one metric that does need truth, held-out
anchor error, lives in `replay.py` because it has to re-run the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..skeleton import TrackerRole
from .session import Session

# A device moving slower than this counts as the user standing still. Chosen
# above real headset noise but well below deliberate motion.
STILL_SPEED_MPS = 0.05

# A foot lower than this is taken to be planted, for the purpose of measuring
# whether it then slides.
CONTACT_HEIGHT_M = 0.12

FOOT_ROLES = (TrackerRole.LEFT_FOOT, TrackerRole.RIGHT_FOOT)


def _speeds(track: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Per-frame speed in m/s, aligned to the later frame of each pair."""
    dt = np.diff(timestamps)
    dt[dt <= 0] = np.nan
    return np.linalg.norm(np.diff(track, axis=0), axis=1) / dt


def still_mask(session: Session) -> np.ndarray:
    """(n,) mask of frames where the user was holding position.

    Derived from the headset rather than from the tracker output, so that a
    tracker which is drifting cannot talk itself out of being measured.
    """
    head, valid = session.device_track("head")
    mask = np.zeros(len(session), dtype=bool)
    if not np.any(valid):
        return mask

    speeds = _speeds(head, session.timestamps)
    mask[1:] = (speeds < STILL_SPEED_MPS) & valid[1:] & valid[:-1]
    return mask


@dataclass(slots=True)
class JitterMetric:
    """Involuntary movement while the user is holding still, in millimetres."""

    frames: int
    rms_speed_mm_s: float
    p95_step_mm: float


@dataclass(slots=True)
class FootMetric:
    contact_fraction: float
    slide_mm_per_s: float
    slide_frames: int
    penetration_fraction: float
    mean_penetration_mm: float
    max_penetration_mm: float


@dataclass(slots=True)
class LatencyMetric:
    median_ms: float
    p95_ms: float
    stages_median_ms: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class SessionMetrics:
    label: str
    frames: int
    duration_s: float
    fps: float
    track_rate: float
    stale_rate: float
    jitter: dict[TrackerRole, JitterMetric] = field(default_factory=dict)
    feet: dict[TrackerRole, FootMetric] = field(default_factory=dict)
    latency: LatencyMetric | None = None

    def report(self) -> str:
        lines = [
            f"{self.label or 'session'}: {self.frames} frames, "
            f"{self.duration_s:.1f} s, {self.fps:.0f} fps",
            f"  tracked {self.track_rate * 100:.1f}%  held-over {self.stale_rate * 100:.1f}%",
        ]
        if self.latency is not None:
            lines.append(
                f"  latency  median {self.latency.median_ms:.0f} ms  "
                f"p95 {self.latency.p95_ms:.0f} ms"
            )
            for name, value in sorted(self.latency.stages_median_ms.items()):
                lines.append(f"    {name:<14s} {value:6.1f} ms")
        for role, jitter in self.jitter.items():
            lines.append(
                f"  jitter {role.value:<11s} {jitter.rms_speed_mm_s:6.1f} mm/s rms, "
                f"p95 step {jitter.p95_step_mm:.1f} mm  ({jitter.frames} still frames)"
            )
        for role, foot in self.feet.items():
            skate = (
                f"skate {foot.slide_mm_per_s:6.1f} mm/s ({foot.slide_frames} planted frames)"
                if foot.slide_frames
                else "skate      n/a (never planted while still)"
            )
            lines.append(
                f"  {role.value:<16s} {skate}, "
                f"below floor {foot.penetration_fraction * 100:4.1f}% "
                f"(mean {foot.mean_penetration_mm:.0f} mm, max {foot.max_penetration_mm:.0f} mm)"
            )
        return "\n".join(lines)


def jitter_for(session: Session, role: TrackerRole, still: np.ndarray) -> JitterMetric | None:
    track, valid = session.target_track(role)
    speeds = _speeds(track, session.timestamps)
    steps = np.linalg.norm(np.diff(track, axis=0), axis=1)

    usable = still[1:] & valid[1:] & valid[:-1] & np.isfinite(speeds)
    if usable.sum() < 2:
        return None
    return JitterMetric(
        frames=int(usable.sum()),
        rms_speed_mm_s=float(np.sqrt(np.mean(speeds[usable] ** 2)) * 1000.0),
        p95_step_mm=float(np.percentile(steps[usable], 95) * 1000.0),
    )


def foot_for(
    session: Session, role: TrackerRole, still: np.ndarray | None = None
) -> FootMetric | None:
    """Foot skate and floor penetration for one foot.

    Skate is horizontal movement of a foot that is on the ground at a moment
    when the headset says the user is not going anywhere. Both conditions are
    needed. Height alone is not enough, because a foot sliding forward during
    an actual step is a person walking, not a tracking error, and a metric that
    cannot tell those apart just reports walking speed.

    It is worth measuring separately from plain positional error because it is
    what people actually notice: a constant offset is invisible once the avatar
    is calibrated, whereas a foot skating on the floor while you stand still
    reads as broken immediately.
    """
    track, valid = session.target_track(role)
    if not np.any(valid):
        return None
    if still is None:
        still = still_mask(session)

    heights = track[:, 1]
    contact = valid & (heights < CONTACT_HEIGHT_M)

    dt = np.diff(session.timestamps)
    dt[dt <= 0] = np.nan
    horizontal = np.linalg.norm(np.diff(track[:, [0, 2]], axis=0), axis=1)
    speed = horizontal / dt

    planted = contact[1:] & contact[:-1] & still[1:] & np.isfinite(speed)
    slide = float(np.mean(speed[planted]) * 1000.0) if np.any(planted) else 0.0

    below = valid & (heights < 0.0)
    depths = -heights[below]
    return FootMetric(
        contact_fraction=float(contact.sum() / max(valid.sum(), 1)),
        slide_mm_per_s=slide,
        slide_frames=int(planted.sum()),
        penetration_fraction=float(below.sum() / max(valid.sum(), 1)),
        mean_penetration_mm=float(np.mean(depths) * 1000.0) if depths.size else 0.0,
        max_penetration_mm=float(np.max(depths) * 1000.0) if depths.size else 0.0,
    )


def latency_for(session: Session) -> LatencyMetric | None:
    finite = session.latency_ms[np.isfinite(session.latency_ms)]
    stages = {
        name: float(np.nanmedian(values))
        for name, values in session.stage_ms.items()
        if np.any(np.isfinite(values))
    }
    if finite.size == 0:
        if not stages:
            return None
        return LatencyMetric(median_ms=float("nan"), p95_ms=float("nan"), stages_median_ms=stages)
    return LatencyMetric(
        median_ms=float(np.median(finite)),
        p95_ms=float(np.percentile(finite, 95)),
        stages_median_ms=stages,
    )


def compute(session: Session) -> SessionMetrics:
    """Every ground-truth-free metric for a session."""
    still = still_mask(session)
    roles = session.roles

    jitter = {}
    for role in roles:
        measured = jitter_for(session, role, still)
        if measured is not None:
            jitter[role] = measured

    feet = {}
    for role in roles:
        if role not in FOOT_ROLES:
            continue
        measured = foot_for(session, role, still)
        if measured is not None:
            feet[role] = measured

    valid = session.target_valid
    return SessionMetrics(
        label=session.meta.label,
        frames=len(session),
        duration_s=session.duration_s,
        fps=session.fps,
        track_rate=float(session.tracked().mean()) if len(session) else 0.0,
        stale_rate=float(session.target_stale[valid].mean()) if np.any(valid) else 0.0,
        jitter=jitter,
        feet=feet,
        latency=latency_for(session),
    )
