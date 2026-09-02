"""Side-by-side comparison of two runs.

The intended use is A/B on the same recorded frames: replay one session through
two different lifters and diff the numbers. That controls for the room, the
lighting, the camera and the motion, which are otherwise large enough to swamp
the effect being measured.

Comparing two separately recorded sessions works too, and is what you do
against another tool, but the result is much noisier and should be treated as
indicative rather than conclusive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import SessionMetrics
from .replay import HeldOutError


@dataclass(slots=True)
class Row:
    name: str
    baseline: float
    candidate: float
    unit: str
    lower_is_better: bool = True

    @property
    def delta_pct(self) -> float:
        if self.baseline == 0:
            return 0.0
        return (self.candidate - self.baseline) / abs(self.baseline) * 100.0

    @property
    def improved(self) -> bool:
        if self.candidate == self.baseline:
            return False
        better = self.candidate < self.baseline
        return better if self.lower_is_better else not better


def _rows(
    baseline: SessionMetrics,
    candidate: SessionMetrics,
    baseline_held_out: dict[str, HeldOutError] | None,
    candidate_held_out: dict[str, HeldOutError] | None,
) -> list[Row]:
    rows: list[Row] = []

    if baseline_held_out and candidate_held_out:
        for device in baseline_held_out:
            if device not in candidate_held_out:
                continue
            rows.append(
                Row(
                    f"held-out {device}",
                    baseline_held_out[device].median_m * 100,
                    candidate_held_out[device].median_m * 100,
                    "cm",
                )
            )

    for role, value in baseline.jitter.items():
        other = candidate.jitter.get(role)
        if other is not None:
            rows.append(
                Row(f"jitter {role.value}", value.rms_speed_mm_s, other.rms_speed_mm_s, "mm/s")
            )

    for role, value in baseline.feet.items():
        other = candidate.feet.get(role)
        if other is None:
            continue
        if value.slide_frames and other.slide_frames:
            rows.append(
                Row(f"skate {role.value}", value.slide_mm_per_s, other.slide_mm_per_s, "mm/s")
            )
        rows.append(
            Row(
                f"below floor {role.value}",
                value.penetration_fraction * 100,
                other.penetration_fraction * 100,
                "%",
            )
        )

    rows.append(Row("tracked", baseline.track_rate * 100, candidate.track_rate * 100, "%", False))
    if baseline.latency is not None and candidate.latency is not None:
        rows.append(Row("latency", baseline.latency.median_ms, candidate.latency.median_ms, "ms"))

    # Replayed sessions carry no wall-clock latency, so those rows would be all
    # NaN. Printing them invites the reader to compare nothing with nothing.
    return [row for row in rows if np.isfinite(row.baseline) and np.isfinite(row.candidate)]


def report(
    baseline: SessionMetrics,
    candidate: SessionMetrics,
    *,
    baseline_held_out: dict[str, HeldOutError] | None = None,
    candidate_held_out: dict[str, HeldOutError] | None = None,
) -> str:
    rows = _rows(baseline, candidate, baseline_held_out, candidate_held_out)
    base_name = (baseline.label or "baseline")[:16]
    cand_name = (candidate.label or "candidate")[:16]

    lines = [
        f"{'metric':<24s} {base_name:>16s} {cand_name:>16s} {'change':>10s}",
        "-" * 70,
    ]
    for row in rows:
        marker = "  better" if row.improved else ""
        lines.append(
            f"{row.name:<24s} {row.baseline:>13.1f} {row.unit:<2s} "
            f"{row.candidate:>13.1f} {row.unit:<2s} {row.delta_pct:>+9.1f}%{marker}"
        )
    return "\n".join(lines)
