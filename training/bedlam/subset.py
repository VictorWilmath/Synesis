"""Choosing which BEDLAM scenes to download.

BEDLAM 2.0 is 12.4 TB of PNGs. Downloading it is not an option, and most of it
would be wasted anyway: orbiting cameras, phone footage and egocentric headset
views teach a model about situations a webcam on a shelf will never be in.

The selection here is tiered rather than a single filter, because the strictest
criteria alone would leave too little data. Scenes are scored by how closely
they resemble the target setup and the caller decides how far down the list to
go.

One thing worth saying plainly, because it inverts the obvious reasoning: the
lifter does not consume pixels. It consumes 2D keypoints and headset poses. All
of that can be derived from the ground truth, which is about 4 GB for the whole
of BEDLAM 2.0 against 12.4 TB for the images. Images are only needed for a
small calibration subset, to measure how badly the 2D front-end degrades on
poor footage. So in practice this module picks a handful of scenes to fetch
pixels for, and the ground truth is simply taken whole.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .names import SceneName, try_parse

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Criteria:
    """What a scene needs to look like to be worth its download size."""

    require_static_camera: bool = True
    require_full_body: bool = True
    # Crowds are not excluded by default. Every body in a crowd scene is an
    # independent training sample, and the occlusion they cause each other is
    # the single most useful thing BEDLAM offers that a clean capture does
    # not: in a real room a chair or a desk hides your legs half the time.
    prefer_single_subject: bool = True
    max_bodies: int = 10
    # An indoor scene is a closer match to a living room, but the environment
    # only affects the pixels, and the lifter never sees pixels.
    prefer_indoor: bool = True
    # A webcam sits somewhere between a desk and a shelf, so it looks slightly
    # down. Steeply pitched renders are a different problem.
    max_pitch_deg: float = 30.0


@dataclass(slots=True)
class Scored:
    scene: SceneName
    score: float
    reasons: list[str] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        return self.score > 0.0


def score(scene: SceneName, criteria: Criteria | None = None) -> Scored:
    """How well one scene matches a webcam setup. Zero means excluded."""
    criteria = criteria or Criteria()
    reasons: list[str] = []

    if criteria.require_static_camera and not scene.camera_static:
        return Scored(scene, 0.0, ["camera moves"])
    if criteria.require_full_body and not scene.full_body_framing:
        return Scored(scene, 0.0, [f"{scene.framing} framing"])
    if abs(scene.pitch_deg) > criteria.max_pitch_deg:
        return Scored(scene, 0.0, [f"camera pitched {scene.pitch_deg:.0f} degrees"])
    if scene.bodies_min > criteria.max_bodies:
        return Scored(scene, 0.0, [f"at least {scene.bodies_min} bodies"])

    value = 1.0
    if scene.single_subject:
        if criteria.prefer_single_subject:
            value += 1.0
        reasons.append("single subject")
    else:
        # Crowds still earn their place, just less of it. More bodies means
        # more samples per frame but also more mutual occlusion, and past a
        # handful the frame is mostly people.
        crowd_penalty = min(scene.bodies_max, 10) / 20.0
        value += 0.5 - crowd_penalty
        reasons.append(f"{scene.bodies_min}-{scene.bodies_max} bodies")

    if scene.indoor is True:
        if criteria.prefer_indoor:
            value += 0.5
        reasons.append(f"indoor ({scene.scene})")
    elif scene.indoor is False:
        reasons.append(f"outdoor ({scene.scene})")

    if scene.pitch_deg:
        reasons.append(f"pitched {scene.pitch_deg:.0f} degrees")
    if scene.fps:
        reasons.append(f"{scene.fps:.0f}fps")

    return Scored(scene, value, reasons)


def select(
    folders: list[str],
    criteria: Criteria | None = None,
    *,
    limit: int | None = None,
) -> list[Scored]:
    """Rank scene folder names best-first, dropping the ineligible ones."""
    scored = []
    for folder in folders:
        scene = try_parse(folder)
        if scene is None:
            log.debug("skipping unparseable folder name %r", folder)
            continue
        result = score(scene, criteria)
        if result.eligible:
            scored.append(result)

    # Sequence count as the tie-break, so a given score prefers more data.
    scored.sort(key=lambda s: (-s.score, -s.scene.sequences, s.scene.raw))
    return scored[:limit] if limit else scored


def rejected(folders: list[str], criteria: Criteria | None = None) -> list[Scored]:
    """The scenes that did not make it, and why.

    Worth printing. If the criteria are accidentally too strict this is where
    it shows, rather than in a mysteriously small training set weeks later.
    """
    out = []
    for folder in folders:
        scene = try_parse(folder)
        if scene is None:
            continue
        result = score(scene, criteria)
        if not result.eligible:
            out.append(result)
    return out


def read_scene_list(path: Path | str) -> list[str]:
    """Folder names from BEDLAM's own ``bedlam*_scene_names.csv``.

    Clone ``pixelite1201/BEDLAM`` and point at ``data_processing/``; the first
    column of those files is the authoritative scene list. Reading it beats
    hard-coding a list here, which would silently go stale.
    """
    path = Path(path)
    names: list[str] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row or not row[0].strip():
                continue
            candidate = row[0].strip()
            if try_parse(candidate) is not None:
                names.append(candidate)
    return names


def scan_download(root: Path | str) -> list[str]:
    """Scene folder names already present in a partial download."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and try_parse(p.name))


def report(selected: list[Scored], skipped: list[Scored] | None = None) -> str:
    """A summary to read before committing to a download."""
    lines = [f"{len(selected)} scenes selected"]
    total_sequences = sum(s.scene.sequences for s in selected)
    lines.append(f"  {total_sequences} sequences in total")
    lines.append("")
    for item in selected:
        why = ", ".join(item.reasons) or "no distinguishing features"
        lines.append(f"  {item.score:4.1f}  {item.scene.raw}")
        lines.append(f"        {item.scene.sequences} sequences; {why}")

    if skipped:
        lines.append("")
        lines.append(f"{len(skipped)} scenes skipped")
        for item in skipped:
            lines.append(f"        {item.scene.raw}: {', '.join(item.reasons)}")
    return "\n".join(lines)
