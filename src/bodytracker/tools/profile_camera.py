"""Measure what your webcam does to an image, so training can reproduce it.

Run this once against the camera you will actually track with. It captures a
static scene twice, locked and unlocked, and writes a `CameraProfile` that the
degradation pipeline centres its distribution on.

Two passes, because the two measurements want opposite settings. Noise wants
the exposure locked, since a camera still hunting adds brightness wobble that
looks exactly like sensor noise in the temporal variance. Drift wants it
unlocked, since hunting is the thing being measured. Doing both and taking the
useful half of each is cheaper than asking the user to run two commands.

The one instruction that matters: point the camera at something that does not
move and leave the room, or at least stay out of frame. Any real motion is
counted as noise and inflates the profile.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..config import CameraConfig
from ..degrade.profile import CameraProfile, estimate_drift, profile_frames

log = logging.getLogger(__name__)

DEFAULT_FRAMES = 60
# Discarded at the start of each pass. Webcams deliver garbage for the first
# fraction of a second and take longer than that to settle their gain.
WARMUP_FRAMES = 20


def _grab(config: CameraConfig, count: int, *, warmup: int = WARMUP_FRAMES) -> np.ndarray:
    from ..capture.webcam import Webcam

    frames: list[np.ndarray] = []
    with Webcam(config) as camera:
        for _ in range(warmup):
            camera.read(timeout=2.0)
        while len(frames) < count:
            got = camera.read(timeout=2.0)
            if got is None:
                raise RuntimeError("camera stopped delivering frames")
            frames.append(got[1])
    return np.stack(frames)


def _describe_scene(frames: np.ndarray) -> str:
    """Warn about the two ways a profiling capture is usually spoiled."""
    mean = frames.mean(axis=0) / 255.0
    dark = float((mean < 0.15).mean())
    bright = float((mean > 0.85).mean())
    problems = []
    if dark + bright > 0.85:
        problems.append(
            "the scene is almost entirely black or blown out, so there is no "
            "midtone to fit the noise curve against"
        )
    spread = float(mean.max() - mean.min())
    if spread < 0.3:
        problems.append(
            "the scene has very little contrast; aim at something with both a "
            "shadow and a bright area"
        )
    return "; ".join(problems)


def _motion_warning(frames: np.ndarray) -> float:
    """Largest fraction of pixels that changed a lot between adjacent frames.

    Sensor noise moves every pixel a little. A person walking through moves a
    few percent of them a great deal, which is the signature to catch.
    """
    stack = frames.astype(np.float32) / 255.0
    deltas = np.abs(np.diff(stack, axis=0)).mean(axis=3)
    return float((deltas > 0.15).mean(axis=(1, 2)).max())


def profile_camera(
    config: CameraConfig,
    *,
    frames: int = DEFAULT_FRAMES,
    name: str = "",
    output: Path | str | None = None,
) -> CameraProfile:
    """Capture and measure, returning the profile and optionally saving it."""
    log.info("capturing %d frames with the exposure locked", frames)
    locked = _grab(replace(config, lock_exposure=True), frames)

    motion = _motion_warning(locked)
    if motion > 0.02:
        log.warning(
            "%.1f%% of pixels changed sharply between frames: something is "
            "moving in shot, and it will be measured as noise",
            motion * 100,
        )
    complaint = _describe_scene(locked)
    if complaint:
        log.warning("%s", complaint)

    profile = profile_frames(
        locked,
        name=name or f"camera{config.index}",
        fps=config.fps,
        notes=f"{frames} locked frames, {time.strftime('%Y-%m-%d')}",
    )

    log.info("capturing %d frames with auto exposure to measure drift", frames)
    try:
        auto = _grab(replace(config, lock_exposure=False), frames)
        exposure_drift, balance_drift = estimate_drift(auto.astype(np.float32) / 255.0)
        profile.exposure_drift = exposure_drift
        profile.white_balance_drift = balance_drift
    except RuntimeError:
        log.warning("could not complete the auto-exposure pass; keeping defaults")

    if output is not None:
        profile.save(output)
    return profile


def format_profile(profile: CameraProfile) -> str:
    """A human-readable summary, with the numbers put in context.

    The raw coefficients mean nothing to anyone. What the user wants to know is
    whether their camera is good or bad, so each line says so.
    """

    def verdict(value: float, good: float, bad: float) -> str:
        if value <= good:
            return "clean"
        if value >= bad:
            return "poor"
        return "typical"

    # Noise at mid grey is the number people can actually reason about: it is
    # roughly how many 8-bit levels a flat wall shimmers by.
    mid = float(profile.noise_sigma(np.array([0.18]))[0]) * 255.0

    lines = [
        f"camera profile: {profile.name}",
        f"  {profile.width}x{profile.height} @ {profile.fps:.0f}fps",
        f"  shot noise      {profile.shot_noise:.4f}",
        f"  read noise      {profile.read_noise:.4f}",
        f"  noise at 18% grey  {mid:5.2f}/255  ({verdict(mid, 1.0, 4.0)})",
        f"  exposure drift  {profile.exposure_drift * 100:5.2f}%  "
        f"({verdict(profile.exposure_drift, 0.005, 0.03)})",
        f"  white balance   {profile.white_balance_drift * 100:5.2f}%  "
        f"({verdict(profile.white_balance_drift, 0.005, 0.03)})",
        f"  jpeg quality    ~{profile.jpeg_quality}",
    ]
    if profile.samples:
        lines.append("  measured noise curve (linear level -> sigma):")
        for level, sigma in profile.samples:
            bar = "#" * int(min(sigma * 255.0 * 4, 40))
            lines.append(f"    {level:5.3f}  {sigma * 255.0:5.2f}  {bar}")
    return "\n".join(lines)


def preview(
    profile: CameraProfile,
    image_path: Path | str,
    output_dir: Path | str,
    *,
    count: int = 6,
    seed: int = 0,
) -> list[Path]:
    """Write a few degraded versions of an image so the result can be eyeballed.

    Numbers do not tell you whether the augmentation looks like your camera.
    Looking at it does, and it catches gross mistakes (an inverted gamma, a
    blur an order of magnitude too strong) that no unit test would.
    """
    import cv2

    from ..degrade.pipeline import ClipDegrader

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read {image_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i in range(count):
        degrader = ClipDegrader(profile=profile, seed=seed + i)
        path = output_dir / f"degraded_{i:02d}.png"
        cv2.imwrite(str(path), degrader.as_uint8(image))
        written.append(path)
    log.info("wrote %d previews to %s", len(written), output_dir)
    return written
