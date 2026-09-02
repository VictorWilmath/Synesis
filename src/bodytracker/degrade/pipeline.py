"""Turning a clean render into something that looks like it came off a webcam.

Two properties matter more than the individual effects.

The first is ordering. Degradations are applied in the order a real imaging
chain applies them, and in the right colour space at each step, so the
artefacts interact the way they really do. Noise that the JPEG encoder then has
to represent looks quite different from noise painted on top of a decoded JPEG.

The second is temporal coherence. Auto-exposure hunts over half a second, white
balance wanders over several, and the lens does not change sharpness between
frames. Sampling every parameter independently per frame produces a strobing
mess that teaches a temporal model that the world flickers. Parameters that
belong to the shot are drawn once per clip; parameters that evolve are given a
random walk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import ops
from .profile import CameraProfile


@dataclass(slots=True)
class DegradationConfig:
    """Ranges to sample from. Defaults span "decent webcam" to "bad webcam".

    Wider than any single camera on purpose. The model should be robust across
    the hardware the user might plug in, not tuned to one device, and the cost
    of training on slightly too much degradation is far lower than the cost of
    meeting a camera worse than anything in the training set.
    """

    # Multiplies the profile's measured noise, so a profiled camera stays the
    # centre of the distribution rather than its upper bound.
    noise_scale: tuple[float, float] = (0.5, 2.5)
    blur_sigma: tuple[float, float] = (0.0, 1.6)
    motion_blur_px: tuple[float, float] = (0.0, 9.0)
    jpeg_quality: tuple[int, int] = (45, 95)
    resample_scale: tuple[float, float] = (0.55, 1.0)
    vignette: tuple[float, float] = (0.0, 0.35)
    chromatic_px: tuple[float, float] = (0.0, 2.0)
    rolling_shutter_px: tuple[float, float] = (0.0, 6.0)
    sharpen: tuple[float, float] = (0.0, 0.9)
    # Exposure and white balance as a random walk over the clip.
    exposure_step: tuple[float, float] = (0.0, 0.02)
    exposure_range: tuple[float, float] = (0.55, 1.5)
    white_balance_step: tuple[float, float] = (0.0, 0.015)
    white_balance_range: tuple[float, float] = (0.9, 1.1)
    # Probability that a given clip gets each optional effect at all, so the
    # model also sees plenty of merely-mediocre footage.
    motion_blur_probability: float = 0.5
    rolling_shutter_probability: float = 0.3
    chromatic_probability: float = 0.4


@dataclass(slots=True)
class ClipParameters:
    """Everything held fixed for the duration of one clip."""

    noise_shot: float
    noise_read: float
    blur_sigma: float
    jpeg_quality: int
    resample_scale: float
    vignette: float
    chromatic_px: float
    sharpen: float
    motion_blur_px: float
    rolling_shutter_px: float
    exposure_step: float
    white_balance_step: float
    exposure_range: tuple[float, float]
    white_balance_range: tuple[float, float]
    # Evolving state, advanced once per frame.
    exposure: float = 1.0
    blue_gain: float = 1.0
    red_gain: float = 1.0
    extras: dict = field(default_factory=dict)


def sample_clip(
    config: DegradationConfig,
    profile: CameraProfile,
    rng: np.random.Generator,
) -> ClipParameters:
    """Draw the settings for one clip.

    The camera does not swap lenses between frames, so anything physical is
    fixed here and only the auto-adjusting parts are left to move.
    """

    def uniform(bounds: tuple[float, float]) -> float:
        return float(rng.uniform(*bounds))

    scale = uniform(config.noise_scale)
    motion = (
        uniform(config.motion_blur_px)
        if rng.random() < config.motion_blur_probability
        else 0.0
    )
    skew = (
        uniform(config.rolling_shutter_px)
        if rng.random() < config.rolling_shutter_probability
        else 0.0
    )
    fringe = (
        uniform(config.chromatic_px) if rng.random() < config.chromatic_probability else 0.0
    )

    return ClipParameters(
        noise_shot=profile.shot_noise * scale,
        noise_read=profile.read_noise * scale,
        blur_sigma=uniform(config.blur_sigma),
        jpeg_quality=int(rng.integers(config.jpeg_quality[0], config.jpeg_quality[1] + 1)),
        resample_scale=uniform(config.resample_scale),
        vignette=uniform(config.vignette),
        chromatic_px=fringe,
        sharpen=uniform(config.sharpen),
        motion_blur_px=motion,
        rolling_shutter_px=skew,
        exposure_step=uniform(config.exposure_step),
        white_balance_step=uniform(config.white_balance_step),
        exposure_range=config.exposure_range,
        white_balance_range=config.white_balance_range,
    )


def advance(params: ClipParameters, rng: np.random.Generator) -> None:
    """Move the auto-exposure and white balance on by one frame.

    A bounded random walk rather than fresh samples, because that is what
    hunting looks like: a slow wander with occasional corrections, not
    independent noise.
    """
    params.exposure = float(
        np.clip(
            params.exposure + rng.normal(0.0, params.exposure_step),
            *params.exposure_range,
        )
    )
    params.blue_gain = float(
        np.clip(
            params.blue_gain + rng.normal(0.0, params.white_balance_step),
            *params.white_balance_range,
        )
    )
    params.red_gain = float(
        np.clip(
            params.red_gain + rng.normal(0.0, params.white_balance_step),
            *params.white_balance_range,
        )
    )


def degrade_frame(
    image: np.ndarray,
    params: ClipParameters,
    rng: np.random.Generator,
    *,
    motion_angle_deg: float = 0.0,
) -> np.ndarray:
    """Push one clean sRGB uint8 frame through the imaging chain."""
    working = np.asarray(image, dtype=np.float32)
    if working.max() > 1.5:
        working = working / 255.0

    # Optics and sensor, in linear light.
    linear = ops.to_linear(working)
    linear = ops.defocus(linear, params.blur_sigma)
    linear = ops.chromatic_aberration(linear, params.chromatic_px)
    linear = ops.motion_blur(linear, params.motion_blur_px, motion_angle_deg)
    linear = ops.rolling_shutter(linear, params.rolling_shutter_px)
    linear = ops.vignette(linear, params.vignette)
    linear = ops.apply_exposure(linear, params.exposure)
    linear = ops.sensor_noise(linear, params.noise_shot, params.noise_read, rng)
    linear = ops.apply_white_balance(linear, params.blue_gain, params.red_gain)

    # ISP and transport, in sRGB.
    srgb = ops.to_srgb(linear)
    srgb = ops.resample(srgb, params.resample_scale)
    srgb = ops.sharpen(srgb, params.sharpen)
    srgb = ops.jpeg(srgb, params.jpeg_quality)
    return np.clip(srgb, 0.0, 1.0)


class ClipDegrader:
    """Applies a consistent degradation across the frames of one clip."""

    def __init__(
        self,
        config: DegradationConfig | None = None,
        profile: CameraProfile | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.config = config or DegradationConfig()
        self.profile = profile or CameraProfile()
        self.rng = np.random.default_rng(seed)
        self.params = sample_clip(self.config, self.profile, self.rng)

    def new_clip(self) -> ClipParameters:
        self.params = sample_clip(self.config, self.profile, self.rng)
        return self.params

    def __call__(self, image: np.ndarray, *, motion_angle_deg: float = 0.0) -> np.ndarray:
        advance(self.params, self.rng)
        return degrade_frame(image, self.params, self.rng, motion_angle_deg=motion_angle_deg)

    def as_uint8(self, image: np.ndarray, *, motion_angle_deg: float = 0.0) -> np.ndarray:
        degraded = self(image, motion_angle_deg=motion_angle_deg)
        return np.clip(degraded * 255.0, 0, 255).astype(np.uint8)
