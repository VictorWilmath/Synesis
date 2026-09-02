"""What a specific webcam actually does to an image.

Augmentation is only useful if it lands in the same place as the real camera.
Guessed noise levels produce a model that is robust to imaginary problems, so
these numbers are measured from the user's own hardware rather than invented.

The measurement that matters most is the noise: real sensors are
Poisson-Gaussian, meaning the noise grows with brightness (photon shot noise,
proportional to the square root of the signal) on top of a constant floor (read
noise). Fitting those two coefficients takes nothing more than a static scene
and a few dozen frames, and it pins down the single most important axis of the
domain gap.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Rough defaults for a mid-range 720p USB webcam under office lighting, used
# when the user has not profiled their own. Deliberately on the noisy side:
# training for more degradation than you get is much cheaper than the reverse.
DEFAULT_SHOT_NOISE = 0.010
DEFAULT_READ_NOISE = 0.006


@dataclass(slots=True)
class CameraProfile:
    """A measured description of one camera's imperfections.

    All levels are in linear light, normalised to [0, 1], so they transfer
    between bit depths and do not depend on the display gamma.
    """

    # Noise variance model: var(signal) = shot^2 * signal + read^2
    shot_noise: float = DEFAULT_SHOT_NOISE
    read_noise: float = DEFAULT_READ_NOISE
    # Standard deviation of frame-to-frame brightness wobble, as a fraction of
    # the mean. Auto-exposure hunting, mostly.
    exposure_drift: float = 0.01
    # Same for the red and blue channel gains, from auto white balance.
    white_balance_drift: float = 0.01
    # Estimated JPEG quality of the MJPG stream, 0-100.
    jpeg_quality: int = 85
    # How soft the lens is, as a Gaussian sigma in pixels at native resolution.
    blur_sigma: float = 0.6
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    name: str = "default"
    notes: str = ""
    # Per-decile measurement of the noise curve, kept for plotting and for
    # checking the two-parameter fit was not a fantasy.
    samples: list[tuple[float, float]] = field(default_factory=list)

    def noise_sigma(self, signal: np.ndarray) -> np.ndarray:
        """Expected noise standard deviation at each linear intensity."""
        variance = (self.shot_noise**2) * np.clip(signal, 0.0, None) + self.read_noise**2
        return np.sqrt(variance)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["samples"] = [list(s) for s in self.samples]
        path.write_text(json.dumps(data, indent=2))
        log.info("wrote camera profile to %s", path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> CameraProfile | None:
        path = Path(path)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        data["samples"] = [tuple(s) for s in data.get("samples", [])]
        return cls(**data)


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """Undo the display gamma, because sensor noise happens in linear light.

    Applying noise directly to sRGB values overstates it in the shadows and
    understates it in the highlights, which is the wrong shape entirely. It is
    a cheap step and it is the difference between augmentation that matches a
    real sensor and augmentation that merely looks grainy.
    """
    x = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


def estimate_noise(
    frames: np.ndarray,
    bins: int = 10,
    *,
    compensate_exposure: bool = True,
) -> tuple[float, float, list]:
    """Fit a Poisson-Gaussian noise model to frames of a static scene.

    Temporal differences isolate noise from content: point the camera at
    something that is not moving and whatever changes between frames is the
    sensor. Binning by brightness then separates the signal-dependent part from
    the constant part, which a single global standard deviation cannot do.

    `compensate_exposure` divides out each frame's overall brightness first.
    Without it, a camera whose auto-exposure is still breathing looks far
    noisier than it is, because a global gain wobble shows up in the temporal
    variance as though it were sensor noise. It is scaled away here rather than
    guarded against, since a user will inevitably profile with auto-exposure
    left on at least once.

    Returns ``(shot, read, samples)`` where samples are ``(level, sigma)``
    pairs, kept so the fit can be sanity-checked rather than trusted.
    """
    if len(frames) < 2:
        raise ValueError("need at least two frames of a static scene")

    stack = np.stack([srgb_to_linear(f) for f in frames])
    if compensate_exposure:
        per_frame = stack.mean(axis=(1, 2, 3), keepdims=True)
        stack = stack * (per_frame.mean() / np.clip(per_frame, 1e-6, None))
    mean = stack.mean(axis=0)
    # The unbiased per-pixel standard deviation over time.
    sigma = stack.std(axis=0, ddof=1)

    levels = mean.ravel()
    sigmas = sigma.ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    samples: list[tuple[float, float]] = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        selected = (levels >= low) & (levels < high)
        # A bin with a handful of pixels tells you about those pixels, not the
        # sensor, so require a real population before believing it.
        if selected.sum() < 100:
            continue
        samples.append((float(levels[selected].mean()), float(np.median(sigmas[selected]))))

    if len(samples) < 2:
        raise ValueError(
            "not enough brightness variation to fit a noise model; "
            "point the camera at a scene with both dark and bright areas"
        )

    # var = shot^2 * level + read^2 is linear in level, so least squares on the
    # variance recovers both coefficients at once.
    level = np.array([s[0] for s in samples])
    variance = np.array([s[1] ** 2 for s in samples])
    design = np.stack([level, np.ones_like(level)], axis=1)
    solution, *_ = np.linalg.lstsq(design, variance, rcond=None)

    shot = float(np.sqrt(max(solution[0], 0.0)))
    read = float(np.sqrt(max(solution[1], 0.0)))
    return shot, read, samples


def estimate_drift(frames: np.ndarray) -> tuple[float, float]:
    """How much the camera's own auto-exposure and white balance wander.

    Measured as the relative variation of overall brightness, and of the red
    and blue gains against green, across frames of a static scene. Both are
    things a webcam does on its own and a rendered dataset never does.
    """
    if len(frames) < 3:
        return 0.0, 0.0

    stack = np.stack([np.asarray(f, dtype=np.float32) for f in frames])
    brightness = stack.mean(axis=(1, 2, 3))
    exposure = float(np.std(brightness) / max(np.mean(brightness), 1e-6))

    # OpenCV order: blue, green, red.
    channels = stack.mean(axis=(1, 2))
    green = np.clip(channels[:, 1], 1e-6, None)
    ratios = np.stack([channels[:, 0] / green, channels[:, 2] / green], axis=1)
    balance = float(np.mean(np.std(ratios, axis=0) / np.clip(np.mean(ratios, axis=0), 1e-6, None)))
    return exposure, balance


def reencode_distortion(image: np.ndarray, quality: int) -> float:
    """Mean absolute change, in 8-bit levels, from a JPEG round trip at `quality`."""
    import cv2

    source = np.clip(np.asarray(image, dtype=np.float32), 0, 255).astype(np.uint8)
    if source.ndim == 2:
        source = np.stack([source] * 3, axis=2)
    ok, buffer = cv2.imencode(".jpg", source, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return float("inf")
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR).astype(np.float32)
    return float(np.abs(decoded - source.astype(np.float32)).mean())


def estimate_jpeg_quality(
    image: np.ndarray,
    candidates: range | None = None,
    *,
    tolerance: float = 1.0,
) -> int:
    """Guess the quality setting of the MJPG stream the frame arrived on.

    Rests on JPEG being close to idempotent: re-encoding at the quality the
    image already carries barely changes it, because the coefficients are
    already sitting on that quantisation grid, while re-encoding any coarser
    visibly damages it. So the distortion curve has a knee at the original
    quality, and the lowest setting that survives a round trip cleanly is the
    answer.

    Approximate, and it saturates at the top of the candidate range for an
    image that was never compressed. It is enough to tell a cheap camera
    pushing quality 50 from a good one at 90, which is the distinction the
    augmentation ranges need.
    """
    candidates = candidates or range(30, 101, 5)
    for quality in candidates:
        if reencode_distortion(image, quality) < tolerance:
            return int(quality)
    return int(max(candidates))


def profile_frames(
    frames: np.ndarray,
    *,
    name: str = "measured",
    fps: float = 30.0,
    notes: str = "",
    estimate_quality: bool = True,
) -> CameraProfile:
    """Build a profile from frames of a static scene, as uint8 BGR."""
    normalised = [np.asarray(f, dtype=np.float32) / 255.0 for f in frames]
    shot, read, samples = estimate_noise(np.stack(normalised))
    exposure, balance = estimate_drift(np.stack(normalised))

    # Averaging first, because blockiness is measured against the noise floor
    # and a single noisy frame buries the grid it is trying to find.
    quality = (
        estimate_jpeg_quality(np.stack(normalised).mean(axis=0) * 255.0)
        if estimate_quality
        else 85
    )

    height, width = frames[0].shape[:2]
    return CameraProfile(
        shot_noise=shot,
        read_noise=read,
        exposure_drift=exposure,
        white_balance_drift=balance,
        jpeg_quality=quality,
        width=width,
        height=height,
        fps=fps,
        name=name,
        notes=notes,
        samples=samples,
    )
