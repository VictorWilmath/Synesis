"""Individual degradations, each modelling one stage of a real camera.

Order is not cosmetic. A photon hits the lens before it hits the sensor, and
the sensor reads it before the ISP white-balances it and the encoder
compresses it. Applying noise to an already-compressed image, which is what
most augmentation pipelines do, produces something no camera has ever
produced: JPEG blocks with grain painted evenly on top, rather than grain that
the encoder then had to struggle to represent.

Every function here takes and returns float32 in [0, 1]. The colour space is
noted per function, because it matters: optics and sensor effects belong in
linear light, everything after the ISP belongs in sRGB.
"""

from __future__ import annotations

import cv2
import numpy as np

from .profile import linear_to_srgb, srgb_to_linear


def defocus(image: np.ndarray, sigma: float) -> np.ndarray:
    """Lens softness. Linear light.

    Cheap webcams have fixed-focus plastic lenses that are soft at the corners
    and never quite sharp anywhere, which blurs exactly the fine detail a pose
    model uses to localise wrists and ankles.
    """
    if sigma <= 0.01:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def motion_blur(image: np.ndarray, length: float, angle_deg: float) -> np.ndarray:
    """Directional smear from a slow shutter. Linear light.

    A webcam indoors often runs at 1/30 s, so a moving limb is drawn as a
    streak. This is the single most damaging degradation for keypoint accuracy
    and the one a model trained on sharp renders has never seen.
    """
    if length < 1.0:
        return image

    size = int(np.ceil(length)) | 1
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (size, size))
    total = kernel.sum()
    if total <= 0:
        return image
    return cv2.filter2D(image, -1, kernel / total)


def vignette(image: np.ndarray, strength: float) -> np.ndarray:
    """Corner darkening. Linear light.

    Matters more than it sounds for full-body tracking, because the feet and
    hands are usually the things out at the edge of the frame where the light
    falls off and the noise is therefore relatively worse.
    """
    if strength <= 0.0:
        return image

    height, width = image.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    radius = np.sqrt(((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2) / np.sqrt(2.0)
    falloff = 1.0 - strength * radius**2
    return image * falloff[:, :, None]


def chromatic_aberration(image: np.ndarray, pixels: float) -> np.ndarray:
    """Colour fringing from a lens that focuses red and blue differently.

    Modelled as a slight radial scale difference between channels.
    """
    if pixels <= 0.01:
        return image

    height, width = image.shape[:2]
    out = np.empty_like(image)
    # Blue and red scale in opposite directions; green is the reference.
    for channel, direction in ((0, -1.0), (1, 0.0), (2, 1.0)):
        if direction == 0.0:
            out[:, :, channel] = image[:, :, channel]
            continue
        scale = 1.0 + direction * pixels / max(width, height)
        matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), 0.0, scale)
        out[:, :, channel] = cv2.warpAffine(
            image[:, :, channel], matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
    return out


def rolling_shutter(image: np.ndarray, shift_px: float) -> np.ndarray:
    """Skew from a sensor that reads out row by row. Linear light.

    Every rolling-shutter camera does this and no renderer does. During a fast
    pan or a quick sidestep it shears the whole body, which systematically
    biases limb angles rather than merely adding noise to them.
    """
    if abs(shift_px) < 0.5:
        return image

    height, width = image.shape[:2]
    rows = np.arange(height, dtype=np.float32)
    shifts = shift_px * (rows / max(height - 1, 1) - 0.5)
    map_x = np.tile(np.arange(width, dtype=np.float32), (height, 1)) + shifts[:, None]
    map_y = np.tile(rows[:, None], (1, width))
    return cv2.remap(
        image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def apply_exposure(image: np.ndarray, gain: float) -> np.ndarray:
    """Auto-exposure moving the whole frame's brightness. Linear light."""
    return image * gain


def apply_white_balance(image: np.ndarray, blue_gain: float, red_gain: float) -> np.ndarray:
    """Auto white balance drifting the colour cast. Linear light, BGR order."""
    out = image.copy()
    out[:, :, 0] *= blue_gain
    out[:, :, 2] *= red_gain
    return out


def sensor_noise(
    image: np.ndarray,
    shot: float,
    read: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Poisson-Gaussian read noise. Linear light.

    The signal-dependent term is what makes this look like a sensor rather
    than like added grain: dark parts of the frame are relatively far noisier
    than bright ones, which is exactly why tracking a foot in the shadow under
    a desk is harder than tracking a hand near a lamp.
    """
    signal = np.clip(image, 0.0, None)
    sigma = np.sqrt((shot**2) * signal + read**2)
    return image + rng.normal(0.0, 1.0, size=image.shape).astype(np.float32) * sigma


def jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    """An MJPG round trip. sRGB.

    USB webcams almost always deliver MJPG, because uncompressed 720p60 does
    not fit down USB 2.0. The resulting 8x8 blocking and chroma subsampling are
    present in every frame the tracker will ever see and in none of the
    training renders.
    """
    quality = int(np.clip(quality, 1, 100))
    encoded = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    ok, buffer = cv2.imencode(".jpg", encoded, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR).astype(np.float32) / 255.0


def resample(image: np.ndarray, scale: float) -> np.ndarray:
    """Throw away resolution and put it back. sRGB.

    Models a sensor whose true resolution is below what it reports, and the
    upscaling some drivers do to hit a requested mode.
    """
    if scale >= 0.999:
        return image
    height, width = image.shape[:2]
    small = cv2.resize(
        image,
        (max(int(width * scale), 16), max(int(height * scale), 16)),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def sharpen(image: np.ndarray, amount: float) -> np.ndarray:
    """The ISP's unsharp mask. sRGB.

    Webcam firmware oversharpens to look crisp on a video call, which leaves
    halos around limbs. Worth modelling because it is a systematic edge
    artefact and edges are what a keypoint model keys off.
    """
    if amount <= 0.0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    return np.clip(image + amount * (image - blurred), 0.0, 1.0)


def to_linear(image: np.ndarray) -> np.ndarray:
    return srgb_to_linear(image)


def to_srgb(image: np.ndarray) -> np.ndarray:
    return linear_to_srgb(np.clip(image, 0.0, 1.0))
