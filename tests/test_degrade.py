"""Tests for the webcam degradation pipeline.

The load-bearing test is the noise round trip: synthesise frames with a known
Poisson-Gaussian noise model, measure them, and check the measurement recovers
what went in. If that is wrong then every augmented image is wrong in the same
direction, and a model trained on them is robust to a camera that does not
exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from bodytracker.degrade import (
    CameraProfile,
    ClipDegrader,
    DegradationConfig,
    degrade_frame,
    estimate_drift,
    estimate_noise,
    linear_to_srgb,
    profile_frames,
    sample_clip,
    srgb_to_linear,
)
from bodytracker.degrade import ops


def gradient_image(height: int = 120, width: int = 160) -> np.ndarray:
    """A scene spanning the full brightness range, in sRGB [0, 1]."""
    ramp = np.linspace(0.02, 0.98, width, dtype=np.float32)
    base = np.tile(ramp, (height, 1))
    image = np.stack([base, base * 0.9 + 0.05, base * 0.8 + 0.1], axis=2)
    return image.astype(np.float32)


def noisy_frames(shot: float, read: float, count: int = 40, seed: int = 0) -> np.ndarray:
    """Frames of a static scene with a known noise model applied."""
    rng = np.random.default_rng(seed)
    clean_linear = srgb_to_linear(gradient_image())
    out = []
    for _ in range(count):
        noisy = ops.sensor_noise(clean_linear, shot, read, rng)
        out.append(linear_to_srgb(np.clip(noisy, 0.0, 1.0)))
    return np.stack(out)


class TestColourSpace:
    def test_srgb_round_trips(self):
        values = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(1, -1, 1)
        assert np.allclose(linear_to_srgb(srgb_to_linear(values)), values, atol=1e-5)

    def test_linear_is_darker_in_the_midtones(self):
        """Sanity check the gamma is applied the right way round."""
        assert srgb_to_linear(np.array([0.5])) < 0.5

    def test_endpoints_are_fixed(self):
        assert srgb_to_linear(np.array([0.0]))[0] == pytest.approx(0.0)
        assert srgb_to_linear(np.array([1.0]))[0] == pytest.approx(1.0)


class TestNoiseEstimation:
    @pytest.mark.parametrize(
        ("shot", "read"),
        [(0.02, 0.004), (0.05, 0.010), (0.005, 0.002)],
    )
    def test_recovers_a_known_noise_model(self, shot, read):
        measured_shot, measured_read, _ = estimate_noise(noisy_frames(shot, read, count=60))
        assert measured_shot == pytest.approx(shot, rel=0.25)
        assert measured_read == pytest.approx(read, rel=0.35)

    def test_the_noise_really_is_signal_dependent(self):
        """If it were not, a constant sigma would fit and shot would be zero."""
        _, _, samples = estimate_noise(noisy_frames(0.05, 0.002, count=60))
        levels = np.array([s[0] for s in samples])
        sigmas = np.array([s[1] for s in samples])
        assert np.corrcoef(levels, sigmas)[0, 1] > 0.9

    def test_a_clean_scene_measures_as_clean(self):
        clean = np.stack([gradient_image()] * 10)
        shot, read, _ = estimate_noise(clean)
        assert shot < 1e-3
        assert read < 1e-3

    def test_rejects_a_single_frame(self):
        with pytest.raises(ValueError, match="two frames"):
            estimate_noise(np.stack([gradient_image()]))

    def test_rejects_a_flat_scene(self):
        """One brightness level cannot separate shot noise from read noise."""
        flat = np.full((8, 60, 60, 3), 0.5, dtype=np.float32)
        with pytest.raises(ValueError, match="brightness variation"):
            estimate_noise(flat)


class TestExposureCompensation:
    """Auto-exposure wobble must not be counted as sensor noise."""

    @staticmethod
    def _breathing_frames(gain_sigma: float, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        clean = srgb_to_linear(gradient_image())
        out = []
        for _ in range(60):
            gain = 1.0 + rng.normal(0.0, gain_sigma)
            noisy = ops.sensor_noise(clean * gain, 0.02, 0.004, rng)
            out.append(linear_to_srgb(np.clip(noisy, 0.0, 1.0)))
        return np.stack(out)

    def test_a_breathing_exposure_does_not_inflate_the_noise(self):
        frames = self._breathing_frames(0.04)
        shot, _, _ = estimate_noise(frames, compensate_exposure=True)
        assert shot == pytest.approx(0.02, rel=0.3)

    def test_and_would_without_the_compensation(self):
        """Shows the compensation is load-bearing rather than decorative."""
        frames = self._breathing_frames(0.04)
        naive, _, _ = estimate_noise(frames, compensate_exposure=False)
        corrected, _, _ = estimate_noise(frames, compensate_exposure=True)
        assert naive > corrected * 2.0

    def test_it_does_not_distort_a_steady_camera(self):
        frames = noisy_frames(0.02, 0.004, count=60)
        with_it, _, _ = estimate_noise(frames, compensate_exposure=True)
        without, _, _ = estimate_noise(frames, compensate_exposure=False)
        assert with_it == pytest.approx(without, rel=0.05)


class TestJpegEstimation:
    """A photo, since the estimator keys off content a codec has to work at."""

    @staticmethod
    def scene() -> np.ndarray:
        rng = np.random.default_rng(0)
        base = rng.random((96, 128, 3)).astype(np.float32)
        # Smooth it into blobs, so it has structure rather than being pure
        # noise, which no quantiser can represent at any setting.
        return np.clip(ops.defocus(base, 3.0) * 1.5, 0.0, 1.0)

    @pytest.mark.parametrize("quality", [40, 60, 80])
    def test_recovers_a_known_quality(self, quality):
        from bodytracker.degrade.profile import estimate_jpeg_quality

        compressed = ops.jpeg(self.scene(), quality) * 255.0
        assert estimate_jpeg_quality(compressed) == pytest.approx(quality, abs=15)

    def test_orders_qualities_correctly(self):
        from bodytracker.degrade.profile import estimate_jpeg_quality

        low = estimate_jpeg_quality(ops.jpeg(self.scene(), 35) * 255.0)
        high = estimate_jpeg_quality(ops.jpeg(self.scene(), 92) * 255.0)
        assert low < high

    def test_a_pristine_image_reads_as_high_quality(self):
        from bodytracker.degrade.profile import estimate_jpeg_quality

        assert estimate_jpeg_quality(self.scene() * 255.0) >= 80

    def test_reencoding_at_the_original_quality_barely_changes_anything(self):
        """The idempotence the estimator depends on."""
        from bodytracker.degrade.profile import reencode_distortion

        compressed = ops.jpeg(self.scene(), 50) * 255.0
        assert reencode_distortion(compressed, 50) < reencode_distortion(compressed, 25)


class TestDriftEstimation:
    def test_measures_exposure_wobble(self):
        base = gradient_image()
        rng = np.random.default_rng(3)
        frames = np.stack([np.clip(base * (1.0 + rng.normal(0, 0.05)), 0, 1) for _ in range(40)])
        exposure, _ = estimate_drift(frames)
        assert exposure == pytest.approx(0.05, rel=0.4)

    def test_a_static_feed_has_no_drift(self):
        frames = np.stack([gradient_image()] * 10)
        exposure, balance = estimate_drift(frames)
        assert exposure < 1e-5
        assert balance < 1e-5


class TestProfile:
    def test_profiles_frames_end_to_end(self):
        frames = (noisy_frames(0.03, 0.006, count=40) * 255).astype(np.uint8)
        profile = profile_frames(frames, name="fake")
        assert profile.shot_noise == pytest.approx(0.03, rel=0.3)
        assert profile.width == 160
        assert profile.height == 120
        assert profile.name == "fake"

    def test_round_trips_through_json(self, tmp_path):
        profile = CameraProfile(shot_noise=0.02, read_noise=0.004, name="c920")
        loaded = CameraProfile.load(profile.save(tmp_path / "profile.json"))
        assert loaded.shot_noise == pytest.approx(0.02)
        assert loaded.name == "c920"

    def test_missing_profile_is_none(self, tmp_path):
        assert CameraProfile.load(tmp_path / "absent.json") is None

    def test_sigma_grows_with_signal(self):
        profile = CameraProfile(shot_noise=0.03, read_noise=0.005)
        sigma = profile.noise_sigma(np.array([0.0, 0.25, 1.0]))
        assert sigma[0] == pytest.approx(0.005)
        assert sigma[2] > sigma[1] > sigma[0]


class TestOps:
    def test_defocus_reduces_detail(self):
        image = gradient_image()
        edge = np.zeros_like(image)
        edge[:, 80:] = 1.0
        blurred = ops.defocus(edge, 2.0)
        assert np.var(np.diff(blurred[60, :, 0])) < np.var(np.diff(edge[60, :, 0]))

    def test_motion_blur_smears_along_its_angle(self):
        dot = np.zeros((41, 41, 3), dtype=np.float32)
        dot[20, 20] = 1.0
        horizontal = ops.motion_blur(dot, 11, 0.0)
        assert horizontal[20, 15, 0] > 0.0
        assert horizontal[15, 20, 0] == pytest.approx(0.0, abs=1e-6)

    def test_motion_blur_conserves_brightness(self):
        image = gradient_image()
        assert ops.motion_blur(image, 7, 30.0).mean() == pytest.approx(image.mean(), rel=0.02)

    def test_vignette_darkens_corners_not_the_centre(self):
        image = np.ones((100, 100, 3), dtype=np.float32)
        out = ops.vignette(image, 0.4)
        assert out[50, 50, 0] == pytest.approx(1.0, abs=0.02)
        assert out[0, 0, 0] < 0.7

    def test_rolling_shutter_shears_opposite_ways_top_and_bottom(self):
        """A vertical line should come out slanted, not merely displaced.

        The sign follows a rightward pan: rows read later see the scene
        further left. What the test pins down is that the two ends move in
        opposite directions by roughly the requested amount, which is what
        turns a translation into a shear.
        """
        image = np.zeros((100, 100, 3), dtype=np.float32)
        image[:, 48:52] = 1.0
        sheared = ops.rolling_shutter(image, 20.0)

        def centre(row):
            weights = sheared[row, :, 0]
            return float((np.arange(100) * weights).sum() / max(weights.sum(), 1e-6))

        top, bottom = centre(5), centre(95)
        assert top > 55.0
        assert bottom < 45.0
        assert top - bottom == pytest.approx(20.0, abs=3.0)

    def test_jpeg_loses_information(self):
        image = gradient_image()
        assert not np.allclose(ops.jpeg(image, 30), image, atol=0.01)
        assert np.abs(ops.jpeg(image, 30) - image).mean() > np.abs(
            ops.jpeg(image, 95) - image
        ).mean()

    def test_resample_softens(self):
        edge = np.zeros((100, 100, 3), dtype=np.float32)
        edge[:, 50:] = 1.0
        assert np.abs(np.diff(ops.resample(edge, 0.3)[50, :, 0])).max() < 1.0

    def test_sharpen_overshoots_at_an_edge(self):
        """The halo is the point: it is what webcam firmware leaves behind."""
        edge = np.zeros((64, 64, 3), dtype=np.float32)
        edge[:, 32:] = 0.8
        blurred = ops.defocus(edge, 2.0)
        sharpened = ops.sharpen(blurred, 0.9)

        row = 32
        assert np.abs(np.diff(sharpened[row, :, 0])).max() > np.abs(
            np.diff(blurred[row, :, 0])
        ).max()
        # An undershoot on the dark side of the edge that the blur did not have.
        assert sharpened[row, 28, 0] < blurred[row, 28, 0]

    def test_disabled_effects_are_exact_no_ops(self):
        image = gradient_image()
        assert ops.defocus(image, 0.0) is image
        assert ops.motion_blur(image, 0.0, 0.0) is image
        assert ops.vignette(image, 0.0) is image
        assert ops.rolling_shutter(image, 0.0) is image
        assert ops.sharpen(image, 0.0) is image


class TestPipeline:
    def test_produces_a_valid_image(self):
        degrader = ClipDegrader(seed=0)
        out = degrader(gradient_image())
        assert out.shape == (120, 160, 3)
        assert out.min() >= 0.0
        assert out.max() <= 1.0
        assert np.all(np.isfinite(out))

    def test_accepts_uint8_and_returns_uint8(self):
        degrader = ClipDegrader(seed=0)
        out = degrader.as_uint8((gradient_image() * 255).astype(np.uint8))
        assert out.dtype == np.uint8
        assert out.shape == (120, 160, 3)

    def test_actually_degrades(self):
        degrader = ClipDegrader(seed=1)
        image = gradient_image()
        assert np.abs(degrader(image) - image).mean() > 0.005

    def test_is_deterministic_for_a_seed(self):
        image = gradient_image()
        first = ClipDegrader(seed=7)(image)
        second = ClipDegrader(seed=7)(image)
        assert np.allclose(first, second)

    def test_different_seeds_differ(self):
        image = gradient_image()
        assert not np.allclose(ClipDegrader(seed=1)(image), ClipDegrader(seed=2)(image))

    def test_a_noisier_profile_gives_a_noisier_image(self):
        image = gradient_image()
        quiet = CameraProfile(shot_noise=0.002, read_noise=0.001)
        loud = CameraProfile(shot_noise=0.06, read_noise=0.03)
        # Hold everything except the noise fixed.
        config = DegradationConfig(
            noise_scale=(1.0, 1.0),
            blur_sigma=(0.0, 0.0),
            motion_blur_probability=0.0,
            rolling_shutter_probability=0.0,
            chromatic_probability=0.0,
            jpeg_quality=(100, 100),
            resample_scale=(1.0, 1.0),
            vignette=(0.0, 0.0),
            sharpen=(0.0, 0.0),
            exposure_step=(0.0, 0.0),
            white_balance_step=(0.0, 0.0),
        )

        def roughness(profile):
            out = ClipDegrader(config, profile, seed=4)(image)
            return float(np.std(np.diff(out[:, :, 1], axis=1)))

        assert roughness(loud) > roughness(quiet) * 3.0


class TestTemporalCoherence:
    """A temporal model must not be taught that the world flickers."""

    def test_optics_stay_fixed_across_a_clip(self):
        degrader = ClipDegrader(seed=2)
        first = degrader.params
        for _ in range(10):
            degrader(gradient_image())
        assert degrader.params.blur_sigma == first.blur_sigma
        assert degrader.params.jpeg_quality == first.jpeg_quality
        assert degrader.params.vignette == first.vignette

    def test_exposure_wanders_rather_than_jumping(self):
        config = DegradationConfig(exposure_step=(0.02, 0.02))
        degrader = ClipDegrader(config, seed=3)
        history = []
        for _ in range(60):
            degrader(gradient_image())
            history.append(degrader.params.exposure)

        steps = np.abs(np.diff(history))
        assert steps.max() < 0.1, "exposure jumped"
        assert np.ptp(history) > 0.01, "exposure never moved"

    def test_exposure_stays_within_bounds(self):
        config = DegradationConfig(exposure_step=(0.2, 0.2), exposure_range=(0.7, 1.3))
        degrader = ClipDegrader(config, seed=5)
        for _ in range(200):
            degrader(gradient_image())
            assert 0.7 <= degrader.params.exposure <= 1.3

    def test_a_new_clip_resamples_the_optics(self):
        degrader = ClipDegrader(seed=6)
        before = degrader.params
        for _ in range(20):
            degrader.new_clip()
            if degrader.params.jpeg_quality != before.jpeg_quality:
                return
        pytest.fail("new_clip never changed the settings")

    def test_a_still_scene_stays_still_apart_from_noise(self):
        """Guards against a degradation that shifts geometry frame to frame."""
        config = DegradationConfig(
            noise_scale=(0.0, 0.0),
            motion_blur_probability=0.0,
            rolling_shutter_probability=0.0,
            exposure_step=(0.0, 0.0),
            white_balance_step=(0.0, 0.0),
        )
        degrader = ClipDegrader(config, seed=8)
        image = gradient_image()
        frames = [degrader(image) for _ in range(5)]
        for frame in frames[1:]:
            assert np.allclose(frame, frames[0], atol=1e-6)


class TestSampling:
    def test_respects_the_configured_ranges(self):
        config = DegradationConfig(jpeg_quality=(60, 62), blur_sigma=(0.3, 0.4))
        rng = np.random.default_rng(0)
        for _ in range(50):
            params = sample_clip(config, CameraProfile(), rng)
            assert 60 <= params.jpeg_quality <= 62
            assert 0.3 <= params.blur_sigma <= 0.4

    def test_probabilities_gate_optional_effects(self):
        rng = np.random.default_rng(0)
        never = DegradationConfig(motion_blur_probability=0.0, motion_blur_px=(5.0, 9.0))
        always = DegradationConfig(motion_blur_probability=1.0, motion_blur_px=(5.0, 9.0))
        assert all(
            sample_clip(never, CameraProfile(), rng).motion_blur_px == 0.0 for _ in range(20)
        )
        assert all(
            sample_clip(always, CameraProfile(), rng).motion_blur_px >= 5.0 for _ in range(20)
        )

    def test_profile_scales_the_noise(self):
        config = DegradationConfig(noise_scale=(2.0, 2.0))
        params = sample_clip(
            config, CameraProfile(shot_noise=0.01, read_noise=0.002), np.random.default_rng(0)
        )
        assert params.noise_shot == pytest.approx(0.02)
        assert params.noise_read == pytest.approx(0.004)


class TestOrdering:
    def test_noise_goes_through_the_encoder(self):
        """Noise applied before compression should be partly smoothed by it.

        Distinguishes the correct order from painting grain onto a decoded
        JPEG, which leaves the noise untouched by the encoder entirely.
        """
        image = gradient_image()
        rng = np.random.default_rng(0)
        noisy = ops.to_srgb(ops.sensor_noise(ops.to_linear(image), 0.05, 0.01, rng))

        before = ops.jpeg(noisy, 40)
        after = noisy.copy()

        def roughness(x):
            return float(np.std(np.diff(x[:, :, 1], axis=1)))

        assert roughness(before) < roughness(after)

    def test_the_pipeline_ends_with_jpeg_artefacts(self):
        config = DegradationConfig(
            jpeg_quality=(20, 20),
            noise_scale=(0.0, 0.0),
            blur_sigma=(0.0, 0.0),
            resample_scale=(1.0, 1.0),
            motion_blur_probability=0.0,
            rolling_shutter_probability=0.0,
            chromatic_probability=0.0,
            sharpen=(0.0, 0.0),
            vignette=(0.0, 0.0),
        )
        params = sample_clip(config, CameraProfile(), np.random.default_rng(0))
        flat = np.full((64, 64, 3), 0.5, dtype=np.float32)
        # Deliberately not on a multiple of 8: an edge sitting exactly on a
        # block boundary is the one case JPEG codes perfectly, and would hide
        # the ringing this test is looking for.
        flat[:, 35:] = 0.9
        out = degrade_frame(flat, params, np.random.default_rng(0))

        # The flat region either side of the edge should no longer be flat.
        assert np.ptp(out[:, 32:35, 0]) > 0.002
        # And the block containing the edge should overshoot past the input range.
        assert out[:, 32:40, 0].max() > 0.9 or out[:, 32:40, 0].min() < 0.5
