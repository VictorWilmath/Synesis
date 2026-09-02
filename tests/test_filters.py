"""Tests for temporal smoothing and dropout handling."""

from __future__ import annotations

import numpy as np
import pytest

from bodytracker.config import FilterConfig
from bodytracker.filters import (
    HoldLastValid,
    OneEuroFilter,
    RotationSmoother,
    SkeletonFilter,
    interpolate_missing,
)
from bodytracker.geom import euler_zxy_degrees_from_matrix, matrix_from_euler_zxy_degrees
from bodytracker.skeleton import HIP, LEFT_ANKLE, NUM_KEYPOINTS, RIGHT_ANKLE
from bodytracker.tools.synthetic import reference_skeleton
from bodytracker.types import Skeleton3D


class TestOneEuroFilter:
    def test_first_sample_passes_through(self):
        f = OneEuroFilter()
        x = np.array([1.0, 2.0, 3.0])
        assert np.allclose(f(x, 0.0), x)

    def test_converges_to_a_constant(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        target = np.array([5.0])
        f(np.array([0.0]), 0.0)
        for i in range(1, 200):
            out = f(target, i / 60.0)
        assert out == pytest.approx(target, abs=1e-3)

    def test_suppresses_noise_on_a_stationary_signal(self):
        rng = np.random.default_rng(0)
        f = OneEuroFilter(min_cutoff=0.5, beta=0.1)
        truth = np.array([1.0, 1.0, 1.0])

        raw_error, filtered_error = [], []
        for i in range(300):
            noisy = truth + rng.normal(scale=0.02, size=3)
            out = f(noisy, i / 60.0)
            if i > 60:
                raw_error.append(np.linalg.norm(noisy - truth))
                filtered_error.append(np.linalg.norm(out - truth))

        assert np.mean(filtered_error) < np.mean(raw_error) * 0.5

    def test_ramp_lag_matches_the_filter_time_constant(self):
        """Steady-state lag on a ramp should be speed * tau, and no worse.

        For a first-order low pass the lag settles at tau = 1/(2*pi*fc), where
        the adaptive cutoff is fc = min_cutoff + beta * speed. Pinning the
        measured lag to that prediction catches an error in the adaptation term
        that a loose bound would miss.
        """
        min_cutoff, beta, speed = 1.0, 1.0, 1.0
        f = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        for i in range(400):
            t = i / 60.0
            out = f(np.array([speed * t]), t)

        expected_lag = speed / (2 * np.pi * (min_cutoff + beta * speed))
        actual_lag = speed * (399 / 60.0) - out[0]
        assert actual_lag == pytest.approx(expected_lag, rel=0.05)

    def test_beta_reduces_lag_on_fast_motion(self):
        """The whole point of One Euro: speed should open the filter up."""

        def final_lag(beta):
            f = OneEuroFilter(min_cutoff=0.5, beta=beta)
            for i in range(120):
                t = i / 60.0
                out = f(np.array([2.0 * t]), t)
            return abs(2.0 * (119 / 60.0) - out[0])

        assert final_lag(2.0) < final_lag(0.0)

    def test_handles_repeated_timestamps(self):
        f = OneEuroFilter()
        f(np.array([1.0]), 0.0)
        first = f(np.array([2.0]), 0.1)
        repeated = f(np.array([9.0]), 0.1)
        assert np.allclose(first, repeated)

    def test_handles_backwards_timestamps(self):
        f = OneEuroFilter()
        f(np.array([1.0]), 1.0)
        out = f(np.array([2.0]), 0.5)
        assert np.all(np.isfinite(out))

    def test_works_on_a_full_skeleton_shape(self):
        f = OneEuroFilter()
        for i in range(10):
            out = f(np.zeros((NUM_KEYPOINTS, 3)), i / 30.0)
        assert out.shape == (NUM_KEYPOINTS, 3)

    def test_reset_clears_history(self):
        f = OneEuroFilter()
        for i in range(50):
            f(np.array([10.0]), i / 60.0)
        f.reset()
        assert f(np.array([0.0]), 0.0) == pytest.approx([0.0])

    def test_rejects_invalid_cutoffs(self):
        with pytest.raises(ValueError):
            OneEuroFilter(min_cutoff=0.0)


class TestSkeletonFilter:
    def test_passes_shape_and_metadata_through(self):
        f = SkeletonFilter(FilterConfig())
        skeleton = Skeleton3D(
            xyz=reference_skeleton().astype(np.float32),
            scores=np.ones(NUM_KEYPOINTS, dtype=np.float32),
            timestamp=0.0,
        )
        out = f(skeleton)
        assert out.xyz.shape == (NUM_KEYPOINTS, 3)
        assert out.space == "play"
        assert out.timestamp == 0.0

    def test_holds_low_confidence_joints(self):
        """A joint whose detection collapsed must not drag the estimate."""
        f = SkeletonFilter(FilterConfig(min_cutoff=1.0, beta=0.0), min_score=0.5)
        base = reference_skeleton()

        scores = np.ones(NUM_KEYPOINTS, dtype=np.float32)
        f(Skeleton3D(base.astype(np.float32), scores, timestamp=0.0))

        # The ankle jumps 5 m away but reports no confidence.
        corrupted = base.copy()
        corrupted[LEFT_ANKLE] += 5.0
        weak = scores.copy()
        weak[LEFT_ANKLE] = 0.01

        out = None
        for i in range(1, 30):
            out = f(Skeleton3D(corrupted.astype(np.float32), weak, timestamp=i / 30.0))
        assert np.linalg.norm(out.xyz[LEFT_ANKLE] - base[LEFT_ANKLE]) < 0.1

    def test_smooths_a_noisy_sequence(self):
        rng = np.random.default_rng(1)
        f = SkeletonFilter(FilterConfig(min_cutoff=0.5, beta=0.1))
        base = reference_skeleton()
        scores = np.ones(NUM_KEYPOINTS, dtype=np.float32)

        errors = []
        for i in range(200):
            noisy = base + rng.normal(scale=0.03, size=base.shape)
            out = f(Skeleton3D(noisy.astype(np.float32), scores, timestamp=i / 60.0))
            if i > 60:
                errors.append(np.mean(np.linalg.norm(out.xyz - base, axis=1)))
        assert np.mean(errors) < 0.03


class TestRotationSmoother:
    def test_first_value_passes_through(self):
        s = RotationSmoother(0.3)
        rot = matrix_from_euler_zxy_degrees(0, 45, 0)
        assert np.allclose(s(rot, 0.0), rot)

    def test_converges_to_the_target(self):
        s = RotationSmoother(0.5)
        s(np.eye(3), 0.0)
        target = matrix_from_euler_zxy_degrees(0, 60, 0)
        for i in range(1, 100):
            out = s(target, i / 60.0)
        assert np.allclose(out, target, atol=1e-6)

    def test_output_stays_a_rotation(self):
        s = RotationSmoother(0.25)
        rng = np.random.default_rng(2)
        for i in range(100):
            angles = rng.uniform(-180, 180, size=3)
            out = s(matrix_from_euler_zxy_degrees(*angles), i / 60.0)
            assert np.allclose(out @ out.T, np.eye(3), atol=1e-9)
            assert np.isclose(np.linalg.det(out), 1.0, atol=1e-9)

    def test_framerate_independent(self):
        """Halving the framerate must not halve the perceived smoothing."""

        def settle(rate):
            s = RotationSmoother(0.3)
            s(np.eye(3), 0.0)
            target = matrix_from_euler_zxy_degrees(0, 90, 0)
            for i in range(1, int(rate) + 1):
                out = s(target, i / rate)
            return euler_zxy_degrees_from_matrix(out)[1]

        assert settle(30.0) == pytest.approx(settle(120.0), abs=1.0)

    def test_rejects_invalid_alpha(self):
        with pytest.raises(ValueError):
            RotationSmoother(0.0)


class TestHoldLastValid:
    def test_returns_fresh_values_as_not_stale(self):
        hold = HoldLastValid(timeout=0.5)
        hold.update("hip", 42, 1.0)
        value, stale = hold.get("hip", 1.0)
        assert value == 42 and stale is False

    def test_marks_older_values_stale(self):
        hold = HoldLastValid(timeout=0.5)
        hold.update("hip", 42, 1.0)
        value, stale = hold.get("hip", 1.2)
        assert value == 42 and stale is True

    def test_expires_after_the_timeout(self):
        hold = HoldLastValid(timeout=0.5)
        hold.update("hip", 42, 1.0)
        assert hold.get("hip", 1.6) is None

    def test_missing_key(self):
        assert HoldLastValid().get("nope", 0.0) is None

    def test_reset(self):
        hold = HoldLastValid()
        hold.update("hip", 1, 0.0)
        hold.reset()
        assert hold.get("hip", 0.0) is None


class TestInterpolateMissing:
    def test_mirrors_an_occluded_limb(self):
        xyz = reference_skeleton()
        scores = np.ones(NUM_KEYPOINTS)
        scores[LEFT_ANKLE] = 0.0

        corrupted = xyz.copy()
        corrupted[LEFT_ANKLE] = [99.0, 99.0, 99.0]

        filled = interpolate_missing(corrupted, scores, 0.3)
        # The reference pose is symmetric, so the mirror should be close.
        assert np.linalg.norm(filled[LEFT_ANKLE] - xyz[LEFT_ANKLE]) < 0.05

    def test_leaves_confident_joints_alone(self):
        xyz = reference_skeleton()
        filled = interpolate_missing(xyz, np.ones(NUM_KEYPOINTS), 0.3)
        assert np.allclose(filled, xyz)

    def test_gives_up_when_both_sides_are_missing(self):
        xyz = reference_skeleton()
        scores = np.ones(NUM_KEYPOINTS)
        scores[[LEFT_ANKLE, RIGHT_ANKLE]] = 0.0
        filled = interpolate_missing(xyz, scores, 0.3)
        assert np.allclose(filled[LEFT_ANKLE], xyz[LEFT_ANKLE])

    def test_gives_up_when_the_root_is_missing(self):
        xyz = reference_skeleton()
        scores = np.ones(NUM_KEYPOINTS)
        scores[HIP] = 0.0
        scores[LEFT_ANKLE] = 0.0
        assert np.allclose(interpolate_missing(xyz, scores, 0.3), xyz)
