"""Tests for solving every joint's depth at once.

A webcam sees where each joint is across the image but not how far away it is.
The tree walk guesses each depth from a single bone and goes badly wrong when a
bone is seen side-on. These check that the joint solve puts limbs back where
they belong, and that it uses what SteamVR and the floor say to do it.
"""

from __future__ import annotations

import numpy as np
import pytest

from bodytracker.geom import camera_from_play, play_from_camera
from bodytracker.geom.spaces import intrinsics_from_fov
from bodytracker.lift import AnchoredLifter, LiftContext
from bodytracker.lift.kinematics import backproject_rays, bone_lengths_from_height
from bodytracker.lift.refine import refine_depths
from bodytracker.skeleton import (
    DEPTH_COPY_JOINTS,
    HEAD,
    HIP,
    LEFT_ANKLE,
    LEFT_BIG_TOE,
    LEFT_HEEL,
    LEFT_KNEE,
    LEFT_SMALL_TOE,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_BIG_TOE,
    RIGHT_HEEL,
    RIGHT_SMALL_TOE,
    RIGHT_WRIST,
)
from bodytracker.tools.scene import build_scene

INTRINSICS = intrinsics_from_fov(640, 480, 60.0)
HEIGHT = 1.75
SOLVED = [j for j in range(26) if j not in DEPTH_COPY_JOINTS]
LEFT_LEG = [LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL, LEFT_BIG_TOE, LEFT_SMALL_TOE]
FEET = [LEFT_HEEL, RIGHT_HEEL, LEFT_BIG_TOE, RIGHT_BIG_TOE, LEFT_SMALL_TOE, RIGHT_SMALL_TOE]


@pytest.fixture(scope="module")
def scene():
    return build_scene(intrinsics=INTRINSICS, height_m=HEIGHT, frames=40, motion="in_place")


class Frame:
    """One scene frame, seen the way the solver sees it."""

    def __init__(self, scene, index: int) -> None:
        frame = scene.frames[index]
        self.scene = scene
        self.camera_xyz = camera_from_play(frame.play_xyz, scene.rotation_cw, scene.translation_cw)
        self.truth = self.camera_xyz[:, 2].copy()
        self.rays = backproject_rays(np.asarray(frame.keypoints.xy, dtype=np.float64), INTRINSICS)
        self.up = scene.rotation_cw @ np.array([0.0, 1.0, 0.0])
        self.anchors = {
            joint: float(self.truth[joint]) for joint in (HEAD, LEFT_WRIST, RIGHT_WRIST)
        }

    def solve(self, start: np.ndarray, **kwargs) -> np.ndarray:
        kwargs.setdefault("anchor_depths", self.anchors)
        return refine_depths(
            self.rays,
            start,
            bone_lengths_from_height(HEIGHT),
            up_camera=self.up,
            camera_height_m=float(self.scene.camera_position[1]),
            height_m=HEIGHT,
            **kwargs,
        )

    def points(self, depths: np.ndarray) -> np.ndarray:
        return self.rays * depths[:, None]

    def error(self, depths: np.ndarray, joints) -> np.ndarray:
        return np.linalg.norm(self.points(depths)[joints] - self.camera_xyz[joints], axis=1)

    def heights(self, depths: np.ndarray) -> np.ndarray:
        play = play_from_camera(
            self.points(depths), self.scene.rotation_cw, self.scene.translation_cw
        )
        return play[:, 1]


class TestSolve:
    def test_leaves_a_correct_skeleton_where_it_is(self, scene):
        frame = Frame(scene, 0)
        solved = frame.solve(frame.truth)
        assert frame.error(solved, SOLVED).max() < 0.02

    def test_brings_a_leg_on_the_wrong_depth_root_back(self, scene):
        """The walk's failure: a leg tipped toward the camera, held there."""
        frame = Frame(scene, 10)
        start = frame.truth.copy()
        start[LEFT_LEG] -= 0.35
        assert frame.error(start, [LEFT_ANKLE])[0] > 0.3
        solved = frame.solve(start)
        assert frame.error(solved, LEFT_LEG).max() < 0.05

    def test_moves_the_body_to_the_headset(self, scene):
        """The headset's depth wins, and the bones bring the rest along."""
        frame = Frame(scene, 5)
        start = frame.truth + 0.2
        solved = frame.solve(start)
        assert solved[HEAD] == pytest.approx(frame.truth[HEAD], abs=0.02)
        assert abs(solved[HIP] - frame.truth[HIP]) < 0.05

    def test_keeps_the_feet_out_of_the_floor(self, scene):
        frame = Frame(scene, 0)
        start = frame.truth.copy()
        start[FEET + [LEFT_ANKLE, RIGHT_ANKLE]] += 0.8
        assert frame.heights(start)[FEET].min() < -0.2
        solved = frame.solve(start)
        assert frame.heights(solved)[FEET].min() > -0.04

    def test_face_joints_follow_the_head(self, scene):
        frame = Frame(scene, 0)
        solved = frame.solve(frame.truth + 0.1)
        for joint in DEPTH_COPY_JOINTS:
            assert solved[joint] == solved[HEAD]

    def test_works_without_anchors_or_history(self, scene):
        frame = Frame(scene, 0)
        solved = frame.solve(frame.truth, anchor_depths={}, previous=None)
        assert frame.error(solved, SOLVED).max() < 0.05

    def test_ignores_a_previous_frame_with_gaps(self, scene):
        frame = Frame(scene, 0)
        previous = frame.truth.copy()
        previous[LEFT_ANKLE] = np.nan
        solved = frame.solve(frame.truth, previous=previous)
        assert np.isfinite(solved).all()


def foot_errors(scene, lifter) -> np.ndarray:
    context = LiftContext(
        intrinsics=INTRINSICS,
        bone_lengths=bone_lengths_from_height(HEIGHT),
        height_m=HEIGHT,
        rotation_cw=scene.rotation_cw,
        translation_cw=scene.translation_cw,
    )
    errors = []
    for frame in scene.frames:
        context.vr = frame.vr
        skeleton = lifter(frame.keypoints, context)
        solved = np.asarray(skeleton.xyz, dtype=np.float64)
        errors.extend(
            np.linalg.norm(
                solved[[LEFT_ANKLE, RIGHT_ANKLE]] - frame.play_xyz[[LEFT_ANKLE, RIGHT_ANKLE]],
                axis=1,
            )
        )
    return np.asarray(errors)


class TestInTheLifter:
    def test_keeps_feet_near_the_truth_on_a_noisy_webcam(self):
        """Two pixels of noise at 640x480 is a good webcam on a good day.

        That alone is enough to tip the walk's feet tens of centimetres
        toward or away from the camera.
        """
        noisy = build_scene(
            intrinsics=INTRINSICS, height_m=HEIGHT, frames=150, motion="in_place", noise_px=2.0
        )
        refined = foot_errors(noisy, AnchoredLifter(refine=True))
        walked = foot_errors(noisy, AnchoredLifter(refine=False))
        assert np.percentile(refined, 90) < 0.10
        assert np.percentile(refined, 90) < np.percentile(walked, 90) / 2.0
