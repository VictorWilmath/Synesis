"""Tests for the BEDLAM coordinate conversions.

Written against the conventions rather than against the implementation: each
test states a physical fact (up is up, the camera looks along +z, a taller
camera sees the subject lower in frame) and checks the code agrees. A test that
merely reproduced the same matrix algebra twice would pass just as happily with
a sign error in both copies.
"""

from __future__ import annotations

import numpy as np
import pytest

from bodytracker.geom.spaces import play_from_camera
from training.bedlam import coords


def cam_ext_from(pitch_deg=0.0, roll_deg=0.0, translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    """A BEDLAM-style 4x4 for a camera at a given orientation."""
    matrix = np.eye(4)
    matrix[:3, :3] = coords.camera_rotation(pitch_deg=pitch_deg, roll_deg=roll_deg)
    matrix[:3, 3] = translation
    return matrix


class TestUnrealConversion:
    def test_maps_the_axes_the_way_bedlam_does(self):
        """cv_x = unreal_y, cv_y = -unreal_z, cv_z = unreal_x."""
        got = coords.unreal_to_opencv(np.array([[1.0, 2.0, 3.0]]))
        assert np.allclose(got, [[2.0, -3.0, 1.0]])

    def test_unreal_up_becomes_camera_up(self):
        """Unreal +z is up; OpenCV up is -y."""
        assert np.allclose(coords.unreal_to_opencv(np.array([[0.0, 0.0, 1.0]])), [[0, -1, 0]])

    def test_unreal_forward_becomes_camera_forward(self):
        assert np.allclose(coords.unreal_to_opencv(np.array([[1.0, 0.0, 0.0]])), [[0, 0, 1]])

    def test_flips_handedness_because_unreal_is_left_handed(self):
        """Determinant -1 is required here, not a bug.

        Unreal's world is left-handed and OpenCV's camera frame is
        right-handed, so describing the same scene in both demands an
        orientation-reversing map. Nothing is mirrored: the scene is unchanged
        and only the description of it flips.
        """
        basis = coords.unreal_to_opencv(np.eye(3))
        assert np.linalg.det(basis) == pytest.approx(-1.0)

    def test_opencv_axes_come_out_right_handed(self):
        """After conversion, right × down = forward, which is OpenCV's convention.

        Unreal's labelled axes as numpy arrays are a right-handed triple under
        the ordinary cross product; the left-handedness of Unreal lives in how
        yaw is applied, not in the stored basis. What this test pins down is
        the converted frame, which is the one the rest of the pipeline uses.
        """
        forward, right, up = np.eye(3)
        converted = coords.unreal_to_opencv(np.stack([forward, right, up]))
        cv_forward, cv_right, cv_up = converted
        cv_down = -cv_up
        assert np.dot(np.cross(cv_right, cv_down), cv_forward) == pytest.approx(1.0)

    def test_preserves_lengths(self):
        points = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 0.0]])
        assert np.allclose(
            np.linalg.norm(coords.unreal_to_opencv(points), axis=1),
            np.linalg.norm(points, axis=1),
        )

    def test_keeps_the_shape_of_a_single_point(self):
        assert coords.unreal_to_opencv(np.array([1.0, 2.0, 3.0])).shape == (3,)


class TestIntrinsics:
    def test_matches_the_documented_focal_conversion(self):
        """A 28mm lens on BEDLAM's 36mm filmback at 1280 wide."""
        assert coords.focal_mm_to_px(28.0, 36.0, 1280) == pytest.approx(995.56, abs=0.01)

    def test_pixels_come_out_square(self):
        """Both axes divide the same filmback, so fx must equal fy."""
        matrix = coords.intrinsics(36.905)
        assert matrix[0, 0] == pytest.approx(matrix[1, 1])

    def test_the_principal_point_is_the_image_centre(self):
        matrix = coords.intrinsics(28.0)
        assert (matrix[0, 2], matrix[1, 2]) == (640.0, 360.0)

    def test_agrees_with_the_recorded_horizontal_fov(self):
        """The CSV carries both focal_length and hfov; they must be consistent."""
        for focal_mm, hfov_deg in [(36.905, 52.0), (28.0, 65.470451)]:
            matrix = coords.intrinsics(focal_mm)
            recovered = 2 * np.degrees(np.arctan((coords.IMAGE_WIDTH / 2) / matrix[0, 0]))
            assert recovered == pytest.approx(hfov_deg, abs=0.01)

    def test_a_rotated_scene_swaps_the_filmback(self):
        matrix = coords.intrinsics(
            28.0, width=720, height=1280, sensor_width_mm=20.25, sensor_height_mm=36.0
        )
        assert matrix[0, 0] == pytest.approx(matrix[1, 1])
        assert (matrix[0, 2], matrix[1, 2]) == (360.0, 640.0)


class TestProjection:
    def test_a_point_on_the_axis_lands_at_the_centre(self):
        matrix = coords.intrinsics(28.0)
        assert np.allclose(coords.project(np.array([[0.0, 0.0, 3.0]]), matrix)[0, :2], [640, 360])

    def test_further_away_is_smaller(self):
        matrix = coords.intrinsics(28.0)
        near = coords.project(np.array([[0.3, 0.0, 2.0]]), matrix)[0, 0]
        far = coords.project(np.array([[0.3, 0.0, 4.0]]), matrix)[0, 0]
        assert 640 < far < near

    def test_camera_down_is_down_in_the_image(self):
        matrix = coords.intrinsics(28.0)
        assert coords.project(np.array([[0.0, 0.5, 3.0]]), matrix)[0, 1] > 360

    def test_returns_homogeneous_rows(self):
        matrix = coords.intrinsics(28.0)
        assert coords.project(np.array([[0.2, 0.1, 3.0]]), matrix)[0, 2] == pytest.approx(1.0)

    def test_flags_points_behind_the_lens(self):
        points = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -0.5], [0.0, 0.0, 0.05]])
        assert list(coords.behind_camera(points)) == [False, True, True]

    def test_a_point_just_behind_the_camera_would_otherwise_look_plausible(self):
        """Why behind_camera exists: the projection alone gives no warning."""
        matrix = coords.intrinsics(28.0)
        pixel = coords.project(np.array([[0.1, 0.1, -2.0]]), matrix)[0, :2]
        assert 0 <= pixel[0] < 1280 and 0 <= pixel[1] < 720


class TestCameraRotation:
    def test_a_level_camera_is_the_identity(self):
        assert np.allclose(coords.camera_rotation(0.0, 0.0), np.eye(3))

    def test_is_a_rotation(self):
        rotation = coords.camera_rotation(pitch_deg=-12.0, roll_deg=3.0, yaw_deg=40.0)
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)

    def test_pitch_tilts_around_the_camera_x_axis(self):
        rotation = coords.camera_rotation(pitch_deg=20.0, roll_deg=0.0)
        assert np.allclose(rotation @ np.array([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-12)

    def test_roll_turns_around_the_view_direction(self):
        rotation = coords.camera_rotation(pitch_deg=0.0, roll_deg=15.0)
        assert np.allclose(rotation @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-12)


class TestUpVector:
    def test_a_level_camera_sees_up_as_negative_y(self):
        """OpenCV camera space has y pointing down, so up is -y."""
        assert np.allclose(coords.up_in_camera(cam_ext_from()), [0.0, -1.0, 0.0])

    def test_is_always_a_unit_vector(self):
        for pitch in (-40.0, -10.0, 0.0, 25.0):
            for roll in (-8.0, 0.0, 8.0):
                up = coords.up_in_camera(cam_ext_from(pitch, roll))
                assert np.linalg.norm(up) == pytest.approx(1.0)

    def test_a_camera_looking_down_sees_up_leaning_backwards(self):
        """Negative pitch is nose-down in Unreal, which is where a webcam sits.

        Tilt the lens toward the floor and the world's up direction leans back
        over the camera, so it picks up a negative z component. Getting this
        sign backwards would hand the lifter a gravity hint that is wrong by
        twice the camera's tilt, which is worse than giving it none.
        """
        up = coords.up_in_camera(cam_ext_from(pitch_deg=-20.0))
        assert up[1] < 0.0
        assert up[2] < 0.0

    def test_a_camera_looking_up_leans_the_other_way(self):
        assert coords.up_in_camera(cam_ext_from(pitch_deg=20.0))[2] > 0.0

    def test_matches_the_pitch_angle(self):
        for pitch in (-30.0, -10.0, 15.0):
            up = coords.up_in_camera(cam_ext_from(pitch_deg=pitch))
            recovered = np.degrees(np.arctan2(up[2], -up[1]))
            assert recovered == pytest.approx(pitch, abs=1e-9)


class TestPlaySpace:
    def test_the_world_flip_is_a_rotation_not_a_mirror(self):
        assert np.linalg.det(coords.WORLD_FROM_BEDLAM) == pytest.approx(1.0)

    def test_the_camera_ends_up_above_the_origin(self):
        rotation_cw, translation_cw = coords.play_space_transform(cam_ext_from(), 1.2)
        camera_in_play = play_from_camera(np.zeros(3), rotation_cw, translation_cw)
        assert np.allclose(camera_in_play, [0.0, 1.2, 0.0], atol=1e-12)

    @pytest.mark.parametrize("pitch", [-25.0, -10.0, 0.0, 10.0])
    def test_and_still_does_when_the_camera_is_tilted(self, pitch):
        rotation_cw, translation_cw = coords.play_space_transform(cam_ext_from(pitch), 1.4)
        camera_in_play = play_from_camera(np.zeros(3), rotation_cw, translation_cw)
        assert np.allclose(camera_in_play, [0.0, 1.4, 0.0], atol=1e-12)

    def test_the_floor_lands_at_zero_height(self):
        """A point on the floor in front of the camera should have play y of 0."""
        cam_ext = cam_ext_from(pitch_deg=-15.0)
        up = coords.up_in_camera(cam_ext)
        height = 1.3
        # A floor point: start below the camera by `height` along up, then walk
        # forwards along the floor.
        forward = np.array([0.0, 0.0, 1.0])
        along_floor = forward - np.dot(forward, up) * up
        along_floor /= np.linalg.norm(along_floor)
        floor_point = -height * up + 2.0 * along_floor

        rotation_cw, translation_cw = coords.play_space_transform(cam_ext, height)
        assert play_from_camera(floor_point, rotation_cw, translation_cw)[1] == pytest.approx(
            0.0, abs=1e-9
        )

    def test_play_space_up_is_positive_y(self):
        cam_ext = cam_ext_from(pitch_deg=-12.0)
        rotation_cw, translation_cw = coords.play_space_transform(cam_ext, 1.2)
        up = coords.up_in_camera(cam_ext)
        raised = play_from_camera(up, rotation_cw, translation_cw)
        base = play_from_camera(np.zeros(3), rotation_cw, translation_cw)
        assert raised[1] - base[1] == pytest.approx(1.0)

    def test_preserves_distances(self):
        rotation_cw, translation_cw = coords.play_space_transform(cam_ext_from(-15.0), 1.3)
        a, b = np.array([0.2, 0.4, 2.0]), np.array([-0.3, 0.1, 2.6])
        moved = play_from_camera(np.stack([a, b]), rotation_cw, translation_cw)
        assert np.linalg.norm(moved[0] - moved[1]) == pytest.approx(np.linalg.norm(a - b))

    def test_does_not_mirror_the_subject(self):
        """Left must stay left. This is the bug that produces a broken avatar."""
        rotation_cw, _ = coords.play_space_transform(cam_ext_from(-15.0), 1.3)
        assert np.linalg.det(rotation_cw) == pytest.approx(1.0)


class TestCameraHeight:
    def test_measures_a_level_camera(self):
        cam_ext = cam_ext_from()
        feet = np.array([[0.1, 1.5, 2.5], [-0.1, 1.5, 2.6]])
        assert coords.camera_height(feet, cam_ext) == pytest.approx(1.5)

    def test_measures_a_tilted_camera(self):
        cam_ext = cam_ext_from(pitch_deg=-20.0)
        up = coords.up_in_camera(cam_ext)
        forward_on_floor = np.array([0.0, 0.0, 1.0])
        forward_on_floor -= np.dot(forward_on_floor, up) * up
        forward_on_floor /= np.linalg.norm(forward_on_floor)
        feet = -1.25 * up + np.array([0.0, 0.0, 0.0]) + 2.0 * forward_on_floor
        assert coords.camera_height(feet, cam_ext) == pytest.approx(1.25)

    def test_a_foot_sunk_through_the_floor_does_not_win(self):
        """Renders occasionally put a toe under the ground; it must not set the floor."""
        cam_ext = cam_ext_from()
        feet = np.array([[0.0, 1.5, 2.5], [0.0, 1.7, 2.5]])
        assert coords.camera_height(feet, cam_ext) == pytest.approx(1.5)


class TestCameraCsv:
    HEADER = "name,x,y,z,yaw,pitch,roll,focal_length,sensor_width,sensor_height,hfov\n"
    # A real row, quoted from bedlam_render's unreal_coordinate_system.md.
    ROW = "seq_000000_0000.png,5.058579,-9.743741,168.953232,1.468127,-2.905068,2.813995,36.905,36,20.25,52\n"

    def test_reads_a_real_row(self, tmp_path):
        path = tmp_path / "seq_000000_camera.csv"
        path.write_text(self.HEADER + self.ROW)
        frame = coords.read_camera_csv(path)[0]
        assert frame.name == "seq_000000_0000.png"
        assert frame.focal_mm == pytest.approx(36.905)
        assert frame.hfov_deg == pytest.approx(52.0)
        assert np.allclose(frame.position_cm, [5.058579, -9.743741, 168.953232])

    def test_converts_the_position_to_metres_in_camera_axes(self, tmp_path):
        path = tmp_path / "c.csv"
        path.write_text(self.HEADER + self.ROW)
        frame = coords.read_camera_csv(path)[0]
        # Unreal z of 168.95 cm is the height, which becomes -1.69 on the
        # camera's downward y.
        assert frame.position_m[1] == pytest.approx(-1.68953, abs=1e-4)

    def test_a_recorded_focal_length_matches_its_recorded_fov(self, tmp_path):
        path = tmp_path / "c.csv"
        path.write_text(self.HEADER + self.ROW)
        frame = coords.read_camera_csv(path)[0]
        matrix = coords.intrinsics(frame.focal_mm)
        hfov = 2 * np.degrees(np.arctan((coords.IMAGE_WIDTH / 2) / matrix[0, 0]))
        assert hfov == pytest.approx(frame.hfov_deg, abs=0.05)

    def test_detects_a_static_camera(self, tmp_path):
        path = tmp_path / "c.csv"
        path.write_text(self.HEADER + self.ROW * 5)
        assert coords.camera_is_static(coords.read_camera_csv(path))

    def test_detects_a_camera_that_drifts(self, tmp_path):
        path = tmp_path / "c.csv"
        moved = self.ROW.replace("5.058579", "35.058579")
        path.write_text(self.HEADER + self.ROW * 4 + moved)
        assert not coords.camera_is_static(coords.read_camera_csv(path))

    def test_detects_camera_shake_in_the_angles(self, tmp_path):
        """BEDLAM 2.0 layers Perlin shake on some shots regardless of the name."""
        path = tmp_path / "c.csv"
        shaken = self.ROW.replace(",1.468127,", ",4.468127,")
        path.write_text(self.HEADER + self.ROW * 4 + shaken)
        assert not coords.camera_is_static(coords.read_camera_csv(path))


class TestAgainstTheRealCameraRow:
    """One end-to-end check with real recorded numbers.

    Every convention above is a sign that could be flipped, and unit tests of
    signs tend to encode the same misunderstanding twice. This puts a person of
    known height in front of the real camera from BEDLAM's own documentation
    and asks whether the picture comes out the way a picture should.
    """

    PITCH = -2.905068  # slightly nose-down, from the documented row
    FOCAL_MM = 36.905
    HEIGHT_M = 1.68953  # the row's Unreal z of 168.95 cm
    # This lens has a ~29° vertical FOV. A 1.75 m person 3 m away is taller
    # than the frame, so they stand farther back — the same reason a webcam
    # on a shelf has to sit several metres from the play space.
    DISTANCE_M = 7.0
    PERSON_M = 1.75

    def setup_method(self):
        self.cam_ext = cam_ext_from(pitch_deg=self.PITCH)
        self.matrix = coords.intrinsics(self.FOCAL_MM)
        self.up = coords.up_in_camera(self.cam_ext)
        forward = np.array([0.0, 0.0, 1.0])
        forward = forward - np.dot(forward, self.up) * self.up
        forward /= np.linalg.norm(forward)
        self.feet = -self.HEIGHT_M * self.up + self.DISTANCE_M * forward
        self.head = self.feet + self.PERSON_M * self.up

    def test_the_camera_really_is_looking_slightly_down(self):
        """A camera 1.69 m up watching people would be."""
        recovered = np.degrees(np.arctan2(self.up[2], -self.up[1]))
        assert recovered == pytest.approx(self.PITCH, abs=1e-6)
        assert recovered < 0.0

    def test_the_whole_person_is_in_frame(self):
        pixels = coords.project(np.stack([self.feet, self.head]), self.matrix)[:, :2]
        assert np.all(pixels[:, 0] > 0) and np.all(pixels[:, 0] < coords.IMAGE_WIDTH)
        assert np.all(pixels[:, 1] > 0) and np.all(pixels[:, 1] < coords.IMAGE_HEIGHT)

    def test_the_head_is_above_the_feet_in_the_image(self):
        pixels = coords.project(np.stack([self.feet, self.head]), self.matrix)[:, :2]
        assert pixels[1][1] < pixels[0][1]

    def test_the_person_is_about_the_right_size_on_screen(self):
        """A 1.75 m person at 7 m through a 52-degree lens fills a bit of the frame."""
        pixels = coords.project(np.stack([self.feet, self.head]), self.matrix)[:, :2]
        span = abs(pixels[0][1] - pixels[1][1])
        expected = self.PERSON_M * self.matrix[1, 1] / self.DISTANCE_M
        assert span == pytest.approx(expected, rel=0.05)

    def test_the_floor_and_the_head_land_where_play_space_says(self):
        rotation_cw, translation_cw = coords.play_space_transform(self.cam_ext, self.HEIGHT_M)
        in_play = play_from_camera(np.stack([self.feet, self.head]), rotation_cw, translation_cw)
        assert in_play[0][1] == pytest.approx(0.0, abs=1e-9)
        assert in_play[1][1] == pytest.approx(self.PERSON_M, abs=1e-9)

    def test_the_camera_height_is_recovered_from_the_feet(self):
        assert coords.camera_height(self.feet, self.cam_ext) == pytest.approx(self.HEIGHT_M)
