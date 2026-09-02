"""Tests for BEDLAM scene selection.

The scene names below are real, taken from ``bedlam_scene_names.csv`` and
``bedlam2_scene_names.csv`` in the BEDLAM repository. Testing against invented
names would prove only that the parser agrees with itself, and the whole
difficulty here is that the real names are not quite regular.
"""

from __future__ import annotations

import pytest

from training.bedlam import Criteria, parse, read_scene_list, scan_download, select, try_parse
from training.bedlam.subset import rejected, report, score

# Real BEDLAM 1.0 scene folders, with the fps suffix they carry once downloaded.
BEDLAM1 = [
    "20221010_3_1000_batch01hand_6fps",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_6fps",
    "20221011_1_250_batch01hand_closeup_suburb_a_6fps",
    "20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps",
    "20221013_3_250_batch01hand_orbit_bigOffice_6fps",
    "20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps",
    "20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps",
    "20221019_3-8_1000_highbmihand_static_suburb_d_6fps",
    "20221020-3-8_250_highbmihand_zoom_highSchoolGym_a_6fps",
    "20221024_10_100_batch01handhair_zoom_suburb_d_30fps",
]

# Real BEDLAM 2.0 scene folders.
BEDLAM2 = [
    "20240701_1_250_ai0901_lookat",
    "20240701_1_250_ai0901_static",
    "20240702_1_250_batch01hand_orbit_zoom_citysample",
    "20240703_3-10_500_batch01hand_dolly_busstation",
    "20240704_1_250_batch01hand_vcamego_yogastudio",
    "20240705_1_250_batch01hand_portrait_rome",
]


class TestParsing:
    def test_reads_the_documented_form(self):
        scene = parse("20221010_3-10_500_batch01hand_zoom_suburb_d_6fps")
        assert scene.date == "20221010"
        assert (scene.bodies_min, scene.bodies_max) == (3, 10)
        assert scene.sequences == 500
        assert scene.body_batch == "batch01hand"
        assert scene.fps == 6.0
        assert scene.scene == "suburb_d"

    def test_reads_a_fixed_body_count(self):
        scene = parse("20221011_1_250_batch01hand_closeup_suburb_a_6fps")
        assert (scene.bodies_min, scene.bodies_max) == (1, 1)
        assert scene.single_subject

    def test_survives_the_scene_with_the_body_count_stuck_to_the_date(self):
        """20221020-3-8_250_... is genuinely shaped like this in the dataset."""
        scene = parse("20221020-3-8_250_highbmihand_zoom_highSchoolGym_a_6fps")
        assert scene.date == "20221020"
        assert (scene.bodies_min, scene.bodies_max) == (3, 8)
        assert scene.sequences == 250

    def test_handles_a_name_with_no_fps_suffix(self):
        assert parse("20240701_1_250_ai0901_static").fps is None

    def test_every_real_name_parses(self):
        for folder in BEDLAM1 + BEDLAM2:
            assert try_parse(folder) is not None, folder

    def test_rejects_something_that_is_not_a_scene(self):
        assert try_parse("all_npz_12_training.zip") is None
        assert try_parse("images") is None
        with pytest.raises(ValueError):
            parse("readme.md")


class TestCameraClassification:
    @pytest.mark.parametrize(
        "folder",
        [
            "20221019_3-8_1000_highbmihand_static_suburb_d_6fps",
            "20240701_1_250_ai0901_static",
            # No camera token at all: the HDRI backdrop renders, shot fixed.
            "20221010_3_1000_batch01hand_6fps",
        ],
    )
    def test_static_cameras(self, folder):
        assert parse(folder).camera_static

    @pytest.mark.parametrize(
        "folder",
        [
            "20221010_3-10_500_batch01hand_zoom_suburb_d_6fps",
            "20221013_3_250_batch01hand_orbit_bigOffice_6fps",
            "20240701_1_250_ai0901_lookat",
            "20240703_3-10_500_batch01hand_dolly_busstation",
            "20240704_1_250_batch01hand_vcamego_yogastudio",
        ],
    )
    def test_moving_cameras(self, folder):
        assert not parse(folder).camera_static

    def test_a_combined_motion_still_counts_as_moving(self):
        assert parse("20240702_1_250_batch01hand_orbit_zoom_citysample").camera_moves

    def test_reads_the_pitch(self):
        assert parse("20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps").pitch_deg == 52.0

    def test_a_pitched_camera_is_still_static(self):
        """It is aimed steeply, not moving; a shelf-mounted webcam is the same."""
        assert parse("20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps").camera_static


class TestFraming:
    def test_closeup_is_not_full_body(self):
        scene = parse("20221011_1_250_batch01hand_closeup_suburb_a_6fps")
        assert not scene.full_body_framing
        assert scene.framing == "closeup"

    @pytest.mark.parametrize("folder", ["20221011_1_250_batch01hand_closeup_suburb_a_6fps",
                                        "20240705_1_250_batch01hand_portrait_rome"])
    def test_rotated_scenes_are_flagged(self, folder):
        """These PNGs are stored rotated 90 degrees and must be turned back."""
        assert parse(folder).rotated

    def test_an_ordinary_scene_is_not_rotated(self):
        assert not parse("20221010_3_1000_batch01hand_6fps").rotated


class TestEnvironment:
    @pytest.mark.parametrize(
        ("folder", "expected"),
        [
            ("20221011_1_250_batch01hand_closeup_suburb_a_6fps", True),
            ("20221013_3_250_batch01hand_orbit_bigOffice_6fps", True),
            ("20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps", True),
            ("20221010_3-10_500_batch01hand_zoom_suburb_d_6fps", False),
            ("20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps", False),
        ],
    )
    def test_indoor_versus_outdoor(self, folder, expected):
        assert parse(folder).indoor is expected

    def test_suburb_d_is_the_street_not_a_room(self):
        """The suburb scenes are a, b, c indoors and d outside; easy to get wrong."""
        assert parse("20221010_3-10_500_batch01hand_zoom_suburb_d_6fps").indoor is False
        assert parse("20221011_1_250_batch01hand_closeup_suburb_a_6fps").indoor is True

    def test_an_unnamed_environment_is_unknown_rather_than_outdoor(self):
        assert parse("20221010_3_1000_batch01hand_6fps").indoor is None


class TestSelection:
    def test_picks_only_static_full_body_scenes(self):
        chosen = select(BEDLAM1 + BEDLAM2)
        for item in chosen:
            assert item.scene.camera_static
            assert item.scene.full_body_framing

    def test_prefers_single_subject(self):
        chosen = select(BEDLAM1 + BEDLAM2)
        assert chosen[0].scene.single_subject

    def test_crowds_are_kept_rather_than_dropped(self):
        """Occlusion in a crowd is training signal, not contamination."""
        chosen = select(BEDLAM1 + BEDLAM2)
        assert any(not item.scene.single_subject for item in chosen)

    def test_requiring_single_subject_can_be_made_strict(self):
        criteria = Criteria(max_bodies=1)
        chosen = select(BEDLAM1 + BEDLAM2, criteria)
        assert chosen
        assert all(item.scene.single_subject for item in chosen)

    def test_the_steeply_pitched_stadium_is_excluded_by_default(self):
        chosen = {item.scene.raw for item in select(BEDLAM1)}
        assert "20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps" not in chosen

    def test_but_can_be_allowed_in(self):
        criteria = Criteria(max_pitch_deg=60.0)
        chosen = {item.scene.raw for item in select(BEDLAM1, criteria)}
        assert "20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps" in chosen

    def test_respects_a_limit(self):
        assert len(select(BEDLAM1 + BEDLAM2, limit=2)) == 2

    def test_ranking_is_stable(self):
        first = [s.scene.raw for s in select(BEDLAM1 + BEDLAM2)]
        second = [s.scene.raw for s in select(list(reversed(BEDLAM1 + BEDLAM2)))]
        assert first == second

    def test_ignores_names_that_are_not_scenes(self):
        chosen = select([*BEDLAM1, "all_npz_12_training.zip", "README.md"])
        assert all(try_parse(item.scene.raw) is not None for item in chosen)

    def test_rejections_explain_themselves(self):
        reasons = {item.scene.raw: item.reasons for item in rejected(BEDLAM1 + BEDLAM2)}
        assert "camera moves" in reasons["20221013_3_250_batch01hand_orbit_bigOffice_6fps"]
        assert any(
            "closeup" in reason
            for reason in reasons["20221011_1_250_batch01hand_closeup_suburb_a_6fps"]
        )

    def test_everything_is_either_selected_or_rejected(self):
        folders = BEDLAM1 + BEDLAM2
        chosen = {item.scene.raw for item in select(folders)}
        skipped = {item.scene.raw for item in rejected(folders)}
        assert chosen | skipped == set(folders)
        assert not chosen & skipped

    def test_a_scene_score_is_never_negative(self):
        for folder in BEDLAM1 + BEDLAM2:
            assert score(parse(folder)).score >= 0.0


class TestDiscovery:
    def test_reads_a_scene_list_csv(self, tmp_path):
        path = tmp_path / "bedlam_scene_names.csv"
        path.write_text(
            "20221010_3_1000_batch01hand,static-hdri\n"
            "20221013_3_250_batch01hand_orbit_bigOffice,orbit-office\n"
            "\n"
        )
        assert read_scene_list(path) == [
            "20221010_3_1000_batch01hand",
            "20221013_3_250_batch01hand_orbit_bigOffice",
        ]

    def test_scans_a_partial_download(self, tmp_path):
        (tmp_path / "20221010_3_1000_batch01hand_6fps").mkdir()
        (tmp_path / "notes.txt").write_text("hello")
        (tmp_path / "scratch").mkdir()
        assert scan_download(tmp_path) == ["20221010_3_1000_batch01hand_6fps"]

    def test_scanning_a_missing_directory_is_empty_not_an_error(self, tmp_path):
        assert scan_download(tmp_path / "nope") == []


class TestReport:
    def test_mentions_every_selected_scene(self):
        chosen = select(BEDLAM1 + BEDLAM2)
        text = report(chosen, rejected(BEDLAM1 + BEDLAM2))
        for item in chosen:
            assert item.scene.raw in text

    def test_totals_the_sequences(self):
        chosen = select(BEDLAM1 + BEDLAM2)
        total = sum(item.scene.sequences for item in chosen)
        assert f"{total} sequences in total" in report(chosen)
