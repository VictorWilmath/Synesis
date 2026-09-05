"""Unit tests for raw BEDLAM SMPL-X motion validation and pose slicing."""

from __future__ import annotations

import numpy as np
import pytest

from training.bedlam.smplx import SMPLX_POSE_SIZE, _model_directory, joints_from_motion, split_pose


def test_split_pose_uses_the_official_smplx_ranges():
    pose = np.arange(2 * SMPLX_POSE_SIZE, dtype=np.float32).reshape(2, SMPLX_POSE_SIZE)
    parts = split_pose(pose)
    assert parts["global_orient"].shape == (2, 3)
    assert parts["body_pose"].shape == (2, 63)
    assert parts["jaw_pose"].shape == (2, 3)
    assert parts["left_hand_pose"].shape == (2, 45)
    assert parts["right_hand_pose"].shape == (2, 45)
    assert parts["left_hand_pose"][0, 0] == 75
    assert parts["right_hand_pose"][0, -1] == 164


def test_split_pose_rejects_a_non_smplx_vector():
    with pytest.raises(ValueError, match="165"):
        split_pose(np.zeros((3, 10)))


def test_model_directory_accepts_the_extracted_archive_root(tmp_path):
    archive_root = tmp_path / "package"
    model = archive_root / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
    model.parent.mkdir(parents=True)
    model.touch()
    assert _model_directory(archive_root) == archive_root / "models"
    assert _model_directory(archive_root / "models") == archive_root / "models"


def test_raw_motion_reports_missing_fields_before_loading_a_model(tmp_path):
    path = tmp_path / "motion_seq.npz"
    np.savez(path, poses=np.zeros((2, SMPLX_POSE_SIZE), dtype=np.float32))
    with pytest.raises(KeyError, match="betas"):
        joints_from_motion(path, tmp_path)


def test_raw_motion_rejects_bad_translation_shape(tmp_path):
    path = tmp_path / "motion_seq.npz"
    np.savez(
        path,
        poses=np.zeros((2, SMPLX_POSE_SIZE), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        trans=np.zeros((2, 2), dtype=np.float32),
        gender=np.asarray("female"),
    )
    with pytest.raises(ValueError, match="trans shaped"):
        joints_from_motion(path, tmp_path)
