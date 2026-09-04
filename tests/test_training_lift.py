from __future__ import annotations

import sys
from types import SimpleNamespace

import onnx
import torch

from bodytracker.geom import intrinsics_from_fov
from bodytracker.lift.base import LiftContext
from bodytracker.lift.neural import NeuralLifter
from bodytracker.tools.scene import build_scene
from training.bedlam.samples import from_scene, windows
from training.export_lift import export
from training.lift import (
    Batch,
    ShardDataset,
    TemporalLifter,
    centre_target,
    collate,
    features,
    loss,
)


def _batch() -> Batch:
    scene = build_scene(intrinsics=intrinsics_from_fov(1280, 720, 70.0), frames=60, seed=4)
    sample = windows(from_scene(scene), window_s=0.9, stride_s=0.9)[0]
    item = {
        "xy": torch.tensor(sample.xy, dtype=torch.float32),
        "scores": torch.tensor(sample.scores, dtype=torch.float32),
        "xyz_play": torch.tensor(sample.xyz_play, dtype=torch.float32),
        "device_pos": torch.tensor(sample.device_pos, dtype=torch.float32),
        "device_rot": torch.tensor(sample.device_rot, dtype=torch.float32),
        "device_valid": torch.tensor(sample.device_valid),
        "intrinsics": torch.tensor(sample.intrinsics, dtype=torch.float32),
    }
    return collate([item, item])


def test_features_and_model_shapes():
    batch = _batch()
    encoded, origin = features(batch)
    assert encoded.shape == (2, 27, 117)
    assert origin.shape == (2, 3)
    output = TemporalLifter(hidden_size=24, layers=1)(batch)
    assert output.shape == (2, 26, 3)
    assert torch.isfinite(output).all()


def test_loss_prefers_target():
    batch = _batch()
    target = centre_target(batch)
    assert loss(target, target) < loss(torch.zeros_like(target), target)


def test_shard_dataset_rejects_empty_input():
    try:
        ShardDataset([])
    except ValueError as exc:
        assert "at least one shard" in str(exc)
    else:
        raise AssertionError("empty shard list should fail")


def test_exported_checkpoint_is_valid_onnx(tmp_path) -> None:
    model = TemporalLifter(hidden_size=16)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "hidden_size": 16}, checkpoint)
    output = tmp_path / "lifter.onnx"
    export(checkpoint, output, window=5)
    onnx.checker.check_model(onnx.load(output))


def test_neural_lifter_pads_history_and_returns_skeleton(tmp_path, monkeypatch) -> None:
    captured = {}

    class Session:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, _names, feeds):
            captured.update(feeds)
            return [torch.zeros(1, 26, 3).numpy()]

        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=Session))
    model_path = tmp_path / "lifter.onnx"
    model_path.touch()
    scene = build_scene(intrinsics=intrinsics_from_fov(1280, 720, 70.0), frames=3, seed=8)
    frame = scene.frames[0]
    lifter = NeuralLifter(model_path, window=5)
    skeleton = lifter(frame.keypoints, LiftContext(scene.intrinsics, {}, vr=frame.vr))
    assert skeleton is not None
    assert skeleton.xyz.shape == (26, 3)
    assert captured["xy"].shape == (1, 5, 26, 2)
    assert lifter.diagnostics()["neural_window_frames"] == 1
