"""Real PNG/TGZ fixtures exercise units, splits, provenance, and metric leakage."""

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "finetune_depth", Path(__file__).resolve().parents[1] / "tools/finetune_depth.py"
)
assert SPEC is not None and SPEC.loader is not None
depth_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = depth_tool
SPEC.loader.exec_module(depth_tool)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(depth_tool, "COUNTS", {"train": 2, "val": 2, "test": 2})
    archives = tmp_path / "archives"
    archives.mkdir()
    for sequence_index, sequence in enumerate(depth_tool.SEQUENCES.values()):
        name = f"rgbd_dataset_freiburg1_{sequence}"
        with tarfile.open(archives / f"{name}.tgz", "w:gz") as archive:
            for i in range(4):
                rgb = np.full((24, 32, 3), i + sequence_index * 10, np.uint8)
                target = np.full((24, 32), 5000 + i + sequence_index * 100, np.uint16)
                target[:2, :] = 0
                for folder, stamp, array in (("rgb", i + 1.0, rgb),
                                             ("depth", i + 1.005, target)):
                    success, encoded = cv2.imencode(".png", array)
                    assert success
                    data = encoded.tobytes()
                    member = tarfile.TarInfo(f"{name}/{folder}/{stamp:.6f}.png")
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
    output = tmp_path / "dataset"
    manifest = depth_tool.prepare_from_archives(archives, output, seed=42)
    return manifest, archives


def test_prepare_is_deterministic_and_has_no_test_in_training_yaml(prepared, tmp_path):
    manifest_path, archives = prepared
    manifest = depth_tool.load_manifest(manifest_path)
    assert len(manifest.samples) == 6
    assert manifest.depth_scale == 5000
    assert {s.split for s in manifest.samples} == {"train", "val", "test"}
    assert len({s.rgb_sha256 for s in manifest.samples}) == 6
    assert len({s.depth_sha256 for s in manifest.samples}) == 6
    assert all(abs(s.rgb_timestamp - s.depth_timestamp) < 0.02 for s in manifest.samples)
    again = depth_tool.prepare_from_archives(archives, tmp_path / "again", seed=42)
    assert again.read_bytes() == manifest_path.read_bytes()
    yaml = (manifest_path.parent / "dataset.yaml").read_text()
    assert "val: images/val" in yaml
    assert "test:" not in yaml
    sample = manifest.samples[0]
    target = cv2.imread(str(manifest_path.parent / sample.depth), cv2.IMREAD_UNCHANGED)
    meters = target.astype(np.float32) / manifest.depth_scale
    assert meters[0, 0] == 0
    assert 1 <= meters[3, 3] < 1.1
    for source in manifest.sources:
        archive = archives / f"{source.sequence}.tgz"
        assert source.sha256 == depth_tool.sha256(archive)


def test_existing_output_is_preserved(prepared):
    manifest, archives = prepared
    before = manifest.read_bytes()
    with pytest.raises(FileExistsError):
        depth_tool.prepare_from_archives(archives, manifest.parent)
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("command", ["train", "evaluate"])
def test_missing_weights_fail_before_any_gpu_work(prepared, tmp_path, monkeypatch, command):
    manifest, _ = prepared

    def unexpected_gpu(*args, **kwargs):
        pytest.fail("missing checkpoint must not enter the GPU lock or model loader")

    monkeypatch.setattr(depth_tool, "device_lock", unexpected_gpu)
    with pytest.raises(FileNotFoundError):
        match command:
            case "train":
                depth_tool.train(manifest, tmp_path / "missing.pt", tmp_path / "run",
                                 "0", 15, 512, 0)
            case "evaluate":
                depth_tool.evaluate(manifest, tmp_path / "missing.pt", tmp_path / "result.json",
                                    "0", 512)


def test_manifest_rejects_content_corruption(prepared):
    manifest_path, _ = prepared
    manifest = depth_tool.load_manifest(manifest_path)
    (manifest_path.parent / manifest.samples[0].depth).write_bytes(b"corrupted")
    with pytest.raises(depth_tool.DataError, match="hash mismatch"):
        depth_tool.load_manifest(manifest_path)


@pytest.mark.parametrize("violation", ["same_sequence", "same_rgb", "unsafe_path"])
def test_manifest_rejects_leakage_and_unsafe_paths(prepared, violation):
    path, _ = prepared
    raw = json.loads(path.read_text())
    first, other = raw["samples"][0], raw["samples"][2]
    if violation == "same_sequence":
        other["sequence"] = first["sequence"]
    elif violation == "same_rgb":
        other["rgb_sha256"] = first["rgb_sha256"]
    else:
        first["image"] = "../../outside.png"
    path.write_text(json.dumps(raw))
    with pytest.raises(depth_tool.DataError):
        depth_tool.load_manifest(path)


def test_timestamp_association_is_unique_and_enforces_tolerance():
    pairs = depth_tool.associate(
        ["rgb/1.0.png", "rgb/1.01.png", "rgb/2.0.png"],
        ["depth/1.005.png", "depth/2.03.png"],
    )
    assert pairs == [("rgb/1.0.png", "depth/1.005.png")]


def test_raw_metric_exposes_scale_error_and_does_not_clip_upper_predictions():
    target = np.full((12, 16), 2.0, np.float32)
    prediction = target * 2
    target[0, :3] = [0, np.nan, 20]
    result = depth_tool.metric_depth(prediction, target)
    assert result["abs_rel"] == pytest.approx(1)
    assert result["rmse_m"] == pytest.approx(2)
    assert result["delta1"] == 0
    assert result["valid_pixels"] == 189
    result = depth_tool.metric_depth(np.full_like(target, 100), target)
    assert result["rmse_m"] == pytest.approx(98)


def test_raw_metric_rejects_invalid_predictions_instead_of_dropping_pixels():
    target = np.ones((12, 16), np.float32)
    prediction = target.copy()
    prediction[0, 0] = np.nan
    with pytest.raises(depth_tool.DataError, match="nonfinite"):
        depth_tool.metric_depth(prediction, target)
    with pytest.raises(depth_tool.DataError, match="fewer than 10"):
        depth_tool.metric_depth(np.ones((2, 2)), np.zeros((2, 2)))


def test_vendor_alignment_cannot_replace_meter_metrics():
    import torch
    from ultralytics.utils.metrics import DepthMetrics

    target = torch.full((12, 16), 2.0)
    prediction = target * 2
    raw = DepthMetrics(max_depth=10, align="none")
    aligned = DepthMetrics(max_depth=10, align="median")
    for metric in (raw, aligned):
        metric.update_stats(prediction, target)
        metric.process()
    assert raw.results_dict["metrics/abs_rel"] == pytest.approx(1)
    assert raw.results_dict["metrics/delta1"] == 0
    assert aligned.results_dict["metrics/abs_rel"] == pytest.approx(0)
    assert aligned.results_dict["metrics/delta1"] == 1
