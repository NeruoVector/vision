import json
import runpy
from pathlib import Path

import pytest

from neurovector_vision.models import BoundingBox, Detection

MODULE = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "evaluate.py"))
evaluate = MODULE["evaluate"]
load_manifest = MODULE["load_manifest"]


def prediction(label="person", confidence=0.8, distance=3.0, box=(0, 0, 10, 10)):
    return Detection(
        label=label,
        confidence=confidence,
        distance_m=distance,
        box=BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
    )


def sample(tmp_path, annotations):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image bytes")
    return {"image": image, "annotations": annotations}


def test_duplicate_and_wrong_class_predictions_are_false_positives(tmp_path):
    truth = [
        {"label": "person", "box": (0, 0, 10, 10), "distance_m": 2.0},
        {"label": "dog", "box": (20, 20, 30, 30), "distance_m": 1.0},
    ]
    predictions = [
        prediction(confidence=0.7, distance=9),
        prediction(confidence=0.9),
        prediction(label="cat"),
    ]
    report = evaluate([sample(tmp_path, truth)], lambda _: predictions)
    assert report["overall"]["true_positives"] == 1
    assert report["overall"]["false_positives"] == 2
    assert report["overall"]["false_negatives"] == 1
    assert report["per_class"]["person"]["precision"] == 0.5
    assert report["per_class"]["person"]["distance_mae_m"] == 1.0
    assert report["per_class"]["dog"]["recall"] == 0
    assert report["per_class"]["cat"]["recall"] is None


def test_missing_distance_has_coverage_and_no_fake_zero_error(tmp_path):
    truth = [{"label": "person", "box": (0, 0, 10, 10), "distance_m": 2.0}]
    report = evaluate([sample(tmp_path, truth)], lambda _: [prediction(distance=None)])
    assert report["overall"]["distance_mae_m"] is None
    assert report["overall"]["distance_coverage"] == 0


def test_iou_threshold_and_negative_images(tmp_path):
    truth = [{"label": "person", "box": (0, 0, 10, 10)}]
    report = evaluate([sample(tmp_path, truth)], lambda _: [prediction(box=(8, 8, 18, 18))])
    assert report["overall"]["true_positives"] == 0
    assert report["overall"]["false_negatives"] == 1
    empty = evaluate([sample(tmp_path, [])], lambda _: [])
    assert empty["overall"]["precision"] is None
    assert empty["overall"]["recall"] is None
    assert empty["overall"]["distance_mae_m"] is None


def test_warmup_excluded_and_p95_uses_nearest_rank(tmp_path, monkeypatch):
    ticks = iter([0, 0.001, 1, 1.003])
    monkeypatch.setitem(evaluate.__globals__, "perf_counter", lambda: next(ticks))
    calls = []

    def detect(payload):
        calls.append(payload)
        return []

    report = evaluate([sample(tmp_path, []), sample(tmp_path, [])], detect, warmup=2)
    assert len(calls) == 4
    assert report["images"] == 2
    assert report["latency_ms"]["mean"] == pytest.approx(2.0)
    assert report["latency_ms"]["p95"] == pytest.approx(3.0)


def test_manifest_relative_images_and_optional_distance(tmp_path):
    (tmp_path / "image.jpg").write_bytes(b"data")
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image": "image.jpg",
                        "annotations": [{"label": "person", "box": [0, 0, 10, 10]}],
                    }
                ]
            }
        )
    )
    samples = load_manifest(path)
    assert samples[0]["image"] == tmp_path / "image.jpg"
    assert samples[0]["annotations"][0]["distance_m"] is None


@pytest.mark.parametrize(
    "annotation",
    [
        {"label": "person", "box": [0, 0, 0, 10]},
        {"label": "person", "box": [0, 0, 10, 10], "distance_m": -1},
        {"label": "person", "box": [0, 0, 10, 10], "distance_m": float("nan")},
        {"label": "", "box": [0, 0, 10, 10]},
    ],
)
def test_invalid_annotations_rejected_before_inference(tmp_path, annotation):
    (tmp_path / "image.jpg").write_bytes(b"data")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"images": [{"image": "image.jpg", "annotations": [annotation]}]}))
    with pytest.raises(ValueError):
        load_manifest(path)
