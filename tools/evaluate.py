"""Evaluate labeled images with the configured vision pipeline.

Usage (from vision/): python tools/evaluate.py dataset/manifest.json --warmup 2

Manifest JSON format::

    {"images": [
      {"image": "frames/001.jpg", "annotations": [
        {"label": "person", "box": [10, 20, 100, 180], "distance_m": 2.4}
      ]},
      {"image": "frames/empty.jpg", "annotations": []}
    ]}

Image paths are relative to the manifest. Boxes use original image pixel xyxy
coordinates. Every visible target must be labeled; empty annotation lists are
valid negative images. distance_m is optional and must be nonnegative if given.
Class names must match pipeline labels exactly (including body part names).

Predictions are matched per image/class in descending confidence order to an
unmatched annotation at greatest IoU above the threshold. Duplicate predictions
count as false positives. Undefined precision/recall and distance MAE are null.
Distance coverage reports how many matched labeled distances were returned.
Latency includes image decoding and inference, excludes file I/O and warmup;
p95 uses the nearest-rank percentile. Model loading can affect the first measured
sample when --warmup is zero. This tool does not download or supply a dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any


def _box(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("box must be a four-element xyxy list")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise ValueError("box coordinates must be numbers")
    coords = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in coords):
        raise ValueError("box coordinates must be finite")
    if coords[0] < 0 or coords[1] < 0 or coords[2] <= coords[0] or coords[3] <= coords[1]:
        raise ValueError("box must have nonnegative origin and positive width/height")
    return coords


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load and validate annotation structure and image existence before inference."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("images"), list) or not data["images"]:
        raise ValueError("manifest must contain a nonempty images list")
    samples = []
    for item in data["images"]:
        if not isinstance(item, dict) or not isinstance(item.get("image"), str):
            raise ValueError("each image entry needs an image path")
        image = path.parent / item["image"]
        if not image.is_file():
            raise ValueError(f"image does not exist: {image}")
        annotations = item.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError(f"annotations must be a list: {image}")
        checked = []
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise ValueError("each annotation must be an object")
            label = annotation.get("label")
            if not isinstance(label, str) or not label.strip():
                raise ValueError("annotation label must be a nonempty string")
            distance = annotation.get("distance_m")
            if distance is not None and (
                isinstance(distance, bool)
                or not isinstance(distance, (int, float))
                or not math.isfinite(distance)
                or distance < 0
            ):
                raise ValueError("distance_m must be a finite nonnegative number or null")
            checked.append(
                {"label": label, "box": _box(annotation.get("box")), "distance_m": distance}
            )
        samples.append({"image": image, "annotations": checked})
    return samples


def intersection_over_union(a: Any, b: Any) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def evaluate(
    samples: list[dict[str, Any]], detect: Any, *, iou_threshold: float = 0.5, warmup: int = 0
) -> dict[str, Any]:
    """Run a callable taking image bytes and returning Detection objects."""
    if not samples:
        raise ValueError("at least one sample is required")
    if not 0 < iou_threshold <= 1 or warmup < 0:
        raise ValueError("IoU threshold must be in (0, 1] and warmup must be nonnegative")
    for index in range(warmup):
        detect(samples[index % len(samples)]["image"].read_bytes())
    counts: dict[str, dict[str, Any]] = {}
    latencies = []

    def metrics(label: str) -> dict[str, Any]:
        return counts.setdefault(
            label,
            {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "distance_errors": [],
                "distance_eligible_matches": 0,
            },
        )

    for sample in samples:
        payload = sample["image"].read_bytes()
        start = perf_counter()
        predictions = list(detect(payload))
        latencies.append((perf_counter() - start) * 1000)
        annotations = sample["annotations"]
        unmatched = set(range(len(annotations)))
        for annotation in annotations:
            metrics(annotation["label"])
        for prediction in sorted(predictions, key=lambda p: p.confidence, reverse=True):
            row = metrics(prediction.label)
            box = (prediction.box.x1, prediction.box.y1, prediction.box.x2, prediction.box.y2)
            candidates = [
                (intersection_over_union(box, annotations[i]["box"]), i)
                for i in sorted(unmatched)
                if annotations[i]["label"] == prediction.label
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_iou < iou_threshold:
                row["false_positives"] += 1
                continue
            unmatched.remove(best_index)
            row["true_positives"] += 1
            truth_distance = annotations[best_index].get("distance_m")
            if truth_distance is not None:
                row["distance_eligible_matches"] += 1
                estimate = prediction.distance_m
                if estimate is not None and math.isfinite(estimate) and estimate >= 0:
                    row["distance_errors"].append(abs(estimate - truth_distance))
        for index in unmatched:
            metrics(annotations[index]["label"])["false_negatives"] += 1

    def summarize(row: dict[str, Any]) -> dict[str, Any]:
        tp, fp, fn = (row[key] for key in ("true_positives", "false_positives", "false_negatives"))
        errors = row["distance_errors"]
        eligible = row["distance_eligible_matches"]
        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "distance_mae_m": sum(errors) / len(errors) if errors else None,
            "distance_estimates": len(errors),
            "distance_eligible_matches": eligible,
            "distance_coverage": len(errors) / eligible if eligible else None,
        }

    totals = {
        key: sum(row[key] for row in counts.values())
        for key in (
            "true_positives",
            "false_positives",
            "false_negatives",
            "distance_eligible_matches",
        )
    }
    totals["distance_errors"] = [
        error for row in counts.values() for error in row["distance_errors"]
    ]
    return {
        "images": len(samples),
        "iou_threshold": iou_threshold,
        "warmup_runs": warmup,
        "overall": summarize(totals),
        "per_class": {label: summarize(row) for label, row in sorted(counts.items())},
        "latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "p95": sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=0, help="Unmeasured inference runs")
    parser.add_argument("--output", type=Path, help="Save JSON report (also printed to stdout)")
    args = parser.parse_args()
    if not 0 < args.iou_threshold <= 1 or args.warmup < 0:
        parser.error("IoU threshold must be in (0, 1] and warmup must be nonnegative")
    try:
        samples = load_manifest(args.manifest)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    from neurovector_vision.config import Settings
    from neurovector_vision.pipeline import VisionPipeline

    pipeline = VisionPipeline(Settings())
    report = evaluate(
        samples, pipeline.detect, iou_threshold=args.iou_threshold, warmup=args.warmup
    )
    output = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
