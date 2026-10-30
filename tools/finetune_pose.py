#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["ultralytics==8.4.183"]
# ///
# ─── How to run ───
# Use the existing environment; do not install or upgrade local dependencies:
# cd vision && .venv/bin/python tools/finetune_pose.py --help
"""Prepare real COCO17 keypoints and train/evaluate an opt-in YOLO26 nano pose checkpoint."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeAlias, TypedDict, assert_never

if TYPE_CHECKING:
    from ultralytics.engine.trainer import BaseTrainer

LOCK: Final = Path("/tmp/neurovector-vision-finetune-gpu.lock")
# The cloud runtime is Python 3.11, which cannot parse PEP 695's type statement.
Json: TypeAlias = str | int | float | bool | None | list["Json"] | dict[str, "Json"]  # noqa: UP040
KEYPOINTS: Final = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
FLIP: Final = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


class AnnotationError(ValueError):
    """Native annotation structure cannot safely supervise COCO17 pose."""


class ComputeError(RuntimeError):
    """The requested compute device lacks enough available memory."""


class Command(StrEnum):
    PREPARE = "prepare"
    TRAIN = "train"
    EVAL = "eval"


class Person(TypedDict):
    id: int
    image_id: int
    category_id: int
    iscrowd: int
    num_keypoints: int
    bbox: list[float]
    keypoints: list[float]


class ImageInfo(TypedDict):
    id: int
    file_name: str
    width: int
    height: int
    license: int
    flickr_url: str


class Coco(TypedDict):
    images: list[ImageInfo]
    annotations: list[Person]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value: Json) -> None:
    """Write bounded, human-readable provenance and measurements."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def convert_person(person: Person, width: int, height: int) -> str | None:
    """Preserve COCO visibility 0/1/2 and all 17 joints; ignore crowds and nonpersons explicitly."""
    if person["category_id"] != 1 or person["iscrowd"]:
        return None
    points, bbox = person["keypoints"], person["bbox"]
    if width <= 0 or height <= 0 or len(points) != 51 or len(bbox) != 4:
        raise AnnotationError(f"Invalid COCO17 geometry for annotation {person['id']}")
    if not all(math.isfinite(v) for v in points + bbox):
        raise AnnotationError(f"Nonfinite geometry for annotation {person['id']}")
    if not any(points[2::3]):
        return None
    left, top, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        raise AnnotationError(f"Nonpositive box for annotation {person['id']}")
    x1, y1 = max(0, left), max(0, top)
    x2, y2 = min(width, left + bw), min(height, top + bh)
    if x2 <= x1 or y2 <= y1:
        raise AnnotationError(f"Box outside image for annotation {person['id']}")
    values = [
        (x1 + x2) / (2 * width),
        (y1 + y2) / (2 * height),
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]
    for index in range(0, 51, 3):
        x, y, visible = points[index : index + 3]
        if visible not in (0, 1, 2):
            raise AnnotationError(f"Invalid visibility for annotation {person['id']}")
        if visible and not (0 <= x <= width and 0 <= y <= height):
            raise AnnotationError(f"Labeled keypoint outside image for annotation {person['id']}")
        values.extend(
            (
                x / width if visible else 0,
                y / height if visible else 0,
                visible,
            )
        )
    return "0 " + " ".join(f"{v:.8g}" for v in values)


def select_ids(ids: list[int], count: int, seed: int = 42) -> list[int]:
    """Stable image-level split, independent of annotation ordering or download completion."""
    if count <= 0 or count > len(set(ids)):
        raise AnnotationError(f"Requested {count} images from {len(set(ids))} unique images")
    return sorted(set(ids), key=lambda i: digest(f"{seed}:{i}".encode()))[:count]


def download_image(job: tuple[str, Path]) -> None:
    """Fetch a selected official image using installed curl."""
    url, target = job
    if target.is_file():
        return
    temporary = target.with_suffix(".partial")
    subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--retry",
            "3",
            "--max-time",
            "120",
            "--output",
            str(temporary),
            url,
        ],
        check=True,
    )
    temporary.replace(target)


def prepare(args: argparse.Namespace) -> None:
    """Read original annotation ZIP directly and download a deterministic, bounded image subset."""
    from PIL import Image

    output = args.output.resolve()
    if (output / "summary.json").exists():
        raise FileExistsError(f"Completed dataset already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": "COCO2017 original person_keypoints",
        "seed": 42,
        "annotation_zip_sha256": digest(args.annotations.read_bytes()),
        "keypoints": KEYPOINTS,
        "flip_idx": FLIP,
        "selection": "train2017 -> train; disjoint val2017 IDs -> val and untouched test",
        "selection_filter": "at least one noncrowd person with num_keypoints > 0",
        "crowd_policy": "excluded; ignore regions unavailable to native YOLO validator",
    }
    heldout_ids: list[int] = []
    all_hashes: dict[str, str] = {}
    with zipfile.ZipFile(args.annotations) as archive:
        for split, official, count in (
            ("train", "train2017", args.train_images),
            ("val", "val2017", args.val_images),
            ("test", "val2017", args.test_images),
        ):
            raw = archive.read(f"annotations/person_keypoints_{official}.json")
            data: Coco = json.loads(raw)
            people: dict[int, list[Person]] = defaultdict(list)
            for person in data["annotations"]:
                people[person["image_id"]].append(person)
            eligible = [
                im["id"]
                for im in data["images"]
                if any(
                    p["category_id"] == 1 and not p["iscrowd"] and p["num_keypoints"] > 0
                    for p in people[im["id"]]
                )
            ]
            if split == "val":
                heldout_ids = select_ids(eligible, args.val_images + args.test_images)
            selected = (
                select_ids(eligible, count)
                if split == "train"
                else (
                    heldout_ids[: args.val_images]
                    if split == "val"
                    else heldout_ids[args.val_images :]
                )
            )
            info = {im["id"]: im for im in data["images"]}
            image_dir, label_dir = output / "images" / split, output / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            jobs = [
                (
                    f"https://s3.amazonaws.com/images.cocodataset.org/{official}/{info[i]['file_name']}",
                    image_dir / info[i]["file_name"],
                )
                for i in selected
            ]
            if args.image_cache is not None:
                for _, target in jobs:
                    cached = args.image_cache / split / target.name
                    if not cached.is_file():
                        raise FileNotFoundError(f"Selected cached image is missing: {cached}")
                    if not target.exists():
                        target.symlink_to(cached.resolve())
            with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
                for done, _ in enumerate(pool.map(download_image, jobs), 1):
                    if done % 100 == 0:
                        print(f"{split}: downloaded {done}/{len(jobs)}", flush=True)
            records, total_people, labeled_joints, crowds, zero_joint_people = [], 0, 0, 0, 0
            for image_id in selected:
                image_info = info[image_id]
                image = image_dir / image_info["file_name"]
                with Image.open(image) as decoded:
                    if decoded.size != (image_info["width"], image_info["height"]):
                        raise AnnotationError(
                            f"Image dimensions differ from COCO metadata: {image}"
                        )
                    decoded.verify()
                sha256 = digest(image.read_bytes())
                if sha256 in all_hashes and all_hashes[sha256] != split:
                    raise AnnotationError(f"Identical image content crosses splits: {image}")
                all_hashes[sha256] = split
                rows, annotation_ids = [], []
                for person in people[image_id]:
                    crowds += bool(person["iscrowd"])
                    zero_joint_people += bool(
                        not person["iscrowd"] and not any(person["keypoints"][2::3])
                    )
                    row = convert_person(person, image_info["width"], image_info["height"])
                    if row is not None and row not in rows:
                        rows.append(row)
                        annotation_ids.append(person["id"])
                        labeled_joints += sum(v > 0 for v in person["keypoints"][2::3])
                (label_dir / image.with_suffix(".txt").name).write_text("\n".join(rows) + "\n")
                total_people += len(rows)
                records.append(
                    {
                        "image_id": image_id,
                        "image": f"images/{split}/{image.name}",
                        "image_sha256": sha256,
                        "annotation_ids": annotation_ids,
                        "people": len(rows),
                        "official_split": official,
                    }
                )
            save_json(output / f"{split}.json", records)
            summary[split] = {
                "images": len(selected),
                "people": total_people,
                "labeled_joints": labeled_joints,
                "excluded_crowds": crowds,
                "excluded_zero_joint_people": zero_joint_people,
                "image_ids_sha256": digest(json.dumps(sorted(selected)).encode()),
                "manifest_sha256": digest((output / f"{split}.json").read_bytes()),
                "annotation_json_sha256": digest(raw),
            }
    (output / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: person\n"
        "kpt_shape: [17, 3]\nflip_idx: " + str(FLIP) + "\n"
    )
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


@contextmanager
def compute_lock(device: str, memory_fraction: float) -> Iterator[None]:
    """Serialize every CUDA operation with the depth worker and check free VRAM after locking."""
    with LOCK.open("a") as lock:
        if device != "cpu":
            print(f"Waiting for exclusive GPU lock: {LOCK}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            import torch

            index = int(device)
            free, total = torch.cuda.mem_get_info(index)
            print(f"GPU free={free / 2**20:.0f} MiB total={total / 2**20:.0f} MiB", flush=True)
            if free < 1536 * 2**20:
                raise ComputeError("Less than 1536 MiB spare VRAM; explicitly select --device cpu")
            torch.cuda.set_per_process_memory_fraction(memory_fraction, index)
        yield


def train(args: argparse.Namespace) -> None:
    """Run real updates and retain separate resumable and inference checkpoints."""
    if not args.resume and (args.project / args.name).exists():
        raise FileExistsError(f"Training run already exists: {args.project / args.name}")
    with compute_lock(args.device, args.memory_fraction):
        import torch
        import ultralytics
        from ultralytics import YOLO

        model = YOLO(str(args.weights))
        if model.task != "pose" or model.model.yaml.get("scale") != "n":
            raise AnnotationError("Only YOLO nano pose checkpoints are permitted")
        if list(model.model.yaml.get("kpt_shape", [])) != [17, 3]:
            raise AnnotationError("Only native COCO17 pose checkpoints are permitted")
        if args.resume and (
            model.ckpt.get("optimizer") is None
            or not 0 <= model.ckpt.get("epoch", -1) < model.ckpt["train_args"]["epochs"] - 1
        ):
            raise AnnotationError("Resume requires an incomplete checkpoint with optimizer state")
        started = time.time()

        def checkpoint(trainer: BaseTrainer) -> None:
            shutil.copyfile(trainer.last, trainer.save_dir / "resume.pt")

        def finished(trainer: BaseTrainer) -> None:
            save_json(
                trainer.save_dir / "training_evidence.json",
                {
                    "torch": torch.__version__,
                    "ultralytics": ultralytics.__version__,
                    "command": sys.argv,
                    "elapsed_seconds": time.time() - started,
                    "optimizer_updates": int(trainer.ema.updates),
                    "epochs_completed": trainer.epoch + 1,
                    "frozen_layers": list(trainer.freeze_layer_names),
                    "trainable_parameters": sum(
                        p.numel() for p in trainer.model.parameters() if p.requires_grad
                    ),
                    "base_sha256": digest(args.weights.read_bytes()),
                    "script_sha256": digest(Path(__file__).read_bytes()),
                    "dataset_summary_sha256": digest(
                        (args.data.parent / "summary.json").read_bytes()
                    ),
                },
            )

        model.add_callback("on_model_save", checkpoint)
        model.add_callback("on_train_end", finished)
        model.train(
            data=str(args.data.resolve()),
            project=str(args.project.resolve()),
            name=args.name,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            nbs=args.batch,
            device=args.device,
            freeze=args.freeze,
            workers=0,
            seed=42,
            deterministic=True,
            optimizer="AdamW",
            lr0=0.0001,
            lrf=0.1,
            weight_decay=0.0005,
            warmup_epochs=1,
            warmup_bias_lr=0.0001,
            patience=5,
            cache=False,
            amp=False,
            plots=False,
            save_period=-1,
            mosaic=0.0,
            mixup=0.0,
            close_mosaic=0,
            fliplr=0.5,
            resume=args.resume,
            exist_ok=False,
        )


def evaluate(args: argparse.Namespace) -> None:
    """Measure paired box IoU and pose OKS metrics on identical held-out labels."""
    if args.output.exists():
        raise FileExistsError(f"Evaluation report already exists: {args.output}")
    with compute_lock(args.device, args.memory_fraction):
        from ultralytics import YOLO

        report = {
            "split": args.split,
            "imgsz": args.imgsz,
            "classes": [0],
            "single_cls": False,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 1000,
            "command": sys.argv,
            "manifest_sha256": digest((args.data.parent / f"{args.split}.json").read_bytes()),
        }
        for name, weights in (("baseline", args.baseline), ("tuned", args.tuned)):
            model = YOLO(str(weights))
            if model.task != "pose" or model.model.yaml.get("scale") != "n":
                raise AnnotationError("Only nano pose checkpoints are permitted")
            if list(model.model.yaml.get("kpt_shape", [])) != [17, 3]:
                raise AnnotationError("Only native COCO17 pose checkpoints are permitted")
            metrics = model.val(
                data=str(args.data.resolve()),
                split=args.split,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=0,
                classes=[0],
                single_cls=False,
                conf=0.001,
                iou=0.7,
                max_det=1000,
                half=False,
                plots=False,
                save_json=False,
                project=str(args.output.parent),
                name=f"{args.output.stem}_{name}",
                exist_ok=False,
            )
            report[name] = {
                "weights": str(weights.resolve()),
                "sha256": digest(weights.read_bytes()),
                "box": {
                    "mAP50": float(metrics.box.map50),
                    "mAP50_95": float(metrics.box.map),
                    "precision": float(metrics.box.mp),
                    "recall": float(metrics.box.mr),
                },
                "pose": {
                    "mAP50": float(metrics.pose.map50),
                    "mAP50_95": float(metrics.pose.map),
                    "precision": float(metrics.pose.mp),
                    "recall": float(metrics.pose.mr),
                },
            }
            del model
        save_json(args.output, report)
        print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--annotations", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--train-images", type=int, default=1600)
    prep.add_argument("--val-images", type=int, default=250)
    prep.add_argument("--test-images", type=int, default=250)
    prep.add_argument("--download-workers", type=int, default=8)
    prep.add_argument("--image-cache", type=Path)
    fit = commands.add_parser("train")
    fit.add_argument("--weights", type=Path, required=True)
    fit.add_argument("--project", type=Path, required=True)
    fit.add_argument("--name", required=True)
    fit.add_argument("--epochs", type=int, default=20)
    fit.add_argument("--freeze", type=int, default=10)
    fit.add_argument("--resume", action="store_true")
    evaluation = commands.add_parser("eval")
    evaluation.add_argument("--baseline", type=Path, required=True)
    evaluation.add_argument("--tuned", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--split", choices=("val", "test"), default="test")
    for command in (fit, evaluation):
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--device", default="0")
        command.add_argument("--imgsz", type=int, default=640)
        command.add_argument("--batch", type=int, default=1)
        command.add_argument("--memory-fraction", type=float, default=0.3)
    args = parser.parse_args()
    match Command(args.command):
        case Command.PREPARE:
            prepare(args)
        case Command.TRAIN:
            train(args)
        case Command.EVAL:
            evaluate(args)
        case unreachable:
            assert_never(unreachable)


if __name__ == "__main__":
    main()
