#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["ultralytics==8.4.183", "torch==2.7.1", "pydantic>=2"]
# ///
# ─── How to run ───
# Use the existing environment; do not install or upgrade dependencies:
# cd vision && .venv/bin/python tools/finetune_depth.py --help
# ──────────────────
"""Bounded TUM RGB-D preparation, YOLO26 adaptation, and untouched-test scoring."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from bisect import bisect_left
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue

if TYPE_CHECKING:
    from ultralytics.engine.trainer import BaseTrainer

ROOT: Final = Path(__file__).resolve().parents[1]
WORK: Final = ROOT / "training/depth"
GPU_LOCK: Final = Path("/tmp/neurovector-vision-finetune-gpu.lock")
SEQUENCES: Final = {"train": "desk", "val": "xyz", "test": "plant"}
COUNTS: Final = {"train": 512, "val": 64, "test": 128}


class DataError(ValueError):
    """The dataset or predictions violate the experiment contract."""


class Sample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str
    sequence: str
    rgb_member: str
    depth_member: str
    rgb_timestamp: float
    depth_timestamp: float
    image: str
    depth: str
    rgb_sha256: str
    depth_sha256: str


class Source(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    split: str
    sequence: str
    url: str
    sha256: str
    bytes: int


class Manifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    depth_scale: float = Field(gt=0)
    max_depth_m: float = Field(gt=0)
    license: str
    sources: list[Source]
    samples: list[Sample]


class Scores(TypedDict):
    abs_rel: float
    rmse_m: float
    delta1: float
    valid_pixels: int


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def associate(rgb: list[str], depth: list[str], tolerance: float = 0.02) -> list[tuple[str, str]]:
    """Deterministic nearest-timestamp, one-to-one association within 20 ms."""
    rgb = sorted(rgb, key=lambda p: float(Path(p).stem))
    depth = sorted(depth, key=lambda p: float(Path(p).stem))
    timestamps = [float(Path(p).stem) for p in depth]
    used: set[int] = set()
    pairs = []
    for image in rgb:
        stamp = float(Path(image).stem)
        lo = bisect_left(timestamps, stamp - tolerance)
        hi = bisect_left(timestamps, stamp + tolerance)
        candidates = [i for i in range(lo, hi) if i not in used]
        if candidates:
            index = min(candidates, key=lambda i: (abs(timestamps[i] - stamp), i))
            used.add(index)
            pairs.append((image, depth[index]))
    return pairs


def select_pairs(archive: Path, count: int, seed: int) -> list[tuple[str, str]]:
    """Scan member names without extracting the full archive or trusting tar paths."""
    rgb, depth = [], []
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            path = Path(member.name)
            if member.isfile() and path.suffix == ".png":
                if path.parent.name == "rgb":
                    rgb.append(member.name)
                if path.parent.name == "depth":
                    depth.append(member.name)
    pairs = associate(rgb, depth)
    if count <= 0 or len(pairs) < count:
        raise DataError(f"need {count} pairs, found {len(pairs)} in {archive}")
    return sorted(random.Random(seed).sample(pairs, count))


def prepare_from_archives(archives: Path, output: Path, seed: int = 20261030) -> Path:
    """Retain only selected original PNG pairs and record source/content hashes."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    samples, sources = [], []
    for split, sequence in SEQUENCES.items():
        name = f"rgbd_dataset_freiburg1_{sequence}"
        archive = archives / f"{name}.tgz"
        pairs = select_pairs(archive, COUNTS[split], seed)
        destinations = {}
        for rgb, depth in pairs:
            identifier = f"{sequence}_{Path(rgb).stem}"
            destinations[rgb] = output / "images" / split / f"{identifier}.png"
            destinations[depth] = output / "depth" / split / f"{identifier}.png"
        with tarfile.open(archive, "r|gz") as tar:
            for member in tar:
                if member.name in destinations:
                    target = destinations[member.name]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stream = tar.extractfile(member)
                    if stream is None or not member.isfile():
                        raise DataError(f"not a regular file: {member.name}")
                    with stream, target.open("xb") as destination:
                        shutil.copyfileobj(stream, destination)
        for rgb, depth in pairs:
            image_path, depth_path = destinations[rgb], destinations[depth]
            image = cv2.imread(str(image_path))
            target = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if image is None or target is None or target.dtype != np.uint16:
                raise DataError(f"invalid RGB/uint16 depth pair: {image_path}")
            if image.shape[:2] != target.shape or np.count_nonzero(target) < 10:
                raise DataError(f"unregistered or empty depth: {depth_path}")
            samples.append(Sample(
                split=split, sequence=name, rgb_member=rgb, depth_member=depth,
                rgb_timestamp=float(Path(rgb).stem), depth_timestamp=float(Path(depth).stem),
                image=image_path.relative_to(output).as_posix(),
                depth=depth_path.relative_to(output).as_posix(),
                rgb_sha256=sha256(image_path), depth_sha256=sha256(depth_path),
            ))
        sources.append(Source(
            split=split, sequence=name,
            url=f"https://cvg.cit.tum.de/rgbd/dataset/freiburg1/{name}.tgz",
            sha256=sha256(archive), bytes=archive.stat().st_size,
        ))
    manifest = Manifest(seed=seed, depth_scale=5000, max_depth_m=10,
                        license="CC-BY-4.0", sources=sources, samples=samples)
    path = output / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    load_manifest(path)
    # Test deliberately has no entry in the trainer's YAML: final_eval fits only val.
    (output / "dataset.yaml").write_text(
        f"path: {output.resolve()}\ntrain: images/train\nval: images/val\n"
        "nc: 1\nnames: {0: depth}\nchannels: 3\ndepth_scale: 5000\nmax_depth: 10.0\n",
        encoding="utf-8",
    )
    return path


def prepare(output: Path, archives: Path | None, seed: int) -> None:
    """Use supplied archives, or download to a newly owned temporary directory."""
    if archives is not None:
        print(prepare_from_archives(archives, output, seed), flush=True)
        return
    with tempfile.TemporaryDirectory(prefix="orca-depth-", dir="/tmp") as temporary:
        staging = Path(temporary)
        for sequence in SEQUENCES.values():
            name = f"rgbd_dataset_freiburg1_{sequence}.tgz"
            subprocess.run([
                "curl", "--fail", "--location", "--retry", "2", "--max-time", "600",
                "--output", str(staging / name),
                f"https://cvg.cit.tum.de/rgbd/dataset/freiburg1/{name}",
            ], check=True)
        print(prepare_from_archives(staging, output, seed), flush=True)


def load_manifest(path: Path) -> Manifest:
    """Fail on overlap, moved/corrupt files, or split/sequence leakage."""
    manifest = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    seen_images, seen_depth, sequences = set(), set(), {}
    for sample in manifest.samples:
        if sample.split not in SEQUENCES:
            raise DataError(f"unknown split: {sample.split}")
        if sequences.setdefault(sample.sequence, sample.split) != sample.split:
            raise DataError("one sequence appears in multiple splits")
        if sample.rgb_sha256 in seen_images or sample.depth_sha256 in seen_depth:
            raise DataError("duplicate RGB or depth content across samples")
        seen_images.add(sample.rgb_sha256)
        seen_depth.add(sample.depth_sha256)
        for relative, digest in ((sample.image, sample.rgb_sha256),
                                 (sample.depth, sample.depth_sha256)):
            target = (path.parent / relative).resolve()
            if not target.is_relative_to(path.parent.resolve()) or sha256(target) != digest:
                raise DataError(f"unsafe path or hash mismatch: {relative}")
    if not all(any(s.split == split for s in manifest.samples) for split in SEQUENCES):
        raise DataError("every split needs at least one example")
    return manifest


def metric_depth(prediction: np.ndarray, target: np.ndarray, max_depth: float = 10) -> Scores:
    """Raw output meters, equal image weighting, no scale fitting or upper clipping."""
    if prediction.shape != target.shape:
        raise DataError("prediction and target shapes differ")
    mask = np.isfinite(target) & (target > 0.001) & (target < max_depth)
    if np.count_nonzero(mask) < 10:
        raise DataError("fewer than 10 valid GT pixels")
    p, g = prediction[mask].astype(np.float64), target[mask].astype(np.float64)
    if not np.isfinite(p).all() or np.any(p <= 0):
        raise DataError("nonfinite or nonpositive prediction at valid GT")
    return Scores(abs_rel=float(np.mean(np.abs(p - g) / g)),
                  rmse_m=float(np.sqrt(np.mean((p - g) ** 2))),
                  delta1=float(np.mean(np.maximum(p / g, g / p) < 1.25)),
                  valid_pixels=int(g.size))


@contextmanager
def device_lock(device: str, minimum_free_mib: int = 2300) -> Iterator[None]:
    """Hold the shared exclusive flock for every CUDA load/train/evaluation."""
    if device == "cpu":
        yield
        return
    with GPU_LOCK.open("a") as lock:
        print(f"Waiting for exclusive GPU lock: {GPU_LOCK}", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        info = subprocess.run([
            "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
        ], check=True, capture_output=True, text=True)
        free = int(info.stdout.splitlines()[0])
        print(f"Locked GPU; free VRAM={free} MiB", flush=True)
        if free < minimum_free_mib:
            raise DataError(f"only {free} MiB free; use --device cpu or retry later")
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def train(manifest_path: Path, weights: Path, run: Path, device: str,
          epochs: int, imgsz: int, freeze: int, batch: int = 8) -> None:
    """Native trainer with explicit metric loss and calibration restricted to internal val."""
    manifest_path, weights, run = manifest_path.resolve(), weights.resolve(), run.resolve()
    manifest = load_manifest(manifest_path)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if run.exists():
        raise FileExistsError(run)
    with device_lock(device, minimum_free_mib=12000 if batch > 1 else 2300):
        import torch
        import ultralytics
        from ultralytics import YOLO

        torch.set_num_threads(2)
        model = YOLO(str(weights))
        if model.task != "depth" or model.model.yaml.get("scale") != "n":
            raise DataError("this experiment permits YOLO26 nano depth models only")
        initial = {name: parameter.detach().cpu().clone()
                   for name, parameter in model.model.named_parameters()
                   if name.startswith("model.23.")}
        evidence = {"initial_sha256": sha256(weights), "manifest_sha256": sha256(manifest_path),
                    "architecture": weights.name, "torch": torch.__version__,
                    "ultralytics": ultralytics.__version__, "device": device,
                    "epochs": epochs, "imgsz": imgsz, "batch": batch, "nbs": batch,
                    "seed": manifest.seed, "freeze_layers": list(range(freeze)),
                    "calibration_split": "val", "test_used_for_selection": False,
                    "tracked_parameter_prefix": "model.23.",
                    "train_examples": sum(s.split == "train" for s in manifest.samples),
                    "internal_val_examples": sum(s.split == "val" for s in manifest.samples),
                    "test_examples": sum(s.split == "test" for s in manifest.samples)}

        def started(trainer: BaseTrainer) -> None:
            evidence["trainable_parameters"] = sum(
                p.numel() for p in trainer.model.parameters() if p.requires_grad)
            evidence["total_parameters"] = sum(p.numel() for p in trainer.model.parameters())

        def finished(trainer: BaseTrainer) -> None:
            evidence["optimizer_updates"] = trainer.ema.updates
            evidence["epochs_completed"] = trainer.epoch + 1
            evidence["head_parameter_l1_change"] = sum(
                float((parameter.detach().cpu() - initial[name]).abs().sum())
                for name, parameter in trainer.model.named_parameters() if name in initial)
            evidence["peak_allocated_mib"] = (
                torch.cuda.max_memory_allocated() / 2**20 if device != "cpu" else 0)

        model.add_callback("on_train_start", started)
        model.add_callback("on_train_end", finished)
        started_at = time.perf_counter()
        model.train(
            data=str(manifest_path.parent / "dataset.yaml"), split="val", device=device,
            project=str(run.parent), name=run.name, epochs=epochs, imgsz=imgsz,
            batch=batch, nbs=batch, workers=0, freeze=freeze, amp=False,
            optimizer="AdamW", lr0=0.0001, lrf=0.1, weight_decay=0.0001,
            warmup_epochs=0, dlam=0.5, seed=manifest.seed, deterministic=True,
            cache=False, plots=False, save=True, save_period=-1, patience=0,
            mosaic=0, mixup=0, cutmix=0, copy_paste=0, fliplr=0, flipud=0,
            hsv_h=0, hsv_s=0, hsv_v=0, degrees=0, translate=0, scale=0,
            shear=0, perspective=0, close_mosaic=0,
        )
        evidence["elapsed_seconds"] = time.perf_counter() - started_at
        for filename in ("best.pt", "last.pt"):
            checkpoint = run / "weights" / filename
            evidence[f"{filename}_sha256"] = sha256(checkpoint)
        write_json(run / "training_evidence.json", evidence)


def evaluate(manifest_path: Path, weights: Path, output: Path, device: str, imgsz: int) -> None:
    """Inference only on native-resolution held-out RGB+GT; never refit calibration."""
    manifest = load_manifest(manifest_path)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if output.exists():
        raise FileExistsError(output)
    with device_lock(device):
        import torch
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.utils.metrics import DepthMetrics

        torch.set_num_threads(2)
        model = YOLO(str(weights))
        if model.task != "depth" or model.model.yaml.get("scale") != "n":
            raise DataError("this experiment permits YOLO26 nano depth models only")
        vendor_raw = DepthMetrics(max_depth=manifest.max_depth_m, align="none")
        vendor_aligned = DepthMetrics(max_depth=manifest.max_depth_m, align="median")
        rows = []
        start = time.perf_counter()
        for sample in manifest.samples:
            if sample.split != "test":
                continue
            image = manifest_path.parent / sample.image
            target = cv2.imread(str(manifest_path.parent / sample.depth), cv2.IMREAD_UNCHANGED)
            target = target.astype(np.float32) / manifest.depth_scale
            result = model.predict(str(image), device=device, imgsz=imgsz, quantize=32,
                                   verbose=False)[0]
            pred = result.depth.data.detach().cpu().numpy()
            scores = metric_depth(pred, target, manifest.max_depth_m)
            vendor_raw.update_stats(torch.from_numpy(pred), torch.from_numpy(target))
            vendor_aligned.update_stats(torch.from_numpy(pred), torch.from_numpy(target))
            rows.append({"id": sample.image, **scores})
        vendor_raw.process()
        vendor_aligned.process()
        head = model.model.model[-1]
        write_json(output, {
            "weights": str(weights.resolve()), "weights_sha256": sha256(weights),
            "evaluation_source_sha256": sha256(Path(__file__)),
            "evaluation_environment": {
                "python": sys.version.split()[0], "torch": torch.__version__,
                "ultralytics": ultralytics.__version__, "platform": platform.platform(),
                "gpu": torch.cuda.get_device_name() if device != "cpu" else "cpu",
            },
            "manifest_sha256": sha256(manifest_path), "split": "test", "images": len(rows),
            "imgsz": imgsz, "quantize": 32, "device": device,
            "elapsed_seconds": time.perf_counter() - start,
            "calibration": {"a": float(head.cal_a.item()), "b": float(head.cal_b.item())},
            "protocol": "native GT; 0.001<GT<10m; per-image mean; raw has no fitting/clipping",
            "raw_metric": {key: float(np.mean([row[key] for row in rows]))
                           for key in ("abs_rel", "rmse_m", "delta1")},
            "vendor_align_none_clipped": vendor_raw.results_dict,
            "vendor_align_median_clipped": vendor_aligned.results_dict,
            "per_image": rows,
        })


def main() -> None:
    # argparse follows the installed project tools; no new CLI dependencies/environment.
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--output", type=Path, default=WORK / "data/tum-nano")
    prep.add_argument("--archives", type=Path)
    prep.add_argument("--seed", type=int, default=20261030)
    for name in ("train", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--manifest", type=Path, default=WORK / "data/tum-nano/manifest.json")
        command.add_argument("--weights", type=Path, required=True)
        command.add_argument("--device", default="0")
        command.add_argument("--imgsz", type=int, default=512)
    training = subparsers.choices["train"]
    training.add_argument("--run", type=Path, default=WORK / "runs/yolo26n-tum15")
    training.add_argument("--epochs", type=int, default=15)
    training.add_argument("--freeze", type=int, default=0)
    training.add_argument("--batch", type=int, default=8)
    subparsers.choices["evaluate"].add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    match args.command:
        case "prepare":
            prepare(args.output, args.archives, args.seed)
        case "train":
            if (args.epochs <= 0 or args.imgsz < 32 or args.batch <= 0
                    or not 0 <= args.freeze <= 23):
                parser.error("positive epochs/batch, imgsz>=32 and freeze in [0,23] required")
            train(args.manifest, args.weights, args.run, args.device,
                  args.epochs, args.imgsz, args.freeze, args.batch)
        case "evaluate":
            evaluate(args.manifest, args.weights, args.output, args.device, args.imgsz)


if __name__ == "__main__":
    main()
