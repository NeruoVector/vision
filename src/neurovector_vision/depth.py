"""YOLO26-depth 기반 단안 깊이 추정.

Ultralytics YOLO26-depth는 입력 이미지와 같은 크기의 밀도 깊이 맵을
미터 단위로 반환한다. 얼굴 바운딩 박스 안의 중앙 영역에서 중앙값을
사용해 배경/머리카락 픽셀의 영향을 줄인다.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np

from . import depth_sampling
from .config import Settings
from .detector import _decode_image_bytes
from .models import BoundingBox


class DepthModelUnavailable(RuntimeError):
    """깊이 모델을 로드할 수 없을 때 발생."""


class DepthEstimator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self.last_inference_ms: float | None = None
        self._gpu_retry_at = 0.0
        self._active_device: str | None = None
        self.last_device: str | None = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - 환경 의존
            raise DepthModelUnavailable(
                "ultralytics 가 없어 YOLO26-depth 를 사용할 수 없습니다."
            ) from exc
        self._model = YOLO(self._settings.depth_model_path)
        return self._model

    def estimate(self, image: bytes | str | np.ndarray) -> np.ndarray:
        """이미지의 밀도 깊이 맵(미터)을 ``(H, W)`` float 배열로 반환한다."""
        source = _decode_image_bytes(image) if isinstance(image, bytes) else image
        model = self._load_model()
        requested_device = self._settings.resolved_device
        device = "cpu" if time.monotonic() < self._gpu_retry_at else requested_device
        started = time.perf_counter()
        self.last_inference_ms = None
        try:
            try:
                results = self._predict(model, source, device)
            except RuntimeError as exc:
                # Only CUDA runtime failures justify switching hardware. Other errors
                # (bad inputs, unsupported models) must retain their original cause.
                message = str(exc).lower()
                cuda_error = any(token in message for token in ("cuda", "cudnn", "cublas"))
                is_cuda_device = (
                    device.isdigit()
                    or device.startswith("cuda")
                    or ("," in device and all(part.strip().isdigit() for part in device.split(",")))
                )
                if not (
                    self._settings.depth_cpu_fallback_enabled and is_cuda_device and cuda_error
                ):
                    raise
                self._gpu_retry_at = time.monotonic() + self._settings.depth_gpu_retry_interval_s
                results = self._predict(model, source, "cpu")
        finally:
            self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        if not results or results[0].depth is None:
            raise DepthModelUnavailable("깊이 모델이 깊이 맵을 반환하지 않았습니다.")
        data = results[0].depth.data
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        depth = np.asarray(data, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            raise DepthModelUnavailable("깊이 맵 형식이 올바르지 않습니다.")
        if hasattr(source, "shape") and depth.shape != tuple(source.shape[:2]):
            raise DepthModelUnavailable("깊이 맵 크기가 입력 이미지와 일치하지 않습니다.")
        return depth

    def _predict(self, model, source, device: str):
        if self._active_device is not None and self._active_device != device:
            # Ultralytics caches a predictor and its device after the first call.
            # Moving weights and rebuilding the predictor makes device changes real.
            device_index = device.split(",")[0].strip()
            torch_device = f"cuda:{device_index}" if device_index.isdigit() else device
            model.to(torch_device)
            model.predictor = None
        self._active_device = device
        results = model.predict(
            source=source,
            imgsz=self._settings.depth_imgsz,
            device=device,
            quantize=16 if self._settings.inference_fp16 and device != "cpu" else 32,
            verbose=False,
        )
        self.last_device = device
        return results

    def median_depth(
        self, depth_map: np.ndarray, box: BoundingBox, *,
        inset_ratio: float = 0.2, min_pixels: int = 12,
    ) -> float | None:
        return depth_sampling.median_depth(
            depth_map, box, inset_ratio=inset_ratio, min_pixels=min_pixels,
            min_depth=self._settings.depth_min_m, max_depth=self._settings.depth_max_m,
        )

    def median_depth_around_points(
        self, depth_map: np.ndarray, points: Iterable[tuple[float, float]], *,
        radius_px: float = 6.0, min_pixels: int = 8,
    ) -> float | None:
        return depth_sampling.median_depth_around_points(
            depth_map, points, radius_px=radius_px, min_pixels=min_pixels,
            min_depth=self._settings.depth_min_m, max_depth=self._settings.depth_max_m,
        )


class DepthRuntime:
    """프레임 깊이 추론과 오류 복구 상태를 감지기 사이에서 공유한다."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.estimator = DepthEstimator(settings) if settings.depth_enabled else None
        self.retry_at = 0.0
        self.last_error: str | None = None
        self.last_inference_ms: float | None = None

    def estimate(self, image) -> np.ndarray | None:
        if self.estimator is None or time.monotonic() < self.retry_at:
            self.last_inference_ms = None
            return None
        self.last_inference_ms = None
        try:
            depth_map = self.estimator.estimate(image)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_inference_ms = self.estimator.last_inference_ms
            self.retry_at = time.monotonic() + self._settings.depth_retry_interval_s
            return None
        self.last_error = None
        self.retry_at = 0.0
        self.last_inference_ms = self.estimator.last_inference_ms
        return depth_map
