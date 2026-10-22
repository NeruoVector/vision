"""YOLO26-depth 기반 단안 깊이 추정.

Ultralytics YOLO26-depth는 입력 이미지와 같은 크기의 밀도 깊이 맵을
미터 단위로 반환한다. 얼굴 바운딩 박스 안의 중앙 영역에서 중앙값을
사용해 배경/머리카락 픽셀의 영향을 줄인다.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np

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
            verbose=False,
        )
        self.last_device = device
        return results

    def median_depth(
        self,
        depth_map: np.ndarray,
        box: BoundingBox,
        *,
        inset_ratio: float = 0.2,
        min_pixels: int = 12,
    ) -> float | None:
        """바운딩 박스 중앙 영역의 유효 깊이 중앙값을 반환한다."""
        if depth_map.ndim != 2 or depth_map.size == 0:
            return None
        height, width = depth_map.shape
        if not all(np.isfinite(value) for value in (box.x1, box.y1, box.x2, box.y2)):
            return None
        box_width = box.x2 - box.x1
        box_height = box.y2 - box.y1
        if box_width <= 0.0 or box_height <= 0.0:
            return None
        if not 0.0 <= inset_ratio < 0.5:
            raise ValueError("inset_ratio must be between 0 and 0.5")
        inset_x = box_width * inset_ratio
        inset_y = box_height * inset_ratio
        left, right = box.x1 + inset_x, box.x2 - inset_x
        top, bottom = box.y1 + inset_y, box.y2 - inset_y
        if left >= width or right <= 0.0 or top >= height or bottom <= 0.0:
            return None
        x1 = max(0, min(width, int(np.floor(left))))
        x2 = max(0, min(width, int(np.ceil(right))))
        y1 = max(0, min(height, int(np.floor(top))))
        y2 = max(0, min(height, int(np.ceil(bottom))))
        region = depth_map[y1:y2, x1:x2]
        values = region[np.isfinite(region)]
        values = values[
            (values >= self._settings.depth_min_m) & (values <= self._settings.depth_max_m)
        ]
        if values.size < min_pixels:
            return None
        return float(np.median(values))

    def median_depth_around_points(
        self,
        depth_map: np.ndarray,
        points: Iterable[tuple[float, float]],
        *,
        radius_px: float = 6.0,
        min_pixels: int = 8,
    ) -> float | None:
        """키포인트 주변 작은 패치들의 유효 깊이 중앙값을 반환한다.

        팔/다리처럼 얇은 부위는 바운딩 박스 중앙값을 쓰면 배경이 섞이기
        쉽다. 키포인트를 따라 좁은 패치만 샘플링해 실제 신체 표면 깊이에
        가깝게 만든다.
        """
        if depth_map.ndim != 2 or depth_map.size == 0:
            return None
        height, width = depth_map.shape
        radius = max(1, int(round(radius_px)))
        patches: list[np.ndarray] = []
        for x_px, y_px in points:
            if not np.isfinite(x_px) or not np.isfinite(y_px):
                continue
            if not (0.0 <= x_px < width and 0.0 <= y_px < height):
                continue
            center_x = int(round(x_px))
            center_y = int(round(y_px))
            x1 = max(0, center_x - radius)
            x2 = min(width, center_x + radius + 1)
            y1 = max(0, center_y - radius)
            y2 = min(height, center_y + radius + 1)
            region = depth_map[y1:y2, x1:x2]
            if region.size == 0:
                continue
            values = region[np.isfinite(region)]
            values = values[
                (values >= self._settings.depth_min_m) & (values <= self._settings.depth_max_m)
            ]
            if values.size:
                patches.append(values)
        if not patches:
            return None
        values = np.concatenate(patches)
        if values.size < min_pixels:
            return None
        return float(np.median(values))


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
