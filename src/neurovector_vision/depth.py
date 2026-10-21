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
        started = time.perf_counter()
        results = model.predict(
            source=source,
            imgsz=self._settings.depth_imgsz,
            device=self._settings.resolved_device,
            verbose=False,
        )
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        if not results or results[0].depth is None:
            raise DepthModelUnavailable("깊이 모델이 깊이 맵을 반환하지 않았습니다.")
        data = results[0].depth.data
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        depth = np.asarray(data, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            raise DepthModelUnavailable("깊이 맵 형식이 올바르지 않습니다.")
        return depth

    def median_depth(
        self,
        depth_map: np.ndarray,
        box: BoundingBox,
        *,
        inset_ratio: float = 0.2,
        min_pixels: int = 12,
    ) -> float | None:
        """바운딩 박스 중앙 영역의 유효 깊이 중앙값을 반환한다."""
        height, width = depth_map.shape[:2]
        box_width = max(1.0, box.x2 - box.x1)
        box_height = max(1.0, box.y2 - box.y1)
        inset_x = box_width * inset_ratio
        inset_y = box_height * inset_ratio
        x1 = max(0, min(width - 1, int(round(box.x1 + inset_x))))
        x2 = max(x1 + 1, min(width, int(round(box.x2 - inset_x))))
        y1 = max(0, min(height - 1, int(round(box.y1 + inset_y))))
        y2 = max(y1 + 1, min(height, int(round(box.y2 - inset_y))))
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
        height, width = depth_map.shape[:2]
        radius = max(1, int(round(radius_px)))
        patches: list[np.ndarray] = []
        for x_px, y_px in points:
            if not np.isfinite(x_px) or not np.isfinite(y_px):
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
