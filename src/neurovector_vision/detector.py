"""YOLO 기반 장애물 감지기.

`ultralytics` 는 선택 의존성(`pip install -e ".[vision]"`)이라
설치되어 있지 않으면 명확한 에러를 던진다.
"""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .distance import DistanceEstimator
from .models import BoundingBox, Detection


class VisionBackendUnavailable(RuntimeError):
    """ultralytics / opencv 미설치 시 발생."""


class InvalidImage(ValueError):
    """이미지 바이트를 해석할 수 없을 때 발생."""


def _decode_image_bytes(image: bytes):
    """업로드 바이트를 Ultralytics 가 받는 BGR ndarray 로 변환한다."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise VisionBackendUnavailable(
            "opencv/numpy 가 설치되어 있지 않습니다. `uv sync --extra vision` 으로 설치하세요."
        ) from exc

    if not image:
        raise InvalidImage("빈 이미지입니다.")

    array = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        raise InvalidImage("이미지 바이트를 해석할 수 없습니다.")
    return array


class ObstacleDetector:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._distance = DistanceEstimator(
            horizontal_fov_deg=settings.camera_horizontal_fov_deg,
            focal_length_px=settings.camera_focal_length_px,
            object_sizes=settings.resolved_object_sizes,
        )

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - 환경 의존
            raise VisionBackendUnavailable(
                "ultralytics 가 설치되어 있지 않습니다. `uv sync --extra vision` 으로 설치하세요."
            ) from exc

        self._model = YOLO(self._settings.model_path)
        return self._model

    def detect(self, image: str | Path | bytes) -> list[Detection]:
        """이미지 경로 또는 바이트를 받아 감지 결과를 반환한다."""
        source = _decode_image_bytes(image) if isinstance(image, bytes) else image
        model = self._load_model()

        results = model.predict(
            source=source,
            conf=self._settings.confidence_threshold,
            iou=self._settings.iou_threshold,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            names = result.names
            # 원본 이미지 너비를 사용해야 리사이즈 후에도 초점 거리가 맞는다.
            image_width = int(result.orig_shape[1])
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                label = str(names[int(box.cls[0])])
                bounding_box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                detections.append(
                    Detection(
                        label=label,
                        confidence=float(box.conf[0]),
                        box=bounding_box,
                        distance_m=self._distance.estimate(
                            label,
                            bounding_box,
                            image_width,
                        ),
                    )
                )

        return detections
