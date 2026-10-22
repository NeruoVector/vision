"""YOLO 기반 장애물 감지기.

`ultralytics` 는 선택 의존성(`pip install -e ".[vision]"`)이라
설치되어 있지 않으면 명확한 에러를 던진다.
"""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .distance import CameraIntrinsics, DistanceEstimator
from .models import BoundingBox, Detection
from .priority import priority_level_for


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
        self._camera_intrinsics: CameraIntrinsics | None = None
        self._distance = DistanceEstimator(
            horizontal_fov_deg=settings.camera_horizontal_fov_deg,
            focal_length_px=settings.camera_focal_length_px,
            object_sizes=settings.resolved_object_sizes,
            min_axis_agreement=settings.distance_min_axis_agreement,
            edge_margin_ratio=settings.distance_edge_margin_ratio,
        )

    def set_focal_length_px(self, focal_length_px: float | None) -> None:
        """런타임에 카메라 초점 거리 보정값을 적용한다."""
        self._distance.set_focal_length_px(focal_length_px)

    def focal_px(self, image_width: int) -> float:
        """현재 설정 기준 초점 거리(픽셀)를 반환한다."""
        return self._distance.focal_px(image_width)

    def set_camera_intrinsics(self, intrinsics: CameraIntrinsics | None) -> None:
        self._camera_intrinsics = intrinsics

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
            device=self._settings.resolved_device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            names = result.names
            # 원본 이미지 크기를 사용해야 리사이즈 후에도 초점 거리가 맞는다.
            image_width = int(result.orig_shape[1])
            image_height = int(result.orig_shape[0])
            # 화면 가장자리의 광축 거리(Z)를 실제 직선거리로 보정한다.
            try:
                intrinsics = self._camera_intrinsics or CameraIntrinsics.from_image(
                    image_width,
                    image_height,
                    self._distance.focal_px(image_width),
                )
            except ValueError:
                intrinsics = None
            self._distance.set_camera_intrinsics(intrinsics)
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                label = str(names[int(box.cls[0])])
                bounding_box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                estimate = self._distance.estimate_detail(
                    label,
                    bounding_box,
                    image_width,
                    image_height,
                )
                detections.append(
                    Detection(
                        label=label,
                        confidence=float(box.conf[0]),
                        box=bounding_box,
                        distance_m=estimate.distance_m,
                        distance_note=estimate.note,
                        distance_source="bbox" if estimate.distance_m is not None else None,
                        priority_level=priority_level_for(
                            label,
                            self._settings.priority_levels,
                        ),
                    )
                )

        return detections
