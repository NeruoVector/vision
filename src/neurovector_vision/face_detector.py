"""YOLO26-pose 얼굴 키포인트 + YOLO26-depth 융합 거리 추정.

얼굴만 보이는 근거리 장면에서는 사람 전체 키/너비를 쓸 수 없다. 이 모듈은
YOLO26-pose의 눈·코·귀 키포인트로 얼굴 박스를 만들고, YOLO26-depth의
미터 단위 깊이 맵을 얼굴 박스 중앙에서 샘플링해 거리를 계산한다.

- 깊이 모델이 있으면 깊이를 주 거리로 사용한다.
- 깊이가 없으면 눈 사이 픽셀 거리(IPD)로 대체한다.
- 사용자가 실제 거리를 입력하면 IPD 스케일과 깊이 스케일을 함께 보정한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Settings
from .depth import DepthEstimator
from .detector import VisionBackendUnavailable, _decode_image_bytes
from .distance import FaceDistanceEstimator, horizontal_fov_deg
from .models import BoundingBox, Detection

COCO_LEFT_EYE = 1
COCO_RIGHT_EYE = 2
COCO_LEFT_EAR = 3
COCO_RIGHT_EAR = 4
COCO_NOSE = 0


@dataclass(frozen=True)
class FaceMeasurement:
    detection: Detection
    eye_distance_px: float | None = None
    ear_distance_px: float | None = None
    keypoint_distance_m: float | None = None
    raw_depth_m: float | None = None
    depth_distance_m: float | None = None
    distance_source: str | None = None
    used_axis: str | None = None


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _point(keypoints: np.ndarray, index: int) -> tuple[float, float, float]:
    x, y, confidence = keypoints[index][:3]
    return float(x), float(y), float(confidence)


def _valid(point: tuple[float, float, float], minimum: float) -> bool:
    return point[2] >= minimum and math.isfinite(point[0]) and math.isfinite(point[1])


def _distance_px(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _face_box_from_keypoints(
    keypoints: np.ndarray,
    image_width: int,
    image_height: int,
    min_confidence: float,
) -> BoundingBox | None:
    """눈/귀/코 키포인트로 얼굴 영역만 감싸는 박스를 만든다."""
    left_eye = _point(keypoints, COCO_LEFT_EYE)
    right_eye = _point(keypoints, COCO_RIGHT_EYE)
    nose = _point(keypoints, COCO_NOSE)
    left_ear = _point(keypoints, COCO_LEFT_EAR)
    right_ear = _point(keypoints, COCO_RIGHT_EAR)

    eyes_ok = _valid(left_eye, min_confidence) and _valid(right_eye, min_confidence)
    ears_ok = _valid(left_ear, min_confidence) and _valid(right_ear, min_confidence)

    if eyes_ok:
        eye_px = _distance_px(left_eye, right_eye)
        if eye_px < 2.0:
            return None
        center_x = (left_eye[0] + right_eye[0]) / 2.0
        center_y = (left_eye[1] + right_eye[1]) / 2.0
        width = 2.35 * eye_px
        if ears_ok:
            width = max(width, 1.08 * _distance_px(left_ear, right_ear))
        top = center_y - 0.9 * eye_px
        bottom = nose[1] + 1.0 * eye_px if _valid(nose, min_confidence) else center_y + 1.6 * eye_px
    elif ears_ok:
        ear_px = _distance_px(left_ear, right_ear)
        if ear_px < 2.0:
            return None
        center_x = (left_ear[0] + right_ear[0]) / 2.0
        center_y = (left_ear[1] + right_ear[1]) / 2.0
        width = 1.15 * ear_px
        top = center_y - 0.7 * ear_px
        bottom = center_y + 1.3 * ear_px
    else:
        return None

    x1 = max(0.0, center_x - width / 2.0)
    x2 = min(float(image_width), center_x + width / 2.0)
    y1 = max(0.0, top)
    y2 = min(float(image_height), bottom)
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


class FaceDistanceDetector:
    """YOLO26-pose + YOLO26-depth 기반 얼굴 전용 감지기."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._distance = FaceDistanceEstimator(
            horizontal_fov_deg=settings.camera_horizontal_fov_deg,
            focal_length_px=settings.camera_focal_length_px,
            ipd_m=settings.face_ipd_m,
            head_width_m=settings.face_head_width_m,
            ipd_scale_px_m=settings.face_ipd_scale_px_m,
            min_keypoint_confidence=settings.face_keypoint_confidence,
            min_eye_distance_px=settings.face_min_eye_distance_px,
        )
        self._depth = DepthEstimator(settings) if settings.depth_enabled else None
        self._depth_scale = settings.depth_scale
        self.last_depth_ms: float | None = None
        self.last_depth_error: str | None = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - 환경 의존
            raise VisionBackendUnavailable(
                "ultralytics 가 없어 얼굴 감지를 사용할 수 없습니다."
            ) from exc
        self._model = YOLO(self._settings.face_model_path)
        return self._model

    def warmup(self) -> None:
        """첫 프레임 전에 모델을 로드해 UI 시작 지연을 한 번에 처리한다."""
        self._load_model()
        if self._depth is None:
            return
        try:
            self._depth._load_model()
        except Exception as exc:  # 깊이 모델이 없으면 키포인트 거리로 폴백
            self.last_depth_error = f"{type(exc).__name__}: {exc}"
            self._depth = None

    def set_ipd_scale_px_m(self, ipd_scale_px_m: float | None) -> None:
        self._distance.set_ipd_scale_px_m(ipd_scale_px_m)

    def ipd_scale_px_m(self, image_width: int) -> float:
        return self._distance.ipd_scale_px_m(image_width)

    def focal_px(self, image_width: int) -> float:
        return self._distance.focal_px(image_width)

    @property
    def depth_scale(self) -> float:
        return self._depth_scale

    @property
    def depth_enabled(self) -> bool:
        return self._depth is not None

    def detect(self, image: bytes | str | np.ndarray) -> list[Detection]:
        return [measurement.detection for measurement in self.detect_measurements(image)]

    def detect_measurements(
        self,
        image: bytes | str | np.ndarray,
    ) -> list[FaceMeasurement]:
        """얼굴 감지 결과와 거리 계산에 사용한 중간값을 함께 반환한다."""
        source = _decode_image_bytes(image) if isinstance(image, bytes) else image
        model = self._load_model()
        results = model.predict(
            source=source,
            conf=self._settings.confidence_threshold,
            iou=self._settings.iou_threshold,
            device=self._settings.resolved_device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
            return []

        image_height, image_width = (int(v) for v in result.orig_shape[:2])
        keypoint_data = _to_numpy(keypoints.data)
        boxes = result.boxes
        depth_map = self._estimate_depth(source)

        measurements: list[FaceMeasurement] = []
        for index, person_keypoints in enumerate(keypoint_data):
            face_box = _face_box_from_keypoints(
                person_keypoints,
                image_width,
                image_height,
                self._settings.face_keypoint_confidence,
            )
            if face_box is None:
                continue

            left_eye = _point(person_keypoints, COCO_LEFT_EYE)
            right_eye = _point(person_keypoints, COCO_RIGHT_EYE)
            left_ear = _point(person_keypoints, COCO_LEFT_EAR)
            right_ear = _point(person_keypoints, COCO_RIGHT_EAR)
            eyes_ok = _valid(left_eye, self._settings.face_keypoint_confidence) and _valid(
                right_eye, self._settings.face_keypoint_confidence
            )
            ears_ok = _valid(left_ear, self._settings.face_keypoint_confidence) and _valid(
                right_ear, self._settings.face_keypoint_confidence
            )

            eye_distance_px = _distance_px(left_eye, right_eye) if eyes_ok else None
            ear_distance_px = _distance_px(left_ear, right_ear) if ears_ok else None

            keypoint_estimate = self._distance.estimate_eyes(
                left_eye,
                right_eye,
                image_width,
            )
            used_axis = "eyes" if keypoint_estimate.distance_m is not None else None
            if keypoint_estimate.distance_m is None:
                keypoint_estimate = self._distance.estimate_ears(
                    left_ear,
                    right_ear,
                    image_width,
                )
                used_axis = "ears" if keypoint_estimate.distance_m is not None else None

            raw_depth = None
            if depth_map is not None:
                raw_depth = self._depth.median_depth(depth_map, face_box)
            depth_distance = raw_depth * self._depth_scale if raw_depth is not None else None

            note = None
            if depth_distance is not None:
                distance = depth_distance
                distance_source = "depth"
                if keypoint_estimate.distance_m is not None:
                    ratio = max(depth_distance, keypoint_estimate.distance_m) / max(
                        min(depth_distance, keypoint_estimate.distance_m),
                        1e-6,
                    )
                    if ratio > 3.0:
                        note = "depth_keypoint_mismatch"
            elif keypoint_estimate.distance_m is not None:
                distance = keypoint_estimate.distance_m
                distance_source = "keypoint"
                note = keypoint_estimate.note
            else:
                distance = None
                distance_source = None
                note = keypoint_estimate.note

            confidence = 0.0
            if boxes is not None and index < len(boxes):
                confidence = float(boxes[index].conf[0])
            if eyes_ok:
                confidence = min(confidence or 1.0, (left_eye[2] + right_eye[2]) / 2.0)
            elif ears_ok:
                confidence = min(confidence or 1.0, (left_ear[2] + right_ear[2]) / 2.0)
            confidence = max(0.0, min(1.0, confidence))

            measurements.append(
                FaceMeasurement(
                    detection=Detection(
                        label="face",
                        confidence=confidence,
                        box=face_box,
                        distance_m=round(distance, 2) if distance is not None else None,
                        distance_note=note,
                    ),
                    eye_distance_px=eye_distance_px,
                    ear_distance_px=ear_distance_px,
                    keypoint_distance_m=keypoint_estimate.distance_m,
                    raw_depth_m=raw_depth,
                    depth_distance_m=(
                        round(depth_distance, 2) if depth_distance is not None else None
                    ),
                    distance_source=distance_source,
                    used_axis=used_axis,
                )
            )

        return measurements

    def _estimate_depth(self, source) -> np.ndarray | None:
        if self._depth is None:
            return None
        try:
            depth_map = self._depth.estimate(source)
        except Exception as exc:  # 모델 파일/GPU 오류 시 키포인트 거리로 폴백
            self.last_depth_error = f"{type(exc).__name__}: {exc}"
            self.last_depth_ms = self._depth.last_inference_ms
            self._depth = None
            return None
        self.last_depth_error = None
        self.last_depth_ms = self._depth.last_inference_ms
        return depth_map

    def calibrate(
        self,
        actual_distance_m: float,
        measurement: FaceMeasurement,
        image_width: int,
    ) -> dict:
        """실제 거리로 얼굴 IPD 스케일과 깊이 스케일을 동시에 보정한다."""
        if actual_distance_m <= 0.0:
            raise ValueError("actual_distance_m must be positive")
        if measurement.eye_distance_px is None or measurement.eye_distance_px <= 0.0:
            raise ValueError("양쪽 눈 키포인트가 보이는 얼굴로 보정하세요.")

        previous_ipd_scale = self.ipd_scale_px_m(image_width)
        previous_depth_scale = self._depth_scale
        ipd_scale = self._distance.calibrate_from_eyes(
            actual_distance_m,
            measurement.eye_distance_px,
        )
        depth_scale = previous_depth_scale
        if measurement.raw_depth_m is not None and measurement.raw_depth_m > 0.0:
            depth_scale = max(
                0.1,
                min(10.0, actual_distance_m / measurement.raw_depth_m),
            )
            self._depth_scale = depth_scale

        implied_focal = ipd_scale / self._settings.face_ipd_m
        return {
            "label": "face",
            "actual_distance_m": round(actual_distance_m, 2),
            "eye_distance_px": round(measurement.eye_distance_px, 2),
            "ipd_scale_px_m": round(ipd_scale, 2),
            "previous_ipd_scale_px_m": round(previous_ipd_scale, 2),
            "depth_scale": round(depth_scale, 3),
            "previous_depth_scale": round(previous_depth_scale, 3),
            "focal_px": round(implied_focal, 2),
            "fov_deg": round(horizontal_fov_deg(image_width, implied_focal), 2),
            "recommended_env": (
                f"NEUROVECTOR_FACE_IPD_SCALE_PX_M={ipd_scale:.2f}\n"
                f"NEUROVECTOR_DEPTH_SCALE={depth_scale:.3f}"
            ),
        }


__all__ = ["FaceDistanceDetector", "FaceMeasurement"]
