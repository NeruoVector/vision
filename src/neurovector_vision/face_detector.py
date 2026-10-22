"""YOLO26-pose 키포인트 + YOLO26-depth 융합 거리 추정.

얼굴만 보이는 근거리 장면에서는 사람 전체 키/너비를 쓸 수 없다. 이 모듈은
YOLO26-pose의 눈·코·귀 키포인트로 얼굴 박스를 만들고, 어깨/팔꿈치/손목/
엉덩이/무릎/발목 키포인트로 몸통·팔·손·다리·발 박스를 만든다.
YOLO26-depth의 미터 단위 깊이 맵을 각 부위 표면에서 샘플링해 거리를 계산한다.

- 깊이 모델이 있으면 깊이를 주 거리로 사용한다.
- 얼굴은 깊이가 없을 때 눈 사이 픽셀 거리(IPD)로 대체한다.
- 사용자가 실제 거리를 입력하면 IPD 스케일과 깊이 스케일을 함께 보정한다.
- 신체 부위는 깊이 모델이 없으면 박스만 표시하고 거리는 비운다.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .body_parts import BodyPart, body_parts_from_keypoints
from .config import Settings
from .depth import DepthRuntime
from .detector import VisionBackendUnavailable, _decode_image_bytes
from .distance import CameraIntrinsics, FaceDistanceEstimator, horizontal_fov_deg
from .models import BoundingBox, Detection
from .priority import priority_level_for

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
    keypoint_z_m: float | None = None
    raw_depth_m: float | None = None
    raw_radial_depth_m: float | None = None
    depth_distance_m: float | None = None
    distance_source: str | None = None
    used_axis: str | None = None
    ray_factor: float | None = None


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
    """YOLO26-pose + YOLO26-depth 기반 얼굴/신체 부위 감지기."""

    def __init__(self, settings: Settings, *, depth_runtime: DepthRuntime | None = None) -> None:
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
        self._depth_runtime = depth_runtime if depth_runtime is not None else DepthRuntime(settings)
        self._depth_scale = settings.depth_scale
        self._body_parts_enabled = settings.body_parts_enabled
        self._camera_intrinsics: CameraIntrinsics | None = None

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

    def warmup(self, warm_depth: bool = True) -> None:
        """빈 프레임으로 pose와 depth의 장치 초기화까지 완료한다."""
        self._load_model()
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect_measurements(frame, estimate_depth=False)
        if not warm_depth or self._depth is None:
            return
        try:
            self._depth._load_model()
        except Exception as exc:  # 깊이 모델이 없으면 키포인트 거리로 폴백
            self.last_depth_error = f"{type(exc).__name__}: {exc}"
            self._depth_retry_at = time.monotonic() + self._settings.depth_retry_interval_s
            return
        self._estimate_depth(frame)

    def set_ipd_scale_px_m(self, ipd_scale_px_m: float | None) -> None:
        self._distance.set_ipd_scale_px_m(ipd_scale_px_m)

    def set_focal_length_px(self, focal_length_px: float | None) -> None:
        self._distance.set_focal_length_px(focal_length_px)

    def set_camera_intrinsics(self, intrinsics: CameraIntrinsics | None) -> None:
        """화면 가장자리 사선거리 보정에 사용할 카메라 내부 파라미터."""
        self._camera_intrinsics = intrinsics

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

    @property
    def body_parts_enabled(self) -> bool:
        return self._body_parts_enabled

    def detect(
        self,
        image: bytes | str | np.ndarray,
        *,
        depth_map: np.ndarray | None = None,
        estimate_depth: bool = True,
    ) -> list[Detection]:
        return [
            measurement.detection
            for measurement in self.detect_measurements(
                image, depth_map=depth_map, estimate_depth=estimate_depth
            )
        ]

    def detect_measurements(
        self,
        image: bytes | str | np.ndarray,
        *,
        depth_map: np.ndarray | None = None,
        estimate_depth: bool = True,
    ) -> list[FaceMeasurement]:
        """얼굴/신체 부위 감지 결과와 거리 계산 중간값을 함께 반환한다."""
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
        if estimate_depth and depth_map is None:
            depth_map = self._estimate_depth(source)
        if depth_map is not None and depth_map.shape != (image_height, image_width):
            raise ValueError("깊이 맵 크기가 pose 입력 이미지와 일치하지 않습니다.")

        measurements: list[FaceMeasurement] = []
        for index, person_keypoints in enumerate(keypoint_data):
            person_confidence = self._person_confidence(boxes, index)
            face_box = _face_box_from_keypoints(
                person_keypoints,
                image_width,
                image_height,
                self._settings.face_keypoint_confidence,
            )
            if face_box is not None:
                measurements.append(
                    self._face_measurement(
                        person_keypoints,
                        face_box,
                        image_width,
                        depth_map,
                        person_confidence,
                    )
                )

            if not self._body_parts_enabled:
                continue
            for part in body_parts_from_keypoints(
                person_keypoints,
                image_width,
                image_height,
                self._settings.face_keypoint_confidence,
                face_box=face_box,
            ):
                measurements.append(self._body_part_measurement(part, depth_map, person_confidence))

        return measurements

    @staticmethod
    def _person_confidence(boxes, index: int) -> float:
        if boxes is None or index >= len(boxes):
            return 0.0
        try:
            return float(boxes[index].conf[0])
        except (IndexError, TypeError, ValueError):
            return 0.0

    def _ray_factor(self, box: BoundingBox) -> float:
        if self._camera_intrinsics is None:
            return 1.0
        center_x = (box.x1 + box.x2) / 2.0
        center_y = (box.y1 + box.y2) / 2.0
        return self._camera_intrinsics.ray_factor(center_x, center_y)

    def _face_measurement(
        self,
        person_keypoints: np.ndarray,
        face_box: BoundingBox,
        image_width: int,
        depth_map: np.ndarray | None,
        person_confidence: float,
    ) -> FaceMeasurement:
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

        # 깊이 모델과 키포인트는 광축 방향 Z를 준다. 화면 가장자리에서는
        # 실제 직선거리 R = Z * sqrt(1 + x^2 + y^2) 로 보정한다.
        ray_factor = self._ray_factor(face_box)
        keypoint_z = keypoint_estimate.distance_m
        keypoint_distance = keypoint_z * ray_factor if keypoint_z is not None else None

        raw_depth = None
        if depth_map is not None and self._depth is not None:
            raw_depth = self._depth.median_depth(depth_map, face_box)
        raw_radial_depth = raw_depth * ray_factor if raw_depth is not None else None
        depth_distance = (
            raw_radial_depth * self._depth_scale if raw_radial_depth is not None else None
        )

        note = None
        if depth_distance is not None:
            distance = depth_distance
            distance_source = "depth"
            if keypoint_distance is not None:
                ratio = max(depth_distance, keypoint_distance) / max(
                    min(depth_distance, keypoint_distance),
                    1e-6,
                )
                if ratio > 3.0:
                    note = "depth_keypoint_mismatch"
                    distance = None
                    distance_source = None
        elif keypoint_distance is not None:
            distance = keypoint_distance
            distance_source = "keypoint"
            note = keypoint_estimate.note
        else:
            distance = None
            distance_source = None
            note = keypoint_estimate.note

        confidence = person_confidence
        if eyes_ok:
            confidence = min(confidence or 1.0, (left_eye[2] + right_eye[2]) / 2.0)
        elif ears_ok:
            confidence = min(confidence or 1.0, (left_ear[2] + right_ear[2]) / 2.0)
        confidence = max(0.0, min(1.0, confidence))

        return FaceMeasurement(
            detection=Detection(
                label="face",
                confidence=confidence,
                box=face_box,
                distance_m=round(distance, 2) if distance is not None else None,
                distance_note=note,
                distance_source=distance_source,
                distance_confidence=0.0 if note == "depth_keypoint_mismatch" else None,
                region_is_approximate=True,
                priority_level=priority_level_for(
                    "face",
                    self._settings.priority_levels,
                ),
            ),
            eye_distance_px=eye_distance_px,
            ear_distance_px=ear_distance_px,
            keypoint_distance_m=keypoint_distance,
            keypoint_z_m=keypoint_z,
            raw_depth_m=raw_depth,
            raw_radial_depth_m=raw_radial_depth,
            depth_distance_m=(round(depth_distance, 2) if depth_distance is not None else None),
            distance_source=distance_source,
            used_axis=used_axis,
            ray_factor=ray_factor,
        )

    def _body_part_measurement(
        self,
        part: BodyPart,
        depth_map: np.ndarray | None,
        person_confidence: float,
    ) -> FaceMeasurement:
        """몸통/팔/손/다리/발의 깊이 거리. 키포인트 거리 폴백은 얼굴 전용이다."""
        ray_factor = self._ray_factor(part.box)
        raw_depth = None
        if depth_map is not None and self._depth is not None:
            raw_depth = self._depth.median_depth_around_points(
                depth_map,
                part.points,
                radius_px=part.radius_px,
            )
        raw_radial_depth = raw_depth * ray_factor if raw_depth is not None else None
        depth_distance = (
            raw_radial_depth * self._depth_scale if raw_radial_depth is not None else None
        )
        if depth_distance is not None:
            distance = depth_distance
            distance_source = "depth"
            note = None
        else:
            distance = None
            distance_source = None
            note = "depth_unavailable"

        return FaceMeasurement(
            detection=Detection(
                label=part.label,
                confidence=max(0.0, min(1.0, person_confidence, part.keypoint_confidence)),
                box=part.box,
                distance_m=round(distance, 2) if distance is not None else None,
                distance_note=note,
                distance_source=distance_source,
                region_is_approximate=part.region_is_approximate,
                priority_level=priority_level_for(
                    part.label,
                    self._settings.priority_levels,
                ),
            ),
            raw_depth_m=raw_depth,
            raw_radial_depth_m=raw_radial_depth,
            depth_distance_m=(round(depth_distance, 2) if depth_distance is not None else None),
            distance_source=distance_source,
            ray_factor=ray_factor,
        )

    @property
    def _depth(self):
        return self._depth_runtime.estimator

    @_depth.setter
    def _depth(self, estimator) -> None:
        self._depth_runtime.estimator = estimator

    @property
    def last_depth_ms(self) -> float | None:
        return self._depth_runtime.last_inference_ms

    @last_depth_ms.setter
    def last_depth_ms(self, value: float | None) -> None:
        self._depth_runtime.last_inference_ms = value

    @property
    def last_depth_error(self) -> str | None:
        return self._depth_runtime.last_error

    @last_depth_error.setter
    def last_depth_error(self, value: str | None) -> None:
        self._depth_runtime.last_error = value

    @property
    def _depth_retry_at(self) -> float:
        return self._depth_runtime.retry_at

    @_depth_retry_at.setter
    def _depth_retry_at(self, value: float) -> None:
        self._depth_runtime.retry_at = value

    def _estimate_depth(self, source) -> np.ndarray | None:
        return self._depth_runtime.estimate(source)

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
        ray_factor = measurement.ray_factor or 1.0
        ipd_scale = self._distance.calibrate_from_eyes(
            actual_distance_m,
            measurement.eye_distance_px,
            ray_factor,
        )
        depth_scale = previous_depth_scale
        radial_depth = measurement.raw_radial_depth_m
        if radial_depth is None and measurement.raw_depth_m is not None:
            radial_depth = measurement.raw_depth_m * ray_factor
        if radial_depth is not None and radial_depth > 0.0:
            depth_scale = max(
                0.1,
                min(10.0, actual_distance_m / radial_depth),
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
            "ray_factor": round(ray_factor, 4),
            "focal_px": round(implied_focal, 2),
            "fov_deg": round(horizontal_fov_deg(image_width, implied_focal), 2),
            "recommended_env": (
                f"NEUROVECTOR_FACE_IPD_SCALE_PX_M={ipd_scale:.2f}\n"
                f"NEUROVECTOR_DEPTH_SCALE={depth_scale:.3f}"
            ),
        }


__all__ = ["FaceDistanceDetector", "FaceMeasurement"]
