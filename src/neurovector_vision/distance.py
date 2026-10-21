"""단안 카메라(YOLO 바운딩 박스) 기반 거리 추정.

핀홀 카메라 근사식 ``distance_m = real_dimension_m * focal_px / bbox_dimension_px``
를 사용한다. 초점 거리는 수평 화각(FOV) 또는 명시적 픽셀 값을 사용한다.

일반 객체는 전체 형상이 보일 때 가장 정확하다. 주 축이 화면 경계에서 잘리면
보조 축으로 근사하고 ``approximate_bbox_truncated`` 사유를 붙인다. 얼굴 근거리는
별도의 :class:`FaceDistanceEstimator` 가 눈/귀 키포인트로 계산한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .models import BoundingBox


class ObjectSize(BaseModel):
    """클래스별 실제 크기(미터)."""

    width_m: float = Field(gt=0.0)
    height_m: float = Field(gt=0.0)


# 자주 쓰는 YOLO(COCO) 클래스와 시뮬레이션용 장애물 클래스의 기본 크기.
# 직접 학습한 클래스는 이 표를 수정하지 않고 NEUROVECTOR_OBJECT_SIZES 로 추가하면 된다.
DEFAULT_OBJECT_SIZES: dict[str, ObjectSize] = {
    "person": ObjectSize(width_m=0.5, height_m=1.7),
    "bicycle": ObjectSize(width_m=0.6, height_m=1.7),
    "car": ObjectSize(width_m=1.8, height_m=1.5),
    "motorcycle": ObjectSize(width_m=0.8, height_m=1.5),
    "bus": ObjectSize(width_m=2.5, height_m=3.2),
    "truck": ObjectSize(width_m=2.5, height_m=3.5),
    "boat": ObjectSize(width_m=2.5, height_m=3.0),
    "traffic light": ObjectSize(width_m=0.4, height_m=1.2),
    "fire hydrant": ObjectSize(width_m=0.4, height_m=1.0),
    "stop sign": ObjectSize(width_m=0.75, height_m=0.75),
    "bench": ObjectSize(width_m=1.5, height_m=0.9),
    "dog": ObjectSize(width_m=0.5, height_m=0.6),
    "horse": ObjectSize(width_m=2.0, height_m=2.4),
    "tree": ObjectSize(width_m=3.0, height_m=6.0),
    "construction barrel": ObjectSize(width_m=0.6, height_m=1.0),
    "traffic cone": ObjectSize(width_m=0.4, height_m=0.7),
    "jersey barrier": ObjectSize(width_m=2.0, height_m=0.8),
    "dumpster": ObjectSize(width_m=2.0, height_m=1.4),
    "building": ObjectSize(width_m=10.0, height_m=8.0),
    "fence": ObjectSize(width_m=2.0, height_m=1.8),
}

# 너비/높이로 계산한 두 거리가 이 비율보다 크게 어긋나면 신뢰하지 않는다.
DEFAULT_MIN_AXIS_AGREEMENT = 0.6
# 화면 경계에서 이 비율 안쪽에 닿은 변은 잘린 것으로 본다.
DEFAULT_EDGE_MARGIN_RATIO = 0.02
# 높이가 이 값 이상인 객체는 높이를 거리 추정의 주 축으로 사용한다.
# 사람/차량처럼 세로 형상이 안정적인 객체의 너비는 자세에 따라 크게 변한다.
PRIMARY_HEIGHT_MIN_M = 0.5
# 관측된 바운딩 박스 종횡비가 기대값에서 이 배수 이상 벗어나면
# 전체 객체가 아니라 일부만 보이는 것으로 본다.
MAX_ASPECT_RATIO = 5.0


@dataclass(frozen=True)
class DistanceEstimate:
    """거리 추정 결과와 신뢰할 수 없는 경우의 사유."""

    distance_m: float | None
    note: str | None = None


def focal_length_px(image_width: int, horizontal_fov_deg: float) -> float:
    """수평 화각과 이미지 너비로 초점 거리(픽셀)를 계산한다."""
    if image_width <= 0:
        raise ValueError("image_width must be positive")
    if not 0.0 < horizontal_fov_deg < 180.0:
        raise ValueError("horizontal_fov_deg must be between 0 and 180")

    half_fov_rad = math.radians(horizontal_fov_deg) / 2.0
    return image_width / (2.0 * math.tan(half_fov_rad))


def horizontal_fov_deg(image_width: int, focal_px: float) -> float:
    """초점 거리(픽셀)에서 수평 화각을 역산한다."""
    if image_width <= 0:
        raise ValueError("image_width must be positive")
    if focal_px <= 0.0:
        raise ValueError("focal_px must be positive")
    return math.degrees(2.0 * math.atan(image_width / (2.0 * focal_px)))


def focal_length_from_reference(
    actual_distance_m: float,
    bbox_dimension_px: float,
    real_dimension_m: float,
) -> float:
    """알려진 거리의 기준 객체로 실제 초점 거리를 보정한다.

    ``bbox_dimension_px`` 는 기준 객체의 바운딩 박스 너비 또는 높이,
    ``real_dimension_m`` 은 해당 방향의 실제 크기다.
    """
    if actual_distance_m <= 0.0:
        raise ValueError("actual_distance_m must be positive")
    if bbox_dimension_px <= 0.0:
        raise ValueError("bbox_dimension_px must be positive")
    if real_dimension_m <= 0.0:
        raise ValueError("real_dimension_m must be positive")
    return actual_distance_m * bbox_dimension_px / real_dimension_m


class DistanceEstimator:
    """클래스별 실제 크기와 YOLO 바운딩 박스로 거리를 추정한다."""

    def __init__(
        self,
        *,
        horizontal_fov_deg: float = 70.0,
        focal_length_px: float | None = None,
        object_sizes: dict[str, ObjectSize] | None = None,
        min_axis_agreement: float = DEFAULT_MIN_AXIS_AGREEMENT,
        edge_margin_ratio: float = DEFAULT_EDGE_MARGIN_RATIO,
    ) -> None:
        self._horizontal_fov_deg = horizontal_fov_deg
        self._focal_length_override = focal_length_px
        self._object_sizes = {
            label.strip().lower(): size
            for label, size in (object_sizes or DEFAULT_OBJECT_SIZES).items()
        }
        self._min_axis_agreement = min_axis_agreement
        self._edge_margin_ratio = edge_margin_ratio

    @property
    def focal_length_override(self) -> float | None:
        return self._focal_length_override

    def set_focal_length_px(self, focal_length_px: float | None) -> None:
        """런타임에 보정된 초점 거리를 적용한다."""
        if focal_length_px is not None and focal_length_px <= 0.0:
            raise ValueError("focal_length_px must be positive")
        self._focal_length_override = focal_length_px

    def focal_px(self, image_width: int) -> float:
        return self._focal_px(image_width)

    def _focal_px(self, image_width: int) -> float:
        if self._focal_length_override is not None:
            if self._focal_length_override <= 0.0:
                raise ValueError("focal_length_px must be positive")
            return self._focal_length_override
        return focal_length_px(image_width, self._horizontal_fov_deg)

    def _edge_margin(self, image_dimension: int) -> float:
        return max(2.0, image_dimension * self._edge_margin_ratio)

    def estimate(
        self,
        label: str,
        box: BoundingBox,
        image_width: int,
        image_height: int | None = None,
    ) -> float | None:
        """거리(m)를 반환한다. 신뢰할 수 없으면 ``None``."""
        return self.estimate_detail(label, box, image_width, image_height).distance_m

    def estimate_detail(
        self,
        label: str,
        box: BoundingBox,
        image_width: int,
        image_height: int | None = None,
    ) -> DistanceEstimate:
        """거리와 함께 신뢰할 수 없는 경우의 사유를 반환한다."""
        size = self._object_sizes.get(label.strip().lower())
        if size is None:
            return DistanceEstimate(None, "unknown_class")

        try:
            focal = self._focal_px(image_width)
        except ValueError:
            return DistanceEstimate(None, "invalid_focal")

        box_width = box.x2 - box.x1
        box_height = box.y2 - box.y1
        if box_width <= 0.0 or box_height <= 0.0:
            return DistanceEstimate(None, "degenerate_bbox")

        width_reliable = True
        height_reliable = True
        if image_height is not None:
            margin_x = self._edge_margin(image_width)
            margin_y = self._edge_margin(image_height)
            width_reliable = box.x1 > margin_x and box.x2 < image_width - margin_x
            height_reliable = box.y1 > margin_y and box.y2 < image_height - margin_y

        if not width_reliable and not height_reliable:
            return DistanceEstimate(None, "bbox_truncated")

        primary_is_height = size.height_m >= PRIMARY_HEIGHT_MIN_M
        primary_reliable = height_reliable if primary_is_height else width_reliable
        secondary_reliable = width_reliable if primary_is_height else height_reliable

        # 양쪽 축이 모두 온전할 때만 종횡비를 검사한다. 한 축이라도 잘렸으면
        # 종횡비 자체가 실제 객체 형상을 반영하지 못한다.
        if width_reliable and height_reliable:
            expected_aspect = size.width_m / size.height_m
            observed_aspect = box_width / box_height
            aspect_ratio = observed_aspect / expected_aspect
            if aspect_ratio > MAX_ASPECT_RATIO or aspect_ratio < 1.0 / MAX_ASPECT_RATIO:
                return DistanceEstimate(None, "inconsistent_axes")

        width_estimate = size.width_m * focal / box_width if width_reliable else None
        height_estimate = size.height_m * focal / box_height if height_reliable else None

        if primary_reliable:
            distance = height_estimate if primary_is_height else width_estimate
            note = None
            if secondary_reliable:
                secondary = width_estimate if primary_is_height else height_estimate
                if distance is not None and secondary is not None:
                    agreement = min(distance, secondary) / max(distance, secondary)
                    if agreement < self._min_axis_agreement:
                        note = "approximate_axis_mismatch"
        else:
            # 주 축이 화면 경계에서 잘린 경우 보조 축으로 근사한다.
            # 사람의 다리가 잘린 상태의 어깨 너비처럼 오차가 있을 수 있다.
            distance = width_estimate if primary_is_height else height_estimate
            note = "approximate_bbox_truncated"

        if distance is None or not math.isfinite(distance) or distance <= 0.0:
            return DistanceEstimate(None, "invalid_distance")
        return DistanceEstimate(round(distance, 2), note)


# 성인 평균 눈동자 간 거리(IPD)와 머리 너비. 얼굴 근거리 거리 추정에 사용한다.
DEFAULT_IPD_M = 0.063
DEFAULT_HEAD_WIDTH_M = 0.15


class FaceDistanceEstimator:
    """얼굴 키포인트(눈/귀)로 근거리 거리를 추정한다.

    눈 사이 픽셀 거리는 얼굴 크기와 무관하게 거리에 반비례하므로, 바운딩
    박스가 화면 경계에 잘려도 거리를 계산할 수 있다. ``ipd_scale_px_m`` 는
    ``실제 거리(m) × 눈 간격(px)`` 로 보정되는 값이다.
    """

    def __init__(
        self,
        *,
        horizontal_fov_deg: float = 70.0,
        focal_length_px: float | None = None,
        ipd_m: float = DEFAULT_IPD_M,
        head_width_m: float = DEFAULT_HEAD_WIDTH_M,
        ipd_scale_px_m: float | None = None,
        min_keypoint_confidence: float = 0.5,
        min_eye_distance_px: float = 8.0,
    ) -> None:
        if ipd_m <= 0.0:
            raise ValueError("ipd_m must be positive")
        if head_width_m <= 0.0:
            raise ValueError("head_width_m must be positive")
        if ipd_scale_px_m is not None and ipd_scale_px_m <= 0.0:
            raise ValueError("ipd_scale_px_m must be positive")
        if not 0.0 <= min_keypoint_confidence <= 1.0:
            raise ValueError("min_keypoint_confidence must be between 0 and 1")
        if min_eye_distance_px <= 0.0:
            raise ValueError("min_eye_distance_px must be positive")

        self._horizontal_fov_deg = horizontal_fov_deg
        self._focal_length_override = focal_length_px
        self._ipd_m = ipd_m
        self._head_width_m = head_width_m
        self._ipd_scale_override = ipd_scale_px_m
        self._min_keypoint_confidence = min_keypoint_confidence
        self._min_eye_distance_px = min_eye_distance_px

    @property
    def ipd_scale_override(self) -> float | None:
        return self._ipd_scale_override

    def set_focal_length_px(self, focal_length_px: float | None) -> None:
        if focal_length_px is not None and focal_length_px <= 0.0:
            raise ValueError("focal_length_px must be positive")
        self._focal_length_override = focal_length_px

    def set_ipd_scale_px_m(self, ipd_scale_px_m: float | None) -> None:
        """얼굴 거리 보정값을 적용한다. ``None`` 이면 FOV 기반으로 돌아간다."""
        if ipd_scale_px_m is not None and ipd_scale_px_m <= 0.0:
            raise ValueError("ipd_scale_px_m must be positive")
        self._ipd_scale_override = ipd_scale_px_m

    def _focal_px(self, image_width: int) -> float:
        if self._focal_length_override is not None:
            if self._focal_length_override <= 0.0:
                raise ValueError("focal_length_px must be positive")
            return self._focal_length_override
        return focal_length_px(image_width, self._horizontal_fov_deg)

    def ipd_scale_px_m(self, image_width: int) -> float:
        """눈 간격(px)에 곱해 거리(m)를 만드는 스케일을 반환한다."""
        if self._ipd_scale_override is not None:
            return self._ipd_scale_override
        return self._focal_px(image_width) * self._ipd_m

    def focal_px(self, image_width: int) -> float:
        """보정된 얼굴 스케일에서 역산한 초점 거리(픽셀)."""
        return self.ipd_scale_px_m(image_width) / self._ipd_m

    def calibrate_from_eyes(
        self,
        actual_distance_m: float,
        eye_distance_px: float,
    ) -> float:
        """알려진 거리와 눈 간격으로 얼굴 거리 스케일을 보정한다."""
        if actual_distance_m <= 0.0:
            raise ValueError("actual_distance_m must be positive")
        if eye_distance_px <= 0.0:
            raise ValueError("eye_distance_px must be positive")
        scale = actual_distance_m * eye_distance_px
        self.set_ipd_scale_px_m(scale)
        return scale

    def _estimate_from_distance_px(
        self,
        reference_px: float,
        scale_px_m: float,
        *,
        missing_note: str,
        too_close_note: str,
    ) -> DistanceEstimate:
        if reference_px < self._min_eye_distance_px:
            return DistanceEstimate(None, too_close_note)
        if not math.isfinite(reference_px) or reference_px <= 0.0:
            return DistanceEstimate(None, missing_note)
        distance = scale_px_m / reference_px
        if not math.isfinite(distance) or distance <= 0.0:
            return DistanceEstimate(None, "invalid_distance")
        return DistanceEstimate(round(distance, 2))

    def estimate_eyes(
        self,
        left_eye: tuple[float, float, float],
        right_eye: tuple[float, float, float],
        image_width: int,
    ) -> DistanceEstimate:
        """양쪽 눈 키포인트로 거리를 추정한다. 튜플은 ``(x, y, confidence)``."""
        if (
            left_eye[2] < self._min_keypoint_confidence
            or right_eye[2] < self._min_keypoint_confidence
        ):
            return DistanceEstimate(None, "eye_keypoints_missing")
        try:
            scale = self.ipd_scale_px_m(image_width)
        except ValueError:
            return DistanceEstimate(None, "invalid_focal")
        eye_px = math.hypot(left_eye[0] - right_eye[0], left_eye[1] - right_eye[1])
        return self._estimate_from_distance_px(
            eye_px,
            scale,
            missing_note="eye_keypoints_missing",
            too_close_note="eye_keypoints_too_close",
        )

    def estimate_ears(
        self,
        left_ear: tuple[float, float, float],
        right_ear: tuple[float, float, float],
        image_width: int,
    ) -> DistanceEstimate:
        """눈이 가려졌을 때 귀 사이 거리로 거리를 추정한다."""
        if (
            left_ear[2] < self._min_keypoint_confidence
            or right_ear[2] < self._min_keypoint_confidence
        ):
            return DistanceEstimate(None, "ear_keypoints_missing")
        try:
            ipd_scale = self.ipd_scale_px_m(image_width)
        except ValueError:
            return DistanceEstimate(None, "invalid_focal")
        # 머리 너비 스케일은 IPD 스케일에서 비례식으로 구한다.
        ear_scale = ipd_scale * self._head_width_m / self._ipd_m
        ear_px = math.hypot(left_ear[0] - right_ear[0], left_ear[1] - right_ear[1])
        return self._estimate_from_distance_px(
            ear_px,
            ear_scale,
            missing_note="ear_keypoints_missing",
            too_close_note="ear_keypoints_too_close",
        )
