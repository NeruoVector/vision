"""단안 카메라(YOLO 바운딩 박스) 기반 거리 추정.

핀홀 카메라 근사식 ``distance_m = real_dimension_m * focal_px / bbox_dimension_px``
를 사용한다. 초점 거리는 수평 화각(FOV) 또는 명시적 픽셀 값을 사용한다.

단안 거리 추정은 객체의 **전체 형상**이 보일 때만 의미가 있다. 화면 경계에
잘린 객체나 너비/높이 추정값이 크게 어긋나는 객체는 잘못된 거리를 표시하지
않도록 ``None`` 과 사유를 반환한다.
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

        # 주 축이 잘렸다면 보조 축만으로 거리를 만들지 않는다. 예를 들어
        # 사람의 다리가 화면 아래로 잘린 상태에서 어깨 너비만으로 거리를
        # 추정하면 자세에 따라 크게 틀릴 수 있다.
        if size.height_m >= PRIMARY_HEIGHT_MIN_M:
            if not height_reliable:
                return DistanceEstimate(None, "bbox_truncated")
        elif not width_reliable:
            return DistanceEstimate(None, "bbox_truncated")

        width_estimate = size.width_m * focal / box_width if width_reliable else None
        height_estimate = size.height_m * focal / box_height if height_reliable else None

        if width_estimate is None:
            distance = height_estimate
        elif height_estimate is None:
            distance = width_estimate
        else:
            agreement = min(width_estimate, height_estimate) / max(
                width_estimate,
                height_estimate,
            )
            if agreement < self._min_axis_agreement:
                return DistanceEstimate(None, "inconsistent_axes")
            # 두 축이 모두 신뢰 가능할 때는 면적 기반 기하평균을 사용한다.
            distance = math.sqrt(width_estimate * height_estimate)

        if distance is None or not math.isfinite(distance) or distance <= 0.0:
            return DistanceEstimate(None, "invalid_distance")
        return DistanceEstimate(round(distance, 2))
