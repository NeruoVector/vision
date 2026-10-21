"""단안 카메라(YOLO 바운딩 박스) 기반 거리 추정.

핀홀 카메라 근사식 ``distance_m = real_dimension_m * focal_px / bbox_dimension_px``
를 사용한다. 초점 거리는 수평 화각(FOV) 또는 명시적 픽셀 값을 사용한다.

거리는 너비/높이 추정값 중 **더 가까운 값**을 사용한다. 장애물 회피에서는
거리를 과대평가하는 것보다 과소평가하는 편이 안전하기 때문이다.
"""

from __future__ import annotations

import math

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


def focal_length_px(image_width: int, horizontal_fov_deg: float) -> float:
    """수평 화각과 이미지 너비로 초점 거리(픽셀)를 계산한다."""
    if image_width <= 0:
        raise ValueError("image_width must be positive")
    if not 0.0 < horizontal_fov_deg < 180.0:
        raise ValueError("horizontal_fov_deg must be between 0 and 180")

    half_fov_rad = math.radians(horizontal_fov_deg) / 2.0
    return image_width / (2.0 * math.tan(half_fov_rad))


class DistanceEstimator:
    """클래스별 실제 크기와 YOLO 바운딩 박스로 거리를 추정한다."""

    def __init__(
        self,
        *,
        horizontal_fov_deg: float = 70.0,
        focal_length_px: float | None = None,
        object_sizes: dict[str, ObjectSize] | None = None,
    ) -> None:
        self._horizontal_fov_deg = horizontal_fov_deg
        self._focal_length_override = focal_length_px
        self._object_sizes = {
            label.strip().lower(): size
            for label, size in (object_sizes or DEFAULT_OBJECT_SIZES).items()
        }

    def _focal_px(self, image_width: int) -> float:
        if self._focal_length_override is not None:
            if self._focal_length_override <= 0.0:
                raise ValueError("focal_length_px must be positive")
            return self._focal_length_override
        return focal_length_px(image_width, self._horizontal_fov_deg)

    def estimate(
        self,
        label: str,
        box: BoundingBox,
        image_width: int,
    ) -> float | None:
        """거리(m)를 반환한다. 크기 정보가 없는 클래스는 ``None``."""
        size = self._object_sizes.get(label.strip().lower())
        if size is None:
            return None

        try:
            focal = self._focal_px(image_width)
        except ValueError:
            return None

        estimates: list[float] = []
        box_width = box.x2 - box.x1
        box_height = box.y2 - box.y1
        if box_width > 0.0:
            estimates.append(size.width_m * focal / box_width)
        if box_height > 0.0:
            estimates.append(size.height_m * focal / box_height)

        if not estimates:
            return None

        distance = min(estimates)
        if not math.isfinite(distance) or distance <= 0.0:
            return None
        return round(distance, 2)
