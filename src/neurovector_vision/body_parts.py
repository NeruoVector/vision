"""YOLO26-pose 키포인트에서 얼굴 외 신체 부위 박스를 만든다.

COCO 17 키포인트의 어깨/팔꿈치/손목/엉덩이/무릎/발목을 묶어 몸통, 팔,
손, 다리, 발 영역을 근사한다. 손/발은 키포인트 주변에 얼굴 크기 또는 어깨
너비에 비례한 작은 박스를 만든다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .models import BoundingBox

COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_ELBOW = 7
COCO_RIGHT_ELBOW = 8
COCO_LEFT_WRIST = 9
COCO_RIGHT_WRIST = 10
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12
COCO_LEFT_KNEE = 13
COCO_RIGHT_KNEE = 14
COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16


@dataclass(frozen=True)
class BodyPart:
    """감지된 신체 부위와 깊이 샘플링에 사용할 키포인트들."""

    label: str
    box: BoundingBox
    points: tuple[tuple[float, float], ...]
    radius_px: float = 6.0
    keypoint_confidence: float = 1.0
    region_is_approximate: bool = True


def _point(keypoints: np.ndarray, index: int) -> tuple[float, float, float]:
    x, y, confidence = keypoints[index][:3]
    return float(x), float(y), float(confidence)


def _valid(point: tuple[float, float, float], minimum: float) -> bool:
    return point[2] >= minimum and math.isfinite(point[0]) and math.isfinite(point[1])


def _distance_px(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _box_from_points(
    points: list[tuple[float, float]],
    image_width: int,
    image_height: int,
    *,
    margin_ratio: float = 0.12,
    min_thickness_px: float = 10.0,
) -> BoundingBox | None:
    if len(points) < 2:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    width = x2 - x1
    height = y2 - y1
    margin = max(6.0, max(width, height) * margin_ratio)
    if width < min_thickness_px:
        center = (x1 + x2) / 2.0
        x1, x2 = center - min_thickness_px / 2.0, center + min_thickness_px / 2.0
    if height < min_thickness_px:
        center = (y1 + y2) / 2.0
        y1, y2 = center - min_thickness_px / 2.0, center + min_thickness_px / 2.0
    x1 = max(0.0, x1 - margin)
    y1 = max(0.0, y1 - margin)
    x2 = min(float(image_width), x2 + margin)
    y2 = min(float(image_height), y2 + margin)
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _extend_from(
    origin: tuple[float, float, float],
    target: tuple[float, float, float],
    distance_px: float,
) -> tuple[float, float]:
    """origin→target 방향으로 target 너머 distance_px 만큼 연장한 점."""
    delta_x = target[0] - origin[0]
    delta_y = target[1] - origin[1]
    length = math.hypot(delta_x, delta_y)
    if length < 1e-6:
        return target[0], target[1]
    return (
        target[0] + delta_x / length * distance_px,
        target[1] + delta_y / length * distance_px,
    )


def _point_box(
    point: tuple[float, float, float],
    image_width: int,
    image_height: int,
    width: float,
    height: float,
) -> BoundingBox | None:
    x1 = max(0.0, point[0] - width / 2.0)
    y1 = max(0.0, point[1] - height / 2.0)
    x2 = min(float(image_width), point[0] + width / 2.0)
    y2 = min(float(image_height), point[1] + height / 2.0)
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def body_parts_from_keypoints(
    keypoints: np.ndarray,
    image_width: int,
    image_height: int,
    min_confidence: float,
    *,
    face_box: BoundingBox | None = None,
) -> list[BodyPart]:
    """신체 부위 목록을 만든다. 얼굴은 별도 감지기가 담당한다."""
    left_shoulder = _point(keypoints, COCO_LEFT_SHOULDER)
    right_shoulder = _point(keypoints, COCO_RIGHT_SHOULDER)
    left_elbow = _point(keypoints, COCO_LEFT_ELBOW)
    right_elbow = _point(keypoints, COCO_RIGHT_ELBOW)
    left_wrist = _point(keypoints, COCO_LEFT_WRIST)
    right_wrist = _point(keypoints, COCO_RIGHT_WRIST)
    left_hip = _point(keypoints, COCO_LEFT_HIP)
    right_hip = _point(keypoints, COCO_RIGHT_HIP)
    left_knee = _point(keypoints, COCO_LEFT_KNEE)
    right_knee = _point(keypoints, COCO_RIGHT_KNEE)
    left_ankle = _point(keypoints, COCO_LEFT_ANKLE)
    right_ankle = _point(keypoints, COCO_RIGHT_ANKLE)

    shoulder_px = (
        _distance_px(left_shoulder, right_shoulder)
        if _valid(left_shoulder, min_confidence) and _valid(right_shoulder, min_confidence)
        else 0.0
    )
    if face_box is not None:
        face_width = max(1.0, face_box.x2 - face_box.x1)
    elif shoulder_px > 0.0:
        face_width = shoulder_px * 0.28
    else:
        # 얼굴/어깨가 모두 없을 때는 화면 크기 기준의 최소 손 크기를 쓴다.
        face_width = min(image_width, image_height) * 0.08
    hand_width = max(12.0, face_width * 0.65, shoulder_px * 0.22)
    hand_height = max(12.0, face_width * 0.90, shoulder_px * 0.28)
    foot_width = max(10.0, face_width * 0.50, shoulder_px * 0.16)
    foot_height = max(10.0, face_width * 0.30, shoulder_px * 0.10)

    parts: list[BodyPart] = []
    seen: set[str] = set()

    def add(part: BodyPart) -> None:
        if part.label in seen:
            return
        seen.add(part.label)
        parts.append(part)

    # 몸통: 어깨와 엉덩이. 엉덩이가 화면 밖이면 어깨 아래로 근사한다.
    torso_points = [
        (point[0], point[1])
        for point in (left_shoulder, right_shoulder, left_hip, right_hip)
        if _valid(point, min_confidence)
    ]
    if (
        _valid(left_shoulder, min_confidence)
        and _valid(right_shoulder, min_confidence)
        and shoulder_px >= 2.0
    ):
        if len(torso_points) < 4:
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
            torso_points.extend(
                [
                    ((left_shoulder[0] + right_shoulder[0]) / 2.0, shoulder_y),
                    (
                        (left_shoulder[0] + right_shoulder[0]) / 2.0,
                        min(float(image_height), shoulder_y + shoulder_px * 1.2),
                    ),
                ]
            )
        box = _box_from_points(torso_points, image_width, image_height, margin_ratio=0.08)
        if box is not None:
            add(
                BodyPart(
                    label="body",
                    box=box,
                    points=tuple(torso_points),
                    keypoint_confidence=min(
                        point[2]
                        for point in (left_shoulder, right_shoulder, left_hip, right_hip)
                        if _valid(point, min_confidence)
                    ),
                    radius_px=max(6.0, shoulder_px * 0.08),
                )
            )

    # 팔과 손
    for side, shoulder, elbow, wrist in (
        ("left", left_shoulder, left_elbow, left_wrist),
        ("right", right_shoulder, right_elbow, right_wrist),
    ):
        arm_points = [
            (point[0], point[1])
            for point in (shoulder, elbow, wrist)
            if _valid(point, min_confidence)
        ]
        if len(arm_points) >= 2:
            box = _box_from_points(arm_points, image_width, image_height, margin_ratio=0.12)
            if box is not None:
                add(
                    BodyPart(
                        label=f"{side}_arm",
                        box=box,
                        points=tuple(arm_points),
                        keypoint_confidence=min(
                            point[2]
                            for point in (shoulder, elbow, wrist)
                            if _valid(point, min_confidence)
                        ),
                        radius_px=max(
                            5.0,
                            min(box.x2 - box.x1, box.y2 - box.y1) * 0.25,
                        ),
                    )
                )
        if _valid(wrist, min_confidence):
            # 손은 손목에서 팔 방향으로 뻗어 있으므로 손목 중심보다 조금 바깥을
            # 박스 중심으로 잡아야 손바닥/손가락이 박스에 더 잘 들어온다.
            if _valid(elbow, min_confidence):
                hand_center = _extend_from(elbow, wrist, hand_height * 0.35)
            else:
                hand_center = (wrist[0], wrist[1])
            box = _point_box(
                (hand_center[0], hand_center[1], wrist[2]),
                image_width,
                image_height,
                hand_width,
                hand_height,
            )
            if box is not None:
                add(
                    BodyPart(
                        label=f"{side}_hand",
                        box=box,
                        points=((wrist[0], wrist[1]), hand_center),
                        keypoint_confidence=(
                            min(wrist[2], elbow[2]) if _valid(elbow, min_confidence) else wrist[2]
                        ),
                        radius_px=max(5.0, hand_width * 0.30),
                    )
                )

    # 다리와 발
    for side, hip, knee, ankle in (
        ("left", left_hip, left_knee, left_ankle),
        ("right", right_hip, right_knee, right_ankle),
    ):
        leg_points = [
            (point[0], point[1]) for point in (hip, knee, ankle) if _valid(point, min_confidence)
        ]
        if len(leg_points) >= 2:
            box = _box_from_points(leg_points, image_width, image_height, margin_ratio=0.10)
            if box is not None:
                add(
                    BodyPart(
                        label=f"{side}_leg",
                        box=box,
                        points=tuple(leg_points),
                        keypoint_confidence=min(
                            point[2]
                            for point in (hip, knee, ankle)
                            if _valid(point, min_confidence)
                        ),
                        radius_px=max(
                            5.0,
                            min(box.x2 - box.x1, box.y2 - box.y1) * 0.25,
                        ),
                    )
                )
        if _valid(ankle, min_confidence):
            if _valid(knee, min_confidence):
                foot_center = _extend_from(knee, ankle, foot_height * 0.5)
            else:
                foot_center = (ankle[0], ankle[1])
            box = _point_box(
                (foot_center[0], foot_center[1], ankle[2]),
                image_width,
                image_height,
                foot_width,
                foot_height,
            )
            if box is not None:
                add(
                    BodyPart(
                        label=f"{side}_foot",
                        box=box,
                        points=((ankle[0], ankle[1]), foot_center),
                        keypoint_confidence=(
                            min(ankle[2], knee[2]) if _valid(knee, min_confidence) else ankle[2]
                        ),
                        radius_px=max(4.0, foot_width * 0.30),
                    )
                )

    return parts


__all__ = ["BodyPart", "body_parts_from_keypoints"]
