"""감지 객체의 위험 우선순위(1~3단계).

- 1단계: 얼굴/신체/동물처럼 사람과 생명에 직접 영향을 주는 객체
- 2단계: 차량/일반 사물처럼 충돌 시 위험이 있지만 생명 위험은 낮은 객체
- 3단계: 나뭇잎처럼 손상되어도 큰 지장이 없는 자연물

나뭇잎 데이터셋은 아직 없지만 라벨을 미리 등록해 두어, 모델이 추가되면
별도 코드 수정 없이 3단계로 분류된다. ``NEUROVECTOR_PRIORITY_LEVELS`` JSON으로
클래스별 단계를 덮어쓸 수 있다.
"""

from __future__ import annotations

from collections.abc import Mapping

PRIORITY_CRITICAL = 1
PRIORITY_IMPORTANT = 2
PRIORITY_LOW = 3

# COCO 클래스 + 얼굴/신체/한국어 별칭.
DEFAULT_PRIORITY_LEVELS: dict[str, int] = {
    # 1단계: 사람, 신체, 동물
    "face": PRIORITY_CRITICAL,
    "body": PRIORITY_CRITICAL,
    "torso": PRIORITY_CRITICAL,
    "arm": PRIORITY_CRITICAL,
    "hand": PRIORITY_CRITICAL,
    "leg": PRIORITY_CRITICAL,
    "foot": PRIORITY_CRITICAL,
    "left_arm": PRIORITY_CRITICAL,
    "right_arm": PRIORITY_CRITICAL,
    "left_hand": PRIORITY_CRITICAL,
    "right_hand": PRIORITY_CRITICAL,
    "left_leg": PRIORITY_CRITICAL,
    "right_leg": PRIORITY_CRITICAL,
    "left_foot": PRIORITY_CRITICAL,
    "right_foot": PRIORITY_CRITICAL,
    "person": PRIORITY_CRITICAL,
    "people": PRIORITY_CRITICAL,
    "human": PRIORITY_CRITICAL,
    "pedestrian": PRIORITY_CRITICAL,
    "child": PRIORITY_CRITICAL,
    "animal": PRIORITY_CRITICAL,
    "cat": PRIORITY_CRITICAL,
    "dog": PRIORITY_CRITICAL,
    "bird": PRIORITY_CRITICAL,
    "horse": PRIORITY_CRITICAL,
    "sheep": PRIORITY_CRITICAL,
    "cow": PRIORITY_CRITICAL,
    "elephant": PRIORITY_CRITICAL,
    "bear": PRIORITY_CRITICAL,
    "zebra": PRIORITY_CRITICAL,
    "giraffe": PRIORITY_CRITICAL,
    "얼굴": PRIORITY_CRITICAL,
    "신체": PRIORITY_CRITICAL,
    "몸통": PRIORITY_CRITICAL,
    "팔": PRIORITY_CRITICAL,
    "손": PRIORITY_CRITICAL,
    "다리": PRIORITY_CRITICAL,
    "발": PRIORITY_CRITICAL,
    "사람": PRIORITY_CRITICAL,
    "보행자": PRIORITY_CRITICAL,
    "동물": PRIORITY_CRITICAL,
    "고양이": PRIORITY_CRITICAL,
    "개": PRIORITY_CRITICAL,
    "강아지": PRIORITY_CRITICAL,
    "새": PRIORITY_CRITICAL,
    "말": PRIORITY_CRITICAL,
    "양": PRIORITY_CRITICAL,
    "소": PRIORITY_CRITICAL,
    "코끼리": PRIORITY_CRITICAL,
    "곰": PRIORITY_CRITICAL,
    "얼룩말": PRIORITY_CRITICAL,
    "기린": PRIORITY_CRITICAL,
    # 2단계: 차량, 일반 사물
    "bicycle": PRIORITY_IMPORTANT,
    "car": PRIORITY_IMPORTANT,
    "motorcycle": PRIORITY_IMPORTANT,
    "airplane": PRIORITY_IMPORTANT,
    "bus": PRIORITY_IMPORTANT,
    "train": PRIORITY_IMPORTANT,
    "truck": PRIORITY_IMPORTANT,
    "boat": PRIORITY_IMPORTANT,
    "traffic light": PRIORITY_IMPORTANT,
    "fire hydrant": PRIORITY_IMPORTANT,
    "stop sign": PRIORITY_IMPORTANT,
    "parking meter": PRIORITY_IMPORTANT,
    "bench": PRIORITY_IMPORTANT,
    "backpack": PRIORITY_IMPORTANT,
    "umbrella": PRIORITY_IMPORTANT,
    "handbag": PRIORITY_IMPORTANT,
    "tie": PRIORITY_IMPORTANT,
    "suitcase": PRIORITY_IMPORTANT,
    "frisbee": PRIORITY_IMPORTANT,
    "skis": PRIORITY_IMPORTANT,
    "snowboard": PRIORITY_IMPORTANT,
    "sports ball": PRIORITY_IMPORTANT,
    "kite": PRIORITY_IMPORTANT,
    "baseball bat": PRIORITY_IMPORTANT,
    "baseball glove": PRIORITY_IMPORTANT,
    "skateboard": PRIORITY_IMPORTANT,
    "surfboard": PRIORITY_IMPORTANT,
    "tennis racket": PRIORITY_IMPORTANT,
    "bottle": PRIORITY_IMPORTANT,
    "wine glass": PRIORITY_IMPORTANT,
    "cup": PRIORITY_IMPORTANT,
    "fork": PRIORITY_IMPORTANT,
    "knife": PRIORITY_IMPORTANT,
    "spoon": PRIORITY_IMPORTANT,
    "bowl": PRIORITY_IMPORTANT,
    "banana": PRIORITY_IMPORTANT,
    "apple": PRIORITY_IMPORTANT,
    "sandwich": PRIORITY_IMPORTANT,
    "orange": PRIORITY_IMPORTANT,
    "broccoli": PRIORITY_IMPORTANT,
    "carrot": PRIORITY_IMPORTANT,
    "hot dog": PRIORITY_IMPORTANT,
    "pizza": PRIORITY_IMPORTANT,
    "donut": PRIORITY_IMPORTANT,
    "cake": PRIORITY_IMPORTANT,
    "chair": PRIORITY_IMPORTANT,
    "couch": PRIORITY_IMPORTANT,
    "potted plant": PRIORITY_IMPORTANT,
    "bed": PRIORITY_IMPORTANT,
    "dining table": PRIORITY_IMPORTANT,
    "toilet": PRIORITY_IMPORTANT,
    "tv": PRIORITY_IMPORTANT,
    "laptop": PRIORITY_IMPORTANT,
    "mouse": PRIORITY_IMPORTANT,
    "remote": PRIORITY_IMPORTANT,
    "keyboard": PRIORITY_IMPORTANT,
    "cell phone": PRIORITY_IMPORTANT,
    "microwave": PRIORITY_IMPORTANT,
    "oven": PRIORITY_IMPORTANT,
    "toaster": PRIORITY_IMPORTANT,
    "sink": PRIORITY_IMPORTANT,
    "refrigerator": PRIORITY_IMPORTANT,
    "book": PRIORITY_IMPORTANT,
    "clock": PRIORITY_IMPORTANT,
    "vase": PRIORITY_IMPORTANT,
    "scissors": PRIORITY_IMPORTANT,
    "teddy bear": PRIORITY_IMPORTANT,
    "hair drier": PRIORITY_IMPORTANT,
    "toothbrush": PRIORITY_IMPORTANT,
    "vehicle": PRIORITY_IMPORTANT,
    "tree": PRIORITY_IMPORTANT,
    "building": PRIORITY_IMPORTANT,
    "자동차": PRIORITY_IMPORTANT,
    "차": PRIORITY_IMPORTANT,
    "차량": PRIORITY_IMPORTANT,
    "오토바이": PRIORITY_IMPORTANT,
    "자전거": PRIORITY_IMPORTANT,
    "버스": PRIORITY_IMPORTANT,
    "트럭": PRIORITY_IMPORTANT,
    "보트": PRIORITY_IMPORTANT,
    "기차": PRIORITY_IMPORTANT,
    "비행기": PRIORITY_IMPORTANT,
    "사물": PRIORITY_IMPORTANT,
    # 3단계: 손상되어도 큰 지장이 없는 자연물 (나뭇잎 데이터셋 대비)
    "leaf": PRIORITY_LOW,
    "leaves": PRIORITY_LOW,
    "fallen leaf": PRIORITY_LOW,
    "tree leaf": PRIORITY_LOW,
    "leaf litter": PRIORITY_LOW,
    "grass": PRIORITY_LOW,
    "flower": PRIORITY_LOW,
    "flowers": PRIORITY_LOW,
    "petal": PRIORITY_LOW,
    "petals": PRIORITY_LOW,
    "나뭇잎": PRIORITY_LOW,
    "낙엽": PRIORITY_LOW,
    "잎": PRIORITY_LOW,
    "풀": PRIORITY_LOW,
    "꽃": PRIORITY_LOW,
    "꽃잎": PRIORITY_LOW,
}


def normalize_priority_label(label: str) -> str:
    """공백/하이픈/대소문자 차이를 흡수한다."""
    return "_".join(label.strip().lower().replace("-", " ").split())


_NORMALIZED_DEFAULT_PRIORITY_LEVELS = {
    normalize_priority_label(key): value for key, value in DEFAULT_PRIORITY_LEVELS.items()
}


def priority_level_for(
    label: str,
    overrides: Mapping[str, int] | None = None,
) -> int:
    """라벨에 해당하는 1~3단계 우선순위를 반환한다.

    알 수 없는 라벨은 일반 사물로 보고 2단계를 기본값으로 사용한다.
    잘못된 override 값은 무시하고 기본 매핑을 사용한다.
    """
    normalized = normalize_priority_label(label)
    if overrides:
        normalized_overrides = {
            normalize_priority_label(key): value for key, value in overrides.items()
        }
        override = normalized_overrides.get(normalized)
        if isinstance(override, int) and 1 <= override <= 3:
            return override
    default = _NORMALIZED_DEFAULT_PRIORITY_LEVELS.get(normalized)
    if default is not None:
        return default
    # 향후 나뭇잎 데이터셋의 leaf_damage, leaf_spot 같은 라벨도 3단계로.
    if normalized.startswith("leaf") or normalized.startswith("leaves"):
        return PRIORITY_LOW
    # left_forearm, right_wrist 처럼 세분화된 신체 부위 라벨도 1단계로.
    if normalized.startswith(("left_", "right_")) and normalized.split("_", 1)[1] in {
        "arm",
        "forearm",
        "upper_arm",
        "hand",
        "palm",
        "fingers",
        "leg",
        "thigh",
        "calf",
        "shin",
        "foot",
        "toe",
        "toes",
        "torso",
        "shoulder",
        "elbow",
        "wrist",
        "hip",
        "knee",
        "ankle",
        "head",
        "neck",
    }:
        return PRIORITY_CRITICAL
    return PRIORITY_IMPORTANT


def priority_name(level: int) -> str:
    if level == PRIORITY_CRITICAL:
        return "1단계 · 위험"
    if level == PRIORITY_LOW:
        return "3단계 · 낮음"
    return "2단계 · 주의"


__all__ = [
    "DEFAULT_PRIORITY_LEVELS",
    "PRIORITY_CRITICAL",
    "PRIORITY_IMPORTANT",
    "PRIORITY_LOW",
    "normalize_priority_label",
    "priority_level_for",
    "priority_name",
]
