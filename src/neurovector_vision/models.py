from datetime import UTC, datetime

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """픽셀 좌표 기준 바운딩 박스."""

    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    """모델이 검출한 단일 객체."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: BoundingBox
    distance_m: float | None = Field(default=None, ge=0.0)
    # 거리를 계산하지 못한 사유 (예: bbox_truncated, inconsistent_axes).
    distance_note: str | None = None
    # 1=얼굴/신체/동물, 2=차량/사물, 3=나뭇잎 등 저위험 자연물
    priority_level: int = Field(default=2, ge=1, le=3)


class DetectionRequest(BaseModel):
    """백엔드 `/api/v1/detections` 로 보내는 페이로드."""

    drone_id: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    latitude: float | None = None
    longitude: float | None = None
    distance_m: float | None = Field(default=None, ge=0.0)
    priority_level: int = Field(default=2, ge=1, le=3)
    detected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DetectResponse(BaseModel):
    """`POST /detect` 응답."""

    drone_id: str
    detections: list[Detection]
    forwarded: int
    elapsed_ms: float
