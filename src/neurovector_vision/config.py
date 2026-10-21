from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .distance import DEFAULT_OBJECT_SIZES, ObjectSize


class Settings(BaseSettings):
    """환경변수 / .env 로부터 읽어오는 서비스 설정."""

    model_config = SettingsConfigDict(
        env_prefix="NEUROVECTOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # HTTP 서버
    host: str = "0.0.0.0"
    port: int = 8082

    # 백엔드 API (감지 결과 전송 대상)
    backend_base_url: str = "http://localhost:8081"
    backend_timeout_seconds: float = 5.0

    # 모델
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45

    # 단안 거리 추정 (카메라 캘리브레이션)
    # camera_focal_length_px 가 있으면 FOV 대신 이 값을 사용한다.
    camera_horizontal_fov_deg: float = 70.0
    camera_focal_length_px: float | None = None
    # JSON: {"my-obstacle": {"width_m": 1.2, "height_m": 0.8}}
    object_sizes: dict[str, ObjectSize] = Field(default_factory=dict)

    # 드론 식별자 (기체별로 배포 시 변경)
    drone_id: str = "drone-001"

    @property
    def backend_detections_url(self) -> str:
        return f"{self.backend_base_url.rstrip('/')}/api/v1/detections"

    @property
    def resolved_object_sizes(self) -> dict[str, ObjectSize]:
        """기본 크기 표 위에 사용자 정의 클래스 크기를 덮어쓴다."""
        sizes = dict(DEFAULT_OBJECT_SIZES)
        sizes.update({label.strip().lower(): size for label, size in self.object_sizes.items()})
        return sizes


@lru_cache
def get_settings() -> Settings:
    return Settings()
