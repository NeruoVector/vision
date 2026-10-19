from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # 드론 식별자 (기체별로 배포 시 변경)
    drone_id: str = "drone-001"

    @property
    def backend_detections_url(self) -> str:
        return f"{self.backend_base_url.rstrip('/')}/api/v1/detections"


@lru_cache
def get_settings() -> Settings:
    return Settings()
