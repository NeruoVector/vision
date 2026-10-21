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

    # 모델 / 추론 장치
    model_path: str = "yolo26n.pt"
    face_model_path: str = "yolo26n-pose.pt"
    depth_model_path: str = "yolo26n-depth.pt"
    device: str = "auto"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    depth_enabled: bool = True
    depth_imgsz: int = Field(default=768, ge=320, le=1280)
    depth_min_m: float = Field(default=0.05, gt=0.0)
    depth_max_m: float = Field(default=150.0, gt=0.0)
    # 수동 보정으로 저장되는 깊이 모델 스케일. 1.0이면 모델 출력 그대로 사용.
    depth_scale: float = Field(default=1.0, gt=0.0, le=10.0)

    # 얼굴 키포인트 기반 거리 (근거리)
    face_ipd_m: float = Field(default=0.063, gt=0.0)
    face_head_width_m: float = Field(default=0.15, gt=0.0)
    face_keypoint_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    face_ipd_scale_px_m: float | None = Field(default=None, gt=0.0)
    face_min_eye_distance_px: float = Field(default=8.0, gt=0.0)

    # 체커보드로 만든 camera_calibration.json 경로 (없으면 FOV 사용)
    camera_calibration_path: str = "camera_calibration.json"

    # 단안 거리 추정 (카메라 캘리브레이션)
    # camera_focal_length_px 가 있으면 FOV 대신 이 값을 사용한다.
    camera_horizontal_fov_deg: float = 70.0
    camera_focal_length_px: float | None = None
    # 너비/높이 거리 추정값이 이 비율보다 어긋나면 거리를 버린다.
    distance_min_axis_agreement: float = Field(default=0.6, gt=0.0, le=1.0)
    # 화면 경계에서 이 비율 안쪽에 닿은 변은 잘린 것으로 본다.
    distance_edge_margin_ratio: float = Field(default=0.02, ge=0.0, lt=0.1)
    # JSON: {"my-obstacle": {"width_m": 1.2, "height_m": 0.8}}
    object_sizes: dict[str, ObjectSize] = Field(default_factory=dict)

    # 드론 식별자 (기체별로 배포 시 변경)
    drone_id: str = "drone-001"

    @property
    def backend_detections_url(self) -> str:
        return f"{self.backend_base_url.rstrip('/')}/api/v1/detections"

    @property
    def resolved_device(self) -> str:
        """``auto`` 이면 CUDA 사용 가능 여부에 따라 GPU/CPU를 고른다."""
        device = self.device.strip()
        if device and device.lower() != "auto":
            return device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "0" if torch.cuda.is_available() else "cpu"

    @property
    def resolved_object_sizes(self) -> dict[str, ObjectSize]:
        """기본 크기 표 위에 사용자 정의 클래스 크기를 덮어쓴다."""
        sizes = dict(DEFAULT_OBJECT_SIZES)
        sizes.update({label.strip().lower(): size for label, size in self.object_sizes.items()})
        return sizes


@lru_cache
def get_settings() -> Settings:
    return Settings()
