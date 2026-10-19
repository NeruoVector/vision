import pytest
from fastapi.testclient import TestClient

from neurovector_vision.api import app
from neurovector_vision.config import Settings
from neurovector_vision.models import BoundingBox, Detection


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.port == 8082
    assert settings.backend_detections_url.endswith("/api/v1/detections")


def test_detection_model_validation() -> None:
    detection = Detection(
        label="tree",
        confidence=0.9,
        box=BoundingBox(x1=0, y1=0, x2=10, y2=10),
    )
    assert detection.label == "tree"
