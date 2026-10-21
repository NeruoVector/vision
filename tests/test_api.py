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
        distance_m=4.2,
    )
    assert detection.label == "tree"
    assert detection.distance_m == 4.2
    assert detection.priority_level == 2


def test_detect_returns_distance_and_forwards_it(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None)
    forwarded: list[Detection] = []

    class FakeDetector:
        def __init__(self, _settings: Settings) -> None:
            pass

        def detect(self, _payload: bytes) -> list[Detection]:
            return [
                Detection(
                    label="tree",
                    confidence=0.9,
                    box=BoundingBox(x1=0, y1=0, x2=100, y2=200),
                    distance_m=6.4,
                    priority_level=1,
                )
            ]

    async def fake_send_detections(
        _self: object,
        _drone_id: str,
        detections: list[Detection],
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> int:
        forwarded.extend(detections)
        return len(detections)

    monkeypatch.setattr("neurovector_vision.api._settings", lambda: settings)
    monkeypatch.setattr("neurovector_vision.api.ObstacleDetector", FakeDetector)
    monkeypatch.setattr(
        "neurovector_vision.api.BackendClient.send_detections",
        fake_send_detections,
    )

    response = client.post(
        "/detect",
        files={"file": ("sample.jpg", b"not-a-real-image", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["detections"][0]["distance_m"] == 6.4
    assert response.json()["detections"][0]["priority_level"] == 1
    assert forwarded[0].distance_m == 6.4
    assert forwarded[0].priority_level == 1
