import pytest

from neurovector_vision.config import Settings
from neurovector_vision.detector import InvalidImage, ObstacleDetector

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


class FakeModel:
    def __init__(self) -> None:
        self.sources: list[object] = []

    def predict(self, source, **_kwargs):
        self.sources.append(source)
        return []


def test_detector_decodes_bytes_to_bgr_ndarray() -> None:
    ok, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok

    detector = ObstacleDetector(Settings(_env_file=None))
    model = FakeModel()
    detector._model = model

    assert detector.detect(encoded.tobytes()) == []
    assert model.sources[0].shape == (8, 8, 3)


def test_detector_rejects_invalid_bytes() -> None:
    detector = ObstacleDetector(Settings(_env_file=None))
    detector._model = FakeModel()

    with pytest.raises(InvalidImage):
        detector.detect(b"not-an-image")
