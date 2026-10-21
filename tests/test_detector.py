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


class FakeBox:
    def __init__(self, coordinates: list[float]) -> None:
        self.xyxy = [np.array(coordinates)]
        self.cls = [np.array(0)]
        self.conf = [np.array(0.9)]


class FakeResult:
    names = {0: "person"}
    orig_shape = (720, 1280)

    def __init__(self, coordinates: list[float]) -> None:
        self.boxes = [FakeBox(coordinates)]


class FakePersonModel:
    def __init__(self, coordinates: list[float]) -> None:
        self._coordinates = coordinates

    def predict(self, source, **_kwargs):
        return [FakeResult(self._coordinates)]


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


def test_detector_marks_truncated_person_as_unreliable() -> None:
    ok, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok

    detector = ObstacleDetector(Settings(_env_file=None))
    detector._model = FakePersonModel([742.0, 437.0, 1272.0, 712.0])

    detections = detector.detect(encoded.tobytes())

    assert detections[0].label == "person"
    assert detections[0].priority_level == 1
    assert detections[0].distance_m is None
    assert detections[0].distance_note == "bbox_truncated"
