import sys
from types import ModuleType

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


class LabeledModel:
    def __init__(self, label: str, coordinates: list[float], confidence: float = 0.9) -> None:
        self.label = label
        self.coordinates = coordinates
        self.confidence = confidence
        self.arguments = None

    def predict(self, **kwargs):
        self.arguments = kwargs
        result = FakeResult(self.coordinates)
        result.names = {0: self.label}
        result.boxes[0].conf = [np.array(self.confidence)]
        return [result]


class CombinedModel:
    def __init__(self, models: list[LabeledModel]) -> None:
        self.models = models

    def predict(self, **kwargs):
        return [result for model in self.models for result in model.predict(**kwargs)]


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


@pytest.mark.parametrize(
    ("primary_label", "secondary_label"),
    [("person", "person"), ("pedestrian", "person"), ("people", "person"),
     ("motor", "motorcycle"), ("van", "car"), ("bus", "bus")],
)
def test_secondary_skips_overlapping_primary_alias(
    primary_label: str, secondary_label: str
) -> None:
    # Given a primary person and the same supplemental person.
    detector = ObstacleDetector(Settings(_env_file=None, secondary_model_path="secondary.pt"))
    detector._model = LabeledModel(primary_label, [100, 100, 150, 270])
    secondary = LabeledModel(secondary_label, [102, 100, 152, 270])
    detector._secondary_model = secondary

    # When both detectors inspect the frame.
    detections = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))

    # Then the primary identity and confidence survive without a duplicate.
    assert [(d.label, d.confidence) for d in detections] == [(primary_label, 0.9)]
    assert secondary.arguments["imgsz"] == 640
    assert secondary.arguments["conf"] == 0.35


def test_secondary_adds_distinct_person_with_distance_and_priority() -> None:
    # Given a school window and a supplemental person.
    detector = ObstacleDetector(Settings(_env_file=None, secondary_model_path="secondary.pt"))
    detector._model = LabeledModel("window", [100, 100, 150, 270])
    detector._secondary_model = LabeledModel("person", [400, 100, 450, 270])

    # When both detectors inspect the frame.
    detections = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))

    # Then both real boxes remain, and person uses the shared distance estimator.
    assert [d.label for d in detections] == ["window", "person"]
    assert detections[1].distance_m is not None
    assert detections[1].priority_level == 1
    assert detections[1].box.x1 == 400


def test_secondary_keeps_separate_people() -> None:
    # Given two spatially separate people found by different models.
    detector = ObstacleDetector(Settings(_env_file=None, secondary_model_path="secondary.pt"))
    detector._model = LabeledModel("pedestrian", [100, 100, 150, 270])
    detector._secondary_model = LabeledModel("person", [400, 100, 450, 270])

    # When both detectors inspect the frame.
    detections = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))

    # Then semantic aliases do not remove distinct objects.
    assert [d.label for d in detections] == ["pedestrian", "person"]


def test_secondary_discards_classes_outside_allowed_catalog() -> None:
    # Given a secondary model returning a class outside pedestrian/vehicle coverage.
    detector = ObstacleDetector(Settings(_env_file=None, secondary_model_path="secondary.pt"))
    detector._model = FakeModel()
    detector._secondary_model = LabeledModel("dog", [400, 100, 450, 270])

    # When the frame is inspected.
    detections = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))

    # Then supplemental detections stay within the opted-in coverage.
    assert detections == []


def test_secondary_is_disabled_by_default() -> None:
    # Given a default detector with an available secondary object.
    detector = ObstacleDetector(Settings(_env_file=None))
    detector._model = FakeModel()
    secondary = LabeledModel("person", [400, 100, 450, 270])
    detector._secondary_model = secondary

    # When the frame is inspected.
    detections = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))

    # Then no secondary inference is invoked.
    assert detections == []
    assert secondary.arguments is None


def test_secondary_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given an enabled secondary checkpoint that cannot load.
    detector = ObstacleDetector(Settings(_env_file=None, secondary_model_path="missing.pt"))
    detector._model = FakeModel()

    def fail_load(_path):
        raise FileNotFoundError("missing checkpoint")

    ultralytics = ModuleType("ultralytics")
    monkeypatch.setattr(ultralytics, "YOLO", fail_load, raising=False)
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)

    # When inference needs the enabled checkpoint, then failure is visible to the runtime.
    with pytest.raises(FileNotFoundError, match="missing checkpoint"):
        detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    ("primary_score", "secondary_score", "expected_label"),
    [(0.66055, 0.855, "car"), (0.9, 0.855, "truck"), (0.855, 0.855, "truck")],
)
def test_overlapping_vehicle_classes_keep_higher_score(
    primary_score: float, secondary_score: float, expected_label: str
) -> None:
    # Given the live Prius boxes classified differently, plus unrelated school geometry.
    detector = ObstacleDetector(Settings(_env_file=None, secondary_model_path="secondary.pt"))
    detector._model = CombinedModel([
        LabeledModel("truck", [458, 125, 605, 191], primary_score),
        LabeledModel("tree_region", [200, 10, 400, 150], 0.5613),
    ])
    detector._secondary_model = LabeledModel("car", [458, 127, 605, 190], secondary_score)

    # When the models' actual vehicle predictions are merged.
    detections = detector.detect(np.zeros((360, 640, 3), dtype=np.uint8))

    # Then the best vehicle score survives; ties favor primary and school geometry remains.
    vehicle = [d for d in detections if d.label in {"truck", "car"}]
    assert len(vehicle) == 1
    assert vehicle[0].label == expected_label
    assert vehicle[0].confidence == max(primary_score, secondary_score)
    assert [d.confidence for d in detections if d.label == "tree_region"] == [0.5613]


def test_nearby_distinct_vehicles_both_survive() -> None:
    # Given a primary truck and a separate nearby car, with only slight box overlap.
    detector = ObstacleDetector(Settings(_env_file=None, secondary_model_path="secondary.pt"))
    detector._model = LabeledModel("truck", [458, 125, 605, 191], 0.66055)
    detector._secondary_model = LabeledModel("car", [590, 127, 637, 190], 0.855)

    # When supplemental vehicle detection is merged.
    detections = detector.detect(np.zeros((360, 640, 3), dtype=np.uint8))

    # Then distinct physical vehicles are preserved with their actual predictions.
    assert [(d.label, d.confidence) for d in detections] == [("truck", 0.66055), ("car", 0.855)]


def test_primary_only_vehicle_results_are_unchanged() -> None:
    # Given conflicting primary vehicle boxes with supplemental inference disabled.
    detector = ObstacleDetector(Settings(_env_file=None))
    detector._model = CombinedModel([
        LabeledModel("truck", [458, 125, 605, 191], 0.66055),
        LabeledModel("car", [458, 127, 605, 190], 0.855),
    ])

    # When the default primary-only detector processes a frame.
    detections = detector.detect(np.zeros((360, 640, 3), dtype=np.uint8))

    # Then supplemental merging does not alter the existing primary-only behavior.
    assert [(d.label, d.confidence) for d in detections] == [("truck", 0.66055), ("car", 0.855)]
