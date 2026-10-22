import pytest

from neurovector_vision.models import BoundingBox, Detection
from neurovector_vision.tracking import TemporalTracker


def detection(distance: float | None = 3.0, *, label: str = "face", x: float = 0) -> Detection:
    return Detection(
        label=label,
        confidence=0.9,
        box=BoundingBox(x1=x, y1=0, x2=x + 100, y2=100),
        distance_m=distance,
    )


def test_tracks_follow_overlap_when_detection_order_changes() -> None:
    tracker = TemporalTracker()
    first = tracker.update([detection(x=0), detection(x=200)], now=0)
    second = tracker.update([detection(x=205), detection(x=5)], now=0.1)
    assert first[0].track_id == second[1].track_id
    assert first[1].track_id == second[0].track_id
    assert first[0].track_id != first[1].track_id


def test_classes_and_body_part_sides_never_share_tracks() -> None:
    tracker = TemporalTracker()
    first = tracker.update([detection(label="left_hand")], now=0)
    second = tracker.update([detection(label="right_hand"), detection()], now=0.1)
    assert len({first[0].track_id, second[0].track_id, second[1].track_id}) == 3


def test_each_track_matches_at_most_one_detection() -> None:
    tracker = TemporalTracker()
    first = tracker.update([detection()], now=0)
    second = tracker.update([detection(x=1), detection(x=2)], now=0.1)
    assert len({item.track_id for item in second}) == 2
    assert first[0].track_id in {item.track_id for item in second}


def test_expired_tracks_do_not_reuse_distance_history_or_id() -> None:
    tracker = TemporalTracker(max_age_s=0.5)
    first = tracker.update([detection(1.0)], now=0)
    tracker.update([], now=0.6)
    second = tracker.update([detection(10.0)], now=0.7)
    assert first[0].track_id != second[0].track_id
    assert second[0].distance_m == 10.0


def test_farther_outlier_is_filtered_and_closer_reading_is_immediate() -> None:
    tracker = TemporalTracker()
    for now in [0.0, 0.1, 0.2]:
        tracker.update([detection(3.0)], now=now)
    assert tracker.update([detection(15.0)], now=0.3)[0].distance_m == 3.0
    assert tracker.update([detection(0.5)], now=0.4)[0].distance_m == 0.5
    # Sustained retreat eventually updates the output, too.
    for now in [0.5, 0.6, 0.7, 0.8, 0.9]:
        result = tracker.update([detection(5.0)], now=now)
    assert result[0].distance_m == 5.0


def test_missing_distance_preserves_null_and_clears_old_filter() -> None:
    tracker = TemporalTracker()
    first = tracker.update([detection(1.0)], now=0)
    missing = detection(None)
    missing.distance_note = "depth_geometry_conflict"
    unknown = tracker.update([missing], now=0.1)[0]
    assert unknown.track_id == first[0].track_id
    assert unknown.distance_m is None
    assert unknown.distance_note == "depth_geometry_conflict"
    assert tracker.update([detection(5.0)], now=0.2)[0].distance_m == 5.0


def test_tracker_does_not_modify_raw_measurement() -> None:
    tracker = TemporalTracker()
    original = detection()
    tracked = tracker.update([original], now=0)[0]
    assert tracked is not original
    assert original.track_id is None
    assert tracked.track_id is not None


@pytest.mark.parametrize("options", [{"iou_threshold": 0}, {"max_age_s": -1}, {"window_size": 0}])
def test_invalid_tracker_settings(options: dict) -> None:
    with pytest.raises(ValueError):
        TemporalTracker(**options)
