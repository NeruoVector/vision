from neurovector_vision.priority import (
    PRIORITY_CRITICAL,
    PRIORITY_IMPORTANT,
    PRIORITY_LOW,
    priority_level_for,
    priority_name,
)


def test_face_body_and_animals_are_critical() -> None:
    assert priority_level_for("face") == PRIORITY_CRITICAL
    assert priority_level_for("person") == PRIORITY_CRITICAL
    assert priority_level_for("dog") == PRIORITY_CRITICAL
    assert priority_level_for("고양이") == PRIORITY_CRITICAL


def test_body_parts_are_critical() -> None:
    assert priority_level_for("body") == PRIORITY_CRITICAL
    assert priority_level_for("left_hand") == PRIORITY_CRITICAL
    assert priority_level_for("right_arm") == PRIORITY_CRITICAL
    assert priority_level_for("left_leg") == PRIORITY_CRITICAL
    assert priority_level_for("right_foot") == PRIORITY_CRITICAL
    assert priority_level_for("left_forearm") == PRIORITY_CRITICAL
    assert priority_level_for("손") == PRIORITY_CRITICAL


def test_vehicles_and_objects_are_important() -> None:
    assert priority_level_for("car") == PRIORITY_IMPORTANT
    assert priority_level_for("truck") == PRIORITY_IMPORTANT
    assert priority_level_for("unknown-thing") == PRIORITY_IMPORTANT


def test_leaves_and_low_risk_nature_are_low_priority() -> None:
    assert priority_level_for("leaf") == PRIORITY_LOW
    assert priority_level_for("fallen leaf") == PRIORITY_LOW
    assert priority_level_for("나뭇잎") == PRIORITY_LOW
    assert priority_level_for("leaf_damage") == PRIORITY_LOW
    assert priority_level_for("leaves_pile") == PRIORITY_LOW


def test_priority_overrides_are_normalized() -> None:
    assert priority_level_for("Leaf", {"leaf": 1}) == PRIORITY_CRITICAL
    assert priority_level_for("my obstacle", {"My Obstacle": 3}) == PRIORITY_LOW


def test_invalid_override_falls_back_to_default() -> None:
    assert priority_level_for("leaf", {"leaf": 9}) == PRIORITY_LOW
    assert priority_level_for("car", {"car": 0}) == PRIORITY_IMPORTANT


def test_priority_names() -> None:
    assert priority_name(1).startswith("1단계")
    assert priority_name(2).startswith("2단계")
    assert priority_name(3).startswith("3단계")
