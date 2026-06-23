import time

from src.violations.violation_aggregator import deduplicate_violations


def test_deduplicate_violations_within_window():
    now = time.time()
    events = [
        {"vehicle_id": 1, "violation_type": "helmet", "timestamp": now},
        {"vehicle_id": 1, "violation_type": "helmet", "timestamp": now + 1},
        {"vehicle_id": 2, "violation_type": "helmet", "timestamp": now},
    ]
    result = deduplicate_violations(events, window_seconds=5.0)
    assert len(result) == 2


def test_deduplicate_violations_respects_window_boundary():
    now = time.time()
    events = [
        {"vehicle_id": 1, "violation_type": "helmet", "timestamp": now},
        {"vehicle_id": 1, "violation_type": "helmet", "timestamp": now + 10},
    ]
    result = deduplicate_violations(events, window_seconds=5.0)
    assert len(result) == 2


def test_metadata_safe_access_pattern():
    violation = {"violation_type": "helmet"}
    vehicle_id = violation.get("metadata", {}).get("vehicle_id")
    assert vehicle_id is None

    violation_with_meta = {"violation_type": "helmet", "metadata": {"vehicle_id": 42}}
    assert violation_with_meta.get("metadata", {}).get("vehicle_id") == 42
