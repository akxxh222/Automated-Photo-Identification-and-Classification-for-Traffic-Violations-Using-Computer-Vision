import cv2
import numpy as np

from src.detection.red_light_detector import RedLightDetector


def _green_light_frame():
    return np.zeros((640, 640, 3), dtype=np.uint8)


def _red_light_frame():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (550, 10), (630, 80), (0, 0, 255), -1)
    return frame


def test_red_light_cycle_reset_clears_debounce_state():
    detector = RedLightDetector()
    detector.crossing_vehicles[1] = 3
    detector.reported_violations.add(1)
    detector.last_signal_state = "RED"

    detector.detect(_green_light_frame(), [], "CAM_001")

    assert len(detector.crossing_vehicles) == 0
    assert len(detector.reported_violations) == 0


def test_red_light_debounce_fires_once_per_cycle():
    detector = RedLightDetector()
    frame = _red_light_frame()
    tracks = [{"vehicle_id": 7, "bbox": [200, 480, 250, 510]}]

    violations = []
    for _ in range(5):
        violations.extend(detector.detect(frame, tracks, "CAM_001"))

    assert len(violations) == 1
    assert violations[0]["metadata"]["vehicle_id"] == 7
