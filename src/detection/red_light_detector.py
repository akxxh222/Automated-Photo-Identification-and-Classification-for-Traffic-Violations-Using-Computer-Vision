import cv2
import numpy as np
import json
import yaml
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class RedLightDetector:
    def __init__(self, config_path="configs/config.yaml", zones_path="configs/camera_zones.json"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["violations"]
        with open(zones_path, "r") as f:
            self.zones = json.load(f)

        self.debounce_frames = self.config.get("red_light_debounce_frames", 3)
        self.crossing_vehicles = defaultdict(int)  # {vehicle_id: consecutive_frames_count}
        self.last_signal_state = "GREEN"  # Track previous state to detect transitions
        self.reported_violations = set()  # {vehicle_id} already reported in current cycle

    def _get_traffic_light_state(self, frame: np.ndarray, camera_id: str) -> str:
        # In a real system, this ROI would be part of camera_zones.json
        # For demo, we assume a fixed ROI at the top-right
        roi = frame[10:80, 550:630]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Red color range in HSV
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        red_pixel_count = cv2.countNonZero(red_mask)
        
        # If more than a threshold of pixels are red, signal is RED
        if red_pixel_count > 100:
            return "RED"
        return "GREEN" # or "OTHER"

    def detect(self, frame: np.ndarray, tracked_objects: List[Dict[str, Any]], camera_id: str) -> List[Dict[str, Any]]:
        violations = []
        camera_config = self.zones.get(camera_id)
        if not camera_config or "stop_line" not in camera_config:
            return violations

        stop_line = np.array(camera_config["stop_line"], dtype=np.int32)
        signal_state = self._get_traffic_light_state(frame, camera_id)

        # Detect light cycle transition: RED -> GREEN
        if self.last_signal_state == "RED" and signal_state == "GREEN":
            self.crossing_vehicles.clear()
            self.reported_violations.clear()

        if signal_state == "RED":
            for track in tracked_objects:
                vehicle_id = track["vehicle_id"]
                x1, y1, x2, y2 = track["bbox"]
                centroid = (int((x1 + x2) / 2), int(y2))

                if cv2.pointPolygonTest(stop_line, centroid, False) >= 0:
                    self.crossing_vehicles[vehicle_id] += 1

                    if (self.crossing_vehicles[vehicle_id] == self.debounce_frames and
                            vehicle_id not in self.reported_violations):
                        violations.append({
                            "violation_type": "red_light",
                            "confidence": 0.95,
                            "bbox": track["bbox"],
                            "evidence_crop": frame,
                            "metadata": {"vehicle_id": vehicle_id}
                        })
                        self.reported_violations.add(vehicle_id)
                else:
                    self.crossing_vehicles[vehicle_id] = 0

        self.last_signal_state = signal_state
        return violations

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running Red Light Detector smoke test...")
    detector = RedLightDetector()
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    # Make the traffic light ROI red
    cv2.rectangle(dummy_frame, (550, 10), (630, 80), (0, 0, 255), -1)
    dummy_tracks = [{"vehicle_id": 1, "bbox": [200, 510, 250, 560]}]
    
    # Simulate crossing for 3 frames
    for _ in range(3):
        v = detector.detect(dummy_frame, dummy_tracks, "CAM_001")
    print(f"Smoke test complete. Violations detected: {len(v)}")
