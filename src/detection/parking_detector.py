import cv2
import numpy as np
import json
import yaml
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class IllegalParkingDetector:
    def __init__(self, config_path="configs/config.yaml", zones_path="configs/camera_zones.json"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["violations"]
        with open(zones_path, "r") as f:
            self.zones = json.load(f)
            
        self.stationary_time_sec = self.config.get("parking_stationary_time_sec", 30)
        self.stationary_speed_thresh = self.config.get("parking_displacement_px", 1.0)

    def detect(self, frame: np.ndarray, tracked_objects: List[Dict[str, Any]], camera_id: str) -> List[Dict[str, Any]]:
        violations = []
        camera_config = self.zones.get(camera_id)
        if not camera_config or "no_parking_zone" not in camera_config:
            return violations

        no_parking_zone = np.array(camera_config["no_parking_zone"], dtype=np.int32)

        for track in tracked_objects:
            # Check if vehicle is stationary for long enough
            if track["dwell_time"] > self.stationary_time_sec and track["estimated_speed"] < self.stationary_speed_thresh:
                x1, y1, x2, y2 = track["bbox"]
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                if cv2.pointPolygonTest(no_parking_zone, centroid, False) >= 0:
                    violations.append({
                        "violation_type": "illegal_parking",
                        "confidence": 0.9,
                        "bbox": track["bbox"],
                        "evidence_crop": frame,
                        "metadata": {"vehicle_id": track["vehicle_id"], "dwell_time": track["dwell_time"]}
                    })
        return violations

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running Illegal Parking Detector smoke test...")
    detector = IllegalParkingDetector()
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_tracks = [{
        "vehicle_id": 1, "bbox": [10, 10, 100, 100],
        "dwell_time": 45.0, "estimated_speed": 0.5
    }]
    
    v = detector.detect(dummy_frame, dummy_tracks, "CAM_001")
    print(f"Smoke test complete. Violations detected: {len(v)}")
    if len(v) > 0:
        print(f"Violation metadata: {v[0]['metadata']}")
