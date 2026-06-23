import time
import yaml
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.detection.helmet_detector import HelmetDetector
from src.detection.triple_riding_detector import TripleRidingDetector
from src.detection.wrong_side_detector import WrongSideDetector
from src.detection.red_light_detector import RedLightDetector
from src.detection.parking_detector import IllegalParkingDetector
from src.detection.seatbelt_detector import SeatbeltDetector

logger = logging.getLogger(__name__)


class NullDetector:
    """Fallback detector used when a custom model has not been trained yet."""

    def __init__(self, name: str):
        self.name = name

    def detect(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return []


def deduplicate_violations(events: List[Dict[str, Any]], window_seconds: float = 5.0) -> List[Dict[str, Any]]:
    seen = {}
    res = []
    for ev in events:
        vid = ev.get("vehicle_id")
        vtype = ev.get("violation_type")
        ts = ev.get("timestamp", time.time())
        key = (vid, vtype)
        if key not in seen or (ts - seen[key] > window_seconds):
            seen[key] = ts
            res.append(ev)
    return res

class ViolationAggregator:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["violations"]

        self.dedup_window = self.config.get("deduplication_window_sec", 5)
        self.last_seen = {}
        self.executor = ThreadPoolExecutor(max_workers=6)

        logger.info("Initializing Violation Detectors...")
        try:
            self.helmet_detector = HelmetDetector(config_path)
        except Exception as exc:
            logger.warning("Helmet detector unavailable (%s); using fallback no-op detector.", exc)
            self.helmet_detector = NullDetector("helmet")

        try:
            self.triple_riding_detector = TripleRidingDetector(config_path)
        except Exception as exc:
            logger.warning("Triple riding detector unavailable (%s); using fallback no-op detector.", exc)
            self.triple_riding_detector = NullDetector("triple_riding")

        self.wrong_side_detector = WrongSideDetector(config_path)
        self.red_light_detector = RedLightDetector(config_path)
        self.parking_detector = IllegalParkingDetector(config_path)
        self.seatbelt_detector = SeatbeltDetector(config_path)

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

    def detect(self, frame: np.ndarray, tracked_objects: List[Dict[str, Any]], camera_id: str) -> List[Dict[str, Any]]:
        violations = []

        futures = {
            self.executor.submit(self.helmet_detector.detect, frame, tracked_objects): "helmet",
            self.executor.submit(self.triple_riding_detector.detect, frame, tracked_objects): "triple_riding",
            self.executor.submit(self.wrong_side_detector.detect, frame, tracked_objects): "wrong_side",
            self.executor.submit(self.red_light_detector.detect, frame, tracked_objects, camera_id): "red_light",
            self.executor.submit(self.parking_detector.detect, frame, tracked_objects, camera_id): "parking",
            self.executor.submit(self.seatbelt_detector.detect, frame, tracked_objects): "seatbelt"
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    violations.extend(result)
            except Exception as exc:
                logger.error("Error in %s detector: %s", futures[future], exc)

        current_time = time.time()
        deduped_violations = []

        for v in violations:
            vehicle_id = v.get("metadata", {}).get("vehicle_id")
            v_type = v["violation_type"]

            if vehicle_id is None:
                deduped_violations.append(v)
                continue

            key = (vehicle_id, v_type)
            last_time = self.last_seen.get(key, 0)

            if current_time - last_time > self.dedup_window:
                self.last_seen[key] = current_time
                deduped_violations.append(v)

        keys_to_delete = [k for k, v in self.last_seen.items() if current_time - v > self.dedup_window * 2]
        for k in keys_to_delete:
            del self.last_seen[k]

        return deduped_violations

    def run(self, frame: np.ndarray, detections: List[Dict[str, Any]] = None, camera_id: str = "CAM_TEST") -> List[Dict[str, Any]]:
        if detections is None:
            detections = []
        return self.detect(frame, detections, camera_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running Violation Aggregator smoke test...")
    aggregator = ViolationAggregator()
    logger.info("Aggregator initialized and verified.")
