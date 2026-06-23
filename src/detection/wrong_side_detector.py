import cv2
import numpy as np
import yaml
import logging
from typing import List, Dict, Any, Optional
from collections import deque

logger = logging.getLogger(__name__)

class WrongSideDetector:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["violations"]

        self.cosine_threshold = self.config.get("wrong_side_cosine_threshold", -0.5)
        self.prev_gray = {}  # Keyed by camera_id
        self.dominant_flow_vectors = {}  # Keyed by camera_id

    def _calculate_dominant_flow(self, flow: np.ndarray) -> Optional[np.ndarray]:
        # Use a central ROI to calculate dominant flow, avoiding edges
        h, w = flow.shape[:2]
        roi_flow = flow[h//4:3*h//4, w//4:3*w//4]
        
        mag, ang = cv2.cartToPolar(roi_flow[..., 0], roi_flow[..., 1])
        
        # Filter for significant movements
        mag_thresh = np.mean(mag)
        significant_flow = roi_flow[mag > mag_thresh]
        
        if len(significant_flow) > 0:
            avg_flow = np.mean(significant_flow, axis=0)
            norm = np.linalg.norm(avg_flow)
            if norm > 1e-4:
                return avg_flow / norm
        return None

    def detect(self, frame: np.ndarray, tracked_objects: List[Dict[str, Any]], camera_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if camera_id is None:
            camera_id = "default"

        violations = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if camera_id not in self.dominant_flow_vectors:
            self.dominant_flow_vectors[camera_id] = deque(maxlen=10)

        if camera_id in self.prev_gray and self.prev_gray[camera_id] is not None:
            # Farneback Optical Flow
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray[camera_id], gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            dominant_vector = self._calculate_dominant_flow(flow)
            if dominant_vector is not None:
                self.dominant_flow_vectors[camera_id].append(dominant_vector)

            if len(self.dominant_flow_vectors[camera_id]) > 0:
                # Use the averaged dominant vector
                avg_dominant_vector = np.mean(list(self.dominant_flow_vectors[camera_id]), axis=0)

                for track in tracked_objects:
                    traj = track.get("trajectory", [])
                    if len(traj) > 5: # Need a reasonably long trajectory
                        # Vehicle heading vector
                        p1 = np.array(traj[-5])
                        p2 = np.array(traj[-1])
                        vehicle_vector = p2 - p1
                        norm = np.linalg.norm(vehicle_vector)

                        if norm > 1e-4:
                            vehicle_vector_norm = vehicle_vector / norm

                            # Cosine similarity
                            cosine_sim = np.dot(vehicle_vector_norm, avg_dominant_vector)

                            if cosine_sim < self.cosine_threshold:
                                violations.append({
                                    "violation_type": "wrong_side",
                                    "confidence": 1.0 - (cosine_sim + 1.0) / 2.0,
                                    "bbox": track["bbox"],
                                    "evidence_crop": frame, # Full frame is evidence
                                    "metadata": {"vehicle_id": track["vehicle_id"], "cosine_similarity": cosine_sim}
                                })

        self.prev_gray[camera_id] = gray
        return violations

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running Wrong Side Detector smoke test...")
    detector = WrongSideDetector()
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_tracks = [{
        "vehicle_id": 1, "bbox": [10, 10, 50, 50],
        "trajectory": [(20, 300), (20, 250), (20, 200), (20, 150), (20, 100), (20, 50)]
    }]
    detector.dominant_flow_vectors["default"] = deque(maxlen=10)
    detector.dominant_flow_vectors["default"].append(np.array([0, 1]))
    v = detector.detect(dummy_frame, dummy_tracks)
    print(f"Smoke test complete. Violations detected: {len(v)}")
