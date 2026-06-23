import cv2
import numpy as np
import yaml
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

def _init_pose():
    try:
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
        from mediapipe.tasks.python.core.base_options import BaseOptions
        import mediapipe as mp
        model_path = Path(mp.__file__).parent / "models" / "pose_landmarker_lite.task"
        if not model_path.exists():
            logger.debug("MediaPipe pose model not found at %s", model_path)
            return None
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            min_pose_detection_confidence=0.5
        )
        return PoseLandmarker.create_from_options(options)
    except Exception as exc:
        logger.debug("PoseLandmarker init failed: %s", exc)
        try:
            import mediapipe as mp
            logger.info("Falling back to MediaPipe Pose solution API")
            return mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        except Exception as exc2:
            logger.debug("MediaPipe fallback also failed: %s", exc2)
            return None

class SeatbeltDetector:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["violations"]
        self.pose_model = _init_pose()
        self.confidence_threshold = self.config.get("seatbelt_confidence_threshold", 0.5)
        if self.pose_model is None:
            logger.warning("MediaPipe pose not available. Seatbelt detection using fallback edge detection only.")

    @staticmethod
    def _diagonal_edge_response(gray: np.ndarray, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        h, w = gray.shape
        x1, y1 = max(0, int(p1[0])), max(0, int(p1[1]))
        x2, y2 = min(w - 1, int(p2[0])), min(h - 1, int(p2[1]))
        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            return 0.0
        edges = cv2.Canny(gray, 50, 150)
        line_mask = np.zeros_like(edges)
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 6)
        overlap = cv2.bitwise_and(edges, line_mask)
        edge_pixels = np.count_nonzero(overlap)
        total_pixels = np.count_nonzero(line_mask)
        if total_pixels == 0:
            return 0.0
        return edge_pixels / total_pixels

    @staticmethod
    def _estimate_pose_fallback(gray: np.ndarray) -> float:
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=10)
        if lines is None:
            return 0.0
        diag_lines = 0
        total_lines = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            if 30 < angle < 60 or 120 < angle < 150:
                diag_lines += 1
        return diag_lines / max(total_lines, 1)

    def detect(self, frame: np.ndarray, tracked_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        violations = []
        car_tracks = [t for t in tracked_objects if t["class"] == 0]

        for track in car_tracks:
            x1, y1, x2, y2 = map(int, track["bbox"])
            roi_x1, roi_y1 = x1, y1
            roi_x2, roi_y2 = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
            roi_x1, roi_y1 = max(0, roi_x1), max(0, roi_y1)
            roi_x2, roi_y2 = min(frame.shape[1], roi_x2), min(frame.shape[0], roi_y2)
            if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
                continue

            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            h, w, _ = roi.shape
            seatbelt_conf = 0.0

            if self.pose_model is not None:
                try:
                    if hasattr(self.pose_model, "process"):
                        results = self.pose_model.process(roi_rgb)
                        if results.pose_landmarks:
                            lm = results.pose_landmarks.landmark
                            ls = (lm[11].x * w, lm[11].y * h)
                            rs = (lm[12].x * w, lm[12].y * h)
                            lh = (lm[23].x * w, lm[23].y * h)
                            rh = (lm[24].x * w, lm[24].y * h)
                            d1 = self._diagonal_edge_response(gray, ls, rh)
                            d2 = self._diagonal_edge_response(gray, rs, lh)
                            seatbelt_conf = max(d1, d2)
                except Exception as exc:
                    logger.debug("Pose estimation failed for track %s: %s", track.get("vehicle_id"), exc)

            fallback_conf = self._estimate_pose_fallback(gray)
            seatbelt_conf = max(seatbelt_conf, fallback_conf)

            if seatbelt_conf < self.confidence_threshold and seatbelt_conf > 0:
                violations.append({
                    "violation_type": "seatbelt",
                    "confidence": float(1.0 - seatbelt_conf),
                    "bbox": track["bbox"],
                    "evidence_crop": roi,
                    "metadata": {"vehicle_id": track["vehicle_id"], "edge_response": float(seatbelt_conf)}
                })

        return violations

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running Seatbelt Detector smoke test...")
    detector = SeatbeltDetector()
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_tracks = [{"class": 0, "bbox": [100, 100, 400, 300], "vehicle_id": 1}]
    v = detector.detect(dummy_frame, dummy_tracks)
    print(f"Smoke test complete. Violations detected: {len(v)}")
