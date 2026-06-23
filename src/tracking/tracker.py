import yaml
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict, deque

from src.utils.runtime import configure_runtime

configure_runtime()
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class UnifiedTracker:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.det_cfg = config.get("detection", {})
        self.trk_cfg = config.get("tracking", {})
        self.fps = config.get("preprocessing", {}).get("target_fps", 5)

        model_path = self.det_cfg.get("model_path", "models/vehicle_detector.pt")
        self.fallback_mode = not Path(model_path).exists()
        if self.fallback_mode:
            logger.warning(
                "Vehicle detector model not found at %s. "
                "Falling back to yolov8n COCO classes until custom weights are trained.",
                model_path,
            )
            self.model = YOLO("models/pretrained/yolov8n.pt")
            self.class_map = {
                0: 5,  # person -> pedestrian
                1: 3,  # bicycle -> two_wheeler
                2: 0,  # car -> car
                3: 3,  # motorcycle -> two_wheeler
                5: 2,  # bus -> bus
                7: 1,  # truck -> truck
            }
            self.classes = [0, 1, 2, 3, 5, 7]
        else:
            self.model = YOLO(model_path)
            self.class_map = None
            
        self.conf_thresh = self.det_cfg.get("confidence_threshold", 0.5)
        self.iou_thresh = self.det_cfg.get("iou_threshold", 0.45)
        if not self.fallback_mode:
            self.classes = self.det_cfg.get("classes", [0, 1, 2, 3, 4, 5, 6, 7])
        
        self.track_buffer = self.trk_cfg.get("track_buffer", 30)
        
        # State management
        # trajectories stores deque of (x, y) centroids
        self.trajectories = defaultdict(lambda: deque(maxlen=self.track_buffer))
        # track_frames stores total number of frames a track has been seen
        self.track_frames = defaultdict(int)

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        # First try tracking (for video streams)
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.classes,
            verbose=False
        )
        
        tracked_objects = []
        boxes_data = None
        track_ids_data = None
        
        if len(results) > 0 and results[0].boxes is not None:
            if results[0].boxes.id is not None:
                # ByteTrack assigned IDs (video mode)
                boxes_data = results[0].boxes.xyxy.cpu().numpy()
                track_ids_data = results[0].boxes.id.cpu().numpy().astype(int)
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
            else:
                # No track IDs (single image fallback) — run plain detection
                det_results = self.model.predict(
                    frame,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    classes=self.classes,
                    verbose=False
                )
                if len(det_results) > 0 and det_results[0].boxes is not None:
                    boxes_data = det_results[0].boxes.xyxy.cpu().numpy()
                    track_ids_data = np.arange(len(boxes_data), dtype=int)  # fake IDs
                    class_ids = det_results[0].boxes.cls.cpu().numpy().astype(int)
                    confidences = det_results[0].boxes.conf.cpu().numpy()
            
            if boxes_data is not None:
                for bbox, track_id, cls_id, conf in zip(boxes_data, track_ids_data, class_ids, confidences):
                    mapped_cls = self.class_map.get(int(cls_id), int(cls_id)) if self.class_map else int(cls_id)
                    x1, y1, x2, y2 = bbox
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    
                    self.trajectories[track_id].append((cx, cy))
                    self.track_frames[track_id] += 1
                    
                    # Dwell time in seconds
                    dwell_time = self.track_frames[track_id] / self.fps
                    
                    # Estimated speed (px/frame): Displacement over the trajectory window
                    trajectory = self.trajectories[track_id]
                    if len(trajectory) > 1:
                        dx = trajectory[-1][0] - trajectory[0][0]
                        dy = trajectory[-1][1] - trajectory[0][1]
                        dist = np.sqrt(dx**2 + dy**2)
                        speed = dist / len(trajectory)
                    else:
                        speed = 0.0
                        
                    tracked_objects.append({
                        "vehicle_id": int(track_id),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "class": mapped_cls,
                        "confidence": float(conf),
                        "trajectory": list(trajectory),
                        "estimated_speed": float(speed),
                        "dwell_time": float(dwell_time)
                    })
                
        return tracked_objects

if __name__ == "__main__":
    print("Running Unified Tracker smoke test...")
    tracker = UnifiedTracker()
    dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    tracks = tracker.process_frame(dummy_frame)
    print(f"Smoke test complete. Detected {len(tracks)} tracks in dummy frame.")
