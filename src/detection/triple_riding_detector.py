import os
import cv2
import numpy as np
import yaml
import argparse
import shutil
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from src.utils.runtime import configure_runtime
from src.preprocessing.label_harmonizer import get_normalized_data_yaml, harmonize_yolo_datasets

configure_runtime()
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class TripleRidingDetector:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["violations"]

        model_path = self.config.get("triple_riding_model_path", "models/triple_riding_detector.pt")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Triple riding detector model not found at {model_path}. Please train the model first.")
        self.model = YOLO(model_path)

    def _is_inside(self, inner_bbox: List[float], outer_bbox: List[float]) -> bool:
        ix1, iy1, ix2, iy2 = inner_bbox
        ox1, oy1, ox2, oy2 = outer_bbox
        
        # Check if the center of inner is inside outer bounds
        cx, cy = (ix1 + ix2) / 2, (iy1 + iy2) / 2
        return (ox1 <= cx <= ox2) and (oy1 <= cy <= oy2)

    def _count_persons_from_model(self, roi: np.ndarray) -> int:
        results = self.model(roi, verbose=False)
        count = 0
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                if class_name == "single_rider" and conf > 0.4:
                    count = max(count, 1)
                elif class_name == "double_rider" and conf > 0.4:
                    count = max(count, 2)
                elif class_name == "triple_rider" and conf > 0.4:
                    count = max(count, 3)
        return count

    def detect(self, frame: np.ndarray, tracked_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        violations = []
        
        # Stage 2 classes: 0: car, 1: truck, 2: bus, 3: two_wheeler, 4: three_wheeler, 5: pedestrian, 6: rider, 7: pillion
        two_wheelers = [t for t in tracked_objects if t["class"] == 3]
        persons = [t for t in tracked_objects if t["class"] in [5, 6, 7]]
        
        if two_wheelers:
            for tw in two_wheelers:
                x1, y1, x2, y2 = map(int, tw["bbox"])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                roi = frame[y1:y2, x1:x2]
                
                person_count_model = self._count_persons_from_model(roi)
                person_count_fallback = sum(1 for p in persons if self._is_inside(p["bbox"], tw["bbox"]))
                final_count = max(person_count_model, person_count_fallback)
                
                if final_count >= 3:
                    violations.append({
                        "violation_type": "triple_riding",
                        "confidence": 0.85,
                        "bbox": tw["bbox"],
                        "evidence_crop": roi,
                        "metadata": {"vehicle_id": tw["vehicle_id"], "person_count": final_count}
                    })
        else:
            # Fallback: run triple riding model on full frame
            person_count = self._count_persons_from_model(frame)
            if person_count >= 3:
                violations.append({
                    "violation_type": "triple_riding",
                    "confidence": 0.80,
                    "bbox": [0, 0, frame.shape[1], frame.shape[0]],
                    "evidence_crop": frame,
                    "metadata": {"vehicle_id": -1, "person_count": person_count}
                })
                
        return violations

def train(skip_train=False, prepare_labels=True, extract_zips=False, epochs=50, batch=16, imgsz=640, device=None):
    model_out_dir = Path("models")
    model_out_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = model_out_dir / "triple_riding_detector.pt"

    if skip_train:
        print("Skipping triple riding model training. Generating dummy weights...")
        model = YOLO("models/pretrained/yolov8n.pt")
        model.save(str(final_model_path))
        return

    print(f"Training Triple Riding Detector (epochs={epochs}, imgsz={imgsz}, batch={batch})...")
    model = YOLO("models/pretrained/yolov8s.pt")

    if prepare_labels:
        data_yaml = str(harmonize_yolo_datasets(task="triple_riding", extract_zips=extract_zips, strict=False))
    else:
        data_yaml = str(get_normalized_data_yaml("triple_riding", prepare=False) or "data/processed/triple_riding/data.yaml")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        cache=False,
        patience=20,
        project="results/training",
        name="triple_riding_detector",
        exist_ok=True,
        workers=0
    )

    best_weights = Path("results/training/triple_riding_detector/weights/best.pt")
    if best_weights.exists():
        shutil.copy(best_weights, final_model_path)
        print(f"Training complete. Best model saved to {final_model_path}")
    else:
        print("Training failed to produce best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--no-prepare-labels", action="store_true", help="Use an existing normalized data.yaml without rebuilding it")
    parser.add_argument("--extract-zips", action="store_true", help="Extract zip datasets before label harmonization")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    
    train(
        skip_train=args.skip_train,
        prepare_labels=not args.no_prepare_labels,
        extract_zips=args.extract_zips,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )
    
    print("Running Triple Riding Detector smoke test...")
    detector = TripleRidingDetector()
    import numpy as np
    dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Create dummy tracks: 1 two-wheeler and 3 riders positioned inside it
    dummy_tracks = [
        {"class": 3, "bbox": [50, 50, 300, 400], "vehicle_id": 10},
        {"class": 6, "bbox": [100, 100, 150, 200], "vehicle_id": 11},
        {"class": 7, "bbox": [120, 150, 180, 250], "vehicle_id": 12},
        {"class": 7, "bbox": [160, 200, 220, 300], "vehicle_id": 13},
    ]
    
    v = detector.detect(dummy_frame, dummy_tracks)
    print(f"Smoke test complete. Violations detected: {len(v)}")
    if len(v) > 0:
        print(f"Metadata output for detected violation: {v[0]['metadata']}")
