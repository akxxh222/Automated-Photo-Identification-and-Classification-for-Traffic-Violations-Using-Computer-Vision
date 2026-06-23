import os
import cv2
import numpy as np
import yaml
import argparse
import shutil
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.utils.runtime import configure_runtime
from src.preprocessing.label_harmonizer import get_normalized_data_yaml, harmonize_yolo_datasets

configure_runtime()
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class HelmetDetector:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["violations"]

        model_path = self.config.get("helmet_model_path", "models/helmet_detector.pt")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Helmet detector model not found at {model_path}. Please train the model first.")
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray, tracked_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        violations = []
        # Prefer two-wheeler crops, but keep rider/pillion crops if the detector exposes them.
        rider_tracks = [t for t in tracked_objects if t["class"] in [3, 6, 7]]

        if rider_tracks:
            for track in rider_tracks:
                x1, y1, x2, y2 = map(int, track["bbox"])
                
                # Ensure coords are within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                roi = frame[y1:y2, x1:x2]
                
                # Run helmet detection model on the rider ROI
                results = self.model(roi, verbose=False)
                
                if len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = self.model.names.get(cls_id, str(cls_id))
                        
                        if class_name == "helmet_off" and conf > 0.5:
                            violations.append({
                                "violation_type": "helmet",
                                "confidence": conf,
                                "bbox": track["bbox"],
                                "evidence_crop": roi,
                                "metadata": {"vehicle_id": track["vehicle_id"]}
                            })
                            break
        else:
            # Fallback: run helmet detector on the full frame
            results = self.model(frame, verbose=False, conf=0.25)
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.model.names.get(cls_id, str(cls_id))
                    
                    if class_name == "helmet_off" and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        violations.append({
                            "violation_type": "helmet",
                            "confidence": conf,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "evidence_crop": frame[y1:y2, x1:x2],
                            "metadata": {"vehicle_id": -1}
                        })
                        
        return violations

def train(skip_train=False, prepare_labels=True, extract_zips=False, epochs=50, batch=16, imgsz=640, device=None):
    model_out_dir = Path("models")
    model_out_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = model_out_dir / "helmet_detector.pt"

    if skip_train:
        print("Skipping helmet model training. Generating dummy weights...")
        model = YOLO("models/pretrained/yolov8n.pt")
        model.save(str(final_model_path))
        return

    print(f"Training Helmet Detector (epochs={epochs}, imgsz={imgsz}, batch={batch})...")
    model = YOLO("models/pretrained/yolov8s.pt")

    if prepare_labels:
        data_yaml = str(harmonize_yolo_datasets(task="helmet", extract_zips=extract_zips, strict=False))
    else:
        data_yaml = str(get_normalized_data_yaml("helmet", prepare=False) or "data/processed/helmet/data.yaml")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        cache=True,
        patience=20,
        project="results/training",
        name="helmet_detector",
        exist_ok=True
    )

    best_weights = Path("results/training/helmet_detector/weights/best.pt")
    if best_weights.exists():
        shutil.copy(best_weights, final_model_path)
        print(f"Training complete. Best model saved to {final_model_path}")
    else:
        print("Training failed to produce best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="Skip training for demo.")
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
    
    print("Running Helmet Detector smoke test...")
    detector = HelmetDetector()
    import numpy as np
    dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    dummy_tracks = [{"class": 6, "bbox": [100, 100, 200, 300], "vehicle_id": 1}]
    v = detector.detect(dummy_frame, dummy_tracks)
    print(f"Smoke test complete. Violations detected: {len(v)}")
