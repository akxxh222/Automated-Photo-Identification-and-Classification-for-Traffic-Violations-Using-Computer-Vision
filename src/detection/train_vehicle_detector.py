import os
import shutil
import argparse
from pathlib import Path

from src.utils.runtime import configure_runtime
from src.preprocessing.label_harmonizer import get_normalized_data_yaml, harmonize_yolo_datasets

configure_runtime()
from ultralytics import YOLO

def train(skip_train=False, prepare_labels=True, extract_zips=False, epochs=100, batch=16, imgsz=640, device=None):
    model_out_dir = Path("models")
    model_out_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = model_out_dir / "vehicle_detector.pt"

    if skip_train:
        print("Skipping training. Generating a dummy weights file for testing...")
        # If skipping, just download standard yolov8n to mock the vehicle_detector model
        model = YOLO("models/pretrained/yolov8n.pt")
        model.save(str(final_model_path))
        print(f"Saved dummy model to {final_model_path}")
        return

    print("Initializing YOLOv8m for Unified Vehicle Detection Training...")
    pretrained_path = "models/vehicle_detector.pt" if os.path.exists("models/vehicle_detector.pt") else "models/pretrained/yolov8m.pt"
    print(f"Loading from: {pretrained_path}")
    model = YOLO(pretrained_path)
    
    if prepare_labels:
        try:
            data_yaml = str(
                harmonize_yolo_datasets(task="vehicle", extract_zips=extract_zips)
            )
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")
            data_yaml = "data/processed/vehicle/data.yaml"
    else:
        data_yaml = str(get_normalized_data_yaml("vehicle", prepare=False) or "data/processed/vehicle/data.yaml")

    if not os.path.exists(data_yaml):
        print("Warning: normalized vehicle data.yaml not found. Creating a synthetic one for smoke testing...")
        os.makedirs("data/processed/vehicle", exist_ok=True)
        with open(data_yaml, "w") as f:
            f.write("path: .\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("names:\n")
            f.write("  0: car\n  1: truck\n  2: bus\n  3: two_wheeler\n  4: three_wheeler\n  5: pedestrian\n  6: rider\n  7: pillion\n")

    print(f"Starting training process ({epochs} epochs, imgsz={imgsz}, batch={batch})...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        cache=True,
        patience=20,
        project="results/training",
        name="vehicle_detector",
        exist_ok=True
    )

    # Post-training: move the best weights to the top-level models directory
    best_weights = Path("results/training/vehicle_detector/weights/best.pt")
    if best_weights.exists():
        shutil.copy(best_weights, final_model_path)
        print(f"Training complete. Best model saved to {final_model_path}")
        
        # Save dummy detection metrics for the evaluation requirement
        print("Plots and artifacts (PR curves, confusion matrix) have been saved to results/training/vehicle_detector/")
    else:
        print("Training failed to produce best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="Skip actual training and generate dummy model")
    parser.add_argument("--no-prepare-labels", action="store_true", help="Use an existing normalized data.yaml without rebuilding it")
    parser.add_argument("--extract-zips", action="store_true", help="Extract zip datasets before label harmonization")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    
    # A quick smoke-test on load
    print("=== Stage 2: Vehicle Detector Training Script ===")
    train(
        skip_train=args.skip_train,
        prepare_labels=not args.no_prepare_labels,
        extract_zips=args.extract_zips,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )
