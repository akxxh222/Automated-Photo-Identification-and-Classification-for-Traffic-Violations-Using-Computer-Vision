import os
import shutil
import argparse
from pathlib import Path

from src.utils.runtime import configure_runtime
from src.preprocessing.dataset_prep import prepare_plate_dataset

configure_runtime()
from ultralytics import YOLO

def train(skip_train=False, prepare_labels=True, extract_zips=False, epochs=80, batch=16, imgsz=640, device=None):
    model_out_dir = Path("models")
    model_out_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = model_out_dir / "plate_detector.pt"

    if skip_train:
        print("Skipping Plate Detector training. Generating dummy weights...")
        model = YOLO("models/pretrained/yolov8n.pt")
        model.save(str(final_model_path))
        print(f"Saved dummy model to {final_model_path}")
        return

    print(f"Initializing YOLOv8n for License Plate Detection Training (epochs={epochs}, imgsz={imgsz}, batch={batch})...")
    model = YOLO("models/pretrained/yolov8n.pt")
    
    if prepare_labels:
        try:
            data_yaml = str(prepare_plate_dataset())
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")
            data_yaml = "data/processed/license_plate/data.yaml"
    else:
        data_yaml = "data/processed/license_plate/data.yaml"

    if not os.path.exists(data_yaml):
        print("Warning: license_plate_data.yaml not found. Preparing the plate dataset now...")
        data_yaml = str(prepare_plate_dataset())

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        cache=False,
        patience=20,
        project="results/training",
        name="plate_detector",
        exist_ok=True,
        workers=0
    )

    best_weights = Path("results/training/plate_detector/weights/best.pt")
    if best_weights.exists():
        shutil.copy(best_weights, final_model_path)
        print(f"Training complete. Best model saved to {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--no-prepare-labels", action="store_true", help="Use an existing normalized data.yaml without rebuilding it")
    parser.add_argument("--extract-zips", action="store_true", help="Extract zip datasets before label harmonization")
    parser.add_argument("--epochs", type=int, default=80)
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
