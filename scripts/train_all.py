import argparse
from pathlib import Path

from src.preprocessing.dataset_prep import prepare_plate_dataset, prepare_yolo_dataset
from src.detection.train_vehicle_detector import train as train_vehicle
from src.detection.helmet_detector import train as train_helmet
from src.detection.triple_riding_detector import train as train_triple
from src.ocr.train_plate_detector import train as train_plate
from src.violations.forecaster import TrafficForecaster


def _ensure_data_yaml(task: str, prepare_fn, extract_zips: bool, strict: bool = False):
    processed_map = {
        "vehicle": Path("data/processed/vehicle/data.yaml"),
        "helmet": Path("data/processed/helmet/data.yaml"),
        "triple_riding": Path("data/processed/triple_riding/data.yaml"),
        "license_plate": Path("data/processed/license_plate/data.yaml"),
    }
    data_yaml = processed_map[task]
    if data_yaml.exists():
        return data_yaml
    if task == "license_plate":
        return prepare_fn()
    return prepare_fn(task, extract_zips=extract_zips, strict=strict)


def main(
    extract_zips=True,
    epochs_vehicle=100,
    epochs_helmet=80,
    epochs_triple=80,
    epochs_plate=80,
    batch=8,
    device=None,
):
    print("=== Preparing datasets ===")
    vehicle_yaml = _ensure_data_yaml("vehicle", prepare_yolo_dataset, extract_zips, strict=True)
    helmet_yaml = _ensure_data_yaml("helmet", prepare_yolo_dataset, extract_zips, strict=False)
    triple_yaml = _ensure_data_yaml("triple_riding", prepare_yolo_dataset, extract_zips, strict=False)
    plate_yaml = _ensure_data_yaml("license_plate", prepare_plate_dataset, extract_zips=False)

    print(f"Vehicle dataset: {vehicle_yaml}")
    print(f"Helmet dataset: {helmet_yaml}")
    print(f"Triple riding dataset: {triple_yaml}")
    print(f"Plate dataset: {plate_yaml}")

    print("\n=== Training detectors ===")
    train_vehicle(skip_train=False, prepare_labels=False, extract_zips=False, epochs=epochs_vehicle, batch=batch, device=device)
    train_helmet(skip_train=False, prepare_labels=False, extract_zips=False, epochs=epochs_helmet, batch=batch, device=device)
    train_triple(skip_train=False, prepare_labels=False, extract_zips=False, epochs=epochs_triple, batch=batch, device=device)
    train_plate(skip_train=False, prepare_labels=False, extract_zips=False, epochs=epochs_plate, batch=batch, device=device)

    print("\n=== Training forecaster ===")
    forecaster = TrafficForecaster()
    metrics = forecaster.train_models(force_retrain=True)
    print(metrics)

    print("\nAll models have been prepared and trained where possible.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-extract-zips", action="store_true", help="Do not auto-extract archive datasets before training.")
    parser.add_argument("--vehicle-epochs", type=int, default=100)
    parser.add_argument("--helmet-epochs", type=int, default=80)
    parser.add_argument("--triple-epochs", type=int, default=80)
    parser.add_argument("--plate-epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    main(
        extract_zips=not args.no_extract_zips,
        epochs_vehicle=args.vehicle_epochs,
        epochs_helmet=args.helmet_epochs,
        epochs_triple=args.triple_epochs,
        epochs_plate=args.plate_epochs,
        batch=args.batch,
        device=args.device,
    )
