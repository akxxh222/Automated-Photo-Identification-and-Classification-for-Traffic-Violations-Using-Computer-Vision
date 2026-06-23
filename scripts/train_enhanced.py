#!/usr/bin/env python3
"""
Enhanced unified training script for all YOLO models and forecasters.
Optimized hyperparameters targeting >85% mAP@0.5 accuracy.
"""

import argparse
import logging
import shutil
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def configure_training():
    """Configure runtime for optimal training."""
    try:
        from src.utils.runtime import configure_runtime
        configure_runtime()
    except Exception as e:
        logger.warning(f"Could not configure runtime: {e}")

def train_model(model_name, data_yaml, base_model_size, epochs, batch_size, device):
    """Train a single YOLO model with optimized hyperparameters."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install -r requirements.txt")
        return False
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Base Model: {base_model_size}")
    logger.info(f"Epochs: {epochs}, Batch Size: {batch_size}")
    
    if not Path(data_yaml).exists():
        logger.error(f"Data YAML not found: {data_yaml}")
        return False
    
    try:
        # Load pretrained model
        model = YOLO(f"yolov8{base_model_size}.pt")
        
        # Optimized training configuration for high accuracy
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            device=device if device else 0,  # Default to GPU 0
            
            # Data augmentation for better generalization
            augment=True,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            
            # Learning rate schedule
            lr0=0.01,
            lrf=0.01,
            
            # Optimization
            optimizer='SGD',
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            
            # Regularization
            dropout=0.0,
            
            # Cache and performance
            cache='ram',  # or 'disk' if RAM limited
            patience=30,  # Early stopping patience
            
            # Validation
            val=True,
            save_period=10,
            save=True,
            exist_ok=True,
            verbose=True,
            
            # Project and naming
            project="runs/detect",
            name=f"{model_name}_opt",
            
            # Additional metrics
            plots=True,
            conf=0.25,
            iou=0.6,
        )
        
        logger.info(f"✓ Training completed for {model_name}")
        
        # Copy best weights to models directory
        best_weights = Path(f"runs/detect/{model_name}_opt/weights/best.pt")
        if best_weights.exists():
            model_path = Path(f"models/{model_name}.pt")
            shutil.copy2(best_weights, model_path)
            logger.info(f"✓ Best weights saved to: {model_path}")
            
            # Print metrics summary
            logger.info(f"\nTraining Results:")
            logger.info(f"  - Last epoch mAP@0.5: {results.results_dict.get('metrics/mAP50', 'N/A')}")
            logger.info(f"  - Last epoch mAP@0.5:0.95: {results.results_dict.get('metrics/mAP', 'N/A')}")
            return True
        else:
            logger.warning(f"⚠ Best weights not found at {best_weights}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error training {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_vehicle_detector(epochs=100, batch_size=8, device=None):
    """Train vehicle detector with YOLOv8m."""
    data_yaml = "data/processed/vehicle/data.yaml"
    return train_model("vehicle_detector", data_yaml, "m", epochs, batch_size, device)


def train_helmet_detector(epochs=80, batch_size=16, device=None):
    """Train helmet detector with YOLOv8s."""
    data_yaml = "data/processed/helmet/data.yaml"
    return train_model("helmet_detector", data_yaml, "s", epochs, batch_size, device)


def train_triple_riding_detector(epochs=80, batch_size=16, device=None):
    """Train triple riding detector with YOLOv8s."""
    data_yaml = "data/processed/triple_riding/data.yaml"
    return train_model("triple_riding_detector", data_yaml, "s", epochs, batch_size, device)


def train_plate_detector(epochs=80, batch_size=16, device=None):
    """Train license plate detector with YOLOv8n."""
    data_yaml = "data/processed/license_plate/data.yaml"
    return train_model("plate_detector", data_yaml, "n", epochs, batch_size, device)


def train_forecasting_models():
    """Train traffic forecasting models (Prophet + XGBoost)."""
    logger.info(f"\n{'='*60}")
    logger.info("Training Forecasting Models (Prophet + XGBoost)")
    logger.info(f"{'='*60}")
    
    try:
        from src.violations.forecaster import TrafficForecaster
        
        logger.info("Initializing TrafficForecaster...")
        forecaster = TrafficForecaster()
        
        logger.info("Training forecasting models...")
        metrics = forecaster.train_models(force_retrain=True)
        
        logger.info("✓ Forecasting models trained")
        logger.info(f"Metrics: {metrics}")
        
        return True
    except Exception as e:
        logger.warning(f"⚠ Could not train forecasting models: {e}")
        logger.warning("This is optional and can be configured later.")
        return False


def validate_all_models():
    """Validate that all trained models exist."""
    logger.info(f"\n{'='*60}")
    logger.info("Validating Trained Models")
    logger.info(f"{'='*60}")
    
    models_dir = Path("models")
    required_models = [
        "vehicle_detector.pt",
        "helmet_detector.pt",
        "triple_riding_detector.pt",
        "plate_detector.pt",
    ]
    
    all_exist = True
    for model_file in required_models:
        model_path = models_dir / model_file
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024*1024)
            logger.info(f"✓ {model_file} ({size_mb:.1f} MB)")
        else:
            logger.warning(f"⚠ {model_file} NOT FOUND")
            all_exist = False
    
    if all_exist:
        logger.info("\n✓ All required models are ready!")
    else:
        logger.warning("\n⚠ Some models are missing. Please check training results.")
    
    return all_exist


def main():
    """Main training orchestration."""
    parser = argparse.ArgumentParser(
        description="Comprehensive ML model training for traffic violation detection"
    )
    parser.add_argument("--no-extract-zips", action="store_true", 
                        help="Skip dataset extraction (use existing)")
    parser.add_argument("--vehicle-epochs", type=int, default=100,
                        help="Epochs for vehicle detector training")
    parser.add_argument("--helmet-epochs", type=int, default=80,
                        help="Epochs for helmet detector training")
    parser.add_argument("--triple-epochs", type=int, default=80,
                        help="Epochs for triple riding detector training")
    parser.add_argument("--plate-epochs", type=int, default=80,
                        help="Epochs for plate detector training")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--device", default=None,
                        help="Device to use (0 for GPU, cpu for CPU)")
    parser.add_argument("--skip-forecasting", action="store_true",
                        help="Skip forecasting model training")
    parser.add_argument("--models-only", action="store_true",
                        help="Train only detection models (not forecasting)")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("TRAFFIC ENFORCEMENT PLATFORM - ENHANCED TRAINING")
    logger.info("="*60)
    
    # Configure environment
    configure_training()
    
    # Prepare directories
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("runs/detect").mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract and prepare datasets if needed
    if not args.no_extract_zips:
        logger.info("\n[1/5] Extracting and preparing datasets...")
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "scripts/prepare_datasets.py"],
                capture_output=False
            )
            if result.returncode != 0:
                logger.warning("⚠ Dataset preparation had issues. Continuing anyway...")
        except Exception as e:
            logger.warning(f"⚠ Could not run prepare_datasets.py: {e}")
    
    # Step 2-5: Train individual models
    logger.info("\n[2/5] Training Vehicle Detector (YOLOv8m)...")
    vehicle_ok = train_vehicle_detector(
        epochs=args.vehicle_epochs,
        batch_size=args.batch,
        device=args.device
    )
    
    logger.info("\n[3/5] Training Helmet Detector (YOLOv8s)...")
    helmet_ok = train_helmet_detector(
        epochs=args.helmet_epochs,
        batch_size=args.batch,
        device=args.device
    )
    
    logger.info("\n[4/5] Training Triple Riding Detector (YOLOv8s)...")
    triple_ok = train_triple_riding_detector(
        epochs=args.triple_epochs,
        batch_size=args.batch,
        device=args.device
    )
    
    logger.info("\n[5/5] Training Plate Detector (YOLOv8n)...")
    plate_ok = train_plate_detector(
        epochs=args.plate_epochs,
        batch_size=args.batch,
        device=args.device
    )
    
    # Step 6: Train forecasting models (optional)
    forecasting_ok = True
    if not args.skip_forecasting and not args.models_only:
        logger.info("\n[6/6] Training Forecasting Models...")
        forecasting_ok = train_forecasting_models()
    
    # Validation
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    
    results = {
        "Vehicle Detector": vehicle_ok,
        "Helmet Detector": helmet_ok,
        "Triple Riding Detector": triple_ok,
        "Plate Detector": plate_ok,
        "Forecasting Models": forecasting_ok if not args.skip_forecasting else "Skipped"
    }
    
    for name, status in results.items():
        status_str = "✓ OK" if status is True else ("⚠ Skipped" if status == "Skipped" else "✗ FAILED")
        logger.info(f"{name}: {status_str}")
    
    # Validate all models
    all_ok = validate_all_models()
    
    logger.info("\n" + "="*60)
    if all_ok:
        logger.info("✓ ALL TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("\nNext steps:")
        logger.info("1. Test models: python scripts/run_demo.py")
        logger.info("2. Evaluate: python scripts/evaluate_all.py")
        logger.info("3. Dashboard: make dashboard")
        logger.info("4. API: make api")
    else:
        logger.warning("⚠ Some models failed to train. Check logs above.")
    logger.info("="*60)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
