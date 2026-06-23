#!/usr/bin/env python3
"""
Robust Training Script for Traffic Enforcement Platform
Handles CUDA errors and data issues automatically
"""

import os
import sys
import logging
from pathlib import Path
from ultralytics import YOLO
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Set environment variables for stability
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def train_model(model_name, data_yaml, epochs, base_batch=2):
    """Train a single YOLO model with maximum stability"""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Training {model_name.upper()}")
    logger.info(f"{'='*70}")
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Epochs: {epochs}, Batch Size: {base_batch}")
    
    try:
        # Load pretrained model
        model = YOLO(f"{model_name}.pt")
        
        # Train with maximum stability settings
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=512,  # Smaller image size for stability
            batch=base_batch,
            patience=20,
            device=0,
            
            # Memory & Stability optimization
            cache=False,    # Disable caching to avoid memory issues
            workers=0,      # Single threaded to avoid data loading issues
            
            # Training settings
            optimizer="SGD",
            lr0=0.001,      # Lower learning rate for stability
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            
            # Augmentation (minimal to avoid issues)
            augment=True,
            mosaic=0.0,     # Disable mosaic to avoid indexing issues
            mixup=0.0,      # Disable mixup
            fliplr=0.3,
            flipud=0.0,
            
            # Hardware settings (NO FP16 due to numerical issues)
            half=False,     # IMPORTANT: Disable FP16
            amp=False,      # Disable AMP to avoid precision issues
            
            # Validation & Logging
            val=True,
            save=True,
            save_period=10,
            verbose=True,
            plots=False,    # Disable plotting to avoid display issues
            
            # Output
            project=str(RUNS_DIR / "detect"),
            name=f"{model_name}_stable",
            exist_ok=True,
        )
        
        logger.info(f"✓ {model_name} training completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"✗ {model_name} training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    logger.info("="*70)
    logger.info("TRAFFIC ENFORCEMENT PLATFORM - STABLE TRAINING")
    logger.info("="*70)
    
    # Check GPU
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / (1024**3)
        logger.info(f"GPU: {props.name} | Memory: {total_memory:.2f}GB")
    else:
        logger.warning("⚠ No GPU detected. Training will be slow.")
    
    # Define models with reduced batch size
    models = {
        "yolov8n": {
            "name": "vehicle",
            "yaml": DATA_DIR / "vehicle" / "data.yaml",
            "epochs": 30,
            "batch": 2,
        },
        "yolov8n": {
            "name": "helmet",
            "yaml": DATA_DIR / "helmet" / "data.yaml",
            "epochs": 25,
            "batch": 2,
        },
        "yolov8n": {
            "name": "triple_riding",
            "yaml": DATA_DIR / "triple_riding" / "data.yaml",
            "epochs": 25,
            "batch": 2,
        },
        "yolov8n": {
            "name": "plate",
            "yaml": DATA_DIR / "license_plate" / "data.yaml",
            "epochs": 25,
            "batch": 2,
        },
    }
    
    results_summary = {}
    
    # Train each model
    for base_model, config in models.items():
        data_yaml = config["yaml"]
        
        # Skip if data.yaml doesn't exist
        if not data_yaml.exists():
            logger.warning(f"⚠ Dataset not found: {data_yaml}")
            continue
        
        model_name = config["name"]
        epochs = config["epochs"]
        batch = config["batch"]
        
        # Train model
        results = train_model(base_model, data_yaml, epochs, batch)
        results_summary[model_name] = "✓ SUCCESS" if results else "✗ FAILED"
        
        # Clear GPU cache between trainings
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING SUMMARY")
    logger.info("="*70)
    for model_name, status in results_summary.items():
        logger.info(f"{model_name}: {status}")
    
    logger.info("\n✓ Training pipeline completed!")
    logger.info("Models saved to: " + str(RUNS_DIR / "detect"))

if __name__ == "__main__":
    main()
