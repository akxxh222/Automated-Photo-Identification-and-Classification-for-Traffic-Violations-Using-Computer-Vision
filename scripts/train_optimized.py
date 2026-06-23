#!/usr/bin/env python3
"""
Optimized Training Script for Traffic Enforcement Platform
Handles GPU memory constraints automatically for RTX 3050 6GB
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

def check_gpu_memory():
    """Check available GPU memory"""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / (1024**3)  # GB
        logger.info(f"GPU: {props.name} | Total Memory: {total_memory:.2f}GB")
        return total_memory
    return 0

def train_model(base_model, model_name, data_yaml, epochs, base_batch=4, patience=15):
    """Train a single YOLO model with memory-optimized settings"""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*70}")
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Epochs: {epochs}, Batch Size: {base_batch}")
    
    try:
        # Load pretrained model
        model = YOLO(f"{base_model}.pt")
        
        # Train with optimized settings for 6GB GPU
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=640,
            batch=base_batch,
            patience=patience,
            device=0,
            
            # Memory optimization
            cache="disk",  # Use disk cache instead of RAM
            workers=4,     # Reduce dataloader workers
            
            # Training optimization
            optimizer="SGD",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            
            # Augmentation (moderate)
            augment=True,
            mosaic=1.0,
            mixup=0.05,
            fliplr=0.5,
            flipud=0.5,
            
            # Hardware optimization
            half=True,       # FP16 precision for memory savings
            amp=True,        # Automatic Mixed Precision
            
            # Validation & Logging
            val=True,
            save=True,
            save_period=10,
            verbose=True,
            plots=True,
            
            # Output
            project=str(RUNS_DIR / "detect"),
            name=f"{model_name}_optimized",
            exist_ok=True,
        )
        
        logger.info(f"✓ {model_name} training completed")
        return results
        
    except Exception as e:
        logger.error(f"✗ {model_name} training failed: {str(e)}")
        return None

def main():
    logger.info("="*70)
    logger.info("TRAFFIC ENFORCEMENT PLATFORM - OPTIMIZED TRAINING")
    logger.info("="*70)
    
    # Check GPU
    gpu_memory = check_gpu_memory()
    if gpu_memory == 0:
        logger.warning("⚠ No GPU detected. Training will be very slow on CPU.")
    
    # Define models (unique keys, base model specified separately)
    models = {
        "vehicle": {
            "base": "yolov8m",
            "yaml": DATA_DIR / "vehicle" / "data.yaml",
            "epochs": 50,
            "batch": 4,
        },
        "helmet": {
            "base": "yolov8s",
            "yaml": DATA_DIR / "helmet" / "data.yaml",
            "epochs": 40,
            "batch": 4,
        },
        "triple_riding": {
            "base": "yolov8s",
            "yaml": DATA_DIR / "triple_riding" / "data.yaml",
            "epochs": 40,
            "batch": 4,
        },
        "plate": {
            "base": "yolov8n",
            "yaml": DATA_DIR / "license_plate" / "data.yaml",
            "epochs": 40,
            "batch": 4,
        },
    }
    
    results_summary = {}
    
    # Train each model
    for model_name, config in models.items():
        data_yaml = config["yaml"]
        
        # Skip if data.yaml doesn't exist
        if not data_yaml.exists():
            logger.warning(f"⚠ Dataset not found: {data_yaml}")
            continue
        
        base_model = config["base"]
        epochs = config["epochs"]
        batch = config["batch"]
        
        # Train model
        results = train_model(base_model, model_name, data_yaml, epochs, batch)
        results_summary[model_name] = "✓ SUCCESS" if results else "✗ FAILED"
        
        # Clear GPU cache between trainings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING SUMMARY")
    logger.info("="*70)
    for model_name, status in results_summary.items():
        logger.info(f"{model_name}: {status}")
    
    logger.info("\n✓ Training pipeline completed!")
    logger.info("Models saved to: " + str(MODELS_DIR))
    logger.info("Results saved to: " + str(RUNS_DIR / "detect"))

if __name__ == "__main__":
    main()
