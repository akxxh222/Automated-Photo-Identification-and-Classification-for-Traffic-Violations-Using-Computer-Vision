#!/usr/bin/env python3
"""
Comprehensive dataset preparation and model training orchestration.
Extracts ZIP archives, prepares YOLO datasets, and trains all models.
"""

import os
import zipfile
import shutil
from pathlib import Path
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_datasets():
    """Extract all dataset ZIP files from data/datasets/Datasets."""
    dataset_dir = Path("data/datasets/Datasets")
    processed_dir = Path("data/processed")
    raw_dir = Path("data/raw/extracted")
    
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return False
    
    logger.info(f"Found datasets in: {dataset_dir}")
    
    zip_files = list(dataset_dir.glob("*.zip"))
    logger.info(f"Found {len(zip_files)} ZIP files to extract")
    
    for zip_path in zip_files:
        logger.info(f"\nExtracting: {zip_path.name}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to temp location first
                extract_path = raw_dir / zip_path.stem
                extract_path.mkdir(parents=True, exist_ok=True)
                
                zip_ref.extractall(extract_path)
                logger.info(f"✓ Extracted to: {extract_path}")
                
                # List contents
                for root, dirs, files in os.walk(extract_path):
                    level = root.replace(str(extract_path), '').count(os.sep)
                    if level < 2:
                        indent = ' ' * 2 * level
                        logger.info(f"{indent}{os.path.basename(root)}/")
                        
        except Exception as e:
            logger.error(f"✗ Error extracting {zip_path.name}: {e}")
            return False
    
    logger.info("\n=== Dataset Extraction Summary ===")
    logger.info(f"All datasets extracted to: {raw_dir}")
    return True


def inspect_extracted_structure():
    """Inspect the structure of extracted datasets."""
    raw_dir = Path("data/raw/extracted")
    
    logger.info("\n=== Extracted Dataset Structure ===")
    for dataset in raw_dir.iterdir():
        if dataset.is_dir():
            logger.info(f"\n{dataset.name}:")
            for item in dataset.iterdir():
                if item.is_dir():
                    count = len(list(item.rglob("*")))
                    logger.info(f"  └─ {item.name}/ ({count} items)")
                else:
                    logger.info(f"  └─ {item.name}")


def prepare_yolo_datasets():
    """Organize extracted datasets into standard YOLO format."""
    raw_dir = Path("data/raw/extracted")
    processed_dir = Path("data/processed")
    
    logger.info("\n=== Preparing YOLO Datasets ===")
    
    # Map datasets to their destination and type
    dataset_mappings = {
        "Bike Helmet Detection": {
            "dest": "helmet",
            "patterns": ["images", "labels"]
        },
        "Indian Vehicle Dataset": {
            "dest": "vehicle",
            "patterns": ["images", "labels"]
        },
        "Triple Riding Model": {
            "dest": "triple_riding",
            "patterns": ["images", "labels"]
        },
        "Car License Plate Detection": {
            "dest": "license_plate",
            "patterns": ["images", "labels"]
        },
        "TVD": {
            "dest": "tvd",
            "patterns": ["images", "labels"]
        },
    }
    
    for source_name, config in dataset_mappings.items():
        dest_name = config["dest"]
        dest_path = processed_dir / dest_name
        dest_path.mkdir(parents=True, exist_ok=True)
        
        # Find matching source directory
        for source_dir in raw_dir.iterdir():
            if source_name.lower() in source_dir.name.lower():
                logger.info(f"\nProcessing {source_name} → {dest_name}")
                
                # Look for YOLO structure (images/ and labels/)
                images_src = None
                labels_src = None
                
                for pattern in config["patterns"]:
                    for item in source_dir.rglob(pattern):
                        if item.is_dir():
                            if pattern == "images":
                                images_src = item
                            elif pattern == "labels":
                                labels_src = item
                
                if images_src and labels_src:
                    # Copy images and labels
                    for split in ["train", "val", "test"]:
                        src_img = images_src / split
                        src_lbl = labels_src / split
                        dst_img = dest_path / "images" / split
                        dst_lbl = dest_path / "labels" / split
                        
                        if src_img.exists():
                            dst_img.mkdir(parents=True, exist_ok=True)
                            for img_file in src_img.glob("*"):
                                if img_file.is_file():
                                    shutil.copy2(img_file, dst_img)
                            logger.info(f"  ✓ Copied {split} images: {len(list(dst_img.glob('*')))} files")
                        
                        if src_lbl.exists():
                            dst_lbl.mkdir(parents=True, exist_ok=True)
                            for lbl_file in src_lbl.glob("*"):
                                if lbl_file.is_file():
                                    shutil.copy2(lbl_file, dst_lbl)
                            logger.info(f"  ✓ Copied {split} labels: {len(list(dst_lbl.glob('*')))} files")
                
                break
    
    logger.info("\n=== Dataset Preparation Complete ===")


def create_data_yaml_files():
    """Create data.yaml files for YOLO training."""
    processed_dir = Path("data/processed")
    
    logger.info("\n=== Creating data.yaml files ===")
    
    yaml_template = """path: {path}
train: images/train
val: images/val
test: images/test

nc: {nc}
names: {names}
"""
    
    configs = {
        "vehicle": {"nc": 1, "names": "['vehicle']"},
        "helmet": {"nc": 2, "names": "['no_helmet', 'helmet']"},
        "triple_riding": {"nc": 2, "names": "['single', 'triple_riding']"},
        "license_plate": {"nc": 1, "names": "['plate']"},
    }
    
    for dataset_name, config in configs.items():
        yaml_path = processed_dir / dataset_name / "data.yaml"
        yaml_content = yaml_template.format(
            path=str((processed_dir / dataset_name).absolute()),
            nc=config["nc"],
            names=config["names"]
        )
        
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(yaml_content)
        logger.info(f"✓ Created: {yaml_path}")


def verify_dataset_integrity():
    """Verify all datasets are properly prepared."""
    processed_dir = Path("data/processed")
    
    logger.info("\n=== Verifying Dataset Integrity ===")
    
    required_datasets = ["vehicle", "helmet", "triple_riding", "license_plate"]
    all_ok = True
    
    for dataset_name in required_datasets:
        dataset_path = processed_dir / dataset_name
        data_yaml = dataset_path / "data.yaml"
        
        if not dataset_path.exists():
            logger.error(f"✗ {dataset_name}: directory not found")
            all_ok = False
            continue
        
        if not data_yaml.exists():
            logger.error(f"✗ {dataset_name}: data.yaml not found")
            all_ok = False
            continue
        
        # Count images and labels
        train_imgs = len(list((dataset_path / "images" / "train").glob("*"))) if (dataset_path / "images" / "train").exists() else 0
        train_lbls = len(list((dataset_path / "labels" / "train").glob("*"))) if (dataset_path / "labels" / "train").exists() else 0
        
        if train_imgs > 0 and train_lbls > 0:
            logger.info(f"✓ {dataset_name}: {train_imgs} train images, {train_lbls} labels")
        else:
            logger.warning(f"⚠ {dataset_name}: Limited or no training data")
    
    if all_ok:
        logger.info("\n✓ All datasets verified successfully!")
    else:
        logger.warning("\n⚠ Some datasets have issues - please review")
    
    return all_ok


def main():
    """Execute full data preparation pipeline."""
    logger.info("=" * 60)
    logger.info("TRAFFIC ENFORCEMENT PLATFORM - DATA PREPARATION")
    logger.info("=" * 60)
    
    # Create necessary directories
    Path("data/raw/extracted").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract datasets
    if not extract_datasets():
        logger.error("Failed to extract datasets. Exiting.")
        sys.exit(1)
    
    # Step 2: Inspect structure
    inspect_extracted_structure()
    
    # Step 3: Prepare YOLO datasets
    prepare_yolo_datasets()
    
    # Step 4: Create data.yaml files
    create_data_yaml_files()
    
    # Step 5: Verify integrity
    verify_dataset_integrity()
    
    logger.info("\n" + "=" * 60)
    logger.info("DATA PREPARATION COMPLETE!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Run: python scripts/train_all.py")
    logger.info("2. Or use: make train")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
