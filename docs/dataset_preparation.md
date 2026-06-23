# Dataset Preparation & Label Harmonization

## Overview

Six datasets from diverse sources are consolidated into a unified YOLO training format. The core challenge is that each dataset uses different label names for the same visual concept (e.g., "auto_rikshaw", "three wheeler", "three_wheeler"). A custom **label harmonization** system resolves this via a declarative YAML alias map.

## Datasets

| Dataset | Source | Format | Raw Images | Used For |
|---------|--------|--------|-----------|----------|
| Bike Helmet Detection | Roboflow YOLOv8 | YOLO .txt | 1,376 | Helmet/No-Helmet |
| Indian Vehicle Dataset | Roboflow YOLOv8 | YOLO .txt | 250+ | Vehicle classification |
| Triple Riding Model | Roboflow YOLOv8 | YOLO .txt | 6,000+ | Rider counting |
| Car License Plate Detection | VOC XML | Pascal VOC | 433 | ALPR |
| Indian Traffic Violation Kaggle | CSV | Time-series | 95K rows | Forecasting only |
| TVD (Traffic Violation Dataset) | Roboflow YOLOv8 | YOLO .txt | 1,676 | Vehicle + Helmet + Triple |
| Pedestrian Dataset | Roboflow YOLOv8 | YOLO .txt | 1,318 | Pedestrian (class 5) |

## Label Harmonization System

The `label_harmonizer.py` + `configs/label_map.yaml` combo resolves inter-dataset naming conflicts:

### How It Works

1. **Source Discovery**: Scans `data/raw/` and `data/datasets/` for `data.yaml` files matching task keywords
2. **Alias Lookup**: Each dataset's class names are normalized and matched against canonical names via alias mapping
3. **Label Remapping**: YOLO label files are rewritten with the correct canonical class IDs
4. **Consolidation**: Images and labels are copied into a single `data/processed/{task}/` directory with subdirectory-per-source to prevent filename collisions

### Task Configuration

Each task in `label_map.yaml` defines:
- `canonical_names`: The unified class names
- `aliases`: All possible variations across datasets mapped to canonicals
- `include_keywords`: Filters which dataset directories to scan

### Example: Vehicle Task (34 aliases mapped to 8 classes)

```yaml
canonical_names: [car, truck, bus, two_wheeler, three_wheeler, pedestrian, rider, pillion]
aliases:
  car: [car, cars, ambulance, police_vehicle, four_wheeler, vehicle_car, automobile]
  two_wheeler: [two_wheeler, bike, bicycle, motorbike, motorcycle, scooter]
  three_wheeler: [three_wheeler, auto, auto_rikshaw, auto_rickshaw, rickshaw]
  pedestrian: [pedestrian, person, people, walker]
  rider: [rider, motorcyclist, bike_rider, person_on_vehicle]
  pillion: [pillion, passenger, pillion_rider, rear_passenger]
  # ... etc
```

## Current Annotation Counts

### Vehicle Detector (8 classes)

| Class | Canonical Name | Annotations | Source Datasets | Status |
|-------|---------------|-------------|-----------------|--------|
| 0 | car | 643 | Indian Vehicle, TVD | ✅ Trained |
| 1 | truck | 116 | Indian Vehicle, TVD | ✅ Trained |
| 2 | bus | 130 | Indian Vehicle, TVD | ✅ Trained |
| 3 | two_wheeler | 272 | Indian Vehicle, TVD | ✅ Trained |
| 4 | three_wheeler | 410 | Indian Vehicle, TVD | ✅ Trained |
| **5** | **pedestrian** | **10,074** | **Pedestrian Dataset (1,318 images)** | **✅ Generated** |
| **6** | **rider** | **9** | **COCO transfer learning** | **⚠️ Low count** |
| **7** | **pillion** | **124** | **COCO transfer learning** | **⚠️ Low count** |

> Classes 5-7 were zero-shot until `scripts/generate_missing_classes.py` used COCO-pretrained YOLOv8n to detect "person" in training images and classify each as pedestrian/rider/pillion based on overlap with vehicle bounding boxes.

### Helmet Detector (2 classes)

| Class | Annotations |
|-------|------------|
| helmet_on | 6,546 |
| helmet_off | 4,427 |

### Triple Riding Detector (3 classes)

| Class | Annotations |
|-------|------------|
| single_rider | 2,870 |
| double_rider | 1,493 |
| triple_rider | 1,865 |

### License Plate Detector

| Class | Annotations |
|-------|------------|
| license_plate | 433 |

## Dataset Counts After Harmonization

| Task | Train | Val | Test | Total Images | Total Labels |
|------|-------|-----|------|-------------|-------------|
| Vehicle | 2,988 | 179 | 93 | 3.1K | 8 classes |
| Helmet | 5,016 | 263 | 136 | 5.4K | 2 classes |
| Triple Riding | 6,908 | 661 | 320 | 7.9K | 3 classes |
| License Plate | 346 | 43 | 44 | 433 | 1 class |

## Missing Class Generation

The `scripts/generate_missing_classes.py` script addresses the zero-shot problem for classes 5-7:

1. Loads COCO-pretrained YOLOv8n
2. Runs inference on all vehicle training images
3. For each detected "person":
   - Checks IoU overlap with existing vehicle bounding boxes (two_wheeler, car, etc.)
   - High overlap with two_wheeler → classifies as **rider** (class 6)
   - Multiple persons on same two_wheeler → second person → **pillion** (class 7)
   - Low/no overlap with vehicles → classifies as **pedestrian** (class 5)
4. Appends new labels to existing training data

**Result**: 10,074 pedestrian + 9 rider + 124 pillion annotations added.

## License Plate Dataset (Special Case)

The Car License Plate Detection dataset uses Pascal VOC XML format (not YOLO). `dataset_prep.py` handles this separately:
- Parses XML annotations with bounding boxes
- Converts to YOLO normalized format (class_id cx cy w h)
- Splits into train/val/test (80/10/10)
- Augmented with online augmentations during training

## Key Design Decisions

1. **Subdirectory per source**: Images stored in subdirectories named after the source dataset to prevent filename collisions when merging
2. **Strict vs lenient mode**: Vehicle training uses `strict=True` (fails on unknown labels), helmet/triple_riding use `strict=False` (skips unknown labels with warning)
3. **Forecaster dataset is separate**: The Indian Traffic Violation Kaggle CSV is used exclusively for time-series forecasting, not for detection training
4. **COCO transfer for missing classes**: Classes without training data are bootstrapped via COCO-pretrained model inference, enabling the unified 8-class detector to output meaningful predictions for all classes
