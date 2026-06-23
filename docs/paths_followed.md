# Design Decisions & Development Paths

## Architecture Philosophy

The system follows a **modular 10-stage pipeline** design where each stage is independently testable and replaceable. Stages communicate through well-defined data structures (`Frame → List[TrackedObject] → List[ViolationEvent] → PlateResult → EvidenceArtifact → RiskScore`).

---

## Key Design Decisions

### 1. Label Harmonization Over Dataset Re-Labeling

**Decision**: Built a YAML-driven alias system (`label_map.yaml`) that maps variant names to canonical classes, rather than manually relabeling all 6 datasets.

**Why**: The 6 datasets use different naming conventions for the same concepts ("auto_rikshaw" vs "three_wheeler" vs "auto rickshaw"). Manual normalization would be error-prone and not reusable. The alias approach is declarative, auditable, and extensible — a new dataset can be added by simply extending the alias map.

**Trade-off**: Requires maintaining the alias map. New datasets may introduce unexpected label names that need to be added. Strict mode (`strict=True`) catches unknown labels during harmonization.

**Result**: 34 aliases mapped to 8 canonical vehicle classes, 2 helmet classes, 3 triple-riding classes.

### 2. Concurrent Violation Detection

**Decision**: All 7 violation detectors run concurrently via `ThreadPoolExecutor(max_workers=6)`.

**Why**: Independent detectors with no shared state. Running them sequentially would increase per-frame latency linearly with the number of detectors.

**Trade-off**: Higher memory usage — all models loaded simultaneously (~400MB GPU memory). Each detector needs its own model instance, but the concurrency yields sub-second total inference.

**Result**: 201ms per frame (all 7 detectors + OCR).

### 3. Unified 8-Class Detector vs Separate Specialized Detectors

**Decision**: Train a single YOLOv8m for all vehicle types (8 classes) rather than separate detectors per vehicle type.

**Why**: Single forward pass for all vehicle detections. The unified model shares features across classes (e.g., wheels, windshields). COCO-pretrained weights provide good initialization.

**Trade-off**: Extreme class imbalance (643 cars vs 9 riders) reduces per-class accuracy. Separate detectors would allow class-specific optimization.

**Result**: 58.4% mAP50 — acceptable for a unified model. Specialized detectors (triple riding at 93.9%) outperform when trained on balanced data.

### 4. Dual OCR with Fallback

**Decision**: PaddleOCR as primary, EasyOCR as fallback when confidence < 0.7.

**Why**: PaddleOCR is more accurate for Indian license plates (trained on multi-lingual data) but has heavier dependencies (~500MB). EasyOCR is lighter and serves as a backup for cases where PaddleOCR fails or times out.

**Trade-off**: ~500MB extra in dependencies. OCR timeout of 5 seconds can cause frame drops. The dual-engine approach adds complexity but increases plate read rate.

**Processing pipeline**: Plate detection → 3× bicubic upscale → adaptive threshold → PaddleOCR → confidence check → EasyOCR fallback → Indian plate regex validation → state code extraction.

### 5. ByteTrack Over DeepSORT

**Decision**: ByteTrack (via Ultralytics) instead of DeepSORT.

**Why**: ByteTrack is natively integrated into Ultralytics YOLO, requires no separate tracking model, and handles occlusions better than simple IoU matching. ByteTrack's association strategy (using detection confidence thresholds rather than appearance features) is simpler and faster.

**Trade-off**: ByteTrack uses motion and appearance cues but doesn't have a learned appearance embedding. May lose tracks after long occlusions (>90 frames). DeepSORT with a ReID model would be more robust for cross-camera tracking.

**Result**: Reliable within-camera tracking for typical traffic scenes (30+ frames of occlusion tolerance).

### 6. Rule-Based Detectors Over Learned Models

**Decision**: Wrong-side, red-light, parking, and seatbelt detectors use rule-based/cv-based approaches rather than trained ML models.

**Why**: These violations are well-defined geometrically:
- **Wrong-side**: Optical flow direction vs dominant traffic flow (cosine similarity)
- **Red-light**: HSV color segmentation in traffic light ROI + debounce state machine
- **Parking**: Geographic zone polygon containment + dwell time threshold
- **Seatbelt**: Edge response along diagonal shoulder-to-hip path

**Trade-off**: Less robust to environmental variation (lighting, camera angles, weather). No labeled data required.

**Mitigation**: Each detector has configurable parameters (thresholds, ROIs, zones) via YAML configs, allowing per-camera calibration.

### 7. SQLite for Prototype, PostgreSQL for Production

**Decision**: SQLite by default, PostgreSQL via `DATABASE_URL` env var.

**Why**: Zero-config for development and hackathon demos. All SQLAlchemy queries are database-agnostic — switching to PostgreSQL requires only changing the connection string.

**Trade-off**: SQLite doesn't support concurrent writes. Not suitable for production with multiple cameras writing simultaneously. Schema auto-creates on first startup.

### 8. Subdirectory Per Source Dataset

**Decision**: When harmonizing datasets, images from different sources are stored in separate subdirectories under `data/processed/{task}/images/{split}/{source_name}/`.

**Why**: Filename collisions between datasets (e.g., both Bike Helmet and TVD might have an `img_001.jpg`). Subdirectories eliminate collisions without renaming files.

**Trade-off**: Slightly non-standard YOLO structure. YOLO training reads recursively by default, so this works. Labels mirror the same subdirectory structure for correct pairing.

### 9. COCO Transfer Learning for Missing Classes

**Decision**: Use COCO-pretrained YOLOv8n to bootstrap labels for classes 5-7 (pedestrian, rider, pillion) that have zero training annotations.

**Why**: Rather than leaving 3 of 8 classes untrained (which would yield zero detections), we generate approximate labels by:
1. Detecting "person" using COCO-pretrained model
2. Classifying each person as pedestrian/rider/pillion based on IoU overlap with vehicle bboxes
3. Appending generated labels to the training set

**Trade-off**: Generated labels are noisier than hand-labeled data. COCO's "person" class doesn't distinguish rider vs pillion — this is inferred from vehicle overlap heuristics.

**Result**: 10,074 pedestrian, 9 rider, 124 pillion annotations added. Rider/pillion counts are low because most detected persons in traffic scenes are pedestrians.

### 10. Workers=0 for Windows Training

**Decision**: Added `workers=0` to all YOLO `.train()` calls on Windows.

**Why**: Windows has a 16GB page file limit per process. YOLO's default `workers=8` spawns multiple worker processes for data loading, each consuming significant memory. On 16GB RAM systems with 4GB VRAM, this causes a Windows paging file crash.

**Trade-off**: `workers=0` disables multi-process data loading, increasing per-epoch training time by ~20%. The system is stable and doesn't crash.

---

## What Didn't Work

### 1. MediaPipe 0.10.x API Migration

MediaPipe 0.10 removed `mp.solutions` — the standard import path for pose estimation. The `mp.solutions.pose.Pose()` API was replaced with a task-based API (`PoseLandmarker` with model asset files). The dependency version in `requirements.txt` (`>=0.10.0`) installs the incompatible version.

**Resolution**: Graceful fallback to Hough line transform edge detection when MediaPipe pose is unavailable. The seatbelt detector logs a warning and continues with reduced accuracy.

### 2. Label Harmonizer Crash on Harmonization

The `harmonize_yolo_datasets()` function in `label_harmonizer.py` crashes when processing certain dataset structures due to unexpected subdirectory layouts. The function expects a specific directory hierarchy but encounters variations.

**Resolution**: Added `--no-prepare-labels` flag to skip label harmonization and use an existing `data.yaml`. Training runs with pre-existing processed datasets.

### 3. PyTorch CUDA Version on Windows

The original `requirements.txt` includes `torch>=2.0.0` which installs the CPU version by default on Windows. Users must manually install the CUDA version.

**Resolution**: Added explicit CUDA installation instructions in the deployment guide. The system works on CPU (at reduced speed) without additional configuration.

### 4. Single-Model All-Class Training

Attempting to train all 8 vehicle classes in a single YOLOv8m model with imbalanced data yields suboptimal results (58.4% mAP50). The model allocates disproportionate capacity to majority classes.

**Resolution**: (a) Class-balanced sampling weights, (b) separate specialized detectors for under-represented classes, (c) data augmentation for minority classes. This is an active area of improvement.

---

## Lessons Learned

1. **Class imbalance is the dominant challenge** in unified detection models — 643 car annotations vs 9 rider annotations makes balanced training difficult
2. **Label harmonization via alias mapping is more robust than manual re-labeling** — the declarative YAML approach caught naming inconsistencies across datasets automatically
3. **Rule-based detectors are surprisingly effective** for well-defined geometric violations (wrong-side, red-light) and serve as zero-data solutions while labeled datasets are sourced
4. **Windows training requires workers=0** — YOLO's default multiprocess data loading crashes on 16GB RAM systems with 4GB VRAM
5. **COCO transfer learning can bootstrap missing classes** — noisy labels are better than zero labels; a model trained with noisy labels can be refined with a small set of clean labels
6. **Concurrent detection with ThreadPoolExecutor scales well** — 7 detectors in ~200ms on RTX 3050, limited primarily by GPU memory rather than CPU

---

## Future Work (Phase 2)

- Multi-camera vehicle ReID for cross-junction tracking and journey mapping
- TensorRT/ONNX model optimization for 40ms inference target (5× improvement)
- Real-time dashboard with WebSocket streaming for sub-second alerts
- Integration with traffic signal controllers (V2I) for predictive enforcement
- Indian state-specific plate format variants (tractor, trailer, embassy, military)
- Kubernetes deployment with GPU node auto-scaling for city-wide coverage
- Mobile officer app for in-field enforcement
- Blockchain-based evidence verification chain
