# System Architecture

## 10-Stage Processing Pipeline

```
[Camera/Video Input]
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 1: Frame Preprocessing                             │
│ • CLAHE contrast enhancement (clip limit 2.0, tile 8×8) │
│ • Bilateral filter denoising (d=9, sigma=75)            │
│ • Unsharp masking (deblur proxy)                         │
│ • Conditional gamma correction (adaptive)                │
│ • Letterbox resize to 640×640                            │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 2: Unified Detection & Tracking                    │
│ • YOLOv8m custom model (trained: 58.4% mAP50)           │
│ • COCO-pretrained YOLOv8n fallback with class mapping    │
│ • ByteTrack multi-object tracking (track_thresh=0.25)   │
│ • Vehicle ID assignment (persistent across frames)       │
│ • Trajectory tracking (centroid deque, maxlen=30)        │
│ • Speed estimation (pixels→world via calibration)        │
│ • Dwell time estimation (frame counters per zone)        │
│ • 8 classes: car, truck, bus, two/three-wheeler,         │
│   pedestrian, rider, pillion                             │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 3: 7 Violation Detectors (concurrent)              │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Helmet       │  │ Triple       │  │ Wrong Side   │   │
│  │ Detector     │  │ Riding       │  │ Detector     │   │
│  │ (YOLOv8s     │  │ (YOLOv8s     │  │ (Farneback   │   │
│  │  trained)    │  │  93.9% mAP50)│  │  Optical Flow│   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Red Light    │  │ Illegal      │  │ Seatbelt     │   │
│  │ (HSV color   │  │ Parking      │  │ (MediaPipe   │   │
│  │  + ROI +     │  │ (Zone +      │  │  Pose + Edge │   │
│  │  debounce)   │  │  Dwell 30s)  │  │  Detection)  │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 4: License Plate Recognition (Dual-Engine)         │
│ • YOLOv8n plate detector (trained, 433 annotations)      │
│ • 3× bicubic upscaling for low-resolution plates         │
│ • Adaptive thresholding (OTSU)                           │
│ • PaddleOCR (primary, conf ≥ 0.7)                        │
│ • EasyOCR (fallback, conf < 0.7)                         │
│ • Indian plate regex: `^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}`   │
│ • OCR correction by positional format (34 state codes)   │
│ • 5-second timeout with graceful degradation             │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 5: Evidence Generation                              │
│ • Color-coded annotated frames (red=violation, green=safe)│
│ • SHA-256 tamper-proof hashing per evidence artifact      │
│ • PDF challan generation (ReportLab) with QR code         │
│   containing verification hash                            │
│ • Fine calculation from configurable fine table           │
│ • Evidence stored with unique event ID (UUID)             │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 6: Risk Intelligence Engine                         │
│ • Severity scoring per violation type (weights 0-10)     │
│ • Risk = Frequency × Severity × Location × Time          │
│ • Location weighting: school zones 1.5×                  │
│ • Time weighting: peak hours 1.3×, night 0.8×            │
│ • DBSCAN hotspot clustering (eps=0.3, min_samples=5)     │
│ • Repeat-offender tracking (30-day sliding window)       │
│ • Risk tiers: CRITICAL (≥80) / HIGH (≥50) / MEDIUM (≥20) │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 7: Predictive Forecasting                           │
│ • Random Forest regressor (300 trees, max_depth=15)      │
│ • Cyclical time features (sin/cos encoding)              │
│ • 7-day lag features for weekly patterns                 │
│ • Rolling window statistics (7-day, 30-day)              │
│ • Confidence intervals (5th-95th percentile ensemble)    │
│ • MAPE: 8.0% on held-out test set                        │
│ • Zone-aware multipliers (school, hospital, market)      │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 8: Enforcement Recommendations                     │
│ • Rule-based resource allocation matrix                  │
│   CRITICAL → 2 officers, HIGH → 1 officer                │
│   MEDIUM → patrol, LOW → no action                       │
│ • Officer deployment planning per zone                   │
│ • Daily schedule auto-generation via APScheduler         │
│ • Priority zone ranking by risk × forecast               │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 9: Command Dashboard (Streamlit)                   │
│ • Live camera feed view with detection overlay           │
│ • Risk map (Folium) with DBSCAN clusters                 │
│ • Hotspot analysis with time-series drill-down           │
│ • Predicted violations chart (7-day forecast)            │
│ • Repeat offender registry with search                   │
│ • Enforcement plan viewer                                │
│ • CSV export for all data views                          │
│ • Auto-refresh (configurable interval)                   │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 10: Performance Evaluation                          │
│ • Per-class precision, recall, F1, mAP metrics           │
│ • Inference latency benchmarking per detector            │
│ • End-to-end frame processing time (target <500ms)       │
│ • Forecast accuracy (MAPE, RMSE, MAE)                    │
│ • PDF evaluation report generation                       │
└──────────────────────────────────────────────────────────┘
```

## Concurrency Model

All 7 violation detectors in Stage 3 run concurrently:

```
Main Thread → FrameProcessor.process_frame()
  ├── Thread 1: HelmetDetector.detect()
  ├── Thread 2: TripleRidingDetector.detect()
  ├── Thread 3: WrongSideDetector.detect()
  ├── Thread 4: RedLightDetector.detect()
  ├── Thread 5: ParkingDetector.detect()
  ├── Thread 6: SeatbeltDetector.detect()
  └── Thread 7: LPREngine.extract_plate() (OCR)
```

Uses `ThreadPoolExecutor(max_workers=6)` — all detectors start simultaneously and results are collected as they complete. Target: 201ms per frame end-to-end on RTX 3050.

## Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Detection | YOLOv8 (Ultralytics) | 8.x |
| Tracking | ByteTrack (via Ultralytics) | built-in |
| OCR | PaddleOCR + EasyOCR | latest |
| Pose Estimation | MediaPipe | ≥0.10 |
| Backend API | FastAPI | ≥0.100 |
| Database ORM | SQLAlchemy | ≥2.0 |
| Dashboard | Streamlit + Folium + Altair | latest |
| Forecasting | Scikit-learn (Random Forest) | ≥1.3 |
| Evidence | ReportLab (PDF), hashlib (SHA-256) | stdlib |
| Scheduler | APScheduler | ≥3.10 |
| Containerization | Docker | Python 3.10-slim |

## Trained Models

| Model | Base Architecture | Training Data | Epochs | Best mAP50 | File Size |
|-------|------------------|---------------|--------|-----------|-----------|
| Vehicle Detector | YOLOv8m | 3,109 images, 8 classes | 100+10 | 58.4% | 296.9 MB |
| Helmet Detector | YOLOv8s | 5,415 images, 2 classes | 80 | — | 21.5 MB |
| Triple Riding | YOLOv8s | 7,889 images, 3 classes | 28 | **93.9%** | 85.4 MB |
| Plate Detector | YOLOv8n | 433 images, 1 class | 37 | **83.7%** | 6.0 MB |
| Traffic Forecaster | Random Forest | 95K rows time-series | 300 trees | 8.0% MAPE | — |

## Data Flow

```
Frame → Preprocessor → Detector → Tracker
  ├──→ Vehicle BBoxes → Helmet ROI Cropper → Helmet Detector
  ├──→ Vehicle BBoxes → Triple Riding Detector
  ├──→ Vehicle BBoxes → Wrong Side Detector (via tracker history)
  ├──→ Red Light ROI → HSV Analyzer → Debounce State Machine
  ├──→ Vehicle BBoxes → Parking Zone Check → Dwell Timer
  ├──→ Vehicle BBoxes → Seatbelt ROI → MediaPipe → Edge Detection
  └──→ Vehicle BBoxes → Plate ROI → Plate Detector → OCR
    
All → Violation Aggregator → Risk Engine → Forecaster → Evidence → Dashboard
```

## Project Structure

```
├── app/                    Streamlit dashboard (7 tabs)
├── configs/                YAML/JSON configs (label_map, fines, cameras, zones)
├── data/
│   ├── datasets/           6 ZIP archives (source)
│   ├── raw/                Extracted datasets
│   └── processed/          Harmonized YOLO datasets with data.yaml
├── Docker/                 Dockerfile
├── docs/                   Documentation
├── models/                 Trained detectors + forecaster
│   ├── pretrained/         Base YOLO weights
│   ├── vehicle_detector.pt (296.9 MB)
│   ├── helmet_detector.pt (21.5 MB)
│   ├── triple_riding_detector.pt (85.4 MB)
│   ├── plate_detector.pt (6.0 MB)
│   └── traffic_forecaster.joblib
├── scripts/                Training, demo, evaluation, orchestration
├── src/
│   ├── api/                FastAPI (9 endpoints)
│   ├── database/           SQLAlchemy models
│   ├── detection/          7 violation detectors
│   ├── enforcement/        Evidence + enforcement engine
│   ├── ocr/                LPR engine + plate training
│   ├── preprocessing/      Frame preprocessor + label harmonizer
│   ├── tracking/           Unified tracker
│   ├── violations/         Aggregator, risk engine, forecaster
│   └── utils/              Runtime configuration
└── tests/                  13+ pytest tests
```

## Error Handling & Graceful Degradation

- **Missing model weights** → COCO-pretrained fallback with class mapping
- **MediaPipe API changes** → Hough line transform fallback
- **OCR timeout** → Returns confidence 0 with empty text
- **Database unavailable** → In-memory fallback for demo mode
- **Frame corruption** → Skip frame, log warning, continue
- **GPU out of memory** → CPU fallback with batched processing
