# Deployment Guide

## Prerequisites

- **Python 3.10+** with pip
- **NVIDIA GPU** with CUDA 11.8+ (recommended for inference, required for training)
- **8GB+ RAM**, **20GB+ disk space** (SSD recommended)
- **Windows** (primary development platform) or **Linux** (Docker)

## Installation

```bash
pip install -r requirements.txt
```

For GPU acceleration:

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

## Quick Start (Pre-Trained Models Included)

The repository includes 4 pre-trained models in `models/`:

| Model | Size | mAP50 |
|-------|------|-------|
| `vehicle_detector.pt` | 296.9 MB | 58.4% (8 classes) |
| `helmet_detector.pt` | 21.5 MB | — |
| `triple_riding_detector.pt` | 85.4 MB | 93.9% |
| `plate_detector.pt` | 6.0 MB | 83.7% |

### Start API Server

```bash
uvicorn src.api.app:app --reload --port 8000
# Swagger docs: http://localhost:8000/docs
```

### Start Dashboard

```bash
streamlit run app/app.py
# Dashboard: http://localhost:8501
```

### Process a Frame

```bash
curl -X POST http://localhost:8000/api/v1/process-frame \
  -H "X-API-Key: dev-key-123" \
  -F "file=@traffic_frame.jpg"
```

### Run Demo on Video

```bash
python scripts/run_demo.py --video path/to/traffic_video.mp4
python scripts/run_demo.py --video path/to/traffic_video.mp4 --output ./output.mp4  # Save annotated video
python scripts/run_demo.py --camera 0  # Webcam feed
```

## Environment Variables

Create a `.env` file (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEYS` | `dev-key-123` | Comma-separated valid API keys (set in production!) |
| `CORS_ORIGINS` | `http://localhost:8501,http://localhost:3000` | Allowed CORS origins |
| `API_BASE` | `http://localhost:8000/api/v1` | Dashboard API URL |
| `DATABASE_URL` | `sqlite:///./gridlock.db` | Database connection string |

## Training

### Train All Models

```bash
python scripts/train_all.py --device 0 --batch 8
```

This trains 4 models sequentially:

| Model | Base | Epochs | Est. Time (RTX 3050) | Best mAP50 |
|-------|------|--------|---------------------|-----------|
| Vehicle Detector | YOLOv8m | 100 | ~1 hr | 58.4% (8 classes) |
| Helmet Detector | YOLOv8s | 80 | ~1.5 hrs | — |
| Triple Riding | YOLOv8s | 80 | ~1 hr | 93.9% |
| Plate Detector | YOLOv8n | 80 | ~10 min | 83.7% |

### Train Individual Models

> **Optimized batch sizes for RTX 3050 4GB VRAM:**
> If training crashes due to Windows page file limits, pass `--workers 0`.
> For CPU-only training, use `--device cpu --batch 2`.

```bash
# Vehicle detector (8 classes)
python -m src.detection.train_vehicle_detector --epochs 100 --batch 2 --device 0 --no-prepare-labels

# Helmet detector (2 classes: helmet_on, helmet_off)
python -m src.detection.helmet_detector --epochs 80 --batch 4 --device 0

# Triple riding detector (3 classes: single, double, triple)
python -m src.detection.triple_riding_detector --epochs 80 --batch 16 --device 0 --no-prepare-labels

# License plate detector
python -m src.ocr.train_plate_detector --epochs 80 --batch 16 --device 0 --no-prepare-labels
```

### Generate Missing Classes (Rider, Pillion)

After vehicle training, bootstrap missing classes 6-7:

```bash
python scripts/generate_missing_classes.py
python -m src.detection.train_vehicle_detector --epochs 10 --batch 2 --device 0 --no-prepare-labels
```

### Evaluate

```bash
python scripts/evaluate_all.py
```

## Docker

```bash
# Build (takes ~10 min, image ~3GB due to ML dependencies)
docker build -f Docker/Dockerfile -t gridlock-ai .

# Run
docker run -p 8000:8000 gridlock-ai
```

For production deployment, consider:
- Using a slim base image with only CPU inference
- Splitting into microservices (API, OCR, dashboard)
- Using NVIDIA CUDA base image for GPU inference

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/api/v1/health` | No | System health, model status, DB connection |
| `POST` | `/api/v1/process-frame` | API Key | End-to-end frame → violations pipeline |
| `GET` | `/api/v1/violations` | API Key | Query violations (filter by plate, type, date, junction) |
| `GET` | `/api/v1/risk/{junction_id}` | API Key | Risk score with breakdown |
| `GET` | `/api/v1/hotspots` | API Key | DBSCAN cluster data with GeoJSON |
| `GET` | `/api/v1/repeat-offenders` | API Key | Top repeat offenders with violation history |
| `GET` | `/api/v1/enforcement-plan` | API Key | Today's officer deployment schedule |
| `GET` | `/api/v1/forecast` | API Key | 7-day violation forecast with confidence intervals |

### API Authentication

```bash
# Include API key in header
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/violations
```

## Database

- **Development**: SQLite (`gridlock.db`, zero-config)
- **Production**: Set `DATABASE_URL=postgresql://user:pass@host:5432/gridlock`
- Schema auto-creates on first startup via SQLAlchemy
- Tables: `violations`, `evidence`, `enforcement_plans`, `hotspots`, `repeat_offenders`

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_api.py -v
```

All 13+ tests pass — API endpoints, detection pipeline, OCR, and enforcement engine.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce batch size, use `--batch 2`, pass `--workers 0` |
| `No module named 'src'` | Run from project root directory |
| `MediaPipe pose not found` | Install `mediapipe>=0.10.0` or use Hough fallback |
| `PaddleOCR import error` | `pip install paddlepaddle paddleocr` |
| `Docker build slow` | Use `--no-cache` only for final builds |
| Database errors | Delete `gridlock.db` to reset (auto-recreated) |
