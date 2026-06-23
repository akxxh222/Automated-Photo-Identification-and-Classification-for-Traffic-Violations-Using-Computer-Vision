import cv2
import os
import base64
import json
import urllib.request
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000"
API_KEY = "dev-key-123"
DATA_DIR = Path("data/processed/vehicle/images/val/tvd_v1i_yolov8")
OUTPUT_VIDEO = "results/demo_video.mp4"
CODEC = "avc1"
FPS = 2
W, H = 1280, 720

SAMPLE_IMAGES = [
    "wtk6ax_jpg.rf.2245a1cae229ea5bc3e4bc0c7e1e3214.jpg",
    "wheelie-3309181_640_jpg.rf.985ba7cf5489ee73a43f817a1abe7925.jpg",
    "w5px1b_jpg.rf.06ef3e52e496229f86d0211412fa5e2e.jpg",
    "Untitled-design-12-1-1280x720_jpg.rf.de62d64265e6baba44761c8971929b1f.jpg",
    "Underage-bike-riders-DH-1562095628_jpg.rf.70443ec4b21839ac77a10486b8ec6e7c.jpg",
    "train-no-helmet-591-_jpg.rf.29168822684525468da3b9ba00f8c7aa.jpg",
    "train-no-helmet-580-_jpg.rf.222dd24bee811194e7f679213e99a9e3.jpg",
    "train-no-helmet-571-_jpg.rf.71c48466f642e42f11ec1ba3b3f506f4.jpg",
    "train-no-helmet-564-_jpg.rf.18b68bd7f53e79f7e576f2147977920e.jpg",
    "train-no-helmet-557-_jpg.rf.4e9402638d2ed35f64329e3f3ee6d7fa.jpg",
]

def put_text(frame, text, x, y, scale=0.6, color=(255, 255, 255), thick=2):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def make_title(text, subtitle="", bg=(20, 20, 40)):
    frame = np.full((H, W, 3), bg, dtype=np.uint8)
    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    put_text(frame, text, (W - ts[0]) // 2, H // 2 - 30, 1.2, (255, 255, 255), 3)
    if subtitle:
        ss = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        put_text(frame, subtitle, (W - ss[0]) // 2, H // 2 + 30, 0.7, (180, 180, 255), 2)
    return frame

def make_processed(original_path, evidence_path, events):
    orig = cv2.imread(str(original_path))
    if orig is None:
        return None
    oh, ow = orig.shape[:2]

    ev = None
    if evidence_path and os.path.exists(evidence_path):
        ev = cv2.imread(evidence_path)

    if ev is not None:
        eh, ew = ev.shape[:2]
        scale = min(H / eh, W / ew)
        ev = cv2.resize(ev, (int(ew * scale), int(eh * scale)))
        fh, fw = ev.shape[:2]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        xo = (W - fw) // 2
        yo = (H - fh) // 2
        canvas[yo:yo+fh, xo:xo+fw] = ev
        frame = canvas
    else:
        scale = min(H / oh, W / ow)
        frame = cv2.resize(orig, (int(ow * scale), int(oh * scale)))
        fh, fw = frame.shape[:2]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        xo = (W - fw) // 2
        yo = (H - fh) // 2
        canvas[yo:yo+fh, xo:xo+fw] = frame
        frame = canvas

    overlay = frame.copy()
    y_end = H - 10
    n = len(events) + 2
    cv2.rectangle(overlay, (5, y_end - n * 28 - 5), (W - 5, y_end), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    v_count = len(events)
    types = [e.get("violation_type", "?").replace("_", " ").title() for e in events]
    put_text(frame, f"Violations: {v_count}  |  {', '.join(types) if types else 'None'}", 15, y_end - (n - 1) * 28, 0.55, (0, 255, 0))
    if events:
        for i, e in enumerate(events[:3]):
            conf = e.get("confidence", 0)
            vtype = e.get("violation_type", "?").replace("_", " ").title()
            plate = e.get("plate_text", "N/A")
            put_text(frame, f"  [{i+1}] {vtype} ({conf:.1%}) Plate: {plate}", 15, y_end - (n - 2 - i) * 28, 0.5, (200, 255, 200))
    put_text(frame, "Gridlock AI | Traffic Intelligence Platform", 15, 25, 0.5, (255, 255, 255))

    return frame

def call_api(image_path):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = json.dumps({"image_base64": img_b64, "camera_id": "demo-cam-1"}).encode("utf-8")
    req = urllib.request.Request(
        f"{API_URL}/api/v1/process-frame", data=payload,
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    Path("results").mkdir(exist_ok=True)
    codecs = ["avc1", "mp4v", "X264"]
    out = None
    for c in codecs:
        fcc = cv2.VideoWriter_fourcc(*c)
        out = cv2.VideoWriter(OUTPUT_VIDEO, fcc, FPS, (W, H))
        if out.isOpened():
            break
    if not out or not out.isOpened():
        fcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(OUTPUT_VIDEO, fcc, FPS, (W, H))

    logger.info("Generating demo video...")

    title_pairs = [
        ("GRIDLOCK AI", "Traffic Intelligence Platform"),
        ("8-Class Vehicle Detector", "Car | Truck | Bus | Two-Wheeler | Three-Wheeler | Pedestrian | Rider | Pillion"),
        ("Violation Detection", "Helmet | Triple Riding | Wrong Side | Red Light | Seatbelt | Illegal Parking"),
        ("License Plate Recognition", "PaddleOCR-based text extraction with validity check"),
        ("Risk Scoring & Forecasting", "Junction-level risk tiers + 7-day violation forecast"),
    ]
    for text, sub in title_pairs:
        f = make_title(text, sub)
        for _ in range(FPS * 3):
            out.write(f)

    available = [f for f in SAMPLE_IMAGES if (DATA_DIR / f).exists()]
    if not available:
        available = sorted(p.name for p in DATA_DIR.glob("*.jpg"))[:12]
        logger.info(f"Using {len(available)} images from dataset")

    logger.info(f"Processing {len(available)} frames (warmup first)...")
    call_api(DATA_DIR / available[0])
    logger.info("Warmup done.")

    for img_name in available:
        img_path = DATA_DIR / img_name
        try:
            result = call_api(img_path)
        except Exception as e:
            logger.warning(f"Failed: {img_name}: {e}")
            continue

        events = result.get("events", [])
        evidence_path = events[0].get("evidence_path") if events else None

        frame = make_processed(img_path, evidence_path, events)
        if frame is None:
            continue

        for _ in range(FPS * 4):
            out.write(frame)

    perf_text = (
        "Model Performance  |  Vehicle: 58.4% mAP50  |  Helmet: 81.8%  |  Triple Riding: 93.9%  |  Plate: 87.4%"
    )
    closing_pairs = [
        ("Model Performance", perf_text),
        ("Tech Stack", "YOLOv8m | ByteTrack | PaddleOCR | Risk Engine | Streamlit | FastAPI"),
        ("3,109 Annotated Images", "Across 7 violation types with 8 vehicle classes"),
        ("Total Pipeline Latency", "~300ms per frame on GPU (preprocess + detect + OCR + evidence)"),
    ]
    for text, sub in closing_pairs:
        f = make_title(text, sub, bg=(10, 30, 20))
        for _ in range(FPS * 3):
            out.write(f)

    f = make_title("THANK YOU", "github.com/akxxh222/Automated-Photo-Identification-and-Classification-for-Traffic-Violations-Using-Computer-Vision", (10, 10, 30))
    for _ in range(FPS * 4):
        out.write(f)

    out.release()
    file_size = os.path.getsize(OUTPUT_VIDEO)
    logger.info(f"Demo video saved: {OUTPUT_VIDEO} ({file_size / 1024:.0f} KB)")
    logger.info("Done!")

if __name__ == "__main__":
    main()
