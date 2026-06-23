import os
import json
import time
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# Import pipeline components for latency benchmarking
from src.violations.forecaster import TrafficForecaster
from src.preprocessing.preprocessor import FramePreprocessor
from src.tracking.tracker import UnifiedTracker
from src.violations.violation_aggregator import ViolationAggregator
from src.ocr.lpr_engine import LPREngine

def evaluate_yolo():
    models_to_eval = [
        {"name": "Vehicle Detector", "path": "models/vehicle_detector.pt", "data": "data/processed/vehicle/data.yaml"},
        {"name": "Helmet Detector", "path": "models/helmet_detector.pt", "data": "data/processed/helmet/data.yaml"},
        {"name": "Triple Riding", "path": "models/triple_riding_detector.pt", "data": "data/processed/triple_riding/data.yaml"},
        {"name": "Plate Detector", "path": "models/plate_detector.pt", "data": "data/processed/license_plate/data.yaml"}
    ]

    results = {}
    for m in models_to_eval:
        print(f"Evaluating {m['name']}...")
        if not os.path.exists(m['path']):
            raise FileNotFoundError(f"Model file not found: {m['path']}")
        if not os.path.exists(m['data']):
            raise FileNotFoundError(f"Dataset file not found: {m['data']}")
        if not YOLO:
            raise ImportError("YOLO package required for evaluation")

        try:
            model = YOLO(m['path'])
            metrics = model.val(data=m['data'])
            results[m['name']] = {
                "Precision": float(metrics.results_dict['metrics/precision(B)']),
                "Recall": float(metrics.results_dict['metrics/recall(B)']),
                "mAP50": float(metrics.results_dict['metrics/mAP50(B)']),
                "mAP50-95": float(metrics.results_dict['metrics/mAP50-95(B)'])
            }
            # Approximate F1 Score
            p = results[m['name']]["Precision"]
            r = results[m['name']]["Recall"]
            results[m['name']]["F1"] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate {m['name']}: {e}")

    return results

def evaluate_inference_latency():
    print("Running Inference Latency Benchmark (100 frames)...")
    preprocessor = FramePreprocessor()
    tracker = UnifiedTracker()
    aggregator = ViolationAggregator()
    lpr = LPREngine()
    
    latencies = []
    
    for i in range(100):
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        start_time = time.time()
        
        proc_frame, _ = preprocessor.process_frame(frame, i)
        tracks = tracker.process_frame(proc_frame)
        _ = aggregator.detect(proc_frame, tracks, "CAM_TEST")
        _ = lpr.process_frame(proc_frame)
        
        latency = (time.time() - start_time) * 1000 # convert to ms
        latencies.append(latency)
        
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    print(f"Mean Latency: {mean_latency:.2f} ms ± {std_latency:.2f} ms")
    return {"mean_latency_ms": float(mean_latency), "std_latency_ms": float(std_latency), "target_met": bool(mean_latency < 40.0)}

def evaluate_forecaster():
    print("Evaluating Forecasting Models...")
    forecaster = TrafficForecaster()
    try:
        metrics = forecaster.train_models("J_EVAL_TEST")
        if not metrics:
            raise ValueError("Forecaster returned empty metrics")
        return metrics
    except Exception as e:
        raise RuntimeError(f"Forecaster evaluation failed: {e}")

def generate_pdf_report(report_data, pdf_path):
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    y = height - 50
    
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "Gridlock Phase 2 - Evaluation & Benchmark Report")
    y -= 30
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 40
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "1. Object Detection Models Performance")
    y -= 20
    c.setFont("Helvetica", 10)
    for model, metrics in report_data.get("yolo_models", {}).items():
        c.drawString(60, y, f"Model: {model}")
        c.drawString(80, y - 15, ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
        y -= 35
        if y < 100: c.showPage(); y = height - 50
        
    y -= 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "2. End-to-End Inference Latency")
    lat = report_data.get("latency", {})
    c.setFont("Helvetica", 10)
    c.drawString(60, y - 20, f"Mean Latency: {lat.get('mean_latency_ms', 0):.2f} ms")
    c.drawString(60, y - 35, f"Standard Deviation: {lat.get('std_latency_ms', 0):.2f} ms")
    c.drawString(60, y - 50, f"Target (<40ms) Met: {lat.get('target_met', False)}")
    y -= 70
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "3. Predictive Forecasting Accuracy")
    for model, metrics in report_data.get("forecasting", {}).items():
        c.setFont("Helvetica", 10)
        c.drawString(60, y - 20, f"Model: {model}")
        c.drawString(80, y - 35, f"MAPE: {metrics.get('MAPE', 0):.2%} | RMSE: {metrics.get('RMSE', 0):.2f}")
        y -= 50
        
    c.save()

def main():
    print("=== Stage 10: Evaluation & Benchmarking ===")
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "yolo_models": evaluate_yolo(),
        "latency": evaluate_inference_latency(),
        "forecasting": evaluate_forecaster()
    }
    
    json_path = out_dir / "evaluation_report.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=4)
        
    pdf_path = out_dir / "evaluation_report.pdf"
    generate_pdf_report(report_data, pdf_path)
    
    print("\n[✔] Evaluation Complete!")
    print(f"Reports saved to: \n -> {json_path}\n -> {pdf_path}")

if __name__ == "__main__":
    main()
