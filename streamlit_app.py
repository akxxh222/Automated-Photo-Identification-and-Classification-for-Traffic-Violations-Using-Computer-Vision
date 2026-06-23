"""Streamlit Cloud entry point. Starts API server in background, then launches dashboard."""
import sys
import os
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scripts.download_models import main as download_models

if not os.path.exists("models/vehicle_detector.pt"):
    print("Models not found. Attempting download...")
    try:
        download_models()
    except Exception as e:
        print(f"Model download failed ({e}). System will use COCO fallback.")

def start_api():
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, log_level="warning")

api_thread = threading.Thread(target=start_api, daemon=True)
api_thread.start()
print("API server started on :8000")

from app.app import *
