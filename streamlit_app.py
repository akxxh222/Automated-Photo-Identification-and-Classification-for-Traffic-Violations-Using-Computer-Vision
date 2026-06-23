"""Streamlit Cloud entry point. Downloads models on first run, then starts the dashboard."""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scripts.download_models import main as download_models

if not os.path.exists("models/vehicle_detector.pt"):
    print("Models not found. Attempting download...")
    try:
        download_models()
    except Exception as e:
        print(f"Model download failed ({e}). System will use COCO fallback.")

from app.app import *
