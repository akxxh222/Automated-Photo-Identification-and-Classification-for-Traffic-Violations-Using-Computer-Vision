"""Download pre-trained model weights for Gridlock AI."""
import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

BASE_URL = os.getenv(
    "MODEL_DOWNLOAD_URL",
    "https://github.com/akxxh222/Automated-Photo-Identification-and-Classification-for-Traffic-Violations-Using-Computer-Vision/releases/download/v1.0"
)

REQUIRED = {
    "vehicle_detector.pt": f"{BASE_URL}/vehicle_detector.pt",
    "helmet_detector.pt": f"{BASE_URL}/helmet_detector.pt",
    "triple_riding_detector.pt": f"{BASE_URL}/triple_riding_detector.pt",
    "plate_detector.pt": f"{BASE_URL}/plate_detector.pt",
    "traffic_forecaster.joblib": f"{BASE_URL}/traffic_forecaster.joblib",
}

def download(url, dest):
    print(f"Downloading {dest.name}...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  Done ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False

def main():
    missing = [name for name in REQUIRED if not (MODELS_DIR / name).exists()]
    if not missing:
        print("All model files already present.")
        return
    print(f"Downloading {len(missing)} model(s) to {MODELS_DIR}/")
    success = 0
    for name in missing:
        if download(REQUIRED[name], MODELS_DIR / name):
            success += 1
    print(f"Downloaded {success}/{len(missing)} models. System can run without custom models (COCO fallback).")

if __name__ == "__main__":
    main()
