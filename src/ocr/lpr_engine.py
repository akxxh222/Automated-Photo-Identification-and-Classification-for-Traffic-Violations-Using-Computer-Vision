import cv2
import numpy as np
import re
import yaml
import logging
from typing import Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

from src.utils.runtime import configure_runtime

configure_runtime()
from ultralytics import YOLO

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

try:
    import easyocr
except ImportError:
    easyocr = None

STATE_CODES = {
    "AN": "Andaman and Nicobar Islands", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CH": "Chandigarh", "CG": "Chhattisgarh",
    "DD": "Dadra and Nagar Haveli and Daman and Diu", "DL": "Delhi", "GA": "Goa",
    "GJ": "Gujarat", "HR": "Haryana", "HP": "Himachal Pradesh", "JK": "Jammu and Kashmir",
    "JH": "Jharkhand", "KA": "Karnataka", "KL": "Kerala", "LA": "Ladakh", "LD": "Lakshadweep",
    "MP": "Madhya Pradesh", "MH": "Maharashtra", "MN": "Manipur", "ML": "Meghalaya",
    "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha", "PY": "Puducherry", "PB": "Punjab",
    "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu", "TG": "Telangana", "TR": "Tripura",
    "UP": "Uttar Pradesh", "UK": "Uttarakhand", "WB": "West Bengal"
}

LOG = logging.getLogger(__name__)

class LPREngine:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["violations"]
        
        model_path = self.config.get("plate_model_path", "models/plate_detector.pt")
        if Path(model_path).exists():
            self.plate_detector = YOLO(model_path)
        else:
            LOG.warning("Plate detector not found at %s. Plate detection disabled.", model_path)
            self.plate_detector = None

        # Initialize OCR models & suppress their verbose internal logging
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        if PaddleOCR:
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        else:
            self.paddle_ocr = None
            
        if easyocr:
            self.easy_ocr = easyocr.Reader(['en'], gpu=False, verbose=False)
        else:
            self.easy_ocr = None

        self.plate_regex = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
        self.ocr_timeout_sec = self.config.get("ocr_timeout_sec", 5)

    def _call_with_timeout(self, func, *args, **kwargs):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=self.ocr_timeout_sec)
            except FuturesTimeoutError:
                LOG.warning("OCR call timed out after %ss", self.ocr_timeout_sec)
                return None

    def _preprocess_crop(self, crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Upscale 3x using Bi-cubic interpolation
        h, w = crop.shape[:2]
        upscaled = cv2.resize(crop, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
        
        # 2. Grayscale conversion
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        
        # 3. Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 4. Morphological operations (Opening to remove noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return upscaled, cleaned

    def _correct_ocr(self, text: str) -> str:
        # Aggressive strip of non-alphanumeric chars
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Apply common OCR corrections based on expected positional format of Indian plates
        # Format is generally: [2 Letters] [2 Numbers] [1-3 Letters] [4 Numbers]
        if len(text) >= 9 and len(text) <= 10:
            chars = list(text)
            # First 2 positions -> Letters
            for i in range(2):
                if chars[i] == '0': chars[i] = 'O'
                if chars[i] == '1': chars[i] = 'I'
                if chars[i] == '5': chars[i] = 'S'
                if chars[i] == '8': chars[i] = 'B'
            # Next 2 positions -> Digits
            for i in range(2, 4):
                if chars[i] == 'O': chars[i] = '0'
                if chars[i] == 'I': chars[i] = '1'
                if chars[i] == 'S': chars[i] = '5'
                if chars[i] == 'B': chars[i] = '8'
            # Last 4 positions -> Digits
            for i in range(len(chars)-4, len(chars)):
                if chars[i] == 'O': chars[i] = '0'
                if chars[i] == 'I': chars[i] = '1'
                if chars[i] == 'S': chars[i] = '5'
                if chars[i] == 'B': chars[i] = '8'
            text = "".join(chars)
            
        return text

    def _run_ocr(self, upscaled: np.ndarray, cleaned: np.ndarray) -> Tuple[str, float]:
        plate_text, ocr_conf = "", 0.0

        if self.paddle_ocr:
            res = self._call_with_timeout(self.paddle_ocr.ocr, upscaled, cls=True)
            if res and res[0]:
                plate_text = "".join([line[1][0] for line in res[0]])
                ocr_conf = np.mean([line[1][1] for line in res[0]])

        if ocr_conf < 0.7 and self.easy_ocr:
            res = self._call_with_timeout(self.easy_ocr.readtext, cleaned)
            if res:
                plate_text = "".join([line[1] for line in res])
                ocr_conf = np.mean([line[2] for line in res])

        return plate_text, float(ocr_conf)

    def recognize_crop(self, crop: np.ndarray) -> Dict[str, Any]:
        """Run OCR on a pre-cropped plate region without running plate detection."""
        if crop is None or crop.size == 0:
            return {"plate_text": "UNKNOWN", "confidence": 0.0, "is_valid": False}

        upscaled, cleaned = self._preprocess_crop(crop)
        plate_text, ocr_conf = self._run_ocr(upscaled, cleaned)
        clean_text = self._correct_ocr(plate_text)
        is_valid = bool(self.plate_regex.match(clean_text))
        state_code = clean_text[:2]
        state_name = STATE_CODES.get(state_code, "Unknown")

        return {
            "plate_text": clean_text,
            "confidence": ocr_conf,
            "state": state_name,
            "is_valid": is_valid,
            "raw_crop": crop,
        }

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.plate_detector is None:
            return []

        results = self.plate_detector(frame, verbose=False)
        found_plates = []
        
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                if conf < 0.4:
                    continue
                    
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                crop = frame[y1:y2, x1:x2]
                upscaled, cleaned = self._preprocess_crop(crop)
                plate_text, ocr_conf = self._run_ocr(upscaled, cleaned)

                clean_text = self._correct_ocr(plate_text)
                is_valid = bool(self.plate_regex.match(clean_text))
                state_code = clean_text[:2]
                state_name = STATE_CODES.get(state_code, "Unknown")
                
                found_plates.append({
                    "plate_text": clean_text,
                    "confidence": float(ocr_conf),
                    "state": state_name,
                    "is_valid": is_valid,
                    "raw_crop": crop
                })
                
        return found_plates
