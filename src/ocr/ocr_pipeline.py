import re
from .lpr_engine import LPREngine

def validate_plate(plate_str):
    if not plate_str:
        return False
    return bool(re.match(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$", plate_str))

class LPRPipeline:
    def __init__(self, config_path="configs/config.yaml"):
        self.engine = LPREngine(config_path)

    def run(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return {"plate_text": "UNKNOWN", "confidence": 0.0, "is_valid": False}
        crop = frame[y1:y2, x1:x2]
        return self.engine.recognize_crop(crop)
