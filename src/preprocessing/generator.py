from src.enforcement.evidence_generator import EvidenceGenerator as _EG

class EvidenceGenerator:
    def __init__(self):
        self.eg = _EG()
        
    def generate(self, frame, violation):
        plate_text = violation.get("plate_text", "UNKNOWN")
        plate_info = {"plate_text": plate_text, "confidence": 0.9}
        cam_id = violation.get("camera_id", "CAM_001")
        return self.eg.process_violation(frame, violation, plate_info, cam_id)
