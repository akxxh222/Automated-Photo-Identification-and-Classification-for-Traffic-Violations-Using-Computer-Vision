import cv2
import yaml
import json
import hashlib
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics import renderPDF

logger = logging.getLogger(__name__)

class EvidenceGenerator:
    def __init__(self, fines_path: str = "configs/fine_table.yaml", loc_path: str = "configs/camera_locations.json"):
        self.evidence_dir = Path("evidence_store")
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        
        with open(fines_path, "r") as f:
            self.fines = yaml.safe_load(f)
        
        with open(loc_path, "r") as f:
            self.locations = json.load(f)
            
        self.colors = {
            "helmet": "red",
            "triple_riding": "orange",
            "wrong_side": "purple",
            "red_light": "red",
            "illegal_parking": "blue",
            "seatbelt": "yellow"
        }

    def _annotate_frame(self, frame: np.ndarray, violation: Dict[str, Any], timestamp: str) -> np.ndarray:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        
        v_type = violation["violation_type"]
        conf = violation["confidence"]
        bbox = violation["bbox"]
        color = self.colors.get(v_type, "red")
        
        # Draw Bounding Box
        draw.rectangle(bbox, outline=color, width=4)
        
        # Label text
        label = f"{v_type.upper()} ({conf*100:.1f}%) - {timestamp}"
        
        try:
            # Try to load a nice truetype font if available on the system
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
            
        # Draw text background
        text_bbox = draw.textbbox((bbox[0], max(0, bbox[1] - 25)), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        
        # Draw text
        draw.text((bbox[0], max(0, bbox[1] - 25)), label, fill="white", font=font)
        
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _compute_tamper_hash(self, raw_frame: np.ndarray, ann_frame: np.ndarray) -> str:
        raw_bytes = cv2.imencode('.jpg', raw_frame)[1].tobytes()
        ann_bytes = cv2.imencode('.jpg', ann_frame)[1].tobytes()
        return hashlib.sha256(raw_bytes + ann_bytes).hexdigest()

    def _generate_pdf(self, pdf_path: Path, summary: str, img_path: Path, fine_amount: int, hash_str: str) -> None:
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4
        
        # Headers
        c.setFont("Helvetica-Bold", 22)
        c.drawString(50, height - 60, "Official Traffic Violation Challan")
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(50, height - 75, "AI-Powered Traffic Enforcement & Risk Intelligence Platform")
        
        # Violation Text
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, height - 110, "Violation Summary:")
        c.setFont("Helvetica", 11)
        c.drawString(50, height - 130, summary)
        
        # Fine details
        c.setFont("Helvetica-Bold", 14)
        c.setFillColorRGB(0.8, 0, 0)
        c.drawString(50, height - 160, f"Fine Amount Due: INR {fine_amount}")
        c.setFillColorRGB(0, 0, 0)
        
        # Embed Image
        c.drawImage(str(img_path), 50, height - 520, width=500, preserveAspectRatio=True)
        
        # Tamper Hash Text
        c.setFont("Courier", 9)
        c.drawString(50, height - 540, f"Cryptographic Evidence Hash (SHA-256): {hash_str}")
        
        # Embed QR Code representation of the hash
        qrw = QrCodeWidget(f"gridlock://verify?hash={hash_str}")
        bounds = qrw.getBounds()
        qr_width = bounds[2] - bounds[0]
        qr_height = bounds[3] - bounds[1]
        
        # Scale QR code to 80x80
        d = Drawing(80, 80, transform=[80/qr_width, 0, 0, 80/qr_height, 0, 0])
        d.add(qrw)
        renderPDF.draw(d, c, width - 130, height - 110)
        
        c.save()

    def process_violation(self, raw_frame: np.ndarray, violation: Dict[str, Any], plate_info: Optional[Dict[str, Any]], camera_id: str) -> Dict[str, Any]:
        if plate_info is None:
            plate_info = {"plate_text": "UNKNOWN", "confidence": 0.0}
        plate_text = plate_info.get('plate_text', 'UNKNOWN')
        
        timestamp = violation.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        event_id = hashlib.md5(f"{plate_text}_{timestamp}".encode()).hexdigest()[:12]
        
        v_type = violation.get("violation_type", "unknown")
        conf = violation.get("confidence", 0.0)
        fine_amount = self.fines.get(v_type, 1000)
        loc_info = self.locations.get(camera_id, {})
        location = f"Junction {loc_info.get('junction_id', 'UNK')} ({loc_info.get('lat', 'N/A')}, {loc_info.get('lon', 'N/A')})"
        
        # Generate strict Natural Language Template
        summary = f"Vehicle {plate_text} detected {v_type} at {location}, {conf*100:.1f}% conf, Camera {camera_id}, {timestamp}."
        
        ann_frame = self._annotate_frame(raw_frame, violation, timestamp)
        img_path = self.evidence_dir / f"{event_id}_annotated.jpg"
        cv2.imwrite(str(img_path), ann_frame)
        
        frame_hash = self._compute_tamper_hash(raw_frame, ann_frame)
        pdf_path = self.evidence_dir / f"{event_id}_challan.pdf"
        
        self._generate_pdf(pdf_path, summary, img_path, fine_amount, frame_hash)
        
        return {"event_id": event_id, "summary": summary, "fine_amount": fine_amount, "frame_hash": frame_hash,
                "evidence_path": str(img_path), "challan_path": str(pdf_path), "timestamp": timestamp}
