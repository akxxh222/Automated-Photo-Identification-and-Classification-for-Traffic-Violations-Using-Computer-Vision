import io
import os
import random
import logging
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

_DEMO_PLATES = ["KA01AB1234", "KA02CD5678", "KA03EF9012", "KA04GH3456", "KA05IJ7890"]
_DEMO_TYPES = ["helmet", "triple_riding", "wrong_side", "red_light", "illegal_parking", "seatbelt"]


class StandalonePipeline:
    """Wraps the full detection pipeline + risk engine for direct use in Streamlit."""

    def __init__(self, db_path: str = "gridlock.db"):
        self.db_path = db_path
        self._set_db_url()
        self._loaded = False
        self.pipeline = None
        self.risk_engine = None
        self.forecaster = None
        self.SessionLocal = None
        self._frame_counter = 0
        self._demo_count = 0

    def _set_db_url(self):
        os.environ.setdefault("DATABASE_URL", f"sqlite:///./{self.db_path}")

    def load(self):
        if self._loaded:
            return True
        try:
            from src.database.database import init_db, SessionLocal
            from src.database.models import Violation as DBViolation, EnforcementPlan
            from src.violations.risk_engine import RiskEngine
            from src.violations.forecaster import TrafficForecaster

            self.SessionLocal = SessionLocal
            self.DBViolation = DBViolation
            self.EnforcementPlan = EnforcementPlan
            init_db()

            from src.api.app import MLPipeline
            self.pipeline = MLPipeline()
            self.pipeline.load()

            self.risk_engine = RiskEngine()
            self.forecaster = TrafficForecaster()

            self._loaded = True
            logger.info("StandalonePipeline fully loaded")
            return True
        except Exception as e:
            logger.warning("Pipeline load failed (non-fatal for dashboard): %s", e)
            return False

    def _demo_event(self) -> Dict[str, Any]:
        self._demo_count += 1
        vtype = random.choice(_DEMO_TYPES)
        return {
            "violation_type": vtype,
            "confidence": random.uniform(0.65, 0.95),
            "plate_text": random.choice(_DEMO_PLATES) if random.random() > 0.3 else "UNKNOWN",
            "fine_amount": random.choice([500, 1000, 1500, 2000]),
            "evidence_path": None,
            "summary": vtype.replace("_", " ").title(),
        }

    def process_image(self, image_bytes: bytes, camera_id: str = "CAM_001") -> Dict[str, Any]:
        if not self._loaded and not self.load():
            events = [self._demo_event() for _ in range(random.randint(1, 3))]
            return {"processed_violations": len(events), "events": events, "junction_risk": {"score": random.uniform(2.0, 8.0), "tier": random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"])}}
        try:
            from PIL import Image
            import numpy as np
            import cv2
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            np_arr = np.array(pil_img)
            frame = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)

            self._frame_counter += 1
            proc_frame, meta = self.pipeline.preprocessor.process_frame(frame, frame_id=self._frame_counter, source=camera_id)
            tracks = self.pipeline.tracker.process_frame(proc_frame)
            violations = self.pipeline.aggregator.detect(proc_frame, tracks, camera_id)
            plates = self.pipeline.lpr.process_frame(proc_frame)

            results = []
            db = self.SessionLocal()
            try:
                for v in violations:
                    matched_plate = {"plate_text": "UNKNOWN", "confidence": 0.0, "is_valid": False}
                    valid_plates = [p for p in plates if p["is_valid"]]
                    if valid_plates:
                        matched_plate = sorted(valid_plates, key=lambda x: x["confidence"], reverse=True)[0]

                    ev_res = self.pipeline.evidence.process_violation(frame, v, matched_plate, camera_id)
                    loc_info = self.risk_engine.locations.get(camera_id, {})
                    db_violation = self.DBViolation(
                        plate_text=matched_plate["plate_text"],
                        plate_confidence=matched_plate["confidence"],
                        violation_type=v["violation_type"],
                        violation_confidence=v["confidence"],
                        camera_id=camera_id,
                        junction_id=loc_info.get("junction_id"),
                        latitude=loc_info.get("lat"),
                        longitude=loc_info.get("lon"),
                        frame_hash=ev_res["frame_hash"],
                        evidence_path=ev_res["evidence_path"],
                        challan_path=ev_res["challan_path"],
                        fine_amount=ev_res["fine_amount"],
                        is_valid_plate=matched_plate["is_valid"]
                    )
                    db.add(db_violation)
                    db.commit()
                    db.refresh(db_violation)
                    ev_res["violation_type"] = v["violation_type"]
                    ev_res["confidence"] = v["confidence"]
                    results.append(ev_res)

                junction_id = self.risk_engine.locations.get(camera_id, {}).get("junction_id")
                risk_data = {"score": 0.0, "tier": "LOW"}
                if junction_id:
                    score, tier, breakdown = self.risk_engine.calculate_risk_score(db, junction_id)
                    risk_data = {"score": score, "tier": tier, "breakdown": breakdown}

                return {"processed_violations": len(results), "events": results, "junction_risk": risk_data}
            finally:
                db.close()
        except Exception as e:
            logger.error("process_image error: %s", e)
            return {"processed_violations": 0, "events": [], "junction_risk": {"score": 0.0, "tier": "LOW"}, "error": str(e)}

    def query_violations(self, plate: Optional[str] = None, violation_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        if not self._loaded and not self.load():
            return [{
                "id": i, "timestamp": (datetime.now() - timedelta(minutes=i*15)).isoformat(),
                "plate_text": random.choice(_DEMO_PLATES), "plate_confidence": random.uniform(0.7, 0.99),
                "violation_type": random.choice(_DEMO_TYPES), "violation_confidence": random.uniform(0.65, 0.95),
                "camera_id": random.choice(["CAM_001", "CAM_002"]),
                "junction_id": f"J00{random.randint(1,2)}",
                "latitude": 12.97 + random.uniform(-0.02, 0.02), "longitude": 77.59 + random.uniform(-0.02, 0.02),
                "fine_amount": random.choice([500, 1000, 1500, 2000]), "is_valid_plate": random.random() > 0.3,
            } for i in range(min(limit, 12))]
        try:
            db = self.SessionLocal()
            try:
                query = db.query(self.DBViolation)
                if plate:
                    query = query.filter(self.DBViolation.plate_text == plate)
                if violation_type and violation_type != "All":
                    query = query.filter(self.DBViolation.violation_type == violation_type)
                results = query.order_by(self.DBViolation.timestamp.desc()).limit(limit).all()
                return [
                    {
                        "id": r.id,
                        "timestamp": r.timestamp.isoformat() if hasattr(r.timestamp, 'isoformat') else str(r.timestamp),
                        "plate_text": r.plate_text,
                        "plate_confidence": float(r.plate_confidence) if r.plate_confidence else 0.0,
                        "violation_type": r.violation_type,
                        "violation_confidence": float(r.violation_confidence) if r.violation_confidence else 0.0,
                        "camera_id": r.camera_id,
                        "junction_id": r.junction_id,
                        "latitude": float(r.latitude) if r.latitude else None,
                        "longitude": float(r.longitude) if r.longitude else None,
                        "fine_amount": float(r.fine_amount) if r.fine_amount else 0.0,
                        "is_valid_plate": bool(r.is_valid_plate) if r.is_valid_plate else False,
                    }
                    for r in results
                ]
            finally:
                db.close()
        except Exception as e:
            logger.error("query_violations error: %s", e)
            return []

    def query_risk(self, junction_id: str) -> Dict[str, Any]:
        if not self._loaded and not self.load():
            return {"junction_id": junction_id, "risk_score": random.uniform(2.0, 8.5), "risk_tier": random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]), "breakdown": {}}
        try:
            db = self.SessionLocal()
            try:
                score, tier, breakdown = self.risk_engine.calculate_risk_score(db, junction_id)
                return {"junction_id": junction_id, "risk_score": score, "risk_tier": tier, "breakdown": breakdown}
            finally:
                db.close()
        except Exception as e:
            logger.error("query_risk error: %s", e)
            return {"junction_id": junction_id, "risk_score": random.uniform(2.0, 8.5), "risk_tier": random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]), "breakdown": {}}

    def query_repeat_offenders(self) -> List[Dict[str, Any]]:
        if not self._loaded and not self.load():
            return [{"plate_text": p, "violation_count": random.randint(2, 8), "risk_tier": random.choice(["HIGH RISK", "MEDIUM RISK"]), "last_seen": (datetime.now() - timedelta(hours=random.randint(1, 48))).isoformat()} for p in _DEMO_PLATES[:3]]
        try:
            db = self.SessionLocal()
            try:
                violations = db.query(self.DBViolation).all()
                return self.risk_engine.get_repeat_offenders(violations)
            finally:
                db.close()
        except Exception as e:
            logger.error("query_repeat_offenders error: %s", e)
            return []

    def query_hotspots(self) -> List[Dict[str, Any]]:
        if not self._loaded and not self.load():
            return [{"cluster_id": i, "centroid": {"lat": 12.97 + random.uniform(-0.01, 0.01), "lon": 77.59 + random.uniform(-0.01, 0.01)}, "violation_count": random.randint(3, 15), "dominant_violation": random.choice(_DEMO_TYPES)} for i in range(1, 4)]
        try:
            db = self.SessionLocal()
            try:
                violations = db.query(self.DBViolation).all()
                return self.risk_engine.detect_hotspots(violations)
            finally:
                db.close()
        except Exception as e:
            logger.error("query_hotspots error: %s", e)
            return []

    def query_enforcement_plan(self) -> Dict[str, Any]:
        if not self._loaded and not self.load():
            return {"date": date.today().isoformat(), "total_officers_needed": random.randint(5, 15), "recommended_allocations": [{"junction": f"J00{i}", "officers": random.randint(2, 6), "priority": p} for i, p in enumerate(["HIGH", "MEDIUM", "LOW"], 1)]}
        try:
            db = self.SessionLocal()
            try:
                today = date.today()
                plan = db.query(self.EnforcementPlan).filter(self.EnforcementPlan.plan_date == today).first()
                return plan.plan_data if plan else {"message": "Plan not generated yet"}
            finally:
                db.close()
        except Exception as e:
            logger.error("query_enforcement_plan error: %s", e)
            return {"message": "Plan not generated yet"}

    def query_forecast(self, junction_id: str, hours: int = 24) -> Dict[str, Any]:
        if not self._loaded and not self.load():
            base = datetime.now().replace(minute=0, second=0, microsecond=0)
            forecast = [{"timestamp": (base + timedelta(hours=h)).isoformat(), "predicted_violations": max(0, int(random.gauss(8, 3))), "confidence_interval": {"lower": max(0, int(random.gauss(5, 2))), "upper": int(random.gauss(12, 4))}, "event_flag": "High traffic expected" if random.random() > 0.8 else None} for h in range(hours)]
            return {"junction_id": junction_id, "hours": hours, "forecast": forecast, "model_status": "demo", "metrics": {}}

        try:
            forecast = self.forecaster.predict(junction_id, hours=hours)
            return {
                "junction_id": junction_id,
                "hours": hours,
                "forecast": forecast,
                "model_status": self.forecaster.validation_metrics.get("status", "unavailable"),
                "metrics": self.forecaster.validation_metrics,
            }
        except Exception as e:
            logger.error("query_forecast error: %s", e)
            return {"junction_id": junction_id, "hours": hours, "forecast": [], "model_status": "error", "metrics": {}}


_pipeline_instance = None


def get_pipeline() -> StandalonePipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = StandalonePipeline()
    return _pipeline_instance
