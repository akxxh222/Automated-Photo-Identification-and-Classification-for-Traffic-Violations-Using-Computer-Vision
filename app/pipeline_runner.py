import io
import os
import logging
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


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
        self._load_err = None

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
            self._load_err = str(e)
            logger.warning("Pipeline load failed (non-fatal for dashboard): %s", e)
            return False

    def process_image(self, image_bytes: bytes, camera_id: str = "CAM_001") -> Dict[str, Any]:
        if not self._loaded and not self.load():
            return {"processed_violations": 0, "events": [], "junction_risk": {"score": 0.0, "tier": "LOW"}}
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
            return []
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
            return {"junction_id": junction_id, "risk_score": 0.0, "risk_tier": "LOW", "breakdown": {}}
        try:
            db = self.SessionLocal()
            try:
                score, tier, breakdown = self.risk_engine.calculate_risk_score(db, junction_id)
                return {"junction_id": junction_id, "risk_score": score, "risk_tier": tier, "breakdown": breakdown}
            finally:
                db.close()
        except Exception as e:
            logger.error("query_risk error: %s", e)
            return {"junction_id": junction_id, "risk_score": 0.0, "risk_tier": "LOW", "breakdown": {}}

    def query_repeat_offenders(self) -> List[Dict[str, Any]]:
        if not self._loaded and not self.load():
            return []
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
            return []
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
            return {"message": "Plan not generated yet"}
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
            return {"junction_id": junction_id, "hours": hours, "forecast": [], "model_status": "unavailable", "metrics": {}}

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
