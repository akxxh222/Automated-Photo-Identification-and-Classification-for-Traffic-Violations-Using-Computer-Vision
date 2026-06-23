import base64
import cv2
import numpy as np
import os
import logging
from datetime import datetime, date, timezone
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.database.database import init_db, SessionLocal
from src.database.models import Violation as DBViolation, EnforcementPlan
from src.violations.risk_engine import RiskEngine
from src.violations.forecaster import TrafficForecaster

logger = logging.getLogger(__name__)

class MLPipeline:
    def __init__(self):
        self.loaded = False

    def load(self):
        if not self.loaded:
            logger.info("Loading ML Pipeline components into memory...")
            from src.preprocessing.preprocessor import FramePreprocessor
            from src.tracking.tracker import UnifiedTracker
            from src.violations.violation_aggregator import ViolationAggregator
            from src.ocr.lpr_engine import LPREngine
            from src.enforcement.evidence_generator import EvidenceGenerator

            self.preprocessor = FramePreprocessor()
            self.tracker = UnifiedTracker()
            self.aggregator = ViolationAggregator()
            self.lpr = LPREngine()
            self.evidence = EvidenceGenerator()
            self.loaded = True

pipeline = MLPipeline()
risk_engine = RiskEngine()
forecaster = TrafficForecaster()

app = FastAPI(
    title="Gridlock AI Traffic Platform",
    description="API for Traffic Enforcement & Risk Intelligence",
    version="1.0.0"
)

# Configure CORS: restrict to specific origins in production
allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

# Eagerly initialize the schema as well so in-memory SQLite tests see the tables
# even when the TestClient does not trigger lifespan startup before the first request.
init_db()

def verify_api_key(x_api_key: str = Header(None)):
    valid_api_keys = os.getenv("API_KEYS", "")
    if not valid_api_keys:
        logger.warning("API_KEYS env var not set. Using insecure default. Set API_KEYS in production.")
        valid_api_keys = "dev-key-123"
    valid_keys = valid_api_keys.split(",")
    if not x_api_key or x_api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return x_api_key

# --- Pydantic Schemas ---
class FrameRequest(BaseModel):
    camera_id: str
    image_base64: str
    timestamp: Optional[str] = None

# --- Sync DB helpers (run in thread pool from async endpoints) ---

def _query_violations(plate, violation_type, junction_id, limit):
    db = SessionLocal()
    try:
        query = db.query(DBViolation)
        if plate:
            query = query.filter(DBViolation.plate_text == plate)
        if violation_type:
            query = query.filter(DBViolation.violation_type == violation_type)
        if junction_id:
            query = query.filter(DBViolation.junction_id == junction_id)
        results = query.order_by(DBViolation.timestamp.desc()).limit(limit).all()
        # Convert ORM objects to dicts for JSON serialization
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

def _query_risk(junction_id):
    db = SessionLocal()
    try:
        score, tier, breakdown = risk_engine.calculate_risk_score(db, junction_id)
        return {"junction_id": junction_id, "risk_score": score, "risk_tier": tier, "breakdown": breakdown}
    finally:
        db.close()

def _query_repeat_offenders():
    db = SessionLocal()
    try:
        violations = db.query(DBViolation).all()
        return risk_engine.get_repeat_offenders(violations)
    finally:
        db.close()

def _query_hotspots():
    from src.database.models import Violation
    db = SessionLocal()
    try:
        violations = db.query(Violation).all()
        return risk_engine.detect_hotspots(violations)
    finally:
        db.close()

def _query_enforcement_plan():
    db = SessionLocal()
    try:
        today = date.today()
        plan = db.query(EnforcementPlan).filter(EnforcementPlan.plan_date == today).first()
        return plan.plan_data if plan else {"message": "Plan not generated yet"}
    finally:
        db.close()

def _query_forecast(junction_id: str, hours: int):
    forecast = forecaster.predict(junction_id, hours=hours)
    return {
        "junction_id": junction_id,
        "hours": hours,
        "forecast": forecast,
        "model_status": forecaster.validation_metrics.get("status", "untrained"),
        "metrics": forecaster.validation_metrics,
    }

def _process_frame_sync(req: FrameRequest):
    db = SessionLocal()
    try:
        pipeline.load()

        try:
            img_data = base64.b64decode(req.image_base64)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        proc_frame, meta = pipeline.preprocessor.process_frame(frame, frame_id=0, source=req.camera_id)
        tracks = pipeline.tracker.process_frame(proc_frame)
        violations = pipeline.aggregator.detect(proc_frame, tracks, req.camera_id)
        plates = pipeline.lpr.process_frame(proc_frame)

        results = []
        for v in violations:
            matched_plate = {"plate_text": "UNKNOWN", "confidence": 0.0, "is_valid": False}
            valid_plates = [p for p in plates if p["is_valid"]]
            if valid_plates:
                matched_plate = sorted(valid_plates, key=lambda x: x["confidence"], reverse=True)[0]

            ev_res = pipeline.evidence.process_violation(frame, v, matched_plate, req.camera_id)

            loc_info = risk_engine.locations.get(req.camera_id, {})
            db_violation = DBViolation(
                plate_text=matched_plate["plate_text"],
                plate_confidence=matched_plate["confidence"],
                violation_type=v["violation_type"],
                violation_confidence=v["confidence"],
                camera_id=req.camera_id,
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

        junction_id = risk_engine.locations.get(req.camera_id, {}).get("junction_id")
        risk_data = {"score": 0.0, "tier": "LOW"}
        if junction_id:
            score, tier, breakdown = risk_engine.calculate_risk_score(db, junction_id)
            risk_data = {"score": score, "tier": tier, "breakdown": breakdown}

        return {"processed_violations": len(results), "events": results, "junction_risk": risk_data}
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# --- Endpoints ---

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "ml_pipeline_loaded": pipeline.loaded, "timestamp": datetime.now(timezone.utc).isoformat()}

@app.post("/api/v1/process-frame")
async def process_frame(req: FrameRequest, api_key: str = Depends(verify_api_key)):
    """End-to-End Pipeline: Ingests a frame, detects violations, generates evidence, updates risk."""
    return await run_in_threadpool(_process_frame_sync, req)

@app.get("/api/v1/violations")
async def get_violations(
    plate: Optional[str] = Query(None),
    violation_type: Optional[str] = Query(None),
    junction_id: Optional[str] = Query(None),
    limit: int = 50,
    api_key: str = Depends(verify_api_key)
):
    return await run_in_threadpool(_query_violations, plate, violation_type, junction_id, limit)

@app.get("/api/v1/risk/{junction_id}")
async def get_risk(junction_id: str, api_key: str = Depends(verify_api_key)):
    return await run_in_threadpool(_query_risk, junction_id)

@app.get("/api/v1/repeat-offenders")
async def get_repeat_offenders(api_key: str = Depends(verify_api_key)):
    return await run_in_threadpool(_query_repeat_offenders)

@app.get("/api/v1/hotspots")
async def get_hotspots(api_key: str = Depends(verify_api_key)):
    return await run_in_threadpool(_query_hotspots)

@app.get("/api/v1/enforcement-plan")
async def get_enforcement_plan(api_key: str = Depends(verify_api_key)):
    return await run_in_threadpool(_query_enforcement_plan)


@app.get("/api/v1/forecast")
async def get_forecast(
    junction_id: str = Query(...),
    hours: int = Query(24, ge=1, le=168),
    api_key: str = Depends(verify_api_key),
):
    return await run_in_threadpool(_query_forecast, junction_id, hours)
