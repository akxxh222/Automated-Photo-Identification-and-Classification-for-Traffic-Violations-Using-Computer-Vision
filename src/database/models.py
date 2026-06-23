import uuid
from datetime import datetime, date
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Date, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Violation(Base):
    __tablename__ = "violations"

    # Using String(36) to store UUIDs ensures max compatibility between Postgres and SQLite
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    plate_text = Column(String(20), index=True)
    plate_confidence = Column(Float)
    violation_type = Column(String(50), nullable=False)
    violation_confidence = Column(Float, nullable=False)
    camera_id = Column(String(20), nullable=False)
    junction_id = Column(String(20), index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    frame_hash = Column(String(64))
    evidence_path = Column(String(255))
    challan_path = Column(String(255))
    fine_amount = Column(Integer)
    is_valid_plate = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class RiskScore(Base):
    __tablename__ = "risk_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    junction_id = Column(String(20), nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_tier = Column(String(20), nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow)
    violation_breakdown = Column(JSON)

class EnforcementPlan(Base):
    __tablename__ = "enforcement_plans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_date = Column(Date, nullable=False)
    plan_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)