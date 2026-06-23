from sqlalchemy.orm import Session
from .models import Violation

def create_violation(db: Session, violation_data: dict):
    db_violation = Violation(**violation_data)
    db.add(db_violation)
    db.commit()
    db.refresh(db_violation)
    return db_violation

def get_violations(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Violation).order_by(Violation.timestamp.desc()).offset(skip).limit(limit).all()