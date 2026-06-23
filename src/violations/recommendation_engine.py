"""Small recommendation-layer helpers used by the enforcement stage."""

from src.database.database import SessionLocal, init_db
from src.enforcement.enforcement_engine import EnforcementEngine


def get_enforcement_tier(score: float) -> str:
    if score >= 8.0:
        return "CRITICAL"
    if score >= 5.0:
        return "HIGH"
    if score >= 2.0:
        return "MEDIUM"
    return "LOW"


def generate_daily_plan(db=None):
    """Generate and persist the daily enforcement plan.

    This is a thin convenience wrapper around ``EnforcementEngine`` so other
    modules can request a plan without needing to manage database sessions.
    """
    owns_db = db is None
    if db is None:
        init_db()
        db = SessionLocal()

    try:
        engine = EnforcementEngine()
        return engine.generate_daily_plan(db)
    finally:
        if owns_db:
            db.close()
