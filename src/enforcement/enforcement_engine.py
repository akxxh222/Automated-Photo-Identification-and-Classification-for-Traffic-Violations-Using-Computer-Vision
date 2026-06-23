import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.database.database import SessionLocal
from src.database.models import RiskScore, EnforcementPlan
from src.violations.forecaster import TrafficForecaster

logger = logging.getLogger(__name__)

class EnforcementEngine:
    def __init__(self):
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.forecaster = TrafficForecaster()

    def _get_action_and_officers(self, score: float) -> Tuple[str, int]:
        if score >= 8.0:
            return "2 officers + 1 patrol unit", 2
        elif score >= 5.0:
            return "1 officer", 1
        elif score >= 2.0:
            return "camera-only monitoring", 0
        else:
            return "no action needed", 0

    def generate_daily_plan(self, db: Session) -> Dict[str, Any]:
        logger.info("Generating Daily Enforcement Recommendation Plan...")
        
        # Subquery to fetch the latest risk score for each junction
        subquery = db.query(
            RiskScore.junction_id,
            func.max(RiskScore.computed_at).label('max_time')
        ).group_by(RiskScore.junction_id).subquery()

        latest_scores = db.query(RiskScore).join(
            subquery,
            (RiskScore.junction_id == subquery.c.junction_id) & 
            (RiskScore.computed_at == subquery.c.max_time)
        ).all()

        allocations = []
        total_officers = 0

        if not latest_scores:
            logger.warning("No risk scores found in DB to generate a plan.")
            
        for rs in latest_scores:
            action, officers = self._get_action_and_officers(rs.risk_score)
            total_officers += officers

            # Obtain the 3-hour short-horizon forecast
            forecast = self.forecaster.predict(rs.junction_id, hours=3)
            total_pred = sum(f["predicted_violations"] for f in forecast)
            
            # Determine the peak traffic window within the forecasted hours
            if forecast:
                peak = max(forecast, key=lambda x: x["predicted_violations"])
                peak_time = datetime.fromisoformat(peak["timestamp"])
                peak_window = f"{peak_time.strftime('%H:%M')} - {(peak_time + timedelta(hours=1)).strftime('%H:%M')}"
            else:
                peak_window = "N/A"

            allocations.append({
                "junction": rs.junction_id,
                "risk_score": round(rs.risk_score, 2),
                "predicted_violations": total_pred,
                "recommended_action": action,
                "peak_window": peak_window
            })

        # Rank priorities critically by Risk Score (Descending)
        allocations.sort(key=lambda x: x["risk_score"], reverse=True)

        plan_data = {
            "date": date.today().isoformat(),
            "total_officers_needed": total_officers,
            "recommended_allocations": allocations
        }

        # 1. Export as static JSON artifact
        report_path = self.reports_dir / f"enforcement_plan_{date.today().isoformat()}.json"
        with open(report_path, "w") as f:
            json.dump(plan_data, f, indent=4)
        logger.info("Successfully exported AI enforcement plan to %s", report_path)

        # 2. Store to PostgreSQL (Update if regenerating for the same day)
        existing_plan = db.query(EnforcementPlan).filter(EnforcementPlan.plan_date == date.today()).first()
        if existing_plan:
            existing_plan.plan_data = plan_data
        else:
            db_plan = EnforcementPlan(plan_date=date.today(), plan_data=plan_data)
            db.add(db_plan)
        db.commit()

        return plan_data

def start_scheduler():
    """Initializes the APScheduler to run the daily plan generation automatically."""
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler()
    
    def scheduled_job():
        db = SessionLocal()
        engine = EnforcementEngine()
        engine.generate_daily_plan(db)
        db.close()

    # Trigger every morning at 06:00 AM
    scheduler.add_job(scheduled_job, 'cron', hour=6, minute=0)
    scheduler.start()
    logger.info("APScheduler active: Daily Enforcement Engine task scheduled for 06:00 AM.")
