from src.database.database import init_db, SessionLocal
from src.enforcement.enforcement_engine import EnforcementEngine
from src.database.models import RiskScore

def run_smoke_test():
    print("=== Stage 8: Enforcement Recommendation Engine Smoke Test ===")
    init_db()
    db = SessionLocal()
    
    # Seed mock risk scores if none exist for testing the engine logic
    if not db.query(RiskScore).first():
        print("Seeding dummy risk scores for testing...")
        rs1 = RiskScore(junction_id="J001", risk_score=8.5, risk_tier="CRITICAL")
        rs2 = RiskScore(junction_id="J002", risk_score=6.2, risk_tier="HIGH")
        rs3 = RiskScore(junction_id="J003", risk_score=3.5, risk_tier="MEDIUM")
        db.add_all([rs1, rs2, rs3])
        db.commit()

    engine = EnforcementEngine()
    plan = engine.generate_daily_plan(db)
    
    print("\n[✔] Daily Enforcement Plan Generated:")
    print(f"Total Officers Needed: {plan['total_officers_needed']}")
    print("-" * 60)
    for alloc in plan['recommended_allocations']:
        print(f"Junction: {alloc['junction']} | Risk: {alloc['risk_score']} | "
              f"Peak: {alloc['peak_window']}")
        print(f"↳ Recommended Action: {alloc['recommended_action']}\n")
              
    db.close()

if __name__ == "__main__":
    run_smoke_test()
