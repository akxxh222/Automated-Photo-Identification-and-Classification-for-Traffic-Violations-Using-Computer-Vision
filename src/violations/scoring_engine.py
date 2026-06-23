def compute_risk_score(violations, location_weight=1.0, time_of_day_weight=1.0):
    if not violations: return 0.0
    SEVERITY = {
        "helmet": 2, "seatbelt": 3, "triple_riding": 5, 
        "illegal_parking": 5, "stop_line": 7, "wrong_side": 8, "red_light": 9
    }
    raw_score = 0.0
    for v in violations:
        sev = SEVERITY.get(v.get("type", "unknown"), 1)
        raw_score += sev * v.get("count", 1)
    final = raw_score * location_weight * time_of_day_weight
    return min(10.0, final / 5.0)