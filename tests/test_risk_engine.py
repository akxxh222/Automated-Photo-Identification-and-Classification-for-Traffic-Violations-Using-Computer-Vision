import json

from src.violations.risk_engine import RiskEngine


def test_location_weight_uses_zone_type():
    engine = RiskEngine(loc_path="configs/camera_locations.json")
    assert engine._get_location_weight("J001") == 1.5
    assert engine._get_location_weight("J002") == 1.5

    with open("configs/camera_locations.json", "r") as f:
        locations = json.load(f)
    assert "zone_type" in locations["CAM_001"]
    assert "zone_type" in locations["CAM_002"]
