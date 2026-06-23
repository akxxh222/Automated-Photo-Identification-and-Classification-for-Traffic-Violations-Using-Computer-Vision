"""
Enhanced Risk Engine for Traffic Violation Assessment.
Stage 6: Traffic Risk Intelligence Engine

Features:
- Violation severity scoring with configurable weights
- Composite risk index calculation
- DBSCAN-based hotspot detection
- Repeat-offender tracking
- Real-time risk scoring per junction
"""

import json
import yaml
import numpy as np
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.cluster import DBSCAN

try:
    from sqlalchemy.orm import Session
    from sqlalchemy import func
    from src.database.models import Violation, RiskScore
    HAS_DB = True
except ImportError:
    HAS_DB = False

logger = logging.getLogger(__name__)

# Configurable violation severity weights (0-10 scale)
SEVERITY = {
    "helmet": 2,
    "seatbelt": 3,
    "triple_riding": 5,
    "illegal_parking": 5,
    "stop_line": 7,
    "wrong_side": 8,
    "red_light": 9
}

class RiskEngine:
    """
    Comprehensive risk assessment and hotspot detection engine.
    Calculates: Risk Score = Frequency × Severity × Location Weight × Time Weight
    """
    
    def __init__(self, config_path="configs/config.yaml", loc_path="configs/camera_locations.json"):
        self.config_path = Path(config_path)
        self.loc_path = Path(loc_path)
        self.locations = {}
        self.junction_zones = {}
        
        self._load_locations()
        logger.info("✓ Risk Engine initialized")
    
    def _load_locations(self) -> None:
        try:
            if self.loc_path.exists():
                with open(self.loc_path, "r") as f:
                    self.locations = json.load(f)
                
                for cam, data in self.locations.items():
                    jid = data.get("junction_id", cam)
                    self.junction_zones[jid] = {
                        'zone_type': data.get("zone_type", "standard"),
                        'is_school': data.get("zone_type") in ["school", "hospital"],
                    }
        except Exception as e:
            logger.warning(f"Could not load locations: {e}")
    
    def _get_time_weight(self, dt: Optional[datetime] = None) -> float:
        if dt is None:
            dt = datetime.now()
        hour = dt.hour
        if (8 <= hour < 10) or (17 <= hour < 20):
            return 1.3
        return 1.0

    def _get_location_weight(self, junction_id: str) -> float:
        zone_info = self.junction_zones.get(junction_id, {})
        if zone_info.get('is_school'):
            return 1.5
        return 1.0

    def calculate_risk_score(self, db: Session, junction_id: str, window_minutes: int = 15) -> Tuple[float, str, Dict[str, int]]:
        if not HAS_DB:
            return 0.0, "LOW", {}
        
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(minutes=window_minutes)
        
        violations = db.query(Violation).filter(
            Violation.junction_id == junction_id,
            Violation.timestamp >= start_time
        ).all()
        
        if not violations:
            return 0.0, "LOW", {}
            
        raw_score = 0.0
        breakdown = Counter()
        
        for v in violations:
            sev = SEVERITY.get(v.violation_type, 1)
            raw_score += sev
            breakdown[v.violation_type] += 1
            
        loc_weight = self._get_location_weight(junction_id)
        time_weight = self._get_time_weight(now)
        
        final_score = raw_score * loc_weight * time_weight
        normalized_score = min(10.0, final_score / 5.0)
        
        if normalized_score >= 8.0:
            tier = "CRITICAL"
        elif normalized_score >= 5.0:
            tier = "HIGH"
        elif normalized_score >= 2.0:
            tier = "MEDIUM"
        else:
            tier = "LOW"
            
        return normalized_score, tier, dict(breakdown)

    def detect_hotspots(self, violations: List[Dict], eps: float = 0.001, min_samples: int = 5) -> List[Dict[str, Any]]:
        coords_list = []
        valid_violations = []
        
        for v in violations:
            if isinstance(v, dict):
                lat = v.get('latitude') or v.get('lat')
                lon = v.get('longitude') or v.get('lon')
            else:
                lat = getattr(v, 'latitude', None)
                lon = getattr(v, 'longitude', None)
            
            if lat and lon:
                coords_list.append([lat, lon])
                valid_violations.append(v)
        
        if len(coords_list) < min_samples:
            return []
            
        coords = np.array(coords_list)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = clustering.labels_
        
        hotspots = []
        for label in set(labels):
            if label == -1:
                continue
                
            cluster_mask = (labels == label)
            cluster_points = coords[cluster_mask]
            centroid = cluster_points.mean(axis=0)
            
            cluster_violations = [v for i, v in enumerate(valid_violations) if cluster_mask[i]]
            violation_types = [v.get('violation_type') if isinstance(v, dict) 
                             else getattr(v, 'violation_type', 'unknown') 
                             for v in cluster_violations]
            
            hotspots.append({
                "cluster_id": int(label),
                "centroid": {"lat": float(centroid[0]), "lon": float(centroid[1])},
                "violation_count": int(sum(cluster_mask)),
                "dominant_violation": Counter(violation_types).most_common(1)[0][0] if violation_types else 'unknown'
            })
            
        return hotspots

    def _get_attr(self, v, *keys):
        for key in keys:
            if hasattr(v, key):
                return getattr(v, key)
            if isinstance(v, dict) and key in v:
                return v[key]
        return None

    def get_repeat_offenders(self, violations, window_days: int = 30, threshold: int = 3) -> List[Dict[str, Any]]:
        plate_violations = defaultdict(list)
        now = datetime.now()
        
        for v in violations:
            plate = self._get_attr(v, 'plate_number', 'plate_text')
            if plate:
                ts = self._get_attr(v, 'timestamp', 'violation_timestamp') or now
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts)
                    except ValueError:
                        ts = now
                
                if isinstance(ts, datetime) and (now - ts).days <= window_days:
                    plate_violations[plate].append(v)
        
        offenders = []
        for plate, viols in plate_violations.items():
            if len(viols) >= threshold:
                tier = "HIGH RISK" if len(viols) >= 5 else "WARNING"
                last_ts = now
                for v in viols:
                    t = self._get_attr(v, 'timestamp', 'violation_timestamp') or now
                    if isinstance(t, str):
                        try:
                            t = datetime.fromisoformat(t)
                        except ValueError:
                            t = now
                    if isinstance(t, datetime) and t > last_ts:
                        last_ts = t
                vtypes = []
                for v in viols:
                    vt = self._get_attr(v, 'violation_type', 'violation_type_name')
                    if vt:
                        vtypes.append(vt)
                offenders.append({
                    "plate_number": plate,
                    "violation_count": len(viols),
                    "last_seen": last_ts,
                    "risk_tier": tier,
                    "violation_types": dict(Counter(vtypes))
                })
        
        return sorted(offenders, key=lambda x: x["violation_count"], reverse=True)
