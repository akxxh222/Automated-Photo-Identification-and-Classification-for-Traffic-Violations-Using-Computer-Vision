CREATE TABLE violations (
  id VARCHAR(36) PRIMARY KEY,
  plate_text VARCHAR(20),
  plate_confidence FLOAT,
  violation_type VARCHAR(50) NOT NULL,
  violation_confidence FLOAT NOT NULL,
  camera_id VARCHAR(20) NOT NULL,
  junction_id VARCHAR(20),
  latitude FLOAT,
  longitude FLOAT,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  frame_hash VARCHAR(64),
  evidence_path VARCHAR(255),
  challan_path VARCHAR(255),
  fine_amount INTEGER,
  is_valid_plate BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_violations_plate ON violations(plate_text);
CREATE INDEX idx_violations_timestamp ON violations(timestamp);
CREATE INDEX idx_violations_junction ON violations(junction_id);

CREATE TABLE risk_scores (
  id SERIAL PRIMARY KEY,
  junction_id VARCHAR(20) NOT NULL,
  risk_score FLOAT NOT NULL,
  risk_tier VARCHAR(20) NOT NULL,
  computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  violation_breakdown JSONB
);

CREATE TABLE enforcement_plans (
  id SERIAL PRIMARY KEY,
  plan_date DATE NOT NULL,
  plan_data JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
