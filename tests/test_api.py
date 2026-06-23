from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)
API_HEADERS = {"X-API-Key": "dev-key-123"}


def test_health_check_no_auth_required():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_violations_requires_api_key():
    response = client.get("/api/v1/violations")
    assert response.status_code == 403


def test_violations_with_valid_api_key():
    response = client.get("/api/v1/violations", headers=API_HEADERS)
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_risk_endpoint_with_valid_api_key():
    response = client.get("/api/v1/risk/J001", headers=API_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["junction_id"] == "J001"
    assert "risk_score" in body
    assert "risk_tier" in body
