import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "TruthLens"

def test_verify_claim_integration(client):
    res = client.post("/api/verify/claim", json={"claim": "ISRO successfully launched the Aditya-L1 spacecraft."})
    assert res.status_code == 200
    data = res.json()
    assert "request_id" in data
    assert "pipeline_steps" in data
    assert len(data["pipeline_steps"]) == 8
    assert data["overall_verdict"]["verdict"] in ["SUPPORTED", "LIKELY_TRUE"]

def test_analytics_integration(client):
    res = client.get("/api/analytics")
    assert res.status_code == 200
    data = res.json()
    assert "total_verifications" in data
    assert "verdict_distribution" in data
    assert "ml_model_evaluation" in data

def test_demo_samples_integration(client):
    res = client.get("/api/demo-samples")
    assert res.status_code == 200
    samples = res.json()
    assert len(samples) >= 6
    assert any(s["id"] == "sample-clearly-true" for s in samples)
    assert any(s["id"] == "sample-clearly-false" for s in samples)
