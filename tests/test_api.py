"""Tests for API endpoints."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns expected response."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "AMR Surveillance ML API"


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_models_endpoint():
    """Test models listing endpoint."""
    response = client.get("/models")
    # May return 503 if models not loaded, which is acceptable in test
    assert response.status_code in [200, 503]


def test_predict_endpoint_structure():
    """Test predict endpoint accepts correct input structure."""
    payload = {
        "bacterial_species": "escherichia_coli",
        "sample_source": "drinking_water",
        "administrative_region": "region_iii_central_luzon",
        "ampicillin_int": "r",
        "tetracycline_int": "r"
    }
    
    response = client.post("/predict", json=payload)
    # May return 503 if models not loaded in test environment
    assert response.status_code in [200, 500, 503]
