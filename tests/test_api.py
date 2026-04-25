"""
API integration tests using standard TestClient.
"""
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.services.prediction_service import prediction_service
from backend.core.registry.model_registry import registry

@pytest.fixture(scope="session", autouse=True)
def init_models():
    """Manually initialize models. Bypasses TestClient lifecycle limits."""
    if not registry.list_models():
        prediction_service.initialize_models()

client = TestClient(app)

POSITIVE_TEXT = "This smartwatch is absolutely incredible! Best purchase I've ever made."
NEGATIVE_TEXT = "Terrible product. Breaks within a week. Total waste of money."

# ─── /health ─────────────────────────────────────────────────────────────────

def test_health_check_status():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_health_check_has_timestamp():
    response = client.get("/health")
    data = response.json()
    assert "timestamp" in data
    assert isinstance(data["timestamp"], float)

# ─── /ready ──────────────────────────────────────────────────────────────────

def test_ready_check_structure():
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    for key in ("ready", "loaded_models", "failed_models", "total_models", "cache_usage"):
        assert key in data

def test_ready_check_at_least_one_model_loaded():
    response = client.get("/ready")
    data = response.json()
    assert data["ready"] is True
    assert len(data["loaded_models"]) > 0

# ─── /models ─────────────────────────────────────────────────────────────────

def test_models_lists_classical_algorithms():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "classical_algorithms" in data
    assert isinstance(data["classical_algorithms"], list)

def test_models_includes_expected_algorithms():
    response = client.get("/models")
    algos = response.json()["classical_algorithms"]
    for expected in ("logistic_regression", "bernoulli_nb", "random_forest"):
        assert expected in algos, f"Expected '{expected}' in /models"

def test_models_is_sorted():
    response = client.get("/models")
    algos = response.json()["classical_algorithms"]
    assert algos == sorted(algos)

# ─── /predict — success cases ────────────────────────────────────────────────

def test_predict_classical_valid_response_shape():
    response = client.post("/predict", json={
        "text": POSITIVE_TEXT,
        "model_type": "classical",
        "algorithm": "logistic_regression",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["sentiment"] in ("Positive", "Neutral", "Negative")
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["model_used"] == "logistic_regression"

def test_predict_all_classical_algorithms_respond():
    """Every registered classical algorithm should return a valid prediction."""
    algorithms = [
        "logistic_regression", "bernoulli_nb", "multinomial_nb",
        "sgd_classifier", "linear_svc", "random_forest",
    ]
    for algo in algorithms:
        response = client.post("/predict", json={
            "text": POSITIVE_TEXT,
            "model_type": "classical",
            "algorithm": algo,
        })
        assert response.status_code == 200, f"{algo} → HTTP {response.status_code}"
        assert response.json()["valid"] is True, f"{algo} returned valid=False: {response.json()}"

def test_predict_positive_text_not_negative():
    response = client.post("/predict", json={
        "text": POSITIVE_TEXT,
        "model_type": "classical",
        "algorithm": "logistic_regression",
    })
    assert response.json()["sentiment"] != "Negative"

def test_predict_negative_text_not_positive():
    response = client.post("/predict", json={
        "text": NEGATIVE_TEXT,
        "model_type": "classical",
        "algorithm": "logistic_regression",
    })
    assert response.json()["sentiment"] != "Positive"

# ─── /predict — validation errors ────────────────────────────────────────────

def test_predict_rejects_empty_text():
    response = client.post("/predict", json={
        "text": "", "model_type": "classical", "algorithm": "logistic_regression"
    })
    assert response.status_code == 422

def test_predict_rejects_text_too_short():
    response = client.post("/predict", json={
        "text": "ok", "model_type": "classical", "algorithm": "logistic_regression"
    })
    assert response.status_code == 422

def test_predict_rejects_invalid_model_type():
    response = client.post("/predict", json={
        "text": POSITIVE_TEXT, "model_type": "neural_network", "algorithm": None
    })
    assert response.status_code == 422

def test_predict_classical_requires_algorithm():
    response = client.post("/predict", json={
        "text": POSITIVE_TEXT, "model_type": "classical", "algorithm": None
    })
    assert response.status_code == 422

def test_predict_missing_fields_returns_422():
    response = client.post("/predict", json={"text": POSITIVE_TEXT})
    assert response.status_code == 422

# ─── /metrics ────────────────────────────────────────────────────────────────

def test_metrics_response_shape():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    for key in ("request_count", "error_count", "error_rate",
                "latency_p50", "latency_p95", "avg_batch_size",
                "model_errors", "circuit_breakers"):
        assert key in data, f"Key '{key}' missing from /metrics"

def test_metrics_error_rate_is_valid_ratio():
    response = client.get("/metrics")
    assert 0.0 <= response.json()["error_rate"] <= 1.0

# ─── /comparison /wordclouds /Images ─────────────────────────────────────────

def test_comparison_returns_200():
    response = client.get("/comparison")
    assert response.status_code in [200, 404]

def test_wordclouds_returns_list():
    response = client.get("/wordclouds")
    assert response.status_code == 200
    assert isinstance(response.json()["wordclouds"], list)

def test_images_returns_list():
    response = client.get("/Images")
    assert response.status_code == 200
    assert isinstance(response.json()["Images"], list)
