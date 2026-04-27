import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Payload válido de ejemplo
VALID_PAYLOAD = {
    "applicant_id": "TEST-001",
    "revolving_util":    0.15,
    "age":               45,
    "late_30_59":        0,
    "debt_ratio":        0.30,
    "monthly_income":    6000.0,
    "open_credit_lines": 7,
    "late_90plus":       0,
    "real_estate_loans": 1,
    "late_60_89":        0,
    "n_dependents":      2,
}


@pytest.fixture
def mock_predictor():
    """Reemplaza el predictor real con un mock para no necesitar modelos entrenados."""
    from api.schemas import PredictionResponse, RiskFactor

    mock_response = PredictionResponse(
        applicant_id="TEST-001",
        score=672,
        probability_default=0.043,
        decision="APPROVE",
        risk_band="Low",
        top_risk_factors=[
            RiskFactor(feature="revolving_util", value=0.15, points=-12, impact="HIGH"),
            RiskFactor(feature="late_90plus",    value=0.0,  points=5,   impact="MEDIUM"),
            RiskFactor(feature="debt_ratio",     value=0.30, points=-8,  impact="LOW"),
        ],
        model_version="1.0.0",
    )

    with patch("api.predictor.predictor") as mock_pred:
        mock_pred._loaded       = True
        mock_pred.predict_one   = MagicMock(return_value=mock_response)
        mock_pred.predict_batch = MagicMock(return_value=[mock_response])
        mock_pred.champion      = "XGBoost"
        mock_pred.model_info    = {
            "model_name":    "CreditScoringML",
            "model_version": "1.0.0",
            "champion":      "XGBoost",
            "scorecard_bins": 120,
            "metrics":       {"AUC-ROC": 0.856},
        }
        yield mock_pred


@pytest.fixture
def client(mock_predictor):
    from api.main import app
    return TestClient(app)


# ── Health & Info ──────────────────────────────────────────────────────────────

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    assert "champion" in data
    assert "metrics" in data


# ── Predict ────────────────────────────────────────────────────────────────────

def test_predict_valid_payload(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert "score" in data
    assert "decision" in data
    assert "probability_default" in data
    assert "top_risk_factors" in data
    assert data["applicant_id"] == "TEST-001"


def test_predict_response_schema(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    data = r.json()
    assert data["decision"] in ("APPROVE", "MANUAL_REVIEW", "REJECT")
    assert data["risk_band"] in ("Low", "Medium", "High")
    assert 0.0 <= data["probability_default"] <= 1.0
    assert isinstance(data["score"], int)
    assert len(data["top_risk_factors"]) <= 3


def test_predict_invalid_age(client):
    payload = {**VALID_PAYLOAD, "age": 15}  # menor de 18
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_invalid_revolving_util(client):
    payload = {**VALID_PAYLOAD, "revolving_util": 5.0}  # > 1.5
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_missing_field(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "monthly_income"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_negative_income(client):
    payload = {**VALID_PAYLOAD, "monthly_income": -100.0}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


# ── Batch ──────────────────────────────────────────────────────────────────────

def test_predict_batch_valid(client):
    payload = {"applicants": [VALID_PAYLOAD]}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 1
    assert len(data["results"]) == 1


def test_predict_batch_empty_list(client):
    r = client.post("/predict/batch", json={"applicants": []})
    assert r.status_code == 422


def test_predict_batch_over_limit(client):
    many = [VALID_PAYLOAD] * 1001
    r = client.post("/predict/batch", json={"applicants": many})
    assert r.status_code == 422
