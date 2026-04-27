import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from api.schemas import ApplicantInput, PredictionResponse, RiskFactor

MODELS_DIR = Path(__file__).parent.parent / "models" / "saved"
MODEL_VERSION = "1.0.0"

_IMPACT_THRESHOLDS = {"HIGH": 0.66, "MEDIUM": 0.33}


def _impact_label(abs_points: int, max_abs: int) -> str:
    ratio = abs_points / max(max_abs, 1)
    if ratio >= _IMPACT_THRESHOLDS["HIGH"]:
        return "HIGH"
    if ratio >= _IMPACT_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


class CreditPredictor:
    """Carga y mantiene en memoria el modelo campeón y la scorecard."""

    def __init__(self) -> None:
        self.model        = None
        self.scorecard    = None
        self.cap_bounds   = None
        self.experiments  = None
        self.champion     = None
        self._loaded      = False

    def load(self) -> None:
        experiments_path = MODELS_DIR / "experiments.json"
        self.experiments  = json.loads(experiments_path.read_text())
        self.champion     = self.experiments["advanced"]["champion"]

        file_map = {
            "XGBoost":  "champion_xgb.joblib",
            "LightGBM": "champion_lgb.joblib",
            "CatBoost": "champion_catboost.joblib",
        }
        self.model     = joblib.load(MODELS_DIR / file_map[self.champion])
        self.scorecard = joblib.load(MODELS_DIR / "scorecard.joblib")
        self.cap_bounds = joblib.load(MODELS_DIR / "cap_bounds.joblib")
        self._loaded   = True

    def _apply_caps(self, features: dict) -> dict:
        capped = features.copy()
        for col, (low, high) in self.cap_bounds.items():
            if col in capped:
                capped[col] = float(np.clip(capped[col], low, high))
        return capped

    def predict_one(self, applicant: ApplicantInput) -> PredictionResponse:
        features = self._apply_caps(applicant.to_feature_dict())

        # Score de la scorecard (interpretable, base para decisión)
        sc_result = self.scorecard.score(features)

        # Probabilidad del modelo campeón (más precisa)
        X = pd.DataFrame([features])[self.scorecard.features_]
        prob = float(self.model.predict_proba(X)[0, 1])

        # Top risk factors desde el breakdown de la scorecard
        breakdown = sc_result["breakdown"]
        abs_points = [abs(f["points"]) for f in breakdown]
        max_abs    = max(abs_points) if abs_points else 1

        top3 = [
            RiskFactor(
                feature=f["feature"],
                value=round(f["value"], 4),
                points=f["points"],
                impact=_impact_label(abs(f["points"]), max_abs),
            )
            for f in breakdown[:3]
        ]

        return PredictionResponse(
            applicant_id=applicant.applicant_id,
            score=sc_result["score"],
            probability_default=round(prob, 4),
            decision=sc_result["decision"],
            risk_band=sc_result["risk_band"],
            top_risk_factors=top3,
            model_version=MODEL_VERSION,
        )

    def predict_batch(self, applicants: list[ApplicantInput]) -> list[PredictionResponse]:
        return [self.predict_one(a) for a in applicants]

    @property
    def model_info(self) -> dict:
        advanced = self.experiments.get("advanced", {})
        results  = advanced.get("results", {}).get("AUC-ROC", {})
        return {
            "model_name":    "CreditScoringML",
            "model_version": MODEL_VERSION,
            "champion":      self.champion,
            "scorecard_bins": len(self.scorecard.scorecard_table_),
            "metrics": {
                "AUC-ROC": results.get(self.champion),
                "xgb_optuna_best_auc": advanced.get("xgb_best_auc"),
            },
        }


predictor = CreditPredictor()
