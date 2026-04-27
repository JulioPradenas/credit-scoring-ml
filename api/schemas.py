from pydantic import BaseModel, Field, model_validator
from typing import Literal


class ApplicantInput(BaseModel):
    applicant_id: str = Field(..., examples=["CLI-00123"])

    revolving_util:    float = Field(..., ge=0.0, le=1.5,  description="Uso de líneas de crédito rotativo")
    age:               int   = Field(..., ge=18,  le=109,  description="Edad del solicitante")
    late_30_59:        int   = Field(..., ge=0,   le=98,   description="Veces con 30-59 días de atraso")
    debt_ratio:        float = Field(..., ge=0.0,           description="Ratio deuda / ingresos")
    monthly_income:    float = Field(..., ge=0.0,           description="Ingreso mensual en USD")
    open_credit_lines: int   = Field(..., ge=0,   le=58,   description="Líneas de crédito abiertas")
    late_90plus:       int   = Field(..., ge=0,   le=98,   description="Veces con 90+ días de atraso")
    real_estate_loans: int   = Field(..., ge=0,   le=54,   description="Préstamos hipotecarios")
    late_60_89:        int   = Field(..., ge=0,   le=98,   description="Veces con 60-89 días de atraso")
    n_dependents:      int   = Field(..., ge=0,   le=20,   description="Número de dependientes")

    @model_validator(mode='after')
    def compute_derived_features(self) -> "ApplicantInput":
        self._debt_to_income      = self.debt_ratio * self.monthly_income
        self._total_late_payments = self.late_30_59 + self.late_60_89 + self.late_90plus
        self._income_per_dependent = self.monthly_income / (self.n_dependents + 1)
        return self

    def to_feature_dict(self) -> dict:
        return {
            "revolving_util":     self.revolving_util,
            "age":                self.age,
            "late_30_59":         self.late_30_59,
            "debt_ratio":         self.debt_ratio,
            "monthly_income":     self.monthly_income,
            "open_credit_lines":  self.open_credit_lines,
            "late_90plus":        self.late_90plus,
            "real_estate_loans":  self.real_estate_loans,
            "late_60_89":         self.late_60_89,
            "n_dependents":       self.n_dependents,
            "debt_to_income":     self._debt_to_income,
            "total_late_payments": self._total_late_payments,
            "income_per_dependent": self._income_per_dependent,
        }


class RiskFactor(BaseModel):
    feature: str
    value:   float
    points:  int
    impact:  Literal["HIGH", "MEDIUM", "LOW"]


class PredictionResponse(BaseModel):
    applicant_id:       str
    score:              int
    probability_default: float
    decision:           Literal["APPROVE", "MANUAL_REVIEW", "REJECT"]
    risk_band:          Literal["Low", "Medium", "High"]
    top_risk_factors:   list[RiskFactor]
    model_version:      str


class BatchInput(BaseModel):
    applicants: list[ApplicantInput] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    total:   int
    results: list[PredictionResponse]


class HealthResponse(BaseModel):
    status:  str
    version: str


class ModelInfoResponse(BaseModel):
    model_name:    str
    model_version: str
    champion:      str
    scorecard_bins: int
    metrics:       dict
