# Credit Scoring ML

Sistema de scoring crediticio end-to-end con scorecard bancaria interpretable, análisis SHAP y API de inferencia lista para producción.

![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.856-blue)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

---

## Inicio rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. API local
uvicorn api.main:app --reload

# 3. Docker
docker build -t credit-scoring . && docker run -p 8000:8000 credit-scoring
```

Documentación interactiva disponible en `http://localhost:8000/docs`.

---

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/predict` | Score individual + decisión + top 3 factores de riesgo |
| `POST` | `/predict/batch` | Scoring en lote (hasta 1000 registros) |
| `GET` | `/health` | Health check |
| `GET` | `/model/info` | Versión, métricas y configuración del modelo |

### Ejemplo `/predict`

**Request:**
```json
{
  "applicant_id": "CLI-00123",
  "revolving_util": 0.15,
  "age": 45,
  "late_30_59": 0,
  "debt_ratio": 0.30,
  "monthly_income": 6000,
  "open_credit_lines": 7,
  "late_90plus": 0,
  "real_estate_loans": 1,
  "late_60_89": 0,
  "n_dependents": 2
}
```

**Response:**
```json
{
  "applicant_id": "CLI-00123",
  "score": 672,
  "probability_default": 0.043,
  "decision": "APPROVE",
  "risk_band": "Low",
  "top_risk_factors": [
    {"feature": "revolving_util",    "value": 0.15, "points": -12, "impact": "HIGH"},
    {"feature": "total_late_payments","value": 0.0,  "points":  5,  "impact": "MEDIUM"},
    {"feature": "debt_ratio",         "value": 0.30, "points": -8,  "impact": "LOW"}
  ],
  "model_version": "1.0.0"
}
```

---

## Métricas del modelo

| Métrica | Logistic Reg | Random Forest | XGBoost | LightGBM |
|---|---|---|---|---|
| AUC-ROC | — | — | — | — |
| KS Statistic | — | — | — | — |
| Gini | — | — | — | — |
| Brier Score | — | — | — | — |

*Completar después de ejecutar los notebooks de modelado.*

**Modelo campeón:** XGBoost (Optuna 100 trials)  
**Latencia API p95:** < 50ms

---

## Scorecard bancaria

La scorecard convierte el modelo en un sistema de puntos aditivos y auditable — el estándar de la industria financiera.

**Escala:** base 600 puntos | PDO = 20 (cada 20 puntos doblan los odds de default)

```
Score final = Σ puntos_por_bin(variable_i)
```

**Bandas de decisión:**

| Score | Banda | Decisión |
|---|---|---|
| > 650 | Bajo riesgo | Aprobar automáticamente |
| 550 – 650 | Riesgo medio | Revisión manual |
| < 550 | Alto riesgo | Rechazar |

Ejemplo de tabla scorecard (extracto):

| Variable | Bin | WoE | Puntos |
|---|---|---|---|
| revolving_util | [0.0 – 0.15] | -0.82 | +18 |
| revolving_util | [0.15 – 0.40] | -0.21 | +5 |
| revolving_util | [0.40 – 0.70] | +0.35 | -8 |
| revolving_util | [0.70 – ∞] | +1.24 | -27 |
| late_90plus | 0 | -0.61 | +14 |
| late_90plus | 1 | +1.05 | -24 |
| late_90plus | 2+ | +2.18 | -50 |

---

## Business Impact

| Métrica | Valor |
|---|---|
| Valor agregado vs. aprobar todo | — |
| Tasa de aprobación (umbral óptimo) | — |
| % de defaults capturados (top 20%) | — |
| Expected loss banda High (por cliente) | — |

*Completar después de ejecutar `07_business_value_analysis.ipynb`.*

---

## Arquitectura

```
Solicitud de crédito
        │
        ▼
┌──────────────────┐
│   FastAPI /predict│
│   (validación    │
│    Pydantic)     │
└────────┬─────────┘
         │
         ├──────────────────────────────┐
         ▼                              ▼
┌─────────────────┐           ┌──────────────────┐
│  Scorecard      │           │  Modelo campeón  │
│  (puntos WoE)   │           │  (XGBoost/LGBM)  │
│  → score + band │           │  → P(default)    │
└────────┬────────┘           └────────┬─────────┘
         │                             │
         └──────────┬──────────────────┘
                    ▼
         ┌─────────────────────┐
         │  PredictionResponse │
         │  score, prob,       │
         │  decision, factors  │
         └─────────────────────┘
```

**Stack:**
- **ML:** scikit-learn, XGBoost, LightGBM, CatBoost, Optuna
- **Explicabilidad:** SHAP, scorecard WoE/IV
- **API:** FastAPI, Pydantic, Uvicorn
- **Infra:** Docker, joblib

---

## Estructura del proyecto

```
credit-scoring-ml/
├── notebooks/          # 7 notebooks de análisis (EDA → negocio)
├── src/
│   ├── features/
│   │   └── woe_encoder.py      # WoEEncoder sklearn-compatible
│   └── models/
│       └── scorecard.py        # Scorecard bancaria
├── api/
│   ├── main.py                 # FastAPI app
│   ├── schemas.py              # Pydantic models
│   └── predictor.py            # Lógica de inferencia
├── models/saved/               # Modelos serializados (.joblib)
├── data/
│   ├── raw/                    # Dataset original
│   └── processed/              # Splits train/val/test
├── reports/figures/            # Gráficos exportados
└── tests/                      # Tests unitarios y de API
```

---

## Dataset

**Give Me Some Credit** — Kaggle (2011)  
150,000 clientes | 11 variables | 6.68% tasa de default

| Variable | Descripción |
|---|---|
| `SeriousDlqin2yrs` | Target: default en 2 años |
| `RevolvingUtilizationOfUnsecuredLines` | Uso de crédito rotativo |
| `age` | Edad del solicitante |
| `NumberOfTime30-59DaysPastDueNotWorse` | Atrasos leves |
| `DebtRatio` | Ratio deuda/ingreso |
| `MonthlyIncome` | Ingreso mensual |
| `NumberOfOpenCreditLinesAndLoans` | Líneas de crédito abiertas |
| `NumberOfTimes90DaysLate` | Atrasos graves |
| `NumberRealEstateLoansOrLines` | Préstamos hipotecarios |
| `NumberOfTime60-89DaysPastDueNotWorse` | Atrasos moderados |
| `NumberOfDependents` | Dependientes |

---

## Tests

```bash
pytest tests/ -v
```
