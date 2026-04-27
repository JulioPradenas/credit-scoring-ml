# Credit Scoring ML

Sistema de scoring crediticio end-to-end con scorecard bancaria interpretable, análisis SHAP y API de inferencia lista para producción.

![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.866-blue)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Tests](https://github.com/JulioPradenas/credit-scoring-ml/actions/workflows/tests.yml/badge.svg)

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

## Reproducir el proyecto

### 1. Clonar e instalar

```bash
git clone https://github.com/JulioPradenas/credit-scoring-ml.git
cd credit-scoring-ml
pip install -r requirements.txt
```

### 2. Descargar el dataset

```bash
# Requiere cuenta en kaggle.com y token en ~/.kaggle/kaggle.json
pip install kaggle
kaggle datasets download -d brycecf/give-me-some-credit-dataset -p data/raw/
unzip data/raw/give-me-some-credit-dataset.zip -d data/raw/
```

### 3. Ejecutar el pipeline completo

```bash
# Opción A — notebooks en orden (recomendado para explorar el análisis)
jupyter notebook

# Opción B — solo entrenar los modelos avanzados directamente
python3 train_advanced.py
```

Orden de notebooks: `01_eda` → `02_preprocessing` → `03_baseline` → `04_advanced` → `05_scorecard` → `06_shap` → `07_business`

### 4. Verificar la API

```bash
uvicorn api.main:app --reload

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_id": "CLI-001",
    "revolving_util": 0.15, "age": 45, "late_30_59": 0,
    "debt_ratio": 0.30, "monthly_income": 6000, "open_credit_lines": 7,
    "late_90plus": 0, "real_estate_loans": 1, "late_60_89": 0, "n_dependents": 2
  }'
```

### 5. Correr los tests

```bash
pytest tests/test_preprocessing.py -v
```

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

| Métrica | Logistic Reg | Random Forest | Gradient Boost | XGBoost | LightGBM | CatBoost |
|---|---|---|---|---|---|---|
| AUC-ROC | 0.746 | 0.864 | 0.865 | **0.866** | 0.864 | 0.866 |
| KS Statistic | 0.391 | 0.567 | 0.567 | **0.574** | 0.572 | 0.569 |
| Gini | 0.492 | 0.727 | 0.730 | **0.733** | 0.728 | 0.731 |
| Brier Score | 0.060 | **0.050** | **0.050** | 0.139 | 0.112 | 0.141 |

**Modelo campeón:** XGBoost (Optuna 100 trials) — AUC-ROC 0.866  
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
| Valor agregado vs. aprobar todo | $30,889,500 (set de test) |
| Tasa de aprobación (umbral óptimo 0.269) | 55.1% |
| % de defaults capturados revisando top 20% | 73.3% |
| Umbral óptimo de negocio | 0.269 |

---

## Fairness Analysis

El modelo fue auditado para detectar sesgo injustificado por grupo de edad (`notebook 06`).

**Default rate real vs. probabilidad predicha por grupo etario (set de test, n=30,000):**

| Grupo de edad | N | Default rate real | P(default) predicha |
|---|---|---|---|
| 18–30 | 2,124 | 12.85% | 47.69% |
| 31–45 | 8,129 | 9.50% | 40.86% |
| 46–60 | 10,721 | 6.66% | 34.03% |
| 60+ | 9,026 | 2.73% | 19.58% |

**Conclusión:** El riesgo predicho decrece monotónicamente con la edad, siguiendo la tasa de default real. El modelo captura el patrón legítimo de riesgo crediticio (mayor historial financiero → menor tasa de default) sin introducir discriminación injustificada por edad. Las probabilidades absolutas están infladas por el `scale_pos_weight` aplicado para manejar el desbalanceo de clases — el ranking relativo es correcto.

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
