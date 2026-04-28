# Credit Scoring ML

Sistema de scoring crediticio end-to-end: scorecard bancaria WoE/IV, XGBoost con Optuna, análisis SHAP, FastAPI lista para producción y dashboard interactivo.

**[Live Demo →](https://credit-scoring-ml.streamlit.app)** *(actualizar con URL real tras deploy en Streamlit Cloud)*

![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.866-blue)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Tests](https://github.com/JulioPradenas/credit-scoring-ml/actions/workflows/tests.yml/badge.svg)
![Docker CI](https://github.com/JulioPradenas/credit-scoring-ml/actions/workflows/docker.yml/badge.svg)

---

## Project Overview

**Problema:** Los bancos aprueban créditos con modelos black-box o revisión manual. Ninguno cumple los dos requisitos regulatorios simultáneamente: precisión predictiva y explicabilidad auditable.

**Usuario final:** Analista de riesgo crediticio en una institución financiera que necesita justificar cada decisión ante el regulador.

**Datos:** [Give Me Some Credit](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset) (Kaggle 2011) — 150,000 clientes, 11 variables, 6.68% tasa de default.

**Output del modelo:**
- Score 300–850 (estilo FICO, sistema de puntos aditivo y auditable)
- P(default) del modelo campeón XGBoost
- Decisión: `APPROVE` / `MANUAL_REVIEW` / `REJECT`
- Top-3 factores de riesgo con impacto cuantificado

**Decisión clave de diseño:** La scorecard WoE/IV corre en paralelo con XGBoost — no en lugar de. El score es auditable para regulación; la probabilidad del modelo es más precisa para el umbral de negocio. Ambos outputs se devuelven en cada predicción.

---

## Architecture

```
  CSV crudo (150k filas)
        │
        ▼
┌─────────────────────┐
│  Feature Engineering │  debt_to_income, total_late_payments,
│  src/features/       │  income_per_dependent, high_utilization,
│  engineering.py      │  has_severe_late, util_bucket, age_group
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  WoE Encoding        │  Binning + Weight of Evidence por variable
│  src/features/       │  Information Value para selección de features
│  woe_encoder.py      │
└────────┬────────────┘
         │
         ├──────────────────────────────┐
         ▼                              ▼
┌──────────────────┐        ┌──────────────────────┐
│  Scorecard       │        │  XGBoost + Optuna     │
│  (WoE → puntos)  │        │  100 trials           │
│  base 600, PDO=20│        │  AUC-ROC 0.866        │
└────────┬─────────┘        └──────────┬───────────┘
         │                             │
         └──────────────┬──────────────┘
                        ▼
           ┌─────────────────────────┐
           │  FastAPI /predict        │
           │  score + prob + decision │
           │  + top-3 risk factors    │
           └────────────┬────────────┘
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
     REST API (8000)      Dashboard (8501)
     /docs Swagger        Streamlit UI
```

---

## Results

| Modelo | AUC-ROC | KS | Gini | Brier | vs. baseline |
|---|---|---|---|---|---|
| LogReg WoE *(baseline)* | 0.746 | 0.391 | 0.492 | 0.060 | — |
| Decision Tree | 0.852 | 0.552 | 0.704 | 0.051 | +14.2% AUC |
| Random Forest | 0.864 | 0.567 | 0.727 | 0.050 | +15.8% AUC |
| Gradient Boost | 0.865 | 0.567 | 0.730 | 0.050 | +16.0% AUC |
| LightGBM | 0.864 | 0.572 | 0.728 | 0.112 | +15.9% AUC |
| CatBoost | 0.866 | 0.569 | 0.731 | 0.141 | +16.1% AUC |
| **XGBoost + Optuna** | **0.866** | **0.574** | **0.733** | 0.139 | **+16.1% AUC** |

**Modelo campeón:** XGBoost (Optuna 100 trials) — AUC-ROC 0.866, KS 0.574  
**Latencia API p95:** < 50 ms  
**Valor agregado vs. aprobar todo:** $30,889,500 (set de test, 30k clientes)

---

## Tech Stack

| Herramienta | Propósito |
|---|---|
| XGBoost + Optuna | Modelo campeón con tuning bayesiano (100 trials) |
| scikit-learn | Pipeline ML, validación cruzada, métricas |
| LightGBM / CatBoost | Modelos challenger para comparación |
| WoEEncoder (custom) | Encoding WoE sklearn-compatible con IV automático |
| SHAP TreeExplainer | Explicabilidad global/local y auditoría de fairness |
| Scorecard (custom) | Sistema de puntos aditivos y auditables (estándar bancario) |
| FastAPI + Pydantic | API REST con validación de inputs y docs automáticas |
| Streamlit + Plotly | Dashboard interactivo para presentar el modelo a stakeholders |
| Docker + Compose | Imagen única, dos servicios: API (8000) y dashboard (8501) |
| GitHub Actions | CI: pytest en cada PR + Docker build/push a GHCR en cada push |

---

## Setup & Installation

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

### 3. Entrenar los modelos

```bash
# Opción A — notebooks en orden (recomendado para explorar el análisis)
jupyter notebook
# Orden: 01_eda → 02_preprocessing → 03_baseline → 04_advanced → 05_scorecard → 06_shap → 07_business

# Opción B — entrenar modelos avanzados directamente (más rápido)
python train_advanced.py
```

---

## How to Run

### API local

```bash
uvicorn api.main:app --reload
# Documentación interactiva: http://localhost:8000/docs
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_id": "CLI-001",
    "revolving_util": 0.15, "age": 45, "late_30_59": 0,
    "debt_ratio": 0.30, "monthly_income": 6000, "open_credit_lines": 7,
    "late_90plus": 0, "real_estate_loans": 1, "late_60_89": 0, "n_dependents": 2
  }'
```

### Dashboard Streamlit

```bash
streamlit run dashboard/app.py
# Dashboard: http://localhost:8501
```

### Docker Compose (API + Dashboard)

```bash
docker compose up --build
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

### Tests

```bash
pytest tests/test_preprocessing.py -v
```

---

## API Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/predict` | Score individual + decisión + top 3 factores de riesgo |
| `POST` | `/predict/batch` | Scoring en lote (hasta 1,000 registros) |
| `GET` | `/health` | Health check |
| `GET` | `/model/info` | Versión, métricas y configuración del modelo |

**Ejemplo de respuesta `/predict`:**
```json
{
  "applicant_id": "CLI-001",
  "score": 672,
  "probability_default": 0.043,
  "decision": "APPROVE",
  "risk_band": "Low",
  "top_risk_factors": [
    {"feature": "revolving_util",     "value": 0.15, "points": -12, "impact": "HIGH"},
    {"feature": "total_late_payments","value": 0.0,  "points":   5, "impact": "MEDIUM"},
    {"feature": "debt_ratio",         "value": 0.30, "points":  -8, "impact": "LOW"}
  ],
  "model_version": "1.0.0"
}
```

---

## Feature Engineering

Todas las features se crean en `src/features/engineering.py` — fuente única importada por la API y el dashboard.

| Feature engineered | Fórmula | IV | Razón |
|---|---|---|---|
| `revolving_util` *(original)* | — | **1.11** | Predictor más fuerte: estrés de crédito rotativo |
| `total_late_payments` | `late_30_59 + late_60_89 + late_90plus` | **1.06** | Agrega historial de atrasos; evita colapso de bins por capping p99 |
| `late_30_59` *(original)* | — | **0.48** | Señal fuerte independiente de atrasos leves |
| `age` *(original)* | — | **0.25** | Historial crediticio — más edad, menor riesgo |
| `debt_to_income` | `debt_ratio × monthly_income` | — | Deuda en USD absoluta; ratio × ingreso da carga real |
| `income_per_dependent` | `monthly_income / (n_dependents + 1)` | — | Ingreso disponible real por persona a cargo |
| `high_utilization` | `revolving_util > 0.8 → 1` | — | Flag de estrés financiero inminente |
| `has_severe_late` | `late_90plus > 0 → 1` | — | Señal binaria de incumplimiento grave pasado |

---

## Scorecard Bancaria

La scorecard convierte el modelo en un sistema de puntos aditivos y auditable — el estándar de la industria financiera para cumplimiento regulatorio.

**Escala:** base 600 puntos | PDO = 20 (cada 20 puntos doblan los odds de default)

```
Score final = Σ puntos_por_bin(variable_i)
```

| Score | Banda | Decisión |
|---|---|---|
| > 650 | Bajo riesgo | Aprobar automáticamente |
| 550 – 650 | Riesgo medio | Revisión manual |
| < 550 | Alto riesgo | Rechazar |

Extracto de la tabla scorecard:

| Variable | Bin | WoE | Puntos |
|---|---|---|---|
| revolving_util | [0.00 – 0.15] | -0.82 | +18 |
| revolving_util | [0.15 – 0.40] | -0.21 | +5 |
| revolving_util | [0.40 – 0.70] | +0.35 | -8 |
| revolving_util | [0.70 – ∞] | +1.24 | -27 |
| total_late_payments | 0 | -0.61 | +14 |
| total_late_payments | 1 | +1.05 | -24 |
| total_late_payments | 2+ | +2.18 | -50 |

---

## Fairness Analysis

El modelo fue auditado para detectar sesgo injustificado por grupo de edad (notebook 06, set de test n=30,000).

| Grupo de edad | N | Default rate real | P(default) predicha |
|---|---|---|---|
| 18–30 | 2,124 | 12.85% | 47.69% |
| 31–45 | 8,129 | 9.50% | 40.86% |
| 46–60 | 10,721 | 6.66% | 34.03% |
| 60+ | 9,026 | 2.73% | 19.58% |

**Conclusión:** El riesgo predicho decrece monotónicamente con la edad, siguiendo la tasa de default real. El modelo captura el patrón legítimo de riesgo crediticio sin introducir discriminación injustificada por edad. Las probabilidades absolutas están infladas por el `scale_pos_weight` aplicado para manejar el desbalanceo de clases — el ranking relativo es correcto.

---

## Key Decisions & Lessons

1. **Scorecard + XGBoost en paralelo, no en competencia.** La scorecard resuelve el requisito regulatorio de auditabilidad; el modelo resuelve el requisito de negocio de precisión. Un solo modelo no puede hacer las dos cosas igual de bien.

2. **Optuna en lugar de GridSearch.** 100 trials bayesianos mejoran AUC de 0.865 → 0.866 usando ~10× menos evaluaciones que un grid equivalente. El gain es marginal en AUC pero la infraestructura de tuning queda lista para versiones futuras.

3. **Feature engineering antes del WoE encoding.** Crear `total_late_payments` antes de aplicar WoE evita que `late_90plus` colapse a un solo bin por el capping p99 (más del 95% de los clientes tiene 0 atrasos graves — su señal queda absorbida en la feature agregada).

4. **Fallo real:** el notebook 04 falló en ejecución porque XGBoost 3.x eliminó el parámetro `use_label_encoder`. Solución: `train_advanced.py` como script standalone de respaldo que no depende del estado del kernel del notebook.

5. **Calibración vs. ranking.** Las probabilidades absolutas del modelo están infladas por `scale_pos_weight` (ajuste de desbalanceo de clases). Para el umbral de negocio el ranking relativo es correcto; para comunicar riesgo a clientes habría que re-calibrar con Platt scaling.

---

## Business Impact

| Métrica | Valor |
|---|---|
| Valor agregado vs. aprobar todo | $30,889,500 (set de test) |
| Tasa de aprobación (umbral óptimo 0.269) | 55.1% |
| % de defaults capturados revisando top 20% | 73.3% |
| Umbral óptimo de negocio | 0.269 |

---

## File Structure

```
credit-scoring-ml/
├── .github/
│   └── workflows/
│       ├── tests.yml          # CI: pytest en cada push/PR
│       └── docker.yml         # CI: build + push a GHCR en cada push a main
├── notebooks/                 # 7 notebooks de análisis (EDA → negocio)
│   ├── 01_eda_exploratory.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   ├── 03_modeling_baseline.ipynb
│   ├── 04_modeling_advanced.ipynb
│   ├── 05_scorecard_woe_iv.ipynb
│   ├── 06_model_explainability_shap.ipynb
│   └── 07_business_value_analysis.ipynb
├── src/
│   ├── features/
│   │   ├── engineering.py     # Feature engineering (fuente única para API + dashboard)
│   │   └── woe_encoder.py     # WoEEncoder sklearn-compatible
│   └── models/
│       └── scorecard.py       # Scorecard bancaria (WoE → puntos → score)
├── api/
│   ├── main.py                # FastAPI app (4 endpoints)
│   ├── schemas.py             # Pydantic models (validación de inputs)
│   └── predictor.py           # Lógica de inferencia (carga modelos, aplica capping)
├── dashboard/
│   └── app.py                 # Streamlit UI (4 páginas: simulador, métricas, negocio, pipeline)
├── models/saved/              # Modelos serializados (.joblib) — commiteados (~1.6 MB)
│   ├── champion_xgb.joblib    # XGBoost campeón (Optuna)
│   ├── scorecard.joblib       # Scorecard bancaria
│   ├── woe_encoder.joblib     # WoE encoder entrenado
│   ├── cap_bounds.joblib      # Límites de capping p99
│   ├── scorecard_table.csv    # Tabla WoE + puntos por bin
│   └── experiments.json       # Métricas de todos los modelos
├── data/
│   ├── raw/                   # Dataset original (excluido de git)
│   └── processed/             # Splits train/val/test (excluidos de git)
├── tests/
│   └── test_preprocessing.py  # 11 tests: WoEEncoder + IV validation
├── train_advanced.py          # Script standalone para re-entrenar modelos avanzados
├── Dockerfile                 # python:3.11-slim, expone 8000 + 8501
├── docker-compose.yml         # 2 servicios: api (8000) + dashboard (8501)
└── requirements.txt           # Dependencias completas del proyecto
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
