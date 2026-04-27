# 🏦 Credit Scoring con Machine Learning
## Plan de Proyecto para Portafolio de Data Scientist

> **Objetivo:** Construir un sistema de scoring crediticio end-to-end con valor agregado real: modelo interpretable tipo scorecard, análisis de riesgo con SHAP, curva de rentabilidad esperada y API de inferencia lista para producción.

---

## 📁 Estructura del Proyecto

```
credit-scoring-ml/
│
├── data/
│   ├── raw/                        # Dataset original sin modificar
│   ├── processed/                  # Datos limpios y transformados
│   └── external/                   # Datos externos si aplica
│
├── notebooks/
│   ├── 01_eda_exploratory.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   ├── 03_modeling_baseline.ipynb
│   ├── 04_modeling_advanced.ipynb
│   ├── 05_scorecard_woe_iv.ipynb
│   ├── 06_model_explainability_shap.ipynb
│   └── 07_business_value_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── features/
│   │   ├── engineering.py
│   │   └── woe_encoder.py          # Weight of Evidence encoder custom
│   ├── models/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── scorecard.py            # Lógica de scorecard interpretable
│   └── visualization/
│       ├── eda_plots.py
│       └── shap_plots.py
│
├── api/
│   ├── main.py                     # FastAPI app
│   ├── schemas.py                  # Pydantic models
│   └── predictor.py
│
├── models/
│   └── saved/                      # Modelos serializados (.pkl / .joblib)
│
├── reports/
│   └── figures/                    # Gráficas exportadas
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_api.py
│
├── requirements.txt
├── Dockerfile
├── README.md
└── PLAN.md                         # Este archivo
```

---

## 📊 Dataset

**Dataset principal:** [Give Me Some Credit — Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)

| Campo | Descripción |
|---|---|
| `SeriousDlqin2yrs` | **Target:** 1 = default en 2 años |
| `RevolvingUtilizationOfUnsecuredLines` | Uso de líneas de crédito |
| `age` | Edad del solicitante |
| `NumberOfTime30-59DaysPastDueNotWorse` | Atrasos leves |
| `DebtRatio` | Ratio deuda / ingresos |
| `MonthlyIncome` | Ingreso mensual |
| `NumberOfOpenCreditLinesAndLoans` | Líneas abiertas |
| `NumberOfTimes90DaysLate` | Atrasos graves |
| `NumberRealEstateLoansOrLines` | Préstamos inmobiliarios |
| `NumberOfTime60-89DaysPastDueNotWorse` | Atrasos moderados |
| `NumberOfDependents` | Número de dependientes |

**Tamaño:** ~150,000 filas | Tasa de default: ~6.7% (desbalanceo moderado)

---

## 🗺️ Fases del Proyecto

### FASE 1 — Exploración y Análisis (EDA)
**Notebook:** `01_eda_exploratory.ipynb`

**Tareas:**
- [ ] Cargar dataset y revisar shape, dtypes, memoria
- [ ] Análisis de valores nulos (visualizar con `missingno`)
- [ ] Distribución de la variable target — calcular tasa de default
- [ ] Distribuciones de variables numéricas (histogramas + boxplots)
- [ ] Correlaciones (heatmap + pairplot por target)
- [ ] Detección de outliers con IQR y Z-score
- [ ] Análisis bivariado: cada feature vs target
- [ ] **Valor agregado:** Análisis de `Information Value (IV)` preliminar por variable

**Output esperado:**
- Informe EDA con hallazgos clave comentados en el notebook
- Lista de variables con alto/bajo poder predictivo (IV)

---

### FASE 2 — Preprocesamiento y Feature Engineering
**Notebook:** `02_preprocessing_feature_engineering.ipynb`

**Tareas:**
- [ ] Imputación de nulos:
  - `MonthlyIncome`: mediana por grupo de edad
  - `NumberOfDependents`: moda
- [ ] Tratamiento de outliers con capping (percentiles 1% y 99%)
- [ ] Creación de nuevas features:
  - `debt_to_income_ratio` = DebtRatio × MonthlyIncome
  - `total_late_payments` = suma de todos los atrasos
  - `credit_utilization_bucket` = bins de utilización de crédito
  - `age_group` = grupos etarios (18-30, 31-45, 46-60, 60+)
  - `income_per_dependent` = MonthlyIncome / (NumberOfDependents + 1)
- [ ] **WoE Encoding (Weight of Evidence):**
  - Binning óptimo de variables continuas
  - Calcular WoE e IV por bin
  - Reemplazar valores originales por WoE para modelo scorecard
- [ ] Split train/validation/test (60/20/20) con stratify
- [ ] Serializar pipeline de transformación

**Archivos generados:**
- `src/features/woe_encoder.py` — clase WoEEncoder reutilizable
- `data/processed/train.csv`, `val.csv`, `test.csv`

---

### FASE 3 — Modelado Baseline
**Notebook:** `03_modeling_baseline.ipynb`

**Modelos a entrenar:**
- [ ] Logistic Regression (baseline interpretable)
- [ ] Decision Tree
- [ ] Random Forest
- [ ] Gradient Boosting (sklearn)

**Para cada modelo:**
- [ ] Cross-validation con 5-fold estratificado
- [ ] Métricas: AUC-ROC, KS Statistic, Gini, Brier Score
- [ ] Curva ROC y Precision-Recall
- [ ] Matriz de confusión con umbral optimizado

**Manejo del desbalanceo:**
- [ ] Comparar: sin balanceo vs SMOTE vs class_weight='balanced'
- [ ] Documentar impacto en métricas

---

### FASE 4 — Modelado Avanzado
**Notebook:** `04_modeling_advanced.ipynb`

**Modelos:**
- [ ] **XGBoost** con Optuna para hypertuning
- [ ] **LightGBM** con early stopping
- [ ] **CatBoost** (manejo nativo de categoricals)

**Proceso:**
- [ ] Optuna: 100 trials con pruning para XGBoost
- [ ] Calibración de probabilidades (Platt Scaling / Isotonic Regression)
- [ ] Comparativa de modelos en tabla final
- [ ] Selección del modelo campeón por AUC + interpretabilidad

**Tabla comparativa final:**

| Modelo | AUC-ROC | KS | Gini | Brier Score |
|---|---|---|---|---|
| Logistic Reg | - | - | - | - |
| Random Forest | - | - | - | - |
| XGBoost | - | - | - | - |
| LightGBM | - | - | - | - |

---

### FASE 5 — Scorecard Interpretable ⭐ (Valor Agregado #1)
**Notebook:** `05_scorecard_woe_iv.ipynb`

> Esta fase diferencia el proyecto. Un scorecard es el estándar de la industria bancaria.

**Tareas:**
- [ ] Construir scorecard a partir de la Logistic Regression + WoE
- [ ] Calcular puntos por cada bin de cada variable:
  ```
  Score = A - B × ln(odds)
  Puntos_variable = -(WoE × beta_i × B)
  ```
  Donde A=600 (score base), B=20/ln(2) (PDO=20 puntos dobles los odds)
- [ ] Definir bandas de riesgo:
  - Score < 550: Alto Riesgo → Rechazar
  - 550–650: Riesgo Medio → Revisar manualmente
  - Score > 650: Bajo Riesgo → Aprobar
- [ ] Tabla scorecard exportable a Excel/CSV
- [ ] Visualización: puntos por variable (bar chart horizontal)
- [ ] Función `calcular_score(perfil_cliente) -> score, decision`

**Output:**
- `models/saved/scorecard_table.csv`
- `src/models/scorecard.py`

---

### FASE 6 — Explicabilidad con SHAP ⭐ (Valor Agregado #2)
**Notebook:** `06_model_explainability_shap.ipynb`

**Análisis global:**
- [ ] SHAP summary plot (beeswarm) — importancia global
- [ ] SHAP bar plot — magnitud promedio por feature
- [ ] Dependence plots para las top 5 features

**Análisis local (individual):**
- [ ] Waterfall plot para 3 perfiles: aprobado / borderline / rechazado
- [ ] Force plot interactivo

**Análisis de fairness:**
- [ ] Comparar distribución de scores por `age_group`
- [ ] Verificar que el modelo no discrimine por edad de forma injustificada
- [ ] Documentar hallazgos como análisis de sesgo

---

### FASE 7 — Análisis de Valor de Negocio ⭐ (Valor Agregado #3)
**Notebook:** `07_business_value_analysis.ipynb`

> Este análisis convierte el proyecto de "técnico" a "orientado al negocio".

**Curva de Ganancia Esperada:**
- [ ] Definir matriz de costos:
  - Verdadero Negativo (rechazar buen cliente): -$0 (costo oportunidad)
  - Falso Positivo (aprobar cliente que hace default): -$10,000
  - Falso Negativo (rechazar cliente bueno): -$500 (pérdida de ingreso)
  - Verdadero Positivo (rechazar cliente malo): +$10,000 (ahorro)
- [ ] Calcular ganancia esperada para cada umbral de decisión
- [ ] Graficar curva umbral vs ganancia
- [ ] Encontrar umbral óptimo de negocio

**Análisis de KPIs de negocio:**
- [ ] Tasa de aprobación por banda de riesgo
- [ ] Expected Loss por banda
- [ ] ROI estimado vs modelo sin scoring (aprobar todo)
- [ ] Tabla de decisión de política crediticia

---

### FASE 8 — API de Inferencia con FastAPI
**Archivos:** `api/main.py`, `api/schemas.py`

**Endpoints:**
```
POST /predict          → Score individual + decisión + explicación top 3 features
POST /predict/batch    → Scoring en lote (hasta 1000 registros)
GET  /health           → Health check
GET  /model/info       → Versión del modelo, métricas, fecha de entrenamiento
```

**Ejemplo de respuesta `/predict`:**
```json
{
  "applicant_id": "CLI-00123",
  "score": 623,
  "probability_default": 0.087,
  "decision": "MANUAL_REVIEW",
  "risk_band": "Medium",
  "top_risk_factors": [
    {"feature": "total_late_payments", "impact": "HIGH", "value": 2},
    {"feature": "revolving_utilization", "impact": "MEDIUM", "value": 0.82},
    {"feature": "debt_ratio", "impact": "LOW", "value": 0.45}
  ],
  "model_version": "1.0.0"
}
```

**Tareas:**
- [ ] Pydantic schemas para input/output
- [ ] Cargar modelo al iniciar (no en cada request)
- [ ] Manejo de errores con HTTPException
- [ ] Validación de rangos de input
- [ ] Middleware de logging

---

### FASE 9 — Containerización y Documentación Final
**Archivos:** `Dockerfile`, `README.md`

**Docker:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**README.md debe incluir:**
- [ ] Badge de AUC-ROC del modelo
- [ ] GIF o screenshot del API respondiendo
- [ ] Tabla de métricas finales
- [ ] Sección "Business Impact" con resultados del análisis de valor
- [ ] Instrucciones de instalación y ejecución (3 comandos máximo)
- [ ] Descripción de la scorecard con ejemplo visual
- [ ] Sección de arquitectura con diagrama

---

## 📦 Stack Tecnológico

```
# Core ML
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
xgboost==2.0.0
lightgbm==4.1.0
catboost==1.2.0
optuna==3.3.0

# Explicabilidad
shap==0.43.0

# Visualización
matplotlib==3.7.0
seaborn==0.12.0
plotly==5.17.0

# API
fastapi==0.103.0
uvicorn==0.23.0
pydantic==2.3.0

# Utils
joblib==1.3.0
imbalanced-learn==0.11.0
missingno==0.5.2

# Dev
pytest==7.4.0
jupyter==1.0.0
```

---

## 📐 Métricas de Éxito del Proyecto

| Métrica | Mínimo Aceptable | Objetivo |
|---|---|---|
| AUC-ROC | 0.78 | > 0.82 |
| KS Statistic | 0.35 | > 0.45 |
| Gini | 0.56 | > 0.64 |
| Brier Score | < 0.10 | < 0.07 |
| Latencia API (p95) | < 200ms | < 50ms |

---

## 🎯 Valor Agregado vs Proyecto Típico

| Característica | Proyecto Típico | Este Proyecto |
|---|---|---|
| Modelos | Solo XGBoost | Múltiples + selección rigurosa |
| Interpretabilidad | Feature importance básica | SHAP completo + scorecard bancaria |
| Negocio | Solo AUC-ROC | Análisis de ganancia esperada + política crediticia |
| Despliegue | Solo notebook | FastAPI + Docker |
| WoE/IV | No incluye | Sí — estándar bancario |
| Fairness | No incluye | Análisis de sesgo por edad |

---

## ⏱️ Estimación de Tiempo

| Fase | Tiempo estimado |
|---|---|
| EDA | 4–6 horas |
| Preprocesamiento + Feature Engineering | 6–8 horas |
| Modelado Baseline | 3–4 horas |
| Modelado Avanzado + Tuning | 5–7 horas |
| Scorecard (WoE/IV) | 6–8 horas |
| SHAP + Explicabilidad | 4–5 horas |
| Análisis de Negocio | 4–5 horas |
| FastAPI | 4–6 horas |
| Docker + README | 2–3 horas |
| **Total** | **~38–52 horas** |

---

## 🚀 Comandos de Inicio Rápido (para README)

```bash
# 1. Clonar e instalar
git clone https://github.com/tuusuario/credit-scoring-ml
pip install -r requirements.txt

# 2. Ejecutar API local
uvicorn api.main:app --reload

# 3. Docker
docker build -t credit-scoring . && docker run -p 8000:8000 credit-scoring
```

---

## 📌 Notas para Claude Code

- Comenzar siempre por la **Fase 1** antes de tocar modelos
- El `WoEEncoder` en `src/features/woe_encoder.py` debe ser compatible con sklearn Pipeline (`fit`, `transform`, `fit_transform`)
- Los notebooks deben tener **celdas Markdown explicativas** entre cada bloque de código — esto es clave para portafolio
- Todos los gráficos deben guardarse en `reports/figures/` con nombres descriptivos
- El modelo final se serializa con `joblib`, no `pickle`
- La API debe cargar el modelo **una sola vez** al iniciar con `@app.on_event("startup")`
- Mantener un `mlflow` local o un diccionario de experimentos en `models/saved/experiments.json`
