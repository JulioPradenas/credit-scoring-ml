import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.predictor import CreditPredictor
from api.schemas import ApplicantInput

MODELS_DIR = Path(__file__).parent.parent / "models" / "saved"

st.set_page_config(
    page_title="Credit Scoring ML",
    page_icon="📊",
    layout="wide",
)

DECISION_COLOR = {"APPROVE": "#22c55e", "MANUAL_REVIEW": "#f59e0b", "REJECT": "#ef4444"}
BAND_COLOR = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}


@st.cache_resource
def load_predictor() -> CreditPredictor:
    p = CreditPredictor()
    p.load()
    return p


@st.cache_data
def load_experiments() -> dict:
    return json.loads((MODELS_DIR / "experiments.json").read_text())


@st.cache_data
def load_scorecard_table() -> pd.DataFrame:
    return pd.read_csv(MODELS_DIR / "scorecard_table.csv")


@st.cache_data
def load_woe_encoder():
    import joblib
    return joblib.load(MODELS_DIR / "woe_encoder.joblib")


# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.title("Credit Scoring ML")
page = st.sidebar.radio(
    "Navegación",
    ["Simulador de Score", "Resultados del Modelo", "Impacto de Negocio", "Cómo lo Construí"],
)

predictor = load_predictor()

# ── Page 1: Score Simulator ────────────────────────────────────────────────────

if page == "Simulador de Score":
    st.title("Simulador de Score Crediticio")
    st.markdown("Ingresa los datos del solicitante para obtener un score instantáneo con explicación de factores de riesgo.")

    with st.form("applicant_form"):
        col1, col2 = st.columns(2)

        with col1:
            revolving_util = st.slider("Uso de crédito rotativo", 0.0, 1.5, 0.15, 0.01,
                                       help="Porcentaje de líneas de crédito utilizadas")
            age = st.number_input("Edad", min_value=18, max_value=109, value=45)
            late_30_59 = st.number_input("Atrasos 30-59 días", min_value=0, max_value=98, value=0)
            debt_ratio = st.slider("Ratio deuda/ingreso", 0.0, 5.0, 0.30, 0.01)
            monthly_income = st.number_input("Ingreso mensual (USD)", min_value=0.0, value=6000.0, step=500.0)

        with col2:
            open_credit_lines = st.number_input("Líneas de crédito abiertas", min_value=0, max_value=58, value=7)
            late_90plus = st.number_input("Atrasos 90+ días", min_value=0, max_value=98, value=0)
            real_estate_loans = st.number_input("Préstamos hipotecarios", min_value=0, max_value=54, value=1)
            late_60_89 = st.number_input("Atrasos 60-89 días", min_value=0, max_value=98, value=0)
            n_dependents = st.number_input("Número de dependientes", min_value=0, max_value=20, value=2)

        submitted = st.form_submit_button("Evaluar solicitud", use_container_width=True, type="primary")

    if submitted:
        applicant = ApplicantInput(
            applicant_id="DEMO-001",
            revolving_util=revolving_util,
            age=int(age),
            late_30_59=int(late_30_59),
            debt_ratio=debt_ratio,
            monthly_income=float(monthly_income),
            open_credit_lines=int(open_credit_lines),
            late_90plus=int(late_90plus),
            real_estate_loans=int(real_estate_loans),
            late_60_89=int(late_60_89),
            n_dependents=int(n_dependents),
        )
        result = predictor.predict_one(applicant)

        st.divider()

        col_gauge, col_meta = st.columns([1, 1])

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result.score,
                title={"text": "Score crediticio"},
                gauge={
                    "axis": {"range": [300, 850]},
                    "bar": {"color": BAND_COLOR[result.risk_band]},
                    "steps": [
                        {"range": [300, 550], "color": "#fee2e2"},
                        {"range": [550, 650], "color": "#fef9c3"},
                        {"range": [650, 850], "color": "#dcfce7"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": result.score},
                },
            ))
            fig.update_layout(height=300, margin={"t": 30, "b": 0, "l": 20, "r": 20})
            st.plotly_chart(fig, use_container_width=True)

        with col_meta:
            decision_color = DECISION_COLOR[result.decision]
            st.markdown(
                f"<h2 style='color:{decision_color}'>{result.decision}</h2>",
                unsafe_allow_html=True,
            )
            st.metric("Probabilidad de default", f"{result.probability_default:.1%}")
            st.metric("Banda de riesgo", result.risk_band)
            st.metric("Score", result.score)

        st.subheader("Top factores de riesgo")
        impact_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        rows = []
        for f in result.top_risk_factors:
            sign = "+" if f.points >= 0 else ""
            rows.append({
                "Variable": f.feature,
                "Valor": round(f.value, 4),
                "Puntos": f"{sign}{f.points}",
                "Impacto": f"{impact_color[f.impact]} {f.impact}",
            })
        st.table(pd.DataFrame(rows))


# ── Page 2: Model Results ──────────────────────────────────────────────────────

elif page == "Resultados del Modelo":
    st.title("Resultados del Modelo")

    experiments = load_experiments()
    scorecard_df = load_scorecard_table()

    # Leaderboard
    st.subheader("Comparación de modelos")

    baseline_r = experiments["baseline"]["results"]
    advanced_r = experiments["advanced"]["results"]

    all_models = {}
    for metric in ["AUC-ROC", "KS", "Gini", "Brier"]:
        for model, val in {**baseline_r.get(metric, {}), **advanced_r.get(metric, {})}.items():
            all_models.setdefault(model, {})[metric] = val

    leaderboard = pd.DataFrame(all_models).T.reset_index().rename(columns={"index": "Modelo"})
    leaderboard = leaderboard.sort_values("AUC-ROC", ascending=False).reset_index(drop=True)

    champion = experiments["advanced"]["champion"]
    st.dataframe(
        leaderboard.style.apply(
            lambda row: ["background-color: #dcfce7" if row["Modelo"] == champion else "" for _ in row],
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Modelo campeón: **{champion}** (fondo verde)")

    st.divider()

    # IV por variable
    st.subheader("Information Value por variable")
    try:
        enc = load_woe_encoder()
        iv_table = enc.get_iv_table().reset_index().rename(columns={"index": "Variable"})
        iv_table = iv_table.sort_values("IV", ascending=True)
        fig_iv = go.Figure(go.Bar(
            x=iv_table["IV"],
            y=iv_table["Variable"],
            orientation="h",
            marker_color=["#22c55e" if iv >= 0.3 else "#f59e0b" if iv >= 0.1 else "#ef4444"
                          for iv in iv_table["IV"]],
        ))
        fig_iv.update_layout(
            xaxis_title="IV",
            height=max(300, len(iv_table) * 35),
            margin={"l": 150, "r": 20, "t": 20, "b": 40},
        )
        st.plotly_chart(fig_iv, use_container_width=True)
    except Exception:
        pass

    st.divider()

    # Scorecard table
    st.subheader("Scorecard bancaria (WoE + Puntos)")
    variables = ["Todas"] + sorted(scorecard_df["Variable"].unique().tolist())
    selected_var = st.selectbox("Filtrar por variable", variables)
    df_show = scorecard_df if selected_var == "Todas" else scorecard_df[scorecard_df["Variable"] == selected_var]
    st.dataframe(df_show.style.background_gradient(subset=["Puntos"], cmap="RdYlGn"),
                 use_container_width=True, hide_index=True)


# ── Page 3: Business Impact ────────────────────────────────────────────────────

elif page == "Impacto de Negocio":
    st.title("Impacto de Negocio")

    col1, col2, col3 = st.columns(3)
    col1.metric("Valor agregado vs. aprobar todo", "$30,889,500")
    col2.metric("Tasa de aprobación", "55.1%", help="Umbral óptimo 0.269")
    col3.metric("Defaults capturados (top 20%)", "73.3%")

    st.divider()

    # Decision bands
    st.subheader("Bandas de decisión")
    bands = pd.DataFrame({
        "Banda": ["Alto riesgo (Rechazar)", "Riesgo medio (Revisión manual)", "Bajo riesgo (Aprobar)"],
        "Score": ["< 550", "550 – 650", "> 650"],
        "Decisión": ["REJECT", "MANUAL_REVIEW", "APPROVE"],
    })
    st.table(bands)

    st.divider()

    # Fairness analysis
    st.subheader("Fairness Analysis — Default rate por grupo de edad")
    st.markdown("Auditado para detectar sesgo injustificado por grupo etario (notebook 06, set de test n=30,000).")

    fairness_data = pd.DataFrame({
        "Grupo de edad": ["18–30", "31–45", "46–60", "60+"],
        "N": [2124, 8129, 10721, 9026],
        "Default rate real": ["12.85%", "9.50%", "6.66%", "2.73%"],
        "P(default) predicha": ["47.69%", "40.86%", "34.03%", "19.58%"],
    })
    st.table(fairness_data)

    fig_fair = go.Figure()
    groups = ["18–30", "31–45", "46–60", "60+"]
    real_rates = [12.85, 9.50, 6.66, 2.73]
    pred_rates = [47.69, 40.86, 34.03, 19.58]

    fig_fair.add_trace(go.Bar(name="Default rate real (%)", x=groups, y=real_rates,
                              marker_color="#3b82f6"))
    fig_fair.add_trace(go.Bar(name="P(default) predicha (%)", x=groups, y=pred_rates,
                              marker_color="#f97316", opacity=0.7))
    fig_fair.update_layout(
        barmode="group",
        xaxis_title="Grupo de edad",
        yaxis_title="%",
        height=380,
        legend={"orientation": "h", "y": -0.2},
    )
    st.plotly_chart(fig_fair, use_container_width=True)

    st.info(
        "El riesgo predicho decrece monotónicamente con la edad, siguiendo la tasa de default real. "
        "El modelo captura el patrón legítimo de riesgo crediticio sin introducir discriminación "
        "injustificada por edad. Las probabilidades absolutas están infladas por el scale_pos_weight "
        "aplicado para manejar el desbalanceo de clases — el ranking relativo es correcto."
    )


# ── Page 4: How It Was Built ───────────────────────────────────────────────────

elif page == "Cómo lo Construí":
    st.title("Cómo lo Construí")

    st.subheader("Pipeline de 7 etapas")
    stages = [
        ("01 EDA", "Análisis exploratorio: distribuciones, correlaciones, IV preliminar"),
        ("02 Preprocessing", "Imputación, capping p99, WoE encoding, feature engineering"),
        ("03 Baseline", "LogReg + Decision Tree + RF + GBM — benchmark con CV"),
        ("04 Advanced", "XGBoost con Optuna (100 trials) + LightGBM + CatBoost"),
        ("05 Scorecard", "Scorecard bancaria: puntos WoE, escala base 600, PDO=20"),
        ("06 SHAP", "Explicabilidad global/local + fairness analysis por edad"),
        ("07 Business", "Curva de valor, umbral óptimo, análisis costo/beneficio"),
    ]
    for stage, desc in stages:
        with st.expander(stage):
            st.write(desc)

    st.divider()
    st.subheader("Stack tecnológico")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**ML**")
        st.markdown("- scikit-learn\n- XGBoost\n- LightGBM\n- CatBoost\n- Optuna")
    with col2:
        st.markdown("**Explicabilidad**")
        st.markdown("- SHAP TreeExplainer\n- WoE / IV\n- Scorecard bancaria")
    with col3:
        st.markdown("**Infraestructura**")
        st.markdown("- FastAPI + Pydantic\n- Streamlit\n- Docker\n- GitHub Actions")

    st.divider()
    st.markdown(
        "Repositorio: [github.com/JulioPradenas/credit-scoring-ml](https://github.com/JulioPradenas/credit-scoring-ml)"
    )
    st.markdown("Dataset: [Give Me Some Credit — Kaggle (2011)](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset)")
