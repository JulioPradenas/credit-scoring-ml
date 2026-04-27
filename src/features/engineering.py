import numpy as np
import pandas as pd
from pathlib import Path


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features derivadas para credit scoring sobre el dataset base.

    Recibe el DataFrame con las columnas originales (renombradas) y retorna
    un nuevo DataFrame con las columnas originales + las features engineered.
    No modifica el DataFrame de entrada.
    """
    out = df.copy()

    # ── Interacciones: dos variables juntas predicen más que por separado ──────

    # Deuda absoluta mensual real: DebtRatio es un ratio adimensional;
    # multiplicado por el ingreso da el monto en USD que el cliente destina a deuda.
    # Clientes con ratio alto e ingreso alto tienen carga de deuda muy distinta
    # a los de ratio alto e ingreso bajo.
    out["debt_to_income"] = out["debt_ratio"] * out["monthly_income"]

    # Presión financiera por dependiente: un ingreso de $5,000 con 4 dependientes
    # es muy distinto a $5,000 sin dependientes. Divide entre (n+1) para evitar
    # división por cero cuando n_dependents=0.
    out["income_per_dependent"] = out["monthly_income"] / (out["n_dependents"] + 1)

    # ── Agregaciones de dominio ────────────────────────────────────────────────

    # Historial de atrasos total: los atrasos de cualquier gravedad son señal
    # de dificultades de pago. Agregarlos captura el patrón general sin perder
    # los casos donde hay pocos atrasos graves (late_90plus) pero muchos leves.
    out["total_late_payments"] = (
        out["late_30_59"] + out["late_60_89"] + out["late_90plus"]
    )

    # ── Flags binarios de riesgo ───────────────────────────────────────────────

    # Alta utilización de crédito rotativo: superar el 80% de la línea disponible
    # es un indicador clásico de estrés financiero inminente en credit scoring.
    out["high_utilization"] = (out["revolving_util"] > 0.8).astype(int)

    # Cualquier atraso grave (90+ días): señal fuerte de incumplimiento real,
    # no solo de dificultades temporales de pago.
    out["has_severe_late"] = (out["late_90plus"] > 0).astype(int)

    # ── Variables categóricas de segmentación ─────────────────────────────────

    # Bucket de utilización de crédito: permite al modelo capturar efectos
    # no lineales del uso de crédito sin depender de transformaciones WoE.
    out["util_bucket"] = pd.cut(
        out["revolving_util"],
        bins=[-np.inf, 0.3, 0.6, 0.8, np.inf],
        labels=["low", "medium", "high", "very_high"],
    ).astype(str)

    # Grupo etario: la edad tiene efectos distintos en distintos tramos de vida.
    # Jóvenes tienen historial corto; mayores suelen tener mayor estabilidad.
    out["age_group"] = pd.cut(
        out["age"],
        bins=[0, 30, 45, 60, 120],
        labels=["18-30", "31-45", "46-60", "60+"],
    ).astype(str)

    return out


NUMERIC_FEATURES = [
    "revolving_util", "age", "late_30_59", "debt_ratio", "monthly_income",
    "open_credit_lines", "late_90plus", "real_estate_loans", "late_60_89",
    "n_dependents", "debt_to_income", "total_late_payments",
    "income_per_dependent", "high_utilization", "has_severe_late",
]

CATEGORICAL_FEATURES = ["util_bucket", "age_group"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "train_raw.csv"
    df = pd.read_csv(data_path)
    print(f"Input shape:  {df.shape}")

    df_engineered = create_features(df)
    print(f"Output shape: {df_engineered.shape}")
    print(f"\nNuevas features: {[c for c in df_engineered.columns if c not in df.columns]}")
    print(f"\nPrimeras filas:")
    print(df_engineered[ALL_FEATURES].head(3).to_string())
