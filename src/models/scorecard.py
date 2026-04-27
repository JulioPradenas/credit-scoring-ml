import numpy as np
import pandas as pd
import joblib
from pathlib import Path


class Scorecard:
    """Scorecard bancaria construida sobre Logistic Regression + WoE.

    Convierte los coeficientes de la regresión y los WoE de cada bin
    en puntos enteros según la escala estándar de la industria:
        Score = A - B * ln(odds)
        Puntos_variable_bin = -(WoE * beta_i * B)

    Parámetros de escala:
        base_score (A): score asignado a odds de referencia (default 600)
        base_odds:      odds de referencia, e.g. 1:50 → 0.02 (default)
        pdo:            puntos para doblar los odds (default 20)
    """

    def __init__(self, base_score: int = 600, base_odds: float = 1 / 50, pdo: int = 20):
        self.base_score = base_score
        self.base_odds  = base_odds
        self.pdo        = pdo
        self.B          = pdo / np.log(2)
        self.A          = base_score + self.B * np.log(base_odds)

        self.scorecard_table_: pd.DataFrame | None = None
        self.features_: list[str] = []
        self.intercept_: float = 0.0

    def fit(self, logreg, woe_encoder) -> "Scorecard":
        """Construye la tabla de scorecard a partir del modelo y encoder ajustados."""
        self.features_  = woe_encoder.feature_names_in_
        self.intercept_ = logreg.intercept_[0]
        coef_map        = dict(zip(self.features_, logreg.coef_[0]))

        rows = []
        for feat in self.features_:
            woe_map   = woe_encoder.woe_maps_[feat]
            beta      = coef_map[feat]
            edges     = woe_encoder.bin_edges_[feat]

            for i, (interval, woe_val) in enumerate(woe_map.items()):
                low  = edges[i]     if i < len(edges) - 1 else -np.inf
                high = edges[i + 1] if i + 1 < len(edges) else np.inf
                points = -(woe_val * beta * self.B)
                rows.append({
                    'variable':   feat,
                    'bin_index':  i,
                    'bin_label':  str(interval),
                    'lower':      low,
                    'upper':      high,
                    'woe':        round(woe_val, 6),
                    'beta':       round(beta, 6),
                    'points':     round(points, 2),
                    'points_int': int(round(points)),
                })

        # Puntos del intercepto distribuidos como offset base
        intercept_points = -(self.intercept_ * self.B) + self.A
        self.intercept_points_ = round(intercept_points, 2)

        self.scorecard_table_ = pd.DataFrame(rows)
        return self

    def score(self, profile: dict) -> dict:
        """Calcula el score y decisión crediticia para un perfil individual.

        Args:
            profile: dict con los valores originales (antes de WoE) de cada feature.

        Returns:
            dict con score, probability_default, decision, risk_band, y breakdown por variable.
        """
        total = self.intercept_points_
        breakdown = []

        for feat in self.features_:
            value = profile.get(feat)
            if value is None:
                continue
            feat_rows = self.scorecard_table_[self.scorecard_table_['variable'] == feat]
            matched = feat_rows[
                (feat_rows['lower'] <= value) & (value < feat_rows['upper'])
            ]
            if matched.empty:
                # Bin más extremo si cae fuera del rango
                matched = feat_rows.iloc[[-1]] if value >= feat_rows['upper'].max() else feat_rows.iloc[[0]]

            pts = matched['points_int'].values[0]
            total += pts
            breakdown.append({'feature': feat, 'value': value,
                              'bin': matched['bin_label'].values[0], 'points': pts})

        score   = int(round(total))
        prob    = self._score_to_prob(score)
        decision, risk_band = self._decision(score)

        return {
            'score':               score,
            'probability_default': round(prob, 4),
            'decision':            decision,
            'risk_band':           risk_band,
            'breakdown':           sorted(breakdown, key=lambda x: abs(x['points']), reverse=True),
        }

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        results = [self.score(row.to_dict()) for _, row in df[self.features_].iterrows()]
        return pd.DataFrame([{k: v for k, v in r.items() if k != 'breakdown'}
                              for r in results])

    def _score_to_prob(self, score: int) -> float:
        log_odds = (self.A - score) / self.B
        odds     = np.exp(log_odds)
        return odds / (1 + odds)

    @staticmethod
    def _decision(score: int) -> tuple[str, str]:
        if score < 550:
            return 'REJECT',        'High'
        if score <= 650:
            return 'MANUAL_REVIEW', 'Medium'
        return 'APPROVE',           'Low'

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "Scorecard":
        return joblib.load(path)
