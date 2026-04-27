import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Weight of Evidence encoder compatible con sklearn Pipeline.

    Bina variables continuas y reemplaza los valores por WoE calculado
    sobre el set de entrenamiento. Preserva el orden de columnas.
    """

    def __init__(self, n_bins: int = 10, min_bin_size: float = 0.05):
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size  # fracción mínima del total por bin

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).reset_index(drop=True)
        X = X.reset_index(drop=True)

        self.woe_maps_: dict = {}
        self.iv_: dict = {}
        self.bin_edges_: dict = {}
        self.feature_names_in_ = list(X.columns)

        total_events = y.sum()
        total_non_events = len(y) - total_events

        for col in X.columns:
            series = X[col].fillna(X[col].median())
            bins, edges = self._make_bins(series, y)

            df_bin = pd.DataFrame({"bin": bins, "target": y.values})
            grouped = df_bin.groupby("bin", observed=True)["target"]
            events = grouped.sum()
            non_events = grouped.count() - events

            dist_e = (events / total_events).replace(0, 1e-6)
            dist_ne = (non_events / total_non_events).replace(0, 1e-6)

            woe = np.log(dist_e / dist_ne)
            iv = ((dist_e - dist_ne) * woe).sum()

            self.woe_maps_[col] = woe.to_dict()
            self.iv_[col] = iv
            self.bin_edges_[col] = edges

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy().reset_index(drop=True)
        result = pd.DataFrame(index=X.index)

        for col in self.feature_names_in_:
            series = X[col].fillna(X[col].median())
            edges = self.bin_edges_[col]
            bins = pd.cut(series, bins=edges, include_lowest=True)
            result[col] = bins.map(self.woe_maps_[col]).fillna(0.0)

        return result

    def _make_bins(self, series: pd.Series, y: pd.Series):
        try:
            bins = pd.qcut(series, q=self.n_bins, duplicates="drop")
        except ValueError:
            bins = pd.cut(series, bins=self.n_bins)

        # Fusionar bins con pocos eventos para estabilidad
        min_count = int(self.min_bin_size * len(series))
        counts = bins.value_counts()
        small_bins = counts[counts < min_count].index
        if len(small_bins) > 0 and len(counts) > 2:
            try:
                bins = pd.qcut(series, q=max(2, self.n_bins - len(small_bins)), duplicates="drop")
            except ValueError:
                pass

        edges = bins.cat.categories
        # Convertir a lista de floats para pd.cut en transform
        cut_edges = [-np.inf] + [iv.right for iv in edges[:-1]] + [np.inf]
        return bins, cut_edges

    def get_iv_table(self) -> pd.DataFrame:
        return (
            pd.Series(self.iv_, name="IV")
            .sort_values(ascending=False)
            .to_frame()
            .assign(strength=lambda x: x["IV"].map(self._iv_label))
        )

    @staticmethod
    def _iv_label(iv: float) -> str:
        if iv < 0.02:
            return "Inútil"
        if iv < 0.1:
            return "Débil"
        if iv < 0.3:
            return "Medio"
        if iv < 0.5:
            return "Fuerte"
        return "Sospechoso"
