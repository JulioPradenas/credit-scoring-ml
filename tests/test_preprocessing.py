import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.features.woe_encoder import WoEEncoder


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 500
    X = pd.DataFrame({
        "age":            rng.integers(18, 80, n).astype(float),
        "revolving_util": rng.uniform(0, 1, n),
        "debt_ratio":     rng.uniform(0, 1, n),
        "monthly_income": rng.uniform(1000, 10000, n),
    })
    y = pd.Series((rng.uniform(size=n) < 0.1).astype(int))
    return X, y


def test_woe_encoder_fit_transform_shape(sample_data):
    X, y = sample_data
    enc = WoEEncoder(n_bins=5)
    enc.fit(X, y)
    out = enc.transform(X)
    assert out.shape == X.shape


def test_woe_encoder_no_nulls_in_output(sample_data):
    X, y = sample_data
    enc = WoEEncoder(n_bins=5)
    enc.fit(X, y)
    out = enc.transform(X)
    assert out.isnull().sum().sum() == 0


def test_woe_encoder_iv_keys_match_features(sample_data):
    X, y = sample_data
    enc = WoEEncoder(n_bins=5)
    enc.fit(X, y)
    assert set(enc.iv_.keys()) == set(X.columns)


def test_woe_encoder_iv_non_negative(sample_data):
    X, y = sample_data
    enc = WoEEncoder(n_bins=5)
    enc.fit(X, y)
    for feat, iv in enc.iv_.items():
        assert iv >= 0, f"IV negativo para {feat}: {iv}"


def test_woe_encoder_fit_transform_consistent(sample_data):
    X, y = sample_data
    enc = WoEEncoder(n_bins=5)
    out_fit_transform = enc.fit_transform(X, y)
    enc2 = WoEEncoder(n_bins=5)
    enc2.fit(X, y)
    out_separate = enc2.transform(X)
    pd.testing.assert_frame_equal(out_fit_transform, out_separate)


def test_woe_encoder_sklearn_pipeline_compatible(sample_data):
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    X, y = sample_data
    pipe = Pipeline([
        ("woe", WoEEncoder(n_bins=5)),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    pipe.fit(X, y)
    probs = pipe.predict_proba(X)
    assert probs.shape == (len(X), 2)


def test_woe_encoder_get_iv_table(sample_data):
    X, y = sample_data
    enc = WoEEncoder(n_bins=5)
    enc.fit(X, y)
    table = enc.get_iv_table()
    assert "IV" in table.columns
    assert "strength" in table.columns
    assert len(table) == len(X.columns)


def test_woe_encoder_handles_unseen_range(sample_data):
    X, y = sample_data
    enc = WoEEncoder(n_bins=5)
    enc.fit(X, y)
    X_new = X.copy()
    X_new["age"] = 999  # fuera del rango de entrenamiento
    out = enc.transform(X_new)
    assert out.isnull().sum().sum() == 0
