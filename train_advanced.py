"""Script de entrenamiento de modelos avanzados — ejecutar directamente si el notebook falla."""
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PROC = Path("data/processed")
MODELS    = Path("models/saved")

train = pd.read_csv(DATA_PROC / "train_raw.csv")
val   = pd.read_csv(DATA_PROC / "val_raw.csv")
test  = pd.read_csv(DATA_PROC / "test_raw.csv")

X_train, y_train = train.drop("target", axis=1), train["target"]
X_val,   y_val   = val.drop("target", axis=1),   val["target"]
X_test,  y_test  = test.drop("target", axis=1),  test["target"]

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()


def ks_stat(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def metrics(name, y_prob):
    auc = roc_auc_score(y_val, y_prob)
    return {"model": name, "AUC-ROC": auc, "KS": ks_stat(y_val, y_prob),
            "Gini": 2 * auc - 1, "Brier": brier_score_loss(y_val, y_prob)}


# ── XGBoost + Optuna ──────────────────────────────────────────────────────────
print("XGBoost + Optuna (100 trials)...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)

def xgb_objective(trial):
    params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "scale_pos_weight": scale_pos_weight, "verbosity": 0, "seed": 42,
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
    }
    n = trial.suggest_int("n_estimators", 100, 800)
    m = xgb.train(params, dtrain, num_boost_round=n,
                  evals=[(dval, "val")], early_stopping_rounds=30, verbose_eval=False)
    return roc_auc_score(y_val, m.predict(dval))

study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42),
                             pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
study.optimize(xgb_objective, n_trials=100, show_progress_bar=True)

best_p = {k: v for k, v in study.best_params.items() if k != "n_estimators"}
best_p.update({"objective": "binary:logistic", "eval_metric": "auc",
               "scale_pos_weight": scale_pos_weight, "verbosity": 0, "seed": 42})
xgb_booster = xgb.train(best_p, dtrain,
                          num_boost_round=study.best_params.get("n_estimators", 300),
                          evals=[(dval, "val")], early_stopping_rounds=30, verbose_eval=False)

# Wrapper sklearn-compatible sin use_label_encoder
from xgboost import XGBClassifier
xgb_clf = XGBClassifier(**{k: v for k, v in study.best_params.items() if k != "n_estimators"},
                         n_estimators=study.best_params.get("n_estimators", 300),
                         scale_pos_weight=scale_pos_weight,
                         eval_metric="auc", random_state=42, verbosity=0)
xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)],
            verbose=False)

y_prob_xgb = xgb_clf.predict_proba(X_val)[:, 1]
xgb_m = metrics("XGBoost", y_prob_xgb)
print(f"  XGBoost AUC: {xgb_m['AUC-ROC']:.4f}")
joblib.dump(xgb_clf, MODELS / "champion_xgb.joblib")

# ── LightGBM ──────────────────────────────────────────────────────────────────
print("LightGBM...")
lgb_train_ds = lgb.Dataset(X_train, label=y_train)
lgb_val_ds   = lgb.Dataset(X_val,   label=y_val, reference=lgb_train_ds)
lgb_params = {"objective": "binary", "metric": "auc", "is_unbalance": True,
              "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
              "reg_alpha": 0.1, "reg_lambda": 0.1, "verbose": -1, "seed": 42}
lgb_model = lgb.train(lgb_params, lgb_train_ds, num_boost_round=1000,
                       valid_sets=[lgb_val_ds],
                       callbacks=[lgb.early_stopping(50, verbose=False),
                                  lgb.log_evaluation(-1)])
y_prob_lgb = lgb_model.predict(X_val)
lgb_m = metrics("LightGBM", y_prob_lgb)
print(f"  LightGBM AUC: {lgb_m['AUC-ROC']:.4f}  (iters: {lgb_model.best_iteration})")
joblib.dump(lgb_model, MODELS / "champion_lgb.joblib")

# ── CatBoost ─────────────────────────────────────────────────────────────────
print("CatBoost...")
cat_model = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6,
                                l2_leaf_reg=3, auto_class_weights="Balanced",
                                eval_metric="AUC", early_stopping_rounds=50,
                                random_seed=42, verbose=False)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
y_prob_cat = cat_model.predict_proba(X_val)[:, 1]
cat_m = metrics("CatBoost", y_prob_cat)
print(f"  CatBoost AUC: {cat_m['AUC-ROC']:.4f}")
joblib.dump(cat_model, MODELS / "champion_catboost.joblib")

# ── Registrar en experiments.json ────────────────────────────────────────────
adv_results = {m["model"]: {k: round(v, 4) for k, v in m.items() if k != "model"}
               for m in [xgb_m, lgb_m, cat_m]}
champion = max([xgb_m, lgb_m, cat_m], key=lambda x: x["AUC-ROC"])["model"]

exp_path = MODELS / "experiments.json"
experiments = json.loads(exp_path.read_text())
experiments["advanced"] = {
    "champion": champion,
    "xgb_best_params": study.best_params,
    "xgb_best_auc": study.best_value,
    "results": {col: {m: adv_results[m][col] for m in adv_results}
                for col in ["AUC-ROC", "KS", "Gini", "Brier"]},
}
exp_path.write_text(json.dumps(experiments, indent=2))

print(f"\nCampeón: {champion}")
for m in [xgb_m, lgb_m, cat_m]:
    print(f"  {m['model']:12s} AUC={m['AUC-ROC']:.4f}  KS={m['KS']:.4f}  Gini={m['Gini']:.4f}  Brier={m['Brier']:.4f}")
print("\nModelos y experiments.json guardados.")
