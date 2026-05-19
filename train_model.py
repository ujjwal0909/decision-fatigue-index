"""
Train the Decision Fatigue Index predictor.

Prefers sklearn HistGradientBoostingRegressor when available.
Falls back to a numpy-only ridge regression so the prototype is runnable
without external ML dependencies.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from dfi_models import NumpyRidge

BASE = Path(__file__).parent
DATA_CSV = BASE / "data" / "user_activity.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "decisions_count_15m",
    "decisions_count_60m_rolling",
    "decisions_count_today_so_far",
    "avg_stakes_score",
    "max_stakes_score",
    "irreversible_decisions_today",
    "novel_attendees_ratio",
    "novel_topics_ratio",
    "app_switches_15m",
    "tab_domain_entropy",
    "hour_of_day",
    "minutes_since_wake",
    "day_of_week",
    "minutes_since_last_break",
    "lunch_taken_today",
]


def _train_sklearn(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=6, random_state=7
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, "HistGradientBoostingRegressor", float(mean_absolute_error(y_test, preds)), float(r2_score(y_test, preds))


def _train_numpy(X_train, y_train, X_test, y_test):
    model = NumpyRidge(alpha=2.0).fit(X_train.to_numpy(), y_train.to_numpy())
    preds = model.predict(X_test.to_numpy())
    mae = float(np.mean(np.abs(preds - y_test.to_numpy())))
    ss_res = float(np.sum((y_test.to_numpy() - preds) ** 2))
    ss_tot = float(np.sum((y_test.to_numpy() - y_test.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    return model, "NumpyRidge", mae, r2


def main():
    df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    X = df[FEATURES]
    y = df["fatigue"]

    rng = np.random.default_rng(7)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(len(idx) * 0.8)
    train_idx, test_idx = idx[:cut], idx[cut:]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    try:
        model, name, mae, r2 = _train_sklearn(X_train, y_train, X_test, y_test)
    except ImportError:
        model, name, mae, r2 = _train_numpy(X_train, y_train, X_test, y_test)

    print(f"Backend       : {name}")
    print(f"Holdout MAE   : {mae:.2f}   (target < 12)")
    print(f"Holdout R^2   : {r2:.3f}")

    with open(MODEL_DIR / "dfi_model.pkl", "wb") as f:
        pickle.dump({"model": model, "backend": name, "features": FEATURES}, f)
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps({"backend": name, "mae": mae, "r2": r2,
                    "n_train": int(len(X_train)), "n_test": int(len(X_test))}, indent=2)
    )
    print(f"Saved model + metrics to {MODEL_DIR}")


if __name__ == "__main__":
    main()
