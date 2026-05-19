"""
Score today's predicted fatigue curve and identify the optimal
decision window (global minimum of predicted DFI).
"""

import json
import pickle
from pathlib import Path

import pandas as pd

from dfi_models import NumpyRidge  # noqa: F401  needed for pickle resolution

BASE = Path(__file__).parent
DATA = BASE / "data" / "user_activity.csv"
MODEL = BASE / "models" / "dfi_model.pkl"
OUT = BASE / "data" / "todays_curve.csv"


def main():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    with open(MODEL, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    features = bundle["features"]

    df["date"] = df["timestamp"].dt.date
    today = df[df["date"] == df["date"].max()].copy()
    today["predicted_dfi"] = model.predict(today[features].to_numpy())

    best_idx = today["predicted_dfi"].idxmin()
    worst_idx = today["predicted_dfi"].idxmax()
    recommendation = {
        "best_decision_window": today.loc[best_idx, "timestamp"].isoformat(),
        "best_dfi": float(today.loc[best_idx, "predicted_dfi"]),
        "worst_decision_window": today.loc[worst_idx, "timestamp"].isoformat(),
        "worst_dfi": float(today.loc[worst_idx, "predicted_dfi"]),
        "average_dfi_today": float(today["predicted_dfi"].mean()),
    }
    today[["timestamp", "predicted_dfi", "fatigue"]].to_csv(OUT, index=False)
    print(json.dumps(recommendation, indent=2))
    print(f"Saved curve to {OUT}")


if __name__ == "__main__":
    main()
