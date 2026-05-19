"""
Decision Fatigue Index — synthetic data generator.

Produces 30 days of 15-minute-resolution activity for a single fictional user.
Each row contains the engineered features + a "true" decision fatigue score
generated from a known causal model so we can validate the predictor.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)


def true_fatigue(row: dict) -> float:
    """
    Ground-truth causal model the predictor must learn.
    Returns fatigue on a 0-100 scale.
    """
    # Circadian component — fatigue rises through the day, dips after lunch recovery
    hour = row["hour_of_day"]
    circadian = 8 + 4.5 * (hour - 7) + (-12 if row["lunch_taken_today"] and 13 <= hour <= 14 else 0)

    # Density component
    density = 1.6 * row["decisions_count_60m_rolling"]

    # Stakes component
    stakes = 6 * row["avg_stakes_score"] + 2.5 * row["irreversible_decisions_today"]

    # Novelty taxes the brain
    novelty = 9 * row["novel_attendees_ratio"] + 6 * row["novel_topics_ratio"]

    # Context switching
    switching = 0.4 * row["app_switches_15m"] + 12 * row["tab_domain_entropy"]

    # Recovery
    recovery = -0.25 * row["minutes_since_last_break"] * -1  # longer since break => more fatigue
    recovery = 0.25 * row["minutes_since_last_break"]

    score = circadian + density + stakes + novelty + switching + recovery
    score += RNG.normal(0, 4)  # measurement noise
    return float(np.clip(score, 0, 100))


def generate_day(date: pd.Timestamp) -> pd.DataFrame:
    """Generate one workday at 15-minute resolution."""
    slots = pd.date_range(date.replace(hour=7), date.replace(hour=20), freq="15min")
    rows = []
    decisions_today = 0
    irreversible_today = 0
    minutes_since_break = 0
    lunch_taken = False
    is_weekday = date.weekday() < 5

    for slot in slots:
        hour = slot.hour
        # Weekends are quieter
        base_intensity = 1.0 if is_weekday else 0.3

        decisions_15m = int(RNG.poisson(2.5 * base_intensity * (1.3 if 9 <= hour <= 12 else 1.0)))
        decisions_today += decisions_15m
        irreversible_15m = int(RNG.binomial(decisions_15m, 0.08))
        irreversible_today += irreversible_15m

        if 12 <= hour <= 13 and not lunch_taken and RNG.random() < 0.4:
            lunch_taken = True
            minutes_since_break = 0
        else:
            minutes_since_break += 15

        # Random restorative break
        if RNG.random() < 0.05:
            minutes_since_break = 0

        row = {
            "timestamp": slot,
            "decisions_count_15m": decisions_15m,
            "decisions_count_60m_rolling": 0,  # filled later
            "decisions_count_today_so_far": decisions_today,
            "avg_stakes_score": float(RNG.beta(2, 5)),  # 0-1
            "max_stakes_score": float(RNG.beta(2, 3)),
            "irreversible_decisions_today": irreversible_today,
            "novel_attendees_ratio": float(RNG.beta(1.5, 4)),
            "novel_topics_ratio": float(RNG.beta(1.5, 4)),
            "app_switches_15m": int(RNG.poisson(18 * base_intensity)),
            "tab_domain_entropy": float(RNG.uniform(0.2, 0.9)),
            "hour_of_day": hour + slot.minute / 60.0,
            "minutes_since_wake": (slot - slot.replace(hour=6, minute=30)).total_seconds() / 60.0,
            "day_of_week": slot.weekday(),
            "minutes_since_last_break": minutes_since_break,
            "lunch_taken_today": int(lunch_taken),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["decisions_count_60m_rolling"] = (
        df["decisions_count_15m"].rolling(4, min_periods=1).sum()
    )
    df["fatigue"] = df.apply(true_fatigue, axis=1)
    return df


def main():
    start = pd.Timestamp("2026-04-01 00:00:00")
    days = [generate_day(start + pd.Timedelta(days=i)) for i in range(30)]
    full = pd.concat(days, ignore_index=True)
    out = OUT_DIR / "user_activity.csv"
    full.to_csv(out, index=False)
    try:
        full.to_parquet(OUT_DIR / "user_activity.parquet", index=False)
    except Exception:
        pass  # pyarrow optional
    print(f"Wrote {len(full):,} rows to {out}")
    print(full.head())


if __name__ == "__main__":
    main()
