"""
Streamlit dashboard for the Decision Fatigue Index.

Run with: streamlit run dashboard.py
"""

import json
from pathlib import Path

import pickle
import pandas as pd
import streamlit as st
import altair as alt

from dfi_models import NumpyRidge  # noqa: F401  needed for pickle resolution

BASE = Path(__file__).parent
DATA = BASE / "data" / "user_activity.csv"
MODEL = BASE / "models" / "dfi_model.pkl"
METRICS = BASE / "models" / "metrics.json"

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


@st.cache_data
def load():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    with open(MODEL, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    df["predicted_dfi"] = model.predict(df[FEATURES].to_numpy())
    df["date"] = df["timestamp"].dt.date
    metrics = json.loads(METRICS.read_text())
    return df, metrics


def main():
    st.set_page_config(page_title="Decision Fatigue Index", layout="wide")
    st.title("Decision Fatigue Index")
    st.caption("Personal cognitive-load tracker — pick the right hour for hard decisions.")

    df, metrics = load()

    col1, col2, col3 = st.columns(3)
    col1.metric("Model MAE", f"{metrics['mae']:.2f}", "target < 12")
    col2.metric("Model R²", f"{metrics['r2']:.2f}")
    col3.metric("Days observed", df["date"].nunique())

    # Day picker
    dates = sorted(df["date"].unique())
    day = st.select_slider("Pick a day", options=dates, value=dates[-1])
    today = df[df["date"] == day].copy()

    best = today.loc[today["predicted_dfi"].idxmin()]
    worst = today.loc[today["predicted_dfi"].idxmax()]

    st.subheader("Today's recommendation")
    left, right = st.columns(2)
    left.success(
        f"**Best window for hard decisions:**  "
        f"{best['timestamp'].strftime('%H:%M')} (DFI {best['predicted_dfi']:.1f})"
    )
    right.warning(
        f"**Avoid hard decisions around:**  "
        f"{worst['timestamp'].strftime('%H:%M')} (DFI {worst['predicted_dfi']:.1f})"
    )

    # Curve
    curve = today[["timestamp", "predicted_dfi", "fatigue"]].rename(
        columns={"predicted_dfi": "Predicted", "fatigue": "Observed"}
    )
    curve_long = curve.melt("timestamp", var_name="series", value_name="dfi")
    chart = (
        alt.Chart(curve_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("timestamp:T", title="Time of day"),
            y=alt.Y("dfi:Q", title="Decision Fatigue (0-100)"),
            color="series:N",
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Last 30 days — average curve by hour")
    hourly = (
        df.assign(hour=df["timestamp"].dt.hour)
        .groupby("hour", as_index=False)["predicted_dfi"]
        .mean()
    )
    heat = (
        alt.Chart(hourly)
        .mark_bar()
        .encode(
            x=alt.X("hour:O", title="Hour of day"),
            y=alt.Y("predicted_dfi:Q", title="Mean predicted DFI"),
            color=alt.Color("predicted_dfi:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True)),
        )
        .properties(height=260)
    )
    st.altair_chart(heat, use_container_width=True)

    with st.expander("Raw features for selected day"):
        st.dataframe(today.drop(columns=["date"]))


if __name__ == "__main__":
    main()
