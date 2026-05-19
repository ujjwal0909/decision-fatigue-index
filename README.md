# Decision Fatigue Index (DFI)

> A personal cognitive-load tracker that predicts the best hour of the day for you to make hard decisions.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Status: Prototype](https://img.shields.io/badge/status-prototype-orange.svg)](#)

---

## The problem

Knowledge workers make 100 to 300 micro-decisions every workday — which email to answer, which Slack to mute, whether to accept a meeting, what to wear, what to eat. By 3 PM, decision quality measurably collapses. We track our steps, our calories, our sleep — yet nobody tracks the cumulative cognitive cost of all those choices.

Existing tools fail us in different ways:

- Productivity apps measure *output* (tasks done) or *input* (hours worked).
- Wearables measure *physiology* (HRV, sleep) — a proxy, not a measurement.
- Calendar tools measure *meeting load* — just one decision channel.

Nobody puts calendar + email + chat + browser behavior + self-reported quality together into one personal model.

## The hypothesis

Decision fatigue follows a personal, predictable curve driven by decision *density*, *stakes*, *reversibility*, and *novelty*. If we can model that curve, we can recommend the optimal window for high-stakes decisions and tell you when to defer them.

---

## What this repo contains

A working prototype that:

1. **Generates 30 days of realistic activity data** for a fictional knowledge worker (calendar density, email metadata, app-switching, novelty, etc.).
2. **Engineers 15 features** capturing decision density, stakes, novelty, context-switching, circadian rhythm, and recovery signals.
3. **Trains a regression model** to predict fatigue from those features, with a numpy-only fallback so it runs even without scikit-learn.
4. **Scores today's predicted fatigue curve** and identifies the optimal decision window.
5. **Renders an interactive Streamlit dashboard** showing today's curve, the recommended best/worst windows, and historical hourly patterns.

The synthetic data is generated from a known causal model — so you can verify the predictor actually recovers the right relationships.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the synthetic dataset (30 days, 15-minute resolution)
python generate_synthetic_data.py

# 3. Train the model (prints MAE and R²)
python train_model.py

# 4. Score today and print the best/worst decision windows
python predict_today.py

# 5. Launch the interactive dashboard
streamlit run dashboard.py
```

The dashboard opens in your browser at `http://localhost:8501`.

Stop the dashboard with `Ctrl+C` in the terminal.

---

## Sample output

After training, you'll see something like:

```
Backend       : NumpyRidge
Holdout MAE   : 8.86   (target < 12)
Holdout R²    : 0.676
```

And `predict_today.py` will print a recommendation:

```json
{
  "best_decision_window": "2026-04-30T07:00:00",
  "best_dfi": 58.24,
  "worst_decision_window": "2026-04-30T18:45:00",
  "worst_dfi": 109.80,
  "average_dfi_today": 91.26
}
```

Read this as: *make your hardest decision around 7 AM, never around 6:45 PM.*

---

## Project structure

```
project1_decision_fatigue_index/
├── SPEC.md                       # Implementation-ready specification
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── generate_synthetic_data.py    # Builds 30 days of fake activity
├── dfi_models.py                 # Shared model classes (NumpyRidge fallback)
├── train_model.py                # Trains the fatigue predictor
├── predict_today.py              # Scores today's curve, prints recommendation
├── dashboard.py                  # Streamlit dashboard
├── data/                         # Generated CSVs (gitignored if you prefer)
│   ├── user_activity.csv
│   └── todays_curve.csv
└── models/
    ├── dfi_model.pkl
    └── metrics.json
```

---

## How it works

### 1. Feature engineering

For every 15-minute slot, we compute 15 features across 6 categories:

| Category          | Features                                                                  |
| ----------------- | ------------------------------------------------------------------------- |
| Decision density  | `decisions_count_15m`, `decisions_count_60m_rolling`, `decisions_today`  |
| Stakes            | `avg_stakes_score`, `max_stakes_score`, `irreversible_decisions_today`   |
| Novelty           | `novel_attendees_ratio`, `novel_topics_ratio`                            |
| Context switching | `app_switches_15m`, `tab_domain_entropy`                                 |
| Circadian         | `hour_of_day`, `minutes_since_wake`, `day_of_week`                       |
| Recovery          | `minutes_since_last_break`, `lunch_taken_today`                          |

### 2. Modeling

The trainer tries scikit-learn's `HistGradientBoostingRegressor` first, then falls back to a numpy-only ridge regression if sklearn isn't installed. Both achieve MAE under the 12-point target on this dataset.

### 3. Prediction

For each 15-minute slot today, the model predicts a DFI score from 0 to 100. The global minimum of that curve is the recommended decision window; the global maximum is when to defer.

### 4. Personalization roadmap

The current prototype uses a global model. Production version would use a per-user model warm-started from the global one, retrained nightly with the user's actual feedback (3x daily 5-second pings during the first 14 days).

---

## Tech stack

- **Python 3.10+**
- **pandas, numpy** — data wrangling
- **scikit-learn** (optional) — gradient-boosted regression
- **Streamlit + Altair** — dashboard

---

## What's in `SPEC.md`

A full implementation-ready specification including:

- Detailed problem framing and hypothesis
- Success metrics with concrete targets
- All data sources (passive + active) and integration methods
- Modeling roadmap (cold-start → personalized → predictive curve)
- Architecture diagram
- 8-sprint roadmap to MVP
- Risks and mitigations
- Out-of-scope decisions for Phase 1

Open [`SPEC.md`](./SPEC.md) for the full document.

---

## Limitations

- The current data is synthetic — real-world data will be noisier and require active learning to bootstrap the personal model.
- The classifier for stakes/novelty is mocked; production would need a small fine-tuned text classifier on titles + subjects.
- Privacy: the production design only ingests *metadata* (subject lines, attendees, frequencies) — never email/chat bodies.

---

## Roadmap

- [ ] Real calendar OAuth integration (Google + Microsoft Graph)
- [ ] Browser extension for tab-switching telemetry
- [ ] Self-report ping app (mobile + web)
- [ ] Personalized model with nightly retraining
- [ ] Mobile dashboard
- [ ] Phase 2: integrate physiological signals (Oura, Apple Health)

---

## License

MIT
