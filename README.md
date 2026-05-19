# Decision Fatigue Index — Prototype

## Quick start

```bash
pip install -r requirements.txt
python generate_synthetic_data.py   # creates data/user_activity.parquet
python train_model.py               # trains the model, prints MAE/R²
python predict_today.py             # prints today's best/worst decision windows
streamlit run dashboard.py          # interactive dashboard
```

## Files
- `SPEC.md` — full implementation spec (problem, metrics, architecture, roadmap)
- `generate_synthetic_data.py` — 30 days of fake activity with a known causal model
- `train_model.py` — fits HistGradientBoosting on engineered features
- `predict_today.py` — scores today and outputs the optimal decision window
- `dashboard.py` — Streamlit visualization
