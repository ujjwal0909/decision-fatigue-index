# Project 1: Decision Fatigue Index (DFI)
### Implementation-Ready Specification

---

## 1. Problem Statement

Knowledge workers make 100-300 micro-decisions per workday (which email to answer, which Slack to mute, whether to accept a meeting, what to wear, what to eat). Decision quality measurably degrades through the day — but no existing tool *quantifies* the cumulative cognitive cost of those decisions, *personalized* to the individual.

**Hypothesis:** Decision fatigue follows a personal, predictable curve driven by decision *density*, *stakes*, *reversibility*, and *novelty*. If we can model that curve, we can recommend the optimal window for high-stakes decisions.

## 2. Why This Hasn't Been Solved

- Productivity apps measure *output* (tasks done) or *input* (hours worked).
- Wearables measure *physiology* (HRV, sleep) — a proxy, not a measurement.
- Calendar tools measure *meeting load* — only one decision channel.
- Nobody integrates calendar + email + chat metadata + browser behavior + self-reported decision quality into one personal model.

## 3. Success Metrics

| Metric | Target |
| --- | --- |
| Mean Absolute Error (MAE) on next-hour fatigue prediction (0-100 scale) | < 12 |
| Calibration: predicted "low fatigue window" matches user-reported peak focus | ≥ 70% precision |
| Adoption proxy: % of users who reschedule ≥1 decision per week based on DFI | ≥ 30% |
| Onboarding time (signup → first useful prediction) | < 72 hours |

## 4. Data Sources

### Passive (no user effort)
| Source | Signal | Method |
| --- | --- | --- |
| Google/Outlook Calendar | Meeting density, novelty (new attendees), stakes (title keywords) | OAuth + API |
| Gmail/Outlook metadata | Reply count, reply latency, threading depth | OAuth + metadata only (no body) |
| Slack/Teams (optional) | Message volume, response latency, channel-switching | OAuth + events API |
| Browser extension | Tab-switch frequency, domain-switch entropy | Custom Chrome/Firefox extension |
| OS focus events (macOS/Win) | App-switch frequency | Optional native helper |

### Active (light user effort)
- 3x daily 5-second ping: "Rate your current decision-making capacity 1-10."
- Used only for first 14 days to calibrate the personal model, then optional.

## 5. Feature Engineering

For each 15-minute time window:

```
features = {
  # Density
  'decisions_count_15m',
  'decisions_count_60m_rolling',
  'decisions_count_today_so_far',

  # Stakes (NLP classifier on titles/subjects)
  'avg_stakes_score',
  'max_stakes_score',

  # Reversibility
  'irreversible_decisions_today',  # contracts signed, send-once emails, etc.

  # Novelty
  'novel_attendees_ratio',
  'novel_topics_ratio',  # via topic embedding cosine vs. 30-day baseline

  # Context-switching
  'app_switches_15m',
  'tab_domain_entropy',

  # Circadian
  'hour_of_day',
  'minutes_since_wake',  # if sleep data available
  'day_of_week',

  # Recovery signals
  'minutes_since_last_break',
  'lunch_taken_today',
}
```

## 6. Modeling Approach

**Phase 1 — Cold-start (days 0-14):**
A *global* gradient-boosted regressor (LightGBM) trained on a public dataset of cognitive load + activity proxies (we use the simulated dataset for this prototype).

**Phase 2 — Personalized (day 14+):**
Per-user LightGBM with the global model as a warm-start prior. Updates nightly.

**Phase 3 — Predictive curve:**
For each upcoming hour today, predict DFI(t). Identify the global minimum (= best window for hard decisions).

## 7. Architecture

```
┌─────────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Collectors         │───▶│  Feature Store   │───▶│  Model Service  │
│  - Calendar API     │    │  (Postgres +     │    │  (LightGBM)     │
│  - Email metadata   │    │   TimescaleDB)   │    │                 │
│  - Browser ext      │    └──────────────────┘    └────────┬────────┘
│  - Self-report ping │                                     │
└─────────────────────┘                                     ▼
                                                   ┌─────────────────┐
                                                   │  API + Dashboard│
                                                   │  (FastAPI +     │
                                                   │   Streamlit)    │
                                                   └─────────────────┘
```

## 8. Sprint Roadmap (8 weeks to MVP)

| Sprint | Goal | Deliverable |
| --- | --- | --- |
| 1 | Synthetic data + baseline model | Working notebook with curve prediction |
| 2 | Calendar collector + stakes classifier | OAuth working, real data flowing |
| 3 | Browser extension + email metadata | Multi-source feature pipeline |
| 4 | Self-report ping + cold-start model | First personalized predictions |
| 5 | Streamlit dashboard | Daily curve view, "best decision window" recommendation |
| 6 | Personalization loop | Nightly retraining, drift detection |
| 7 | Friends-and-family beta | 20 users, feedback loop |
| 8 | Hardening + privacy review | Encryption at rest, data-deletion endpoint |

## 9. Risks & Mitigations

| Risk | Mitigation |
| --- | --- |
| Privacy concerns over email/calendar access | Metadata only, never content. On-device classifier where possible. |
| Cold-start poor predictions | Global model + actively-solicited self-reports for first 14 days. |
| Goodhart's Law — users gaming the score | Score is descriptive, not target. No leaderboards. |
| Inter-user variance in baseline | Per-user model + z-score normalization for cross-user comparison. |

## 10. Out of Scope (Phase 1)

- Mobile decisions (food choices, ride hailing, shopping) — Phase 2.
- Physiological signals (HRV) — Phase 2 integration with Oura/Apple Health.
- Couples/team DFI ("our combined cognitive bandwidth") — Phase 3.
