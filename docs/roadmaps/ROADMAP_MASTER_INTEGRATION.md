# F1 Strategy Analytics Platform – Master Integration Roadmap

## Overview

The **F1 Strategy Analytics Platform** is a modular, data-driven system for Formula 1 race strategy optimisation. It ingests raw telemetry and timing data via FastF1, processes it through a multi-layer data lake, and produces domain-specific analytics and predictive models.

**Three primary analytics domains:**

| Domain | Focus | Capability |
|--------|-------|------------|
| **D1** | Tyre Degradation | Predict lap-time degradation per compound/track/driver |
| **D2** | Traffic & Spatial | Model position battles, undercut/overcut opportunities |
| **D3** | Expected Threat (xT) | Probabilistic race-outcome value of strategy decisions |

---

## Data Lake Architecture

```
data/
├── cache/
│   └── fastf1/              # FastF1 HTTP cache (not committed)
└── lake/
    ├── bronze/              # Raw ingested data (Parquet, partitioned by year/round)
    │   ├── laps_raw/
    │   │   └── year=YYYY/round=RR/{R|Q}.parquet
    │   ├── positions_raw/
    │   │   └── year=YYYY/round=RR/R.parquet
    │   └── race_control_raw/
    │       └── year=YYYY/round=RR/R.parquet
    ├── silver/              # Cleaned & feature-engineered data
    │   └── domain1/
    │       ├── clean_air_laps.parquet
    │       ├── stints_degradation.parquet
    │       ├── stints_enriched.parquet
    │       ├── ml_features.parquet
    │       ├── driving_style_signals.parquet
    │       └── weather_summary.parquet
    └── gold/                # Aggregated, model-ready outputs (future)
        └── strategy_signals/
```

---

## Phase 0: Data Enablement (Weeks 1–2)

### Goals
- Establish repeatable data ingestion pipeline
- Validate FastF1 API coverage for 2022–2024 seasons
- Set up project scaffolding (CI, linting, paths, tests)

### Tasks

| ID | Task | Output |
|----|------|--------|
| 0A | Create project structure, `requirements.txt`, `.gitignore` | Repo scaffold |
| 0B | Implement `ingest_fastf1.py` for lap data ingestion | Bronze laps Parquet |
| 0C | Implement `ingest_positions.py` for position telemetry | Bronze positions Parquet |
| 0D | Implement `ingest_race_control.py` for RC messages | Bronze race-control Parquet |
| 0E | Validate ingestion coverage across 2022–2024 rounds | Coverage report |
| 0F | Set up `src/utils/paths.py` for consistent path management | `paths.py` |

### Success Metrics
- ≥ 95% of rounds 2022–2024 successfully ingested
- Parquet files readable in < 2s for a full season
- All tests pass in CI

---

## Phase 1: Domain 1 – Tyre Degradation (Weeks 3–9)

### Goal
Build a production-quality tyre degradation model that predicts per-lap pace loss as a function of compound, tyre age, track, and driver style.

### Task 1A – Clean-Air Stint Extraction (Week 3)

**Objective**: Isolate undisturbed stints for degradation analysis.

| Sub-task | Description |
|----------|-------------|
| 1A-1 | Implement `extract_clean_air_stints()` in `domain1_degradation.py` |
| 1A-2 | Define TrackStatus filtering (SC, VSC, Red Flag exclusion) |
| 1A-3 | Assign `stint_id`, `stint_lap`, `tyre_compound` columns |
| 1A-4 | Unit tests: `test_extract_clean_air_stints_*` |

**Output**: `data/lake/silver/domain1/clean_air_laps.parquet`

---

### Task 1B – Degradation Curve Fitting (Week 4)

**Objective**: Fit per-stint degradation models.

| Sub-task | Description |
|----------|-------------|
| 1B-1 | Implement `fit_degradation_curves()` – linear + polynomial + exponential |
| 1B-2 | Compute R², peak lap, degradation slope per stint |
| 1B-3 | Unit tests: `test_fit_degradation_curves_*` |
| 1B-4 | EDA notebook `02_domain1_degradation_analysis.ipynb` |

**Output**: `data/lake/silver/domain1/stints_degradation.parquet`

---

### Task 1C – Stint Context Enrichment (Week 5)

**Objective**: Add contextual features to explain degradation variance.

| Sub-task | Description |
|----------|-------------|
| 1C-1 | `enrich_with_safety_car_proximity()` – pre/post SC tyre age |
| 1C-2 | `enrich_with_track_position()` – stint-start position, gap to leader |
| 1C-3 | `enrich_with_weather_context()` – air/track temp, humidity |
| 1C-4 | `enrich_with_fuel_load()` – estimated fuel kg, fuel-corrected pace |

**Output**: `data/lake/silver/domain1/stints_enriched.parquet`

---

### Task 1D – Driving Style Signals (Week 6)

**Objective**: Extract quantitative driving-style features and classify drivers.

| Sub-task | Description |
|----------|-------------|
| 1D-1 | `compute_consistency_score()` – CV of clean-air lap times |
| 1D-2 | `compute_degradation_management_score()` – vs field average |
| 1D-3 | `compute_tyre_exploitation_score()` – early-stint aggressiveness |
| 1D-4 | `classify_driving_style()` – KMeans k=3 (Preserver/Balanced/Aggressor) |
| 1D-5 | Cluster notebook `03_domain1_cluster_analysis.ipynb` |

**Output**: `data/lake/silver/domain1/driving_style_signals.parquet`

---

### Task 1E – ML Feature Build & Model Training (Weeks 7–9)

**Objective**: Train XGBoost quantile regression model for tyre degradation.

| Sub-task | Description |
|----------|-------------|
| 1E-1 | `feature_builder.py` – compound encoding, interaction features, rolling deg rate |
| 1E-2 | `train_degradation_model.py` – MSE + Q10/Q50/Q90 models |
| 1E-3 | Time-based train/test split (last 4 rounds = test) |
| 1E-4 | ANOVA interaction analysis notebook `05_domain1_interaction_anova.ipynb` |
| 1E-5 | Modelling notebook `04_domain1_modeling.ipynb` |
| 1E-6 | Domain 1 degradation report |

**Output**: `data/models/domain1/degradation_{main|q10|q50|q90}.json`

**Success Metrics (Domain 1)**:
- Linear fit R² ≥ 0.6 for SOFT compound
- XGBoost MAE ≤ 0.05 s/lap on holdout set
- Q90 coverage ≥ 88% of test stints
- ANOVA interaction p-value computed and documented

---

## Phase 2: Domain 2 – Traffic & Spatial (Weeks 10–23)

### Goal
Model the spatial dynamics of on-track traffic, undercut/overcut windows, and DRS train effects.

### Task 2A – Position Telemetry Enrichment (Weeks 10–11)

Enrich position data with:
- Inter-car gap (time delta from timing data)
- Sector-level speed traces
- DRS detection zones (from circuit reference data)

**Output**: `data/lake/silver/domain2/position_enriched.parquet`

---

### Task 2B – Traffic Density Model (Weeks 12–13)

For each lap, compute:
- `cars_within_1s`, `cars_within_3s`, `cars_within_5s` (traffic density)
- `in_drs_train` flag
- `clean_air_probability` (estimated probability of running in clear air)

---

### Task 2C – Undercut / Overcut Window Detection (Weeks 14–15)

Define undercut/overcut opportunity windows:
- `undercut_window_open`: gap to car ahead < 2.5s AND track allows passing
- `overcut_window_score`: estimated net time gain from late pitting
- Track-specific undercut delta estimates

---

### Task 2D – Pit Stop Timing Model (Weeks 16–17)

Predict optimal pit lap given:
- Current tyre age and degradation trajectory
- Undercut/overcut window status
- Safety car probability (from race control history)
- Traffic density ahead

---

### Task 2E – Spatial Clustering of Race Incidents (Week 18)

DBSCAN clustering of overtaking zones:
- Identify high-incident sectors per circuit
- Compute `sector_risk_score`

---

### Task 2F – Track Characterisation Matrix (Weeks 19–20)

Combine:
- Degradation regime (Domain 1)
- Overtaking difficulty (undercut/overcut potential)
- SC probability
- DRS effectiveness score

Into a unified `track_characterisation.parquet`.

---

### Task 2G – ML Model: Undercut Success Probability (Weeks 21–22)

XGBoost classifier:
- **Target**: was the undercut successful? (net gain > 0s at rejoin)
- **Features**: gap pre-pit, deg rate, tyre life, track characterisation, weather

---

### Task 2H – Domain 2 EDA & Reporting (Week 23)

- Notebook: spatial traffic visualisation
- Notebook: undercut success rate analysis
- Domain 2 report

**Success Metrics (Domain 2)**:
- Undercut success classifier AUC ≥ 0.72
- SC probability model calibration error < 5%

---

## Phase 3: Domain 3 – Expected Threat (xT) (Weeks 24–36)

### Goal
Implement an F1-adapted Expected Threat framework to quantify the race-outcome value of strategic decisions.

### Concept

Borrowed from football analytics, xT measures the probability that a given state (position, lap, tyre, gap) leads to gaining a position or winning the race. In F1:

- **State**: (lap, position, tyre compound, tyre age, gap to car ahead, gap to car behind)
- **Action**: pit now, push, back off, defend
- **Value**: probability of finishing in top-N from this state

---

### Task 3A – Markov State Space Definition (Weeks 24–25)

- Discretise (lap, position, tyre_age, gap_ahead, gap_behind) into bins
- Compute observed transition probabilities from historical data

---

### Task 3B – xT Calculation (Weeks 26–28)

- Value iteration to propagate xT from end-of-race states backwards
- Validate xT values against known strategy outcomes

---

### Task 3C – Real-Time xT Updates (Weeks 29–31)

- Stream lap data from live timing during race
- Update xT in rolling window as new laps arrive

---

### Task 3D – Strategy Recommendation Engine (Weeks 32–34)

- Given current xT, recommend: pit now / push / hold
- Compare xT across all available compounds for pit decision

---

### Task 3E – xT Validation & Reporting (Weeks 35–36)

- Back-test on 2022–2024 strategy decisions
- Domain 3 report

---

## Phase 4: Cross-Domain Integration (Weeks 36–39)

### Task 4A – Unified Strategy Signal API

Combine D1 + D2 + D3 outputs into a single `strategy_signal_t` structure:

```python
@dataclass
class StrategySignal:
    lap: int
    driver: str
    tyre_compound: str
    tyre_age: int
    deg_rate_q50: float          # Domain 1
    deg_rate_q10: float
    deg_rate_q90: float
    undercut_window: bool        # Domain 2
    undercut_success_prob: float
    sc_probability: float
    xT_current: float            # Domain 3
    xT_pit_now: float
    xT_push: float
    recommended_action: str
```

---

### Task 4B – Strategy Simulation Engine

Monte Carlo simulation using D1 uncertainty (Q10/Q90 fan), D2 traffic, D3 xT to produce:
- `P(gain_position_this_stint)` distribution
- Optimal pit lap with confidence interval

---

### Task 4C – Dashboard / Visualisation

Interactive Plotly dashboard:
- Live degradation curves per driver
- Quantile uncertainty bands
- Undercut window alerts
- xT heatmap by lap/tyre state

---

## Success Metrics Summary

| Phase | Metric | Target |
|-------|--------|--------|
| D1 | XGBoost MAE (deg rate) | ≤ 0.05 s/lap |
| D1 | Q90 coverage | ≥ 88% |
| D1 | Linear R² (SOFT) | ≥ 0.6 |
| D2 | Undercut classifier AUC | ≥ 0.72 |
| D2 | SC probability calibration | < 5% error |
| D3 | xT back-test accuracy | > 70% correct recommendation |
| Integration | Simulation latency | < 500ms per lap |

---

## Repository Structure

```
f1-strategy/
├── src/
│   ├── ingest/             # Bronze layer ingestion
│   ├── features/           # Silver layer feature engineering
│   ├── models/             # ML model training & inference
│   └── utils/              # Shared utilities (paths, logging)
├── notebooks/              # Jupyter EDA & analysis notebooks
├── tests/                  # Pytest unit tests
├── docs/
│   ├── roadmaps/           # This document
│   └── reports/            # Domain analysis reports
├── data/                   # Data lake (not committed to git)
├── requirements.txt
└── .gitignore
```

---

*Last updated: 2024 – Domain 1 implementation complete.*
