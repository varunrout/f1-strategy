# F1 Strategy — Master Integration Roadmap

## Project Overview

End-to-end F1 race strategy analytics system built on FastF1 data with layered SQLite
storage, progressing from raw data ingestion through tyre-degradation modelling,
traffic/spatial analysis, and culminating in an integrated race-strategy
simulation and pit-stop optimisation engine.

---

## Domain 0: Data Enablement

**Goal**: Reliable, re-runnable ingestion of all FastF1 data into `raw.db`.

| Task | Deliverable | Status |
|------|-------------|--------|
| 0A | `positions_raw` table populated in `raw.db` | ✅ Done |
| 0B | `circuit_info` table with track geometry | ✅ Done |
| 0C | `requirements.txt` with scipy, scikit-learn, xgboost | ✅ Done |
| 0D | `sessions`, `laps_raw`, `telemetry_raw` tables | ✅ Done |
| 0E | `race_control_raw` table (SC/VSC flags) | ✅ Done |
| 0F | `weather_raw` table | ✅ Done |
| 0G | `src/ingest/ingest_session.py` CLI | ✅ Done |
| 0H | `src/ingest/ingest_season.py` batch CLI | ✅ Done |
| 0I | `src/utils/db.py`, `schemas.py`, `fastf1_utils.py` | ✅ Done |

---

## Domain 1: Tyre Degradation

**Goal**: Model per-compound tyre degradation rates and drive compound/stint
recommendations.

| Task | Deliverable | Status |
|------|-------------|--------|
| 1A | Clean-air stint extraction → `clean_air_laps.parquet`, `stints_degradation.parquet` | ✅ Done |
| 1B | Degradation curve fitting (linear + polynomial) → `domain1_degradation.py` | ✅ Done |
| 1C | ML feature engineering → `feature_builder.py`; context enrichment; driving-style signals | ✅ Done |
| 1D | K-means + hierarchical clustering of drivers/tracks → `03_domain1_cluster_analysis.ipynb` | ✅ Done |
| 1E-i | Visualisations (10 model figures, 2 clustering figures) | ✅ Done |
| 1E-ii | Analysis notebooks (EDA, degradation, clustering, modelling) | ✅ Done |
| 1E-iii | Reports (`domain1_degradation_report.md`, model results) | ✅ Done |
| 1F | XGBoost quantile regression model artefacts (Q10/Q50/Q90) | ✅ Done |
| 1G | ANOVA interaction analysis notebook | ✅ Done |

---

## Domain 2: Traffic & Spatial Analysis

**Goal**: Detect proximity events, classify traffic regimes, and quantify the
lap-time cost of dirty-air / being stuck in traffic.

| Task | Deliverable | Status |
|------|-------------|--------|
| 2A | Process `positions_raw` → `lap_positions` table | ✅ Done |
| 2B | Proximity event detection (gap < 1 s / 50 m) | ✅ Done |
| 2C | Traffic regime classification (clean / DRS / stuck) | ✅ Done |
| 2D | Traffic lap-time impact quantification | ✅ Done |
| 2E | Overtake detection & classification | ✅ Done |
| 2F | Spatial hotspot / bottleneck analysis | ✅ Done |
| 2G | Statistical tests on traffic cost | ✅ Done |
| 2H | `07_domain2_traffic_analysis.ipynb` | ✅ Done |
| 2I | `domain2_traffic_report.md` | ✅ Done |

---

## Domain 3: Race Strategy & Pit-Stop Optimisation  ← **current**

**Goal**: Synthesise tyre-degradation (D1) and traffic (D2) signals into a
race-strategy simulation layer that scores pit-stop windows, predicts the
optimal number of stops, and ranks strategies by expected total race time.

### 3A — Cross-Domain Feature Engineering (XT)

Build `features_xt.db` by joining Domain 1 and Domain 2 outputs.

| Sub-task | Deliverable |
|----------|-------------|
| 3A-i | Pit-window features: `laps_to_go`, `tyre_deg_per_lap`, `projected_deg_loss` | `features_xt.db::pit_window_features` |
| 3A-ii | Undercut opportunity score: gap delta, deg advantage over rival | `features_xt.db::undercut_scores` |
| 3A-iii | Safety-car delta: time saved in pit lane under SC vs green | `features_xt.db::sc_delta_features` |
| 3A-iv | Cross-domain strategy score combining 1–3 sub-scores | `features_xt.db::strategy_scores` |

**Script**: `src/features/domain3_strategy.py` + `src/features/build_xt_features.py`

### 3B — Strategy Simulation

| Sub-task | Deliverable |
|----------|-------------|
| 3B-i | Monte-Carlo total-race-time simulator (1-stop, 2-stop, 3-stop) | `src/models/strategy_simulator.py` |
| 3B-ii | Pit-cost model: lane time + position delta as a function of traffic context | embedded in simulator |
| 3B-iii | SC/VSC probability weighting (historical `race_control_raw` frequency) | embedded in simulator |

### 3C — Pit-Stop Timing Prediction Model

| Sub-task | Deliverable |
|----------|-------------|
| 3C-i | Feature matrix: tyre age, deg rate, lap delta, gap-to-pits, traffic score | `src/models/train_pit_strategy_model.py` |
| 3C-ii | XGBoost classifier — "is this lap a good pit lap?" | model artefacts in `data/models/` |
| 3C-iii | Calibrated probability output + SHAP feature importance | saved alongside model |

### 3D — Analysis & Reporting

| Sub-task | Deliverable |
|----------|-------------|
| 3D-i | Strategy EDA notebook | `notebooks/06_domain3_strategy_eda.ipynb` |
| 3D-ii | Pit-window optimisation notebook | `notebooks/07_domain3_pit_optimization.ipynb` |
| 3D-iii | Strategy report | `docs/reports/domain3_strategy_report.md` |

---

## Domain 4 (Future): Real-Time Strategy Dashboard *(out of scope for now)*

| Task | Deliverable |
|------|-------------|
| 4A | FastAPI backend serving strategy scores per lap | `src/api/` |
| 4B | Streamlit / Dash front-end | `app/` |
| 4C | Live race integration via FastF1 streaming | `src/ingest/live_ingest.py` |

---

## Technology Stack

| Layer | Library / Tool |
|-------|----------------|
| Data ingestion | `fastf1`, `pandas` |
| Storage | SQLite 3 (bronze `raw.db`, silver `features_core.db`, domain DBs) |
| Feature engineering | `pandas`, `numpy`, `scipy` |
| ML modelling | `xgboost`, `scikit-learn` |
| Explainability | `shap` |
| Simulation | `numpy` (Monte-Carlo) |
| Notebooks | `jupyter`, `matplotlib`, `seaborn` |
| CLI | `argparse` |

---

## Repository Layout

```
f1-strategy/
├── data/
│   ├── raw.db                  # bronze layer
│   ├── features_core.db        # silver — core lap/gap/segment features
│   ├── features_tyre.db        # domain 1 — tyre artefacts
│   ├── features_traffic.db     # domain 2 — traffic artefacts
│   └── features_xt.db          # domain 3 — cross-domain XT features
├── src/
│   ├── ingest/
│   │   ├── ingest_session.py
│   │   └── ingest_season.py
│   ├── features/
│   │   ├── build_laps_featured.py
│   │   ├── build_gaps_featured.py
│   │   ├── build_segments_featured.py
│   │   ├── domain1_degradation.py
│   │   ├── feature_builder.py
│   │   ├── enrich_stint_context.py
│   │   ├── driving_style_signals.py
│   │   ├── domain2_traffic.py
│   │   ├── build_traffic_features.py
│   │   ├── domain3_strategy.py       # ← Domain 3
│   │   └── build_xt_features.py      # ← Domain 3
│   ├── models/
│   │   ├── train_degradation_model.py
│   │   ├── strategy_simulator.py     # ← Domain 3
│   │   └── train_pit_strategy_model.py  # ← Domain 3
│   └── utils/
│       ├── db.py
│       ├── schemas.py
│       ├── fastf1_utils.py
│       └── logging_utils.py
├── notebooks/
│   ├── 01_domain1_eda.ipynb
│   ├── 02_domain1_degradation_analysis.ipynb
│   ├── 03_domain1_cluster_analysis.ipynb
│   ├── 04_domain1_modeling.ipynb
│   ├── 05_domain1_interaction_anova.ipynb
│   ├── 06_domain3_strategy_eda.ipynb       # ← Domain 3
│   └── 07_domain3_pit_optimization.ipynb   # ← Domain 3
├── docs/
│   ├── roadmaps/
│   │   └── ROADMAP_MASTER_INTEGRATION.md
│   ├── reports/
│   │   ├── domain1_degradation_report.md
│   │   ├── domain2_traffic_report.md
│   │   └── domain3_strategy_report.md      # ← Domain 3
│   └── figures/
│       ├── domain1_modeling/
│       ├── domain1_clustering/
│       ├── domain2_traffic/
│       └── domain3_strategy/               # ← Domain 3
├── requirements.txt
└── README.md
```
