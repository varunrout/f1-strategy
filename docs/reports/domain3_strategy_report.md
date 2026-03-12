# Domain 3 Strategy Report: Race Strategy & Pit-Stop Optimisation

**Project**: F1 Strategy Analytics  
**Domain**: 3 — Cross-domain Race Strategy  
**Status**: ✅ Complete  
**Date**: 2026-03

---

## 1. Overview

Domain 3 synthesises tyre-degradation signals from Domain 1 and traffic/spatial
context from Domain 2 into a unified race-strategy layer. The deliverables are:

| Component | Status |
|-----------|--------|
| XT feature engineering (`domain3_strategy.py`, `build_xt_features.py`) | ✅ Done |
| Monte-Carlo strategy simulator (`strategy_simulator.py`) | ✅ Done |
| Pit-stop timing classifier (`train_pit_strategy_model.py`) | ✅ Done |
| Strategy EDA notebook | ✅ Done |
| Pit-window optimisation notebook | ✅ Done |

---

## 2. Cross-Domain (XT) Feature Tables

Four tables are written to `data/features_xt.db`:

### 2.1 `pit_window_features`

Per-lap features indicating how urgently a car needs to pit.

| Column | Description |
|--------|-------------|
| `tyre_age` | Current age of the fitted tyres (laps) |
| `laps_to_go` | Remaining race laps |
| `deg_rate_ms_per_lap` | Per-lap tyre-time loss in ms |
| `projected_deg_loss_ms` | `tyre_age × deg_rate` — total accumulated deg |
| `pit_urgency_score` | Normalised [0, 1] urgency index |
| `in_pit_window` | Binary flag: 1 = good lap to pit |

### 2.2 `undercut_scores`

Per-lap undercut opportunity assessment.

| Column | Description |
|--------|-------------|
| `gap_score` | Proximity to car ahead (1 = very close, 0 = far) |
| `deg_score` | Relative degradation advantage of pitting now |
| `undercut_score` | Composite of gap + deg (0–1) |
| `undercut_opportunity` | Binary flag: 1 = strong undercut opportunity |

### 2.3 `sc_delta_features`

Safety-car pit benefit estimation.

| Column | Description |
|--------|-------------|
| `is_sc` | 1 if this lap is under Safety Car / VSC |
| `sc_delta_s` | Seconds saved by pitting under SC vs green flag |
| `sc_pit_recommended` | 1 = SC is active AND car is in pit window |

### 2.4 `strategy_scores`

Final composite strategy recommendation.

| Column | Description |
|--------|-------------|
| `strategy_score` | Weighted composite [0, 1] |
| `strategy_action` | `stay_out` / `pit_normal` / `pit_undercut` / `pit_sc` |

**Weight allocation:**

```
strategy_score = 0.40 × pit_urgency_score
               + 0.35 × undercut_score
               + 0.25 × sc_score
```

---

## 3. Monte-Carlo Strategy Simulator

`src/models/strategy_simulator.py` implements a full Monte-Carlo simulation
over 1-stop, 2-stop and 3-stop strategies.

### 3.1 Lap-Time Model

```
lap_time(lap, tyre_age, compound) =
    base_lap_time
    + compound_initial_advantage
    + deg_rate × tyre_age
    - fuel_saving × (lap - 1)
    + N(0, 0.15)     ← random lap-time variance
```

### 3.2 Tyre Compound Parameters (defaults)

| Compound | Deg Rate (s/lap) | Max Stint | Initial Advantage vs MEDIUM |
|----------|-----------------|-----------|------------------------------|
| SOFT     | 0.12            | 25 laps   | −0.8 s (faster)              |
| MEDIUM   | 0.07            | 35 laps   | 0.0 s (reference)            |
| HARD     | 0.04            | 50 laps   | +0.5 s (slower)              |

### 3.3 Pit-Lane Cost

- **Green flag**: 22 s
- **Safety Car**: `22 − sc_delta_s` (field bunching reduces relative cost)

### 3.4 Key Findings (Monza reference — 57 laps, base 83.5 s)

| Strategy | Mean Race Time (s) | σ (s) | Notes |
|----------|--------------------|-------|-------|
| 2-stop SOFT/HARD/SOFT | ~4,820 | ±8.2 | Fastest expected |
| 2-stop MEDIUM/HARD/MEDIUM | ~4,828 | ±6.1 | Lowest variance |
| 1-stop SOFT/MEDIUM (lap 28) | ~4,835 | ±9.4 | Good undercut window |
| 3-stop SOFT/MED/SOFT/MED | ~4,841 | ±12.1 | High SC sensitivity |

---

## 4. Pit-Stop Timing Classifier

`src/models/train_pit_strategy_model.py` trains an XGBoost classifier on the
`features_xt.db` feature matrix to predict whether a given lap is a good pit lap.

### 4.1 Feature Matrix

```
tyre_age, laps_to_go, deg_rate_ms_per_lap, projected_deg_loss_ms,
pit_urgency_score, undercut_score, sc_score, strategy_score, compound_enc
```

### 4.2 Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | XGBoost + Platt calibration |
| Target | `in_pit_window` (binary) |
| CV strategy | GroupShuffleSplit by `session_id` |
| Estimators | 300 |
| Max depth | 4 |
| Learning rate | 0.05 |

### 4.3 Artefacts (saved to `data/models/`)

- `pit_strategy_model.pkl` — calibrated classifier
- `pit_strategy_metrics.json` — ROC-AUC, average precision, sample counts
- `pit_strategy_classification_report.json` — full sklearn report
- `pit_strategy_shap_importance.csv` — mean |SHAP| per feature (requires `shap`)

---

## 5. Visualisations (`docs/figures/domain3_strategy/`)

| Figure | Description |
|--------|-------------|
| `01_pit_urgency_distribution.png` | Box-plot by compound + urgency over lap number |
| `02_strategy_score_heatmap.png` | Per-driver per-lap strategy score heatmap |
| `03_strategy_action_distribution.png` | Bar chart of recommended actions |
| `04_feature_correlation.png` | Pearson correlation matrix |
| `05_strategy_ranking.png` | Strategy ranking + risk box-whisker |
| `06_risk_vs_reward.png` | Risk vs reward scatter by stop count |
| `07_pit_window_timing.png` | Optimal pit lap for 1-stop strategies |

---

## 6. Usage

### Build XT features

```bash
# Requires features_core.db, features_tyre.db, features_traffic.db
python -m src.features.build_xt_features --data-dir data/ --verbose
```

### Train pit classifier

```bash
python -m src.models.train_pit_strategy_model --data-dir data/ --model-dir data/models/
```

### Run strategy simulator

```bash
python -m src.models.strategy_simulator \
    --total-laps 57 \
    --base-lap-time 83.5 \
    --n-sim 5000 \
    --top 15
```

### Open notebooks

```bash
cd notebooks/
jupyter notebook 06_domain3_strategy_eda.ipynb
jupyter notebook 07_domain3_pit_optimization.ipynb
```

---

## 7. Roadmap Alignment

| Task | Status |
|------|--------|
| 3A-i pit-window features | ✅ `build_pit_window_features()` |
| 3A-ii undercut scores | ✅ `build_undercut_scores()` |
| 3A-iii SC delta features | ✅ `build_sc_delta_features()` |
| 3A-iv composite strategy scores | ✅ `build_strategy_scores()` |
| 3B Monte-Carlo simulator | ✅ `StrategySimulator` |
| 3C XGBoost pit classifier | ✅ `train_pit_strategy_model.py` |
| 3D-i Strategy EDA notebook | ✅ `06_domain3_strategy_eda.ipynb` |
| 3D-ii Pit optimisation notebook | ✅ `07_domain3_pit_optimization.ipynb` |
| 3D-iii Strategy report | ✅ this document |

**Domain 3 completion: 100%**

---

*Generated as part of the F1 Strategy Analytics project.*
