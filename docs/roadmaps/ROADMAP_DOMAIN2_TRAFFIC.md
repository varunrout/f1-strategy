# Domain 2: Traffic & Spatial Constraints — Development Roadmap

## Overview

This roadmap outlines the development milestones for the Domain 2 Traffic & Spatial Constraints pipeline, from initial prototype through production-ready deployment.

---

## Milestone 1: Core Pipeline (Complete ✅)

**Target**: Basic traffic regime classification and penalty quantification

### Deliverables
- [x] `build_lap_positions` — GPS-to-lap time join via `np.searchsorted`
- [x] `detect_proximity_events` — KDTree spatial search (20m radius, 0.5s resolution)
- [x] `classify_traffic_regimes` — 5-class labeling (CLEAN_AIR → CLOSE_FOLLOWING)
- [x] `quantify_traffic_penalties` — Baseline-adjusted lap time loss with tyre degradation correction
- [x] `detect_overtakes` — Position-change detection with pit/SC filtering
- [x] `detect_overtake_hotspots` — DBSCAN spatial clustering
- [x] `map_track_bottlenecks` — 25m grid density analysis
- [x] `run_domain2_pipeline` — End-to-end orchestrator
- [x] Unit test suite (35+ tests, >90% coverage)

---

## Milestone 2: Data Integration (In Progress 🔄)

**Target**: Connect pipeline to Bronze/Silver data lake

### Deliverables
- [ ] FastF1 ingestion adapter for position telemetry
- [ ] Bronze layer writer for raw position data
- [ ] Silver layer writer for featured lap data with traffic columns
- [ ] DuckDB query layer for cross-session analysis
- [ ] Incremental ingestion (skip already-processed sessions)

### Dependencies
- FastF1 >= 3.0.0 cache configuration
- PyArrow >= 14.0.0 for Parquet writes
- DuckDB >= 0.9.0 for analytical queries

---

## Milestone 3: Feature Enrichment (Planned 📋)

**Target**: Add gap-to-ahead telemetry and DRS window features

### Deliverables
- [ ] `compute_gap_to_ahead` — Real-time gap calculation from position data
- [ ] `detect_drs_window` — DRS zone detection from track map + gap data
- [ ] `compute_dirty_air_score` — Continuous dirty air exposure metric (0–1 scale)
- [ ] `classify_following_distance` — Sub-1s, 1–2s, 2–5s, 5s+ buckets
- [ ] Integration with Domain 1 tyre degradation features

### Research Questions
- Does dirty air score correlate with tyre temperature outlier laps?
- Is there a non-linear threshold effect at 1.0s gap for dirty air impact?

---

## Milestone 4: Model Integration (Planned 📋)

**Target**: Traffic features as inputs to race strategy prediction model

### Deliverables
- [ ] Traffic regime one-hot encoding for ML pipeline
- [ ] Lap-time penalty prediction model (XGBoost baseline)
- [ ] Overtake probability model per track zone
- [ ] Monte Carlo integration: traffic state transitions × tyre life

### Model Architecture
```
Input: [lap_number, tyre_age, compound, traffic_regime, proximity_pct, gap_to_ahead_s]
  → Feature engineering layer
  → XGBoost lap time predictor
  → Output: predicted_lap_time_s, traffic_penalty_s, overtake_prob
```

---

## Milestone 5: Visualization & Reporting (Planned 📋)

**Target**: Interactive dashboards and automated reports

### Deliverables
- [ ] Track map overlay: proximity density heatmap
- [ ] Lap-by-lap traffic regime timeline chart
- [ ] Overtake hotspot cluster visualization (scatter + convex hulls)
- [ ] Per-driver traffic penalty summary cards
- [ ] Automated PDF report generation for each race

### Notebook Coverage
- [x] `05_domain2_traffic_eda.ipynb` — Regime distribution and penalty analysis
- [x] `06_domain2_overtake_analysis.ipynb` — Overtake detection and classification
- [x] `07_domain2_spatial_analysis.ipynb` — Hotspot and bottleneck mapping
- [x] `08_domain2_visualizations.ipynb` — Full visualization suite

---

## Milestone 6: Production Hardening (Future 🔮)

**Target**: Production-ready pipeline with monitoring and alerting

### Deliverables
- [ ] Pipeline orchestration (Apache Airflow or Prefect DAG)
- [ ] Data quality checks (Great Expectations)
- [ ] Performance profiling (target: < 60s per race session)
- [ ] Error recovery and partial-session handling
- [ ] CI/CD integration with automated test runs on new FastF1 releases

---

## Technical Debt

| Item | Priority | Effort |
|---|---|---|
| Replace `np.searchsorted` with interval tree for large datasets | Medium | 2d |
| Vectorise `_derive_lap_times` for >5000 laps/session | Low | 1d |
| Add 3D KDTree support (x, y, z) for altitude variation circuits | Low | 0.5d |
| Cache binned position data to avoid recomputation | Medium | 1d |

---

## Success Metrics

| Metric | Current | Target |
|---|---|---|
| Test coverage | >90% | >95% |
| Pipeline runtime (1 race) | ~45s | <30s |
| Traffic penalty RMSE | TBD | <0.15s |
| Overtake detection precision | TBD | >85% |
