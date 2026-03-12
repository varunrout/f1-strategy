# F1 Strategy: Multi-Domain Roadmap Integration
## Master Project Plan

**Project Title**: Understanding Performance, Degradation, and Space in Formula 1 Racing  
**Scope**: 3 interconnected research domains across data pipeline  
**Timeline**: 27-40 weeks (~7-10 months)  
**Data Coverage**: 2022-2024 (68 Grand Prix weekends, 337 sessions)

---

## Executive Overview

This project decomposes F1 race performance into three research domains that inform each other:

```
Tyre Degradation (Domain 1)
    ↓ reveals degradation patterns
    ↓
Traffic & Overtaking (Domain 2) ← applies degradation understanding to proximity dynamics
    ↓ identifies where overtakes happen
    ↓
Expected Threat xT (Domain 3) ← contextualizes threat with degradation & traffic effects
```

**Why This Structure?**
- Sequential dependency: Each domain builds on previous
- Shared data foundation: All use same ingested bronze layer
- Natural separation: Three independent research questions
- Clear deliverables: Analysis notebooks, insights docs, optional models

---

## Pipeline Architecture

### Data Layer Progression

```
INGESTION (Done) ✅
  ├─ laps_raw (485K rows)
  ├─ telemetry_raw (48M rows)
  ├─ weather_raw
  ├─ race_control_raw
  └─ positions_raw ❌ MISSING
        ↓
ETL & FEATURE ENGINEERING (This Roadmap)
  ├─ Domain 1: Clean air stints → Degradation curves → Context features
  ├─ Domain 2: Position data + Overtakes → Proximity events → Traffic regimes
  └─ Domain 3: Zone threat → Contextual modifiers → Driver profiles
        ↓
ANALYSIS & INSIGHTS
  ├─ Clustering (behaviours, regimes)
  ├─ Statistical testing (ANOVA, regression)
  ├─ Visualization (heatmaps, profiles, narratives)
  └─ Optional: ML models for prediction
        ↓
RESULTS & DOCUMENTATION
  ├─ Analysis notebooks (per domain)
  ├─ Publication-quality figures
  ├─ Insights documents
  └─ Tools/dashboards (optional)
```

---

## Critical Path to Completion

### Phase 0: Data Enablement (Weeks 1-2) ⚠️ BLOCKER
**Status**: Required before any domain work can progress

#### Checkpoint 0A: Enable Position Ingestion
- **Task**: Implement `ingest_positions()` in `ingest_parquet.py`
- **Effort**: 3-5 days
- **Blockers**: None (FastF1 API supports this)
- **Validation**: Check `positions_raw` table populated for test session
- **Dependencies**: Unblocks Domain 2 entirely

#### Checkpoint 0B: Enable Circuit Info
- **Task**: Implement `ingest_circuit_info()` for corners, track length
- **Effort**: 2-3 days
- **Effort**: Low (FastF1 API call)
- **Validation**: Circuit corners correctly identified
- **Dependencies**: Unblocks Domain 3 entirely

#### Checkpoint 0C: Update Requirements.txt
- **Task**: Add scipy, scikit-learn, statsmodels, xgboost
- **Effort**: 1 day
- **Validation**: `pip install -r requirements.txt` succeeds

**Success Criteria for Phase 0**:
- [ ] positions_raw table has ≥1M rows across test sessions
- [ ] circuit_corners table populated for 2-3 tracks
- [ ] All new dependencies installable

---

### Phase 1: Domain 1 - Tyre Degradation (Weeks 3-9)

#### 1A: Clean Air Data Extraction (Weeks 3-4)
- **Task**: Build `clean_air_stints` table (Job 1)
- **Input**: laps_featured, gaps_featured
- **Filter**: Green flag, clean air, accurate laps
- **Expected Output**: 1,500-2,000 clean stints
- **Validation**: 
  - Average clean stint length > 8 laps
  - No deleted/inaccurate laps in output
  - Clean air gap distribution (median > 2.5s)

#### 1B: Degradation Curve Fitting (Weeks 4-5)
- **Task**: Build `stint_degradation` table (Job 2)
- **Methods**: Linear rate, polynomial fit, rolling slope
- **Expected Output**: Degradation metrics per stint
- **Validation**: R² > 0.8 for curve fits, realistic degradation rates

#### 1C: Context Feature Engineering (Weeks 5-6)
- **Tasks**: 
  - Build `enrich_stint_context` (Job 3) - thermal, strategic, track context
  - Build `extract_driving_style_signals` (Job 4) - aggression, consistency
- **Expected Output**: `stint_degradation_featured` table
- **Validation**: Features correlate with known track/compound characteristics

#### 1D: Unsupervised Analysis (Weeks 6-8)
- **Analysis 1**: K-means clustering for degradation behaviours
  - Validation: Silhouette score > 0.4, ≥4 interpretable clusters
- **Analysis 2**: Hierarchical clustering for track regimes
  - Validation: Regimes align with domain knowledge (high-stress, high-speed, balanced)
- **Analysis 3**: Two-way ANOVA for interaction effects
  - Validation: p < 0.05 for significant interactions

#### 1E: Visualizations & Documentation (Weeks 8-9)
- **Deliverables**: 
  - Behaviour profile plots (4 faceted line plots)
  - Track regime heatmaps
  - Interaction effect bar plots
  - EDA notebook
  - Clustering analysis notebook
  - Insights document

**Checkpoint 1 Success Criteria**:
- [ ] ≥1,500 clean stints extracted
- [ ] Degradation curve fits with R² > 0.75
- [ ] 3-4 distinct behaviour clusters identified
- [ ] ≥3 track regimes clearly separated
- [ ] All visualizations publication-quality
- [ ] Insights align with F1 domain knowledge

---

### Phase 2: Domain 2 - Traffic & Spatial (Weeks 10-23)

**Dependency**: Phase 0 (positions_raw) ⚠️

#### 2A: Position Data Processing (Weeks 10-11)
- **Task**: Build `lap_positions` table (Job 1)
  - Match positions to lap numbers via time-based join
  - Challenge: Need `lap_start_time_s` added to laps_featured
- **Expected Output**: 337 sessions × 2,000-10,000 positions per lap
- **Validation**: 
  - Position data available for ≥90% of race laps
  - Drivers' x,y coordinates sensible (on circuit boundaries)

#### 2B: Proximity Event Detection (Weeks 11-12)
- **Task**: Build `proximity_events` table (Job 2)
- **Method**: KDTree spatial search within 20m radius
- **Expected Output**: 50,000-100,000 proximity events per race
- **Validation**:
  - Events concentrated at bottleneck locations
  - Proximity distance distribution reasonable (5-20m typical)

#### 2C: Traffic Regime Classification (Weeks 12-13)
- **Task**: Build traffic regime labels per lap (Job 3)
- **Classes**: CLEAN_AIR, LIGHT_TRAFFIC, MODERATE_TRAFFIC, HEAVY_TRAFFIC, CLOSE_FOLLOWING
- **Expected Distribution**: ~40%, 25%, 20%, 10%, 5%
- **Validation**: Distribution matches expectations

#### 2D: Traffic Impact Quantification (Weeks 13-14)
- **Task**: Build `traffic_impact_summary` (Job 4)
- **Method**: Baseline-adjusted lap time loss per regime
- **Expected**: LIGHT → +0.1-0.3s, CLOSE → +2-4s
- **Validation**: Time loss increases monotonically with traffic intensity

#### 2E: Overtake Detection & Classification (Weeks 14-16)
- **Task**: Build `overtakes_featured` table (Job 5)
- **Classes**: DRS_ASSISTED, LATE_BRAKING, STRATEGIC, SLIPSTREAM
- **Expected Output**: ≥100 overtakes/season detected
- **Validation**:
  - Overtakes concentrated in expected zones (turn 1s, DRS zones)
  - Overtake types distributed reasonably (60% DRS, 20% braking, etc.)

#### 2F: Spatial Analysis (Weeks 16-17)
- **Tasks**:
  - Identify overtake hotspots (Job 6) - DBSCAN clustering
  - Map track bottlenecks (Job 7) - density analysis
- **Expected Output**: 
  - ≥3 overtake zones per track
  - ≥2 bottleneck zones per track
- **Validation**: Zones align with circuit layout (Turn 1, DRS zones)

#### 2G: Statistical Analysis (Weeks 17-20)
- **Analysis 1**: Traffic penalty quantification (mixed-effects model)
  - Validation: Coefficients significant (p<0.05), consistent across tracks
- **Analysis 2**: Traffic-induced degradation (t-test)
  - Validation: Clear effect size (10-20% increase)
- **Analysis 3**: Overtake success factors (logistic regression)
  - Validation: Model accuracy ≥70%, interpretable coefficients
- **Analysis 4**: Track-specific profiles
  - Validation: ≥3 distinct overtaking profiles identified

#### 2H: Visualizations & Documentation (Weeks 20-23)
- **Deliverables**:
  - Traffic penalty box plots
  - Overtake success probability surfaces
  - Track bottleneck maps
  - EDA, analysis, spatial analysis, visualization notebooks
  - Insights document

**Checkpoint 2 Success Criteria**:
- [ ] ≥90% of race laps have position data
- [ ] ≥100 overtakes detected across 68 GPs
- [ ] Traffic penalties quantified (significant, p<0.05)
- [ ] ≥3 overtake zones identified per track
- [ ] Overtake model accuracy ≥70%
- [ ] All visualizations publication-quality

---

### Phase 3: Domain 3 - Expected Threat xT (Weeks 24-36)

**Dependency**: Phase 2 (overtakes_featured, proximity_events) ⚠️

#### 3A: Track Zone Definition (Weeks 24-25)
- **Task**: Build `track_zones` table
- **Method**: Corner-based zones (approach, corner, exit)
- **Expected Output**: ≥15 zones per circuit (3 per corner)
- **Validation**: Zones capture realistic overtaking locations

#### 3B: Base Threat Computation (Weeks 25-27)
- **Task**: Build `zone_threat` table (Job 1)
- **Method**: P(overtake | position in zone) from historical data
- **Expected**: Range 0.05-0.25 (5%-25% success rate)
- **Validation**: 
  - High-threat zones: DRS zones, turn 1s, long straights (>15% success)
  - Low-threat zones: High-speed corners, chicanes (<5% success)

#### 3C: Contextual Threat Modifiers (Weeks 27-29)
- **Tasks**:
  - Implement contextual threat modifiers (Job 2):
    - Tyre freshness (0.5-1.0x multiplier)
    - Traffic density (0.5-1.0x)
    - DRS activation (1.0-3.0x)
  - Compute cumulative threat (Job 3)
  - Build contextual xT per lap
- **Validation**: Modifiers show expected trends

#### 3D: Driver Threat Profiles (Weeks 29-31)
- **Task**: Build `driver_threat_profiles` (Job 4)
- **Metrics**: Threat generation, conversion efficiency, consistency
- **Expected Output**: Driver philosophies (aggressive, patient, balanced)
- **Validation**: Profiles align with known driver characteristics

#### 3E: Spatial Value Analysis (Weeks 31-33)
- **Analysis 1**: Track-specific xT maps
  - Deliverable: Per-track corner threat heatmaps
  - Validation: Maps interpretable and match domain knowledge
- **Analysis 2**: Contextual threat surfaces
  - Deliverable: 3D surface plots (threat vs tyre age vs gap)
  - Validation: Surfaces show expected trade-offs

#### 3F: Driver Comparison Analysis (Weeks 33-34)
- **Analysis 3**: Threat profile comparison (radar plots)
  - Deliverable: Driver philosophy clustering
  - Validation: Profiles match known driving styles
- **Analysis 4**: Race narrative xT accumulation
  - Deliverable: Lap-by-lap cumulative threat charts
  - Validation: Narratives explain actual race outcomes

#### 3G: Optional Advanced Modeling (Weeks 34-35)
- **Optional Model 1**: Overtake conversion predictor (logistic regression)
- **Optional Model 2**: Driver philosophy classifier (K-means)
- **Validation**: Models show clear insights (not just black-box prediction)

#### 3H: Documentation (Weeks 35-36)
- **Deliverables**:
  - EDA, track analysis, driver profiles, race narratives notebooks
  - xT fundamentals explanation
  - Insights document with strategic implications
  - Publication-quality figures

**Checkpoint 3 Success Criteria**:
- [ ] Threat values range 0.05-0.25, distributed as expected
- [ ] Contextual modifiers show monotonic trends
- [ ] Driver threat profiles interpretable and distinct
- [ ] xT maps align with domain knowledge
- [ ] ≥3 driver philosophies identified
- [ ] All visualizations publication-quality

---

## Cross-Domain Integration (Weeks 36-39)

### Integration Analysis
**Research Question**: How do the three domains interact to produce race outcomes?

#### Task 1: Degradation × Traffic Interaction
```
Q: Do conservative drivers (Domain 1) suffer less in traffic (Domain 2)?
Method: Correlation analysis + interaction terms
```

#### Task 2: Threat Generation × Degradation Trade-off
```
Q: Which driving styles maximize xT while minimizing wear?
Method: Multi-objective optimization visualization
```

#### Task 3: Race Simulation Framework (Optional)
```
Q: Can we predict race outcomes from Domain 1,2,3 features?
Method: Simulation with base pace (Domain 1) + traffic (Domain 2) + xT (Domain 3)
```

### Cross-Domain Notebook
- **`cross_domain_analysis.ipynb`**: Integration of all three domains
- **Deliverables**:
  - Interaction visualizations
  - Unified insights document
  - Case studies (specific races where all 3 domains matter)

---

## Finalization & Release (Weeks 39-40)

### Final Checklist
- [ ] All notebooks run without errors
- [ ] All data outputs validated
- [ ] All visualizations publication-quality
- [ ] All insights documents proofread
- [ ] Documentation complete (methods, assumptions, limitations)
- [ ] Code commented and reproducible
- [ ] Findings consistent across seasons/tracks

### Deliverables Package

```
docs/
  ├─ ROADMAP_DOMAIN1_DEGRADATION.md (this document)
  ├─ ROADMAP_DOMAIN2_TRAFFIC.md
  ├─ ROADMAP_DOMAIN3_XT.md
  └─ METHODOLOGY.md (combined methods document)

notebooks/
  ├─ domain1_degradation_eda.ipynb
  ├─ domain1_clustering_analysis.ipynb
  ├─ domain1_interaction_effects.ipynb
  ├─ domain1_visualizations.ipynb
  ├─ domain2_traffic_eda.ipynb
  ├─ domain2_overtake_analysis.ipynb
  ├─ domain2_spatial_analysis.ipynb
  ├─ domain2_visualizations.ipynb
  ├─ domain3_xt_fundamentals.ipynb
  ├─ domain3_track_specific_analysis.ipynb
  ├─ domain3_driver_profiles.ipynb
  ├─ domain3_race_narratives.ipynb
  └─ cross_domain_analysis.ipynb

results/
  ├─ domain1_insights.md
  ├─ domain2_insights.md
  ├─ domain3_insights.md
  ├─ integration_insights.md
  └─ figures/
      ├─ domain1_*.png (all visualizations)
      ├─ domain2_*.png
      ├─ domain3_*.png
      └─ cross_domain_*.png

src/features/
  ├─ degradation.py (new module)
  ├─ traffic.py (new module)
  ├─ expected_threat.py (new module)
  └─ [updates to existing modules]
```

---

## Resource Requirements

### Human Resources
- **Data Engineer**: 2-3 weeks (Phase 0, ETL setup)
- **Data Analyst**: 8-10 weeks (Analysis & visualizations)
- **Optional**: ML Engineer (1-2 weeks for advanced models)

### Computing Resources
- **Storage**: 
  - Bronze layer: Existing (~10GB for parquet)
  - Silver layer: +20-30GB (features tables)
  - Estimated total: 40-50GB manageable storage

- **Compute**: 
  - Single machine (4-8 core) sufficient
  - DuckDB/SQLite queries <5 min each
  - ML training <1 min per model

### Software
- **Core**: Already installed (pandas, matplotlib, seaborn)
- **New**: scipy, scikit-learn, statsmodels, xgboost, plotly

---

## Risk Management

### High-Risk Items

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Position data incomplete | Blocks Domain 2,3 | Verify 90%+ coverage in Phase 0 |
| Overtake detection misses real passes | Analysis invalid | Manual validation of sample |
| Clusters don't form cleanly | Analysis unclear | Try alternative methods (DBSCAN, HDBSCAN) |
| Context modifiers don't show trends | Model weak | Check data quality, add features |

### Medium-Risk Items

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Performance regressions (slow queries) | Delays | Use DuckDB, profile early |
| Visualization complexity | Communication | Use interactive tools (Plotly) |
| Writing/documentation takes longer | Timeline slip | Start doc early, use templates |

---

## Success Criteria Summary

### By Domain

**Domain 1**: 
- ≥1,500 clean stints, degradation curves with R²>0.75
- 3-4 distinct behaviours, ≥2 track regimes
- Interaction effects significant (p<0.05)

**Domain 2**:
- ≥90% lap coverage, ≥100 overtakes detected
- Traffic penalties significant and track-specific
- ≥3 overtake zones per circuit identified

**Domain 3**:
- Threat values 0.05-0.25, contextual modifiers work
- ≥3 driver philosophies identified
- xT effectively predicts overtake timing/location

### Overall

- **Reproducibility**: Results consistent across 2022-2024 seasons
- **Interpretability**: All findings explainable to F1 domain experts
- **Completeness**: 13+ publication-ready notebooks
- **Quality**: All visualizations and writing publication-grade

---

## Next Steps (Immediate)

**Week 1 Actions**:
1. [ ] Schedule Phase 0 implementation sprint
2. [ ] Assign data engineer to position ingestion
3. [ ] Review this roadmap in team meeting
4. [ ] Set up tracking for ETL job monitoring
5. [ ] Create GitHub issues for each job/task

**Decision Point after Phase 0**:
- If position data incomplete: Adjust Domain 2,3 scope or extend Phase 0
- If new dependencies fail: Troubleshoot, create fresh Python environment
- Otherwise: Proceed to Phase 1 (Domain 1 analysis)

---

## Document Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 29, 2025 | Initial master roadmap |
| | | - 3 domain roadmaps (1,300+ pages total) |
| | | - Critical path & timeline (27-40 weeks) |
| | | - Cross-domain integration framework |

---

**Project Lead**: [To be assigned]  
**Status**: Ready for Phase 0 kickoff  
**Last Updated**: Dec 29, 2025
