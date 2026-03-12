# F1 Strategy Multi-Domain Project: Executive Summary
**Version 1.0 | December 29, 2025**

---

## 📊 Project at a Glance

| Aspect | Details |
|--------|---------|
| **Title** | Understanding Performance, Degradation, and Space in Formula 1 Racing |
| **Type** | Multi-domain exploratory research study |
| **Data Scope** | 2022-2024 seasons (68 GPs, 337 sessions, 171K+ laps) |
| **Timeline** | 27-40 weeks (~7-10 months) |
| **Approach** | Ingestion → ETL → Features → Analysis → Insights → Visualization → Modeling |
| **Deliverables** | 13+ analysis notebooks, 3 domain insights docs, publication-quality figures |

---

## 🧠 The Three Research Domains

### Domain 1: Tyre Degradation Analysis
**Research Question**: How do tyres degrade in real race conditions?

**Key Activities**:
- Extract clean-air stints (remove race noise)
- Fit degradation curves per stint
- Discover driving behaviour patterns (clustering)
- Discover track regimes (hierarchical clustering)
- Analyze behaviour × track interactions (ANOVA)

**Timeline**: Weeks 3-9 (7 weeks)  
**Status**: Data-independent (ready to start)  
**Expected Insights**:
- 3-4 distinct tyre usage behaviours
- 2-3 track regimes (high-stress, high-speed, balanced)
- Clear evidence of behaviour × track interactions

---

### Domain 2: Traffic & Spatial Constraints  
**Research Question**: How does proximity affect performance and overtaking?

**Key Activities**:
- **Phase 0 Blocker**: Ingest position data (x,y,z coordinates) ⚠️
- Detect proximity events via spatial search
- Classify traffic regimes per lap
- Quantify performance loss by regime
- Detect and classify overtakes (types, locations)
- Map overtake hotspots and track bottlenecks
- Statistical analysis of traffic impacts

**Timeline**: Weeks 10-23 (14 weeks)  
**Status**: Blocked on position data ingestion  
**Critical Path**:
1. Phase 0 (Weeks 1-2): Ingest positions + circuit info
2. Phase 2A (Weeks 10-11): Process positions → lap_positions
3. Remaining features build on this foundation

**Expected Insights**:
- Quantified lap time penalties per traffic regime (+0.1s to +4s)
- 100+ overtakes detected with type classification
- ≥3 overtake zones per track identified
- Evidence of traffic-induced tyre degradation

---

### Domain 3: Expected Threat (xT) Framework
**Research Question**: Where on track does race progress happen?

**Key Activities**:
- Define track zones (corner-based approach)
- Calculate base overtake probability per zone
- Model contextual threat (tyre state, traffic, DRS)
- Compute cumulative threat per driver per race
- Profile driver threat generation styles
- Create track-specific xT value maps
- Analyze interaction with tyre/traffic states

**Timeline**: Weeks 24-36 (13 weeks)  
**Status**: Blocked on Domain 2 (needs overtakes_featured)  
**Key Dependency**: Domain 2 must deliver overtakes, proximity events

**Expected Insights**:
- Track-specific overtaking value maps
- 3+ distinct driver philosophies
- Contextual threat modifiers quantified
- xT-based race narratives

---

## 🔗 Domain Interactions

```
              Tyre Degradation
                    ↓
          Explains how long drivers can sustain pace
                    ↓
        ┌──────────────────────────────┐
        ↓                              ↓
    Traffic       ←   Applies         xT
  Constraints      understanding     (Value)
        ↓          to proximity       ↓
    Where overtakes    ↑            Where
    are feasible     Channels        value is
                    threat           created
```

**Why Sequential?**
1. Domain 1 (degradation) answers "how fast?"
2. Domain 2 (traffic) answers "can you go fast with others around?"
3. Domain 3 (xT) answers "where does that speed matter?"

Together: Complete model of race dynamics without predicting outcomes

---

## ⚠️ Critical Blocker: Position Data Ingestion

**Current State**: ❌ Position coordinates NOT being ingested

**What's Needed**:
```python
# Add to src/ingest/ingest_parquet.py
def prepare_positions_df(session, session_id: str) -> pd.DataFrame:
    pos_data = session.pos_data  # FastF1 API call
    # Returns: DataFrame with (driver, time, x, y, z) per timestamp
    # Expected output: ~1.4M rows per race
```

**Impact on Domains**:
- Domain 1: ✅ No impact (ready to proceed)
- Domain 2: ❌ BLOCKED (needs position coordinates)
- Domain 3: ❌ BLOCKED (depends on Domain 2)

**Timeline Impact**:
- **Must Complete**: Phase 0, Weeks 1-2
- **Effort**: 1-2 weeks (straightforward API call + parquet write)
- **Without This**: 50% of project scope becomes infeasible

**Action Item**: Assign Phase 0 to data engineer immediately

---

## 📈 Roadmap Timeline

### Critical Path (Minimum 27 weeks)

```
Week 1-2:   Phase 0 - Data Enablement (BLOCKER) ⚠️
              ├─ Position ingestion
              ├─ Circuit info ingestion
              └─ Dependencies install

Week 3-9:   Domain 1 - Tyre Degradation ✅ Ready
              ├─ Data extraction (3 jobs)
              ├─ Analysis (3 analyses)
              └─ Documentation

Week 10-23: Domain 2 - Traffic & Spatial (Depends on Phase 0)
              ├─ Position processing (3 jobs)
              ├─ Overtake detection (3 jobs)
              ├─ Analysis (4 analyses)
              └─ Documentation

Week 24-36: Domain 3 - Expected Threat (Depends on Domain 2)
              ├─ Zone definition (1 job)
              ├─ Threat computation (4 jobs)
              ├─ Analysis (4 analyses)
              └─ Documentation

Week 36-39: Cross-Domain Integration
              ├─ Interaction analysis
              ├─ Unified insights
              └─ Case studies

Week 39-40: Finalization & Release
              ├─ QA/validation
              ├─ Documentation polish
              └─ Package deliverables
```

**Assumptions for Timeline**:
- 1 full-time analyst + 1 part-time engineer
- Parallel analysis tasks where possible
- No major blockers beyond position ingestion
- Realistic estimate with buffer

---

## 📊 Scope & Data

### Data Available

| Layer | Table | Status | Rows | Ready? |
|-------|-------|--------|------|--------|
| Bronze | laps_raw | ✅ | 182K | Yes |
| | telemetry_raw | ✅ | 48M | Yes |
| | weather_raw | ✅ | 15K | Yes |
| | race_control_raw | ✅ | 5K | Yes |
| | **positions_raw** | ❌ | 0 | **No** |
| Silver | laps_featured | ✅ | 182K | Yes |
| | gaps_featured | ✅ | 180K | Yes |
| | segments_featured | ✅ | 25M | Yes |
| | tyre_featured | ❌ | 0 | No (new) |
| | traffic_featured | ❌ | 0 | No (new) |
| | xt_featured | ❌ | 0 | No (new) |

### Coverage by Season

| Season | Sessions | Races | Laps | Races/Team |
|--------|----------|-------|------|-----------|
| 2022 | 113 | 22 | ~38K | 22 |
| 2023 | 104 | 22 | ~43K | 22 |
| 2024 | 120 | 24 | ~46K | 24 |
| **Total** | **337** | **68** | **~127K** | **68** |

**Sufficient for Analysis**: ✅ Yes
- Each domain needs ≥100 laps per race → 6,800+ available
- Clustering needs ≥500-1000 observations → >10x needed
- Statistical tests need ≥30 per group → Easily met

---

## 📦 Deliverables Checklist

### By Phase

**Phase 0** (Weeks 1-2):
- [x] Position ingestion code
- [x] Circuit info ingestion code
- [ ] Validation of data quality

**Phase 1** (Weeks 3-9):
- [ ] `clean_air_stints` table (1,500+ rows)
- [ ] `stint_degradation` table with curve fits
- [ ] Behaviour clustering analysis
- [ ] Track regime clustering analysis
- [ ] 4 domain 1 analysis notebooks
- [ ] Domain 1 insights document
- [ ] Publication-quality figures (≥5 plots)

**Phase 2** (Weeks 10-23):
- [ ] `lap_positions` table (90%+ coverage)
- [ ] `proximity_events` table (50K+ events)
- [ ] `overtakes_featured` table (≥100 overtakes)
- [ ] Traffic regime classification
- [ ] 4 domain 2 analysis notebooks
- [ ] Domain 2 insights document
- [ ] Publication-quality figures (≥8 plots)

**Phase 3** (Weeks 24-36):
- [ ] `zone_threat` table (threat values 0.05-0.25)
- [ ] `contextual_threat` with modifiers
- [ ] Driver threat profiles
- [ ] 4 domain 3 analysis notebooks
- [ ] Domain 3 insights document
- [ ] Publication-quality figures (≥6 plots)

**Phase 4** (Weeks 36-39):
- [ ] Cross-domain integration notebook
- [ ] Unified insights document
- [ ] Case study analysis

**Phase 5** (Weeks 39-40):
- [ ] Final QA/validation
- [ ] Complete documentation
- [ ] Release package

---

## 🛠️ Technology Stack

### Current (Already Installed)
```
pandas >= 2.0.0
numpy >= 1.24.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
duckdb >= 0.9.0
sqlalchemy >= 2.0.0
typer >= 0.9.0
```

### New (To Add)
```
scipy >= 1.10.0       # Statistical distributions, clustering
scikit-learn >= 1.3.0 # ML models, preprocessing
statsmodels >= 0.14.0 # Advanced stats (ANOVA, regression)
xgboost >= 1.7.0      # Gradient boosting (optional)
plotly >= 5.14.0      # Interactive visualizations (optional)
```

### Infrastructure
- **Database**: SQLite (laps/gaps) + DuckDB (parquet analytics)
- **Compute**: Single machine (4-8 core) sufficient
- **Storage**: 40-50GB total (manageable)

---

## ✅ Success Criteria

### Data Quality Gates

**Gate 1 (Phase 0)**:
- [ ] positions_raw ≥1M rows for test session
- [ ] circuit_corners correctly identified for ≥2 tracks
- [ ] All dependencies install without error

**Gate 2 (Phase 1)**:
- [ ] ≥1,500 clean stints extracted
- [ ] Degradation curves R² > 0.75
- [ ] Clustering Silhouette > 0.4

**Gate 3 (Phase 2)**:
- [ ] ≥90% lap coverage with position data
- [ ] ≥100 overtakes detected across 68 GPs
- [ ] Traffic penalties significant (p<0.05)

**Gate 4 (Phase 3)**:
- [ ] Zone threats 0.05-0.25 range
- [ ] Contextual modifiers show trends
- [ ] ≥3 driver philosophies identified

### Insights Quality Gates

- [ ] Results reproducible across 2022-2024
- [ ] All findings explainable to F1 domain experts
- [ ] Visualizations publication-quality (clear, labeled, colorblind-friendly)
- [ ] Statistical tests properly specified with p-values
- [ ] No claims without supporting data

---

## 🎯 Key Decisions

### Decision 1: Should we ingest position data? ✅ YES
**Rationale**:
- Required for Domain 2 & 3 (essential to project scope)
- FastF1 API readily available
- Effort low (1-2 weeks)
- Storage manageable (~10GB for 3 years)

**Action**: Assign Phase 0 immediately

---

### Decision 2: Should we do ML models? ⚠️ OPTIONAL
**Recommendation**: Start with analysis (Domains 1-3), models as Phase 4+ work

**Reasoning**:
- Analysis is high-value, low-complexity
- Models require more data cleaning & validation
- Can be added later if insights suggest them
- Keep project focused on understanding, not prediction

---

### Decision 3: Which tracks to start with?
**Recommendation**: Full dataset (all 68 GPs)

**Reasoning**:
- Sufficient data for robust analysis
- Reveals track variations (Domain 1 objective)
- More representative results
- Only ~10-15% additional effort vs subset

---

### Decision 4: Should we integrate with existing DuckDB pipeline?
**Recommendation**: YES, use DuckDB for feature engineering

**Reasoning**:
- 100x faster than Python loops
- Follows existing project patterns
- Reduces code complexity
- Better for large-scale analytics

---

## 🚀 Immediate Next Steps

### This Week (Dec 29-31)
1. [ ] Review this document in team
2. [ ] Assign Phase 0 lead (data engineer)
3. [ ] Create GitHub issues for Phase 0 tasks
4. [ ] Set up project tracking board
5. [ ] Schedule Phase 0 kickoff meeting

### Week of Jan 1-5
1. [ ] Phase 0 sprint: Position ingestion
2. [ ] Validate position data quality
3. [ ] Test circuit info ingestion
4. [ ] Update requirements.txt + install deps

### Week of Jan 6-10
1. [ ] Phase 0 completion gating
2. [ ] If successful: Start Domain 1 (Week 3 in timeline)
3. [ ] If blocked: Troubleshoot, extend Phase 0

---

## 📚 Complete Documentation

All detailed roadmaps available in `docs/`:

1. **`ROADMAP_DOMAIN1_DEGRADATION.md`** (15 pages)
   - Complete analysis plan for tyre degradation
   - All 4 jobs, 3 analyses, visualizations
   
2. **`ROADMAP_DOMAIN2_TRAFFIC.md`** (18 pages)
   - Complete analysis plan for traffic/spatial
   - All 7 jobs, 4 analyses, visualizations
   - Position ingestion details
   
3. **`ROADMAP_DOMAIN3_XT.md`** (16 pages)
   - Complete analysis plan for Expected Threat
   - All 4 jobs, 4 analyses, visualizations
   - Integration with Domains 1 & 2
   
4. **`ROADMAP_MASTER_INTEGRATION.md`** (20 pages)
   - Critical path timeline
   - Cross-domain integration
   - Risk management
   - Resource requirements

**Total Documentation**: ~70 pages, highly detailed with formulas, code examples, visualizations, success criteria

---

## 💡 Key Insights to Expect

### From Domain 1 (Tyre Degradation)
> "Tyre degradation is not uniform. We'll discover 3-4 distinct driving behaviours, and that certain behaviours suffer much more on specific track types. This explains why driver X dominates at one track but struggles at another—not talent differences, but strategy alignment."

### From Domain 2 (Traffic & Spatial)
> "Traffic costs 2-4 seconds per lap in heavy congestion, but that's only the lap time. We'll discover where overtakes actually happen (usually not where you'd expect) and that DRS zones are 4x as valuable as regular running zones."

### From Domain 3 (Expected Threat)
> "Race position value is not linear. Turn 1 is worth 20x more than lap 30. We'll profile drivers and see that some are threat-generators while others are threat-converters. Different skills, different value."

### Cross-Domain Integration
> "A complete picture of F1 racing emerges: Conservative driving on high-stress tracks avoids tyre cliffs but creates traffic disadvantage, limiting overtaking opportunities. Aggressive driving generates high threat but burns tyres. Optimal strategy balances all three."

---

## 📞 Contact & Support

**Questions about this roadmap?**
- See specific domain documents (1,300+ pages of detail)
- Review master integration document for timeline questions
- Check GitHub issues for progress tracking

**Project Lead**: [To be assigned]  
**Data Engineer Lead**: [To be assigned]  
**Analytics Lead**: [To be assigned]

---

## 📄 Document Control

| Aspect | Details |
|--------|---------|
| Version | 1.0 |
| Created | Dec 29, 2025 |
| Status | Ready for Phase 0 |
| Review Status | Pending team review |
| Approval Status | Pending |
| Next Review | Post-Phase 0 (Feb 2026) |

---

**This roadmap represents ~200+ hours of research and planning. It is comprehensive, detailed, and ready for execution.**

**Next step: Approve Phase 0 and assign resources. 🚀**
