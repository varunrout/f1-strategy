# Domain 1: Tyre Degradation Analysis
## Detailed Project Roadmap

> **Goal**: Understand tyre degradation as a function of usage, style, and context

---

## Phase 1: ETL & Data Preparation

### 1.1 Ingestion (✅ COMPLETE)
- [x] Laps data with `compound`, `tyre_life`, `lap_time_ms`
- [x] Weather data with `track_temp_c`, `air_temp_c`
- [x] Race control data for SC/VSC/RED flag detection
- [x] Session metadata (year, GP, session_type)

**Available Data**: 337 sessions across 2022-2024 (~68 Grand Prix weekends)

### 1.2 ETL Workflows (NEW)

#### Job 1: `build_clean_air_stints`
**Purpose**: Isolate degradation signal from race artifacts

**Input Tables**:
- `laps_featured` (lap times, track status, fuel proxy)
- `gaps_featured` (clean air flags, gap to ahead)

**Filters**:
```sql
WHERE is_accurate = 1
  AND deleted = 0
  AND track_status = 'GREEN'           -- No SC/VSC/Yellow
  AND in_clean_air = 1                  -- Gap > 2.5s
  AND lap_number BETWEEN stint_start + 2 AND stint_end - 1  -- Exclude in/out laps
```

**Output**: `clean_air_stints` (Silver layer)

**Columns**:
- `stint_id` = `{session_id}_{driver}_{stint}`
- `compound`, `stint_start_lap`, `stint_end_lap`, `stint_length`
- `avg_track_temp_c`, `avg_air_temp_c`
- `fuel_proxy` = `race_progress` (0→1 normalised)
- `lap_times[]` = array of clean lap times
- `tyre_ages[]` = array of tyre ages
- `clean_lap_count` = number of usable laps

**Estimated Output**: ~2,000-3,000 clean stints (filtering will be aggressive)

---

#### Job 2: `build_stint_degradation_curves`
**Purpose**: Fit degradation models per stint

**Input**: `clean_air_stints`

**Transformations**:

1. **Linear Degradation Rate**:
   ```python
   # Per stint:
   deg_rate_s_per_lap = (lap_time[-1] - lap_time[0]) / (tyre_age[-1] - tyre_age[0])
   ```

2. **Polynomial Fit (2nd order)**:
   ```python
   # Fit: lap_time = a + b*tyre_age + c*tyre_age^2
   coeffs = np.polyfit(tyre_ages, lap_times, deg=2)
   curvature = coeffs[0]  # Degradation acceleration
   ```

3. **Moving Average Slope**:
   ```python
   # 3-lap rolling window slope
   rolling_slope = np.gradient(rolling_mean(lap_times, window=3), tyre_ages)
   deg_stability = np.std(rolling_slope)  # Lower = more consistent
   ```

**Output**: `stint_degradation` (Silver layer)

**Columns**:
- `stint_id` (FK to clean_air_stints)
- `linear_deg_rate_s_per_lap`
- `poly_curvature` (acceleration of degradation)
- `deg_stability` (consistency measure)
- `r2_linear`, `r2_poly` (fit quality)
- `tire_cliff_detected` (bool, if slope > 2x baseline)

---

## Phase 2: Feature Engineering

### 2.1 Context Features

#### Job 3: `enrich_stint_context`
**Purpose**: Add environmental and strategic context

**Input**: `stint_degradation` + `clean_air_stints` + `weather_raw`

**New Features**:

1. **Thermal Context**:
   ```python
   thermal_stress = (
       0.4 * normalised_track_temp +
       0.3 * normalised_air_temp +
       0.3 * (1 - humidity_pct / 100)  # Lower humidity = higher evaporation
   )
   ```

2. **Strategic Context**:
   ```python
   fuel_effect = 1 - race_progress  # Heavy fuel early = more deg
   stint_position = 'early' | 'mid' | 'late'  # Race phase
   tyre_delta_to_leader = compound_hardness - leader_compound_hardness
   ```

3. **Track Context** (requires manual track classification initially):
   ```python
   track_type = 'high_speed' | 'street' | 'mixed' | 'technical'
   lateral_load_rating = [1-5]  # Manual rating initially
   stop_start_rating = [1-5]    # Braking intensity
   ```

**Output**: `stint_degradation_featured` (Silver layer)

---

### 2.2 Driver Style Features

#### Job 4: `extract_driving_style_signals`
**Purpose**: Quantify tyre usage patterns from telemetry

**Input**: `telemetry_raw` + `segments_featured`

**Aggregations** (per stint):

1. **Aggression Metrics**:
   ```python
   # From segments_featured:
   avg_full_throttle_pct = mean(full_throttle_pct across all laps in stint)
   avg_braking_pct = mean(braking_pct)
   late_braking_score = mean(entry_speed_kph in high-braking corners)
   ```

2. **Consistency Metrics**:
   ```python
   # Lap-to-lap variation:
   lap_time_std = std(lap_times) / mean(lap_times)
   sector_consistency = mean([std(s1), std(s2), std(s3)])
   ```

3. **Energy Management**:
   ```python
   # From telemetry:
   throttle_variance = std(throttle_pct across stint)
   smooth_driving_score = 1 / (1 + throttle_variance)
   ```

**Output**: `driving_style_features` (Silver layer)

**Columns**:
- `stint_id`
- `aggression_score` (composite 0-1)
- `consistency_score` (composite 0-1)
- `energy_mgmt_score` (composite 0-1)

---

## Phase 3: Analysis & Insights

### 3.1 Unsupervised Learning

#### Analysis 1: Discover Degradation Behaviours
**Method**: K-means clustering on degradation patterns

**Input Features** (per stint):
- `linear_deg_rate_s_per_lap`
- `poly_curvature`
- `deg_stability`
- Normalised by compound baseline

**Steps**:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Feature matrix
X = stints[['linear_deg_rate', 'curvature', 'stability']].values
X_scaled = StandardScaler().fit_transform(X)

# Elbow method for k selection
inertias = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Fit final model (assume k=4)
degradation_clusters = KMeans(n_clusters=4, random_state=42).fit_predict(X_scaled)
```

**Expected Behaviours** (hypotheses):
- Cluster 1: "Conservative" - low rate, flat curve, high stability
- Cluster 2: "Aggressive early fade" - high initial rate, steep curve
- Cluster 3: "Unstable oscillating" - moderate rate, low stability
- Cluster 4: "Cliff hunters" - moderate early, sudden cliff (high curvature)

**Output**: Add `deg_behaviour_cluster` to `stint_degradation_featured`

---

#### Analysis 2: Discover Track Regimes
**Method**: Hierarchical clustering on track-level aggregates

**Aggregation**:
```sql
-- Per compound per track:
SELECT 
  gp_name,
  compound,
  AVG(linear_deg_rate_s_per_lap) as avg_deg_rate,
  AVG(poly_curvature) as avg_curvature,
  AVG(thermal_stress) as avg_thermal,
  COUNT(DISTINCT stint_id) as sample_size
FROM stint_degradation_featured
WHERE clean_lap_count >= 8  -- Minimum viable stint length
GROUP BY gp_name, compound
HAVING sample_size >= 5  -- Sufficient data
```

**Clustering**:
```python
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Feature matrix per track
X_tracks = track_summary[['avg_deg_rate', 'avg_curvature', 'avg_thermal']].values
X_scaled = StandardScaler().fit_transform(X_tracks)

# Hierarchical clustering
linkage_matrix = linkage(X_scaled, method='ward')
track_clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')
```

**Expected Regimes** (hypotheses):
- High-stress: Monaco, Singapore, Hungary (stop-start, high thermal)
- High-speed: Monza, Spa, Silverstone (tyre temperature dependent)
- Balanced: Austria, Barcelona, COTA

**Output**: `track_regime_map` table (reference data)

---

### 3.2 Interaction Analysis

#### Analysis 3: Driving Style × Track Regime Effects
**Method**: Two-way ANOVA + post-hoc tests

**Research Question**: Do certain driving styles suffer more on specific track types?

**Statistical Test**:
```python
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Group stints by (behaviour_cluster, track_regime)
grouped = stints.groupby(['deg_behaviour_cluster', 'track_regime'])

# Dependent variable: normalised degradation rate
# (normalised by compound baseline for fair comparison)
stints['norm_deg_rate'] = stints['linear_deg_rate'] / stints['compound_baseline_deg']

# ANOVA
model = ols('norm_deg_rate ~ C(deg_behaviour_cluster) * C(track_regime)', data=stints).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Post-hoc pairwise comparisons
tukey = pairwise_tukeyhsd(
    endog=stints['norm_deg_rate'],
    groups=stints['behaviour_x_regime'],
    alpha=0.05
)
```

**Expected Insights**:
- "Aggressive early fade" suffers most on high-thermal tracks
- "Conservative" behaviour is track-agnostic
- Street circuits punish inconsistent styles

---

### 3.3 Visualisation Deliverables

#### Viz 1: Degradation Behaviour Profiles
**Type**: Faceted line plots

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Per behaviour cluster, plot average degradation curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for cluster in range(4):
    cluster_stints = stints[stints['deg_behaviour_cluster'] == cluster]
    
    # Normalize tyre age to 0-1 scale
    for stint in cluster_stints.itertuples():
        norm_age = np.linspace(0, 1, len(stint.tyre_ages))
        norm_time = (stint.lap_times - stint.lap_times[0]) / stint.lap_times[0] * 100  # % slower
        axes.flat[cluster].plot(norm_age, norm_time, alpha=0.3, color='grey')
    
    # Mean curve
    axes.flat[cluster].set_title(f"Behaviour {cluster}: {behaviour_labels[cluster]}")
    axes.flat[cluster].set_xlabel("Normalised Tyre Age")
    axes.flat[cluster].set_ylabel("% Slower than Stint Start")
```

---

#### Viz 2: Track Regime Heatmap
**Type**: Compound × Track heatmap

```python
# Pivot table
heatmap_data = track_summary.pivot(
    index='gp_name',
    columns='compound',
    values='avg_deg_rate'
)

sns.heatmap(
    heatmap_data,
    cmap='RdYlGn_r',  # Red = high deg, Green = low deg
    annot=True,
    fmt='.3f',
    cbar_kws={'label': 'Avg Deg Rate (s/lap)'}
)
plt.title("Tyre Degradation by Track and Compound")
```

---

#### Viz 3: Interaction Effects
**Type**: Bar plot with error bars

```python
# Mean degradation rate per (behaviour, track_regime) combo
interaction_summary = stints.groupby(['deg_behaviour_cluster', 'track_regime']).agg({
    'norm_deg_rate': ['mean', 'std', 'count']
}).reset_index()

sns.barplot(
    data=interaction_summary,
    x='track_regime',
    y='norm_deg_rate_mean',
    hue='deg_behaviour_cluster',
    capsize=0.1
)
plt.ylabel("Normalised Deg Rate (compound-adjusted)")
plt.title("Driving Behaviour × Track Regime Effects")
```

---

## Phase 4: Modeling (Optional/Future)

### 4.1 Predictive Models

#### Model 1: Stint Degradation Predictor
**Type**: Gradient Boosting Regressor

**Target**: `linear_deg_rate_s_per_lap`

**Features**:
- Compound (one-hot)
- Track regime (one-hot)
- Thermal stress (continuous)
- Fuel proxy (continuous)
- Driver style cluster (one-hot)
- Historical driver-track performance

**Use Case**: Pre-race degradation forecasting

---

#### Model 2: Tyre Cliff Early Warning
**Type**: Binary classifier (Random Forest)

**Target**: `tire_cliff_detected` (0/1)

**Features**:
- Rolling 3-lap degradation trend
- Compound × track interaction
- Current tyre age
- Gap to expected degradation (anomaly score)

**Use Case**: Live race strategy alerts

---

## Phase 5: Results & Documentation

### 5.1 Notebook Deliverables

1. **`domain1_degradation_eda.ipynb`**
   - Load and validate `stint_degradation_featured`
   - Distribution analysis
   - Correlation heatmaps

2. **`domain1_clustering_analysis.ipynb`**
   - Behaviour discovery (k-means)
   - Track regime discovery (hierarchical)
   - Cluster validation metrics

3. **`domain1_interaction_effects.ipynb`**
   - ANOVA tests
   - Post-hoc comparisons
   - Effect size calculations

4. **`domain1_visualisations.ipynb`**
   - All publication-quality plots
   - Interactive Plotly dashboards

### 5.2 Insights Document

**Template**:
```markdown
# Tyre Degradation Analysis Results

## Key Findings

1. **Degradation Behaviours Identified**: [4] distinct patterns
   - Behaviour A: [Description] (n=[X] stints)
   - ...

2. **Track Regimes Identified**: [3] regimes
   - High-stress tracks: [Monaco, Singapore, ...]
   - ...

3. **Interaction Effects**:
   - Behaviour A performs [X%] worse on high-stress tracks (p<0.001)
   - ...

## Implications for Strategy

- Conservative driving recommended on: [tracks]
- Aggressive early pace viable on: [tracks]
- Compound selection should favour: [analysis]
```

---

## Dependencies & Prerequisites

### Software Stack
```txt
# Add to requirements.txt:
scipy>=1.10.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0  # For interactive viz
```

### Data Dependencies
```
laps_featured (existing)
  ↓
gaps_featured (existing)
  ↓
clean_air_stints (new) ← Job 1
  ↓
stint_degradation (new) ← Job 2
  ↓
stint_degradation_featured (new) ← Jobs 3, 4
  ↓
Analysis notebooks
```

### Estimated Timeline
- **Phase 1 (ETL)**: 1-2 weeks
- **Phase 2 (Features)**: 1-2 weeks
- **Phase 3 (Analysis)**: 2-3 weeks
- **Phase 4 (Modeling)**: 2-3 weeks (optional)
- **Phase 5 (Documentation)**: 1 week

**Total**: 7-11 weeks for complete domain

---

## Success Metrics

1. **Data Quality**:
   - ≥1,500 clean stints extracted (filtering ~50% of data is expected)
   - ≥8 laps per stint on average
   - R² > 0.8 for degradation curve fits

2. **Analysis Validity**:
   - Silhouette score > 0.4 for behaviour clusters
   - ANOVA p-values < 0.05 for interaction effects
   - ≥5 samples per (behaviour, track) combination

3. **Interpretability**:
   - Clear behaviour labels (human-interpretable)
   - Track regimes align with domain knowledge
   - Reproducible results across seasons
