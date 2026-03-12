# Domain 2: Traffic, Dirty Air & Spatial Constraints
## Detailed Project Roadmap

> **Goal**: Quantify how proximity to other cars alters performance, tyre behaviour, and overtaking feasibility

---

## Phase 1: ETL & Data Preparation

### 1.1 Ingestion (PARTIAL - CRITICAL GAP)

**Current State**:
- [x] Laps data with `position`, `lap_time_ms`
- [x] Gaps data with `gap_to_ahead_s`, `in_clean_air`, `within_drs`
- [x] Telemetry data with `speed_kph`, `drs` activation
- [ ] ❌ **MISSING**: Position coordinates (x, y, z) - **BLOCKER**

**Required New Ingestion**:

#### Job: `ingest_positions`
**Priority**: **P0 CRITICAL**

**Implementation**:
```python
# In src/ingest/ingest_parquet.py

def prepare_positions_df(session, session_id: str) -> pd.DataFrame:
    """
    Ingest car position data (x, y, z coordinates).
    
    Critical for:
    - Proximity analysis
    - Overtake location detection
    - Racing line analysis
    """
    try:
        pos_data = session.pos_data  # FastF1 API call
        if pos_data is None or pos_data.empty:
            return pd.DataFrame()
        
        # Reset index to get time as column
        pos_data = pos_data.reset_index()
        
        # Prepare DataFrame
        positions = []
        for driver in pos_data['Driver'].unique():
            driver_pos = pos_data[pos_data['Driver'] == driver].copy()
            
            positions.append(pd.DataFrame({
                'session_id': session_id,
                'driver': driver,
                'time_s': driver_pos['Time'].dt.total_seconds(),
                'x': driver_pos['X'],
                'y': driver_pos['Y'],
                'z': driver_pos['Z'],
                'status': driver_pos['Status']
            }))
        
        return pd.concat(positions, ignore_index=True)
    
    except Exception as e:
        logger.warning(f"Position data unavailable: {e}")
        return pd.DataFrame()

# Add to ingest_session_parquet():
positions_df = prepare_positions_df(session, session_id)
if not positions_df.empty:
    parquet_writer.write_bronze('positions_raw', positions_df, partitions)
```

**Expected Output Size**: ~1.4M rows per race (20 drivers × ~70,000 samples each)

**Storage Optimization**: Down-sample to every 10m traveled (~14,000 rows/race):
```python
# Group by driver and distance bins
positions_df['distance_bin'] = (positions_df['distance_m'] // 10).astype(int) * 10
positions_binned = positions_df.groupby(['driver', 'distance_bin']).first()
```

---

### 1.2 ETL Workflows (NEW)

#### Job 1: `build_lap_positions`
**Purpose**: Match position data to lap numbers

**Input**: `positions_raw` + `laps_featured`

**Method**: Time-based join
```sql
-- DuckDB query
CREATE TABLE lap_positions AS
SELECT 
  p.*,
  l.lap_number,
  l.session_id
FROM positions_raw p
JOIN laps_featured l
  ON p.session_id = l.session_id
  AND p.driver = l.driver
  AND p.time_s BETWEEN l.lap_start_time_s AND l.lap_end_time_s
```

**Challenge**: `lap_start_time_s` doesn't exist in current schema

**Solution**: Add to `laps_featured`:
```python
# In build_laps_featured_duckdb.py:
LAG(cumulative_time_s) OVER (
  PARTITION BY session_id, driver 
  ORDER BY lap_number
) as lap_start_time_s
```

---

#### Job 2: `build_proximity_events`
**Purpose**: Detect when cars are near each other

**Input**: `lap_positions`

**Algorithm**:
```python
def calculate_proximity(positions_df, distance_threshold=20.0):
    """
    For each timestamp, find all car pairs within distance_threshold meters.
    
    Returns: DataFrame with columns:
    - time_s
    - driver_ahead
    - driver_behind
    - distance_m
    - relative_speed_kph
    """
    from scipy.spatial import KDTree
    
    proximity_events = []
    
    for time in positions_df['time_s'].unique():
        frame = positions_df[positions_df['time_s'] == time]
        
        # Build KD-tree for spatial search
        coords = frame[['x', 'y', 'z']].values
        tree = KDTree(coords)
        
        # Find pairs within threshold
        pairs = tree.query_pairs(r=distance_threshold)
        
        for i, j in pairs:
            car_i = frame.iloc[i]
            car_j = frame.iloc[j]
            
            # Determine ahead/behind using track position (y-coordinate typically)
            if car_i['y'] > car_j['y']:  # i is ahead
                ahead, behind = car_i, car_j
            else:
                ahead, behind = car_j, car_i
            
            distance = np.sqrt(
                (ahead['x'] - behind['x'])**2 +
                (ahead['y'] - behind['y'])**2
            )
            
            proximity_events.append({
                'time_s': time,
                'driver_ahead': ahead['driver'],
                'driver_behind': behind['driver'],
                'distance_m': distance,
                'speed_ahead_kph': ahead['speed_kph'],
                'speed_behind_kph': behind['speed_kph']
            })
    
    return pd.DataFrame(proximity_events)
```

**Output**: `proximity_events` (Silver layer)

**Estimated Size**: ~50,000-100,000 events per race

---

#### Job 3: `build_traffic_regimes_per_lap`
**Purpose**: Classify each lap by traffic exposure

**Input**: `proximity_events` + `laps_featured` + `gaps_featured`

**Classification Logic**:
```python
def classify_traffic_regime(lap_row, proximity_data):
    """
    Classifies lap into traffic regime based on:
    1. Gap to car ahead (from gaps_featured)
    2. Time spent in proximity (from proximity_events)
    3. DRS usage patterns
    """
    driver = lap_row['driver']
    lap_num = lap_row['lap_number']
    gap_ahead = lap_row['gap_to_ahead_s']
    
    # Get proximity events for this lap
    lap_proximity = proximity_data[
        (proximity_data['driver_behind'] == driver) &
        (proximity_data['lap_number'] == lap_num)
    ]
    
    time_in_proximity = lap_proximity['duration_s'].sum()
    total_lap_time = lap_row['lap_time_ms'] / 1000.0
    proximity_pct = time_in_proximity / total_lap_time
    
    # Classification
    if gap_ahead > 2.5:
        return 'CLEAN_AIR'
    elif proximity_pct < 0.2:
        return 'LIGHT_TRAFFIC'  # Near someone but not sustained
    elif proximity_pct < 0.5:
        return 'MODERATE_TRAFFIC'
    elif gap_ahead > 1.0:
        return 'HEAVY_TRAFFIC'
    else:
        return 'CLOSE_FOLLOWING'  # Gap < 1s and sustained proximity
```

**Output**: Add `traffic_regime` column to `laps_featured`

**Distribution (expected)**:
- CLEAN_AIR: ~40%
- LIGHT_TRAFFIC: ~25%
- MODERATE_TRAFFIC: ~20%
- HEAVY_TRAFFIC: ~10%
- CLOSE_FOLLOWING: ~5%

---

## Phase 2: Feature Engineering

### 2.1 Traffic Impact Features

#### Job 4: `quantify_traffic_penalties`
**Purpose**: Measure performance loss due to traffic

**Input**: `laps_featured` (with traffic_regime) + `stint_degradation` (from Domain 1)

**Baseline Calculation**:
```python
# For each driver-stint-compound combination:
clean_air_laps = laps[laps['traffic_regime'] == 'CLEAN_AIR']
baseline_pace = clean_air_laps['lap_time_ms'].median()

# Account for tyre degradation
expected_lap_time = baseline_pace + (tyre_age * deg_rate_per_lap)

# Traffic penalty
traffic_laps = laps[laps['traffic_regime'] != 'CLEAN_AIR']
traffic_laps['time_loss_ms'] = traffic_laps['lap_time_ms'] - expected_lap_time
```

**Aggregations per regime**:
```sql
SELECT 
  traffic_regime,
  AVG(time_loss_ms) as avg_time_loss_ms,
  STDDEV(time_loss_ms) as time_loss_variability_ms,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY time_loss_ms) as median_loss_ms
FROM laps_featured
WHERE is_accurate = 1 AND deleted = 0
GROUP BY traffic_regime
```

**Output**: `traffic_impact_summary` (Gold layer)

**Expected Results**:
- LIGHT_TRAFFIC: +0.1-0.3s/lap
- MODERATE_TRAFFIC: +0.5-1.0s/lap
- HEAVY_TRAFFIC: +1.0-2.0s/lap
- CLOSE_FOLLOWING: +2.0-4.0s/lap

---

#### Job 5: `detect_overtakes`
**Purpose**: Identify position changes and classify overtake types

**Input**: `laps_featured` + `proximity_events`

**Overtake Detection**:
```python
def detect_overtakes(laps_df):
    """
    Position changes where:
    1. Position improved by ≥1
    2. Not during pit windows (both drivers on track)
    3. Not under SC/VSC
    """
    laps_df = laps_df.sort_values(['session_id', 'driver', 'lap_number'])
    
    # Lap-to-lap position change
    laps_df['position_change'] = laps_df.groupby(['session_id', 'driver'])['position'].diff()
    
    # Flag overtakes
    overtakes = laps_df[
        (laps_df['position_change'] < 0) &  # Gained position
        (laps_df['is_pit_lap'] == 0) &
        (laps_df['track_status'] == 'GREEN')
    ].copy()
    
    return overtakes
```

**Overtake Classification**:
```python
def classify_overtake_type(overtake_row, proximity_data, drs_data):
    """
    Types:
    - DRS_ASSISTED: DRS active in moments before pass
    - LATE_BRAKING: Significant proximity in braking zone
    - STRATEGIC: Position change without close proximity (undercut/overcut)
    - SLIPSTREAM: High relative speed in straight
    """
    # Get proximity 5 seconds before overtake
    pre_pass_proximity = proximity_data[
        (proximity_data['time_s'] < overtake_row['lap_end_time_s']) &
        (proximity_data['time_s'] > overtake_row['lap_end_time_s'] - 5)
    ]
    
    if overtake_row['drs_activations'] > 0:
        return 'DRS_ASSISTED'
    elif pre_pass_proximity.empty:
        return 'STRATEGIC'
    # ... additional logic
```

**Output**: `overtakes_featured` (Silver layer)

**Columns**:
- `overtake_id`
- `session_id`, `driver`, `lap_number`
- `overtaken_driver`
- `overtake_type`
- `pre_pass_gap_s`
- `proximity_duration_s`
- `track_location` (corner/straight from position data)

---

### 2.2 Spatial Analysis Features

#### Job 6: `identify_overtake_zones`
**Purpose**: Find track locations where overtakes actually occur

**Input**: `overtakes_featured` + `lap_positions` + circuit metadata

**Method**:
```python
from sklearn.cluster import DBSCAN

def find_overtake_hotspots(overtakes_with_positions):
    """
    Cluster overtake locations (x, y coordinates) to find common zones.
    """
    # Extract overtake coordinates
    overtake_coords = overtakes_with_positions[['x', 'y']].values
    
    # DBSCAN clustering
    clustering = DBSCAN(eps=50, min_samples=3).fit(overtake_coords)
    
    overtakes_with_positions['overtake_zone_id'] = clustering.labels_
    
    # Characterize each zone
    zones = []
    for zone_id in set(clustering.labels_):
        if zone_id == -1:  # Noise
            continue
        
        zone_overtakes = overtakes_with_positions[
            overtakes_with_positions['overtake_zone_id'] == zone_id
        ]
        
        zones.append({
            'zone_id': zone_id,
            'center_x': zone_overtakes['x'].mean(),
            'center_y': zone_overtakes['y'].mean(),
            'overtake_count': len(zone_overtakes),
            'drs_assisted_pct': (zone_overtakes['overtake_type'] == 'DRS_ASSISTED').mean(),
            'avg_approach_speed': zone_overtakes['speed_behind_kph'].mean()
        })
    
    return pd.DataFrame(zones)
```

**Output**: `overtake_zones` (per track, Gold layer)

---

#### Job 7: `map_track_bottlenecks`
**Purpose**: Identify where cars naturally bunch up (independent of incidents)

**Input**: `lap_positions` (all cars) + `laps_featured`

**Method**:
```python
def find_bunching_locations(positions_df, lap_filter):
    """
    Calculate local car density at each point on track.
    High density = bottleneck.
    """
    # Filter to green flag laps only
    green_laps = lap_filter[lap_filter['track_status'] == 'GREEN']
    positions_filtered = positions_df.merge(green_laps, on=['session_id', 'lap_number'])
    
    # Discretize track into 50m bins
    positions_filtered['track_bin'] = (positions_filtered['distance_m'] // 50).astype(int)
    
    # Count cars per bin per timestamp
    density = positions_filtered.groupby(['time_s', 'track_bin']).size().reset_index(name='car_count')
    
    # Average density per track bin
    avg_density = density.groupby('track_bin').agg({
        'car_count': ['mean', 'max', 'std']
    })
    
    # Bottlenecks = bins with consistently high density
    bottlenecks = avg_density[avg_density[('car_count', 'mean')] > 3]
    
    return bottlenecks
```

**Output**: `track_bottlenecks` (Gold layer)

---

## Phase 3: Analysis & Insights

### 3.1 Traffic Regime Analysis

#### Analysis 1: Performance Loss Quantification
**Research Question**: How much does each traffic regime cost?

**Method**: Mixed-effects linear model
```python
import statsmodels.formula.api as smf

# Model: lap_time ~ traffic_regime + tyre_age + track_temp + (1|driver) + (1|track)
model = smf.mixedlm(
    "lap_time_ms ~ C(traffic_regime) + tyre_age + track_temp_c",
    data=laps_df,
    groups=laps_df["driver"],  # Random intercept per driver
    re_formula="1"
)
result = model.fit()
print(result.summary())
```

**Output**: Coefficient table showing traffic regime effects

**Expected**:
```
Coefficient (ms):
  CLEAN_AIR: (baseline)
  LIGHT_TRAFFIC: +150-300ms
  MODERATE_TRAFFIC: +500-1000ms
  HEAVY_TRAFFIC: +1000-2000ms
  CLOSE_FOLLOWING: +2000-4000ms
```

---

#### Analysis 2: Traffic-Induced Degradation
**Research Question**: Does traffic accelerate tyre wear?

**Method**: Compare degradation rates in traffic vs clean air

```python
from scipy.stats import ttest_ind

# Split stints by traffic exposure
high_traffic_stints = stints[stints['traffic_exposure_pct'] > 0.5]
low_traffic_stints = stints[stints['traffic_exposure_pct'] < 0.2]

# Compare degradation rates
t_stat, p_value = ttest_ind(
    high_traffic_stints['linear_deg_rate_s_per_lap'],
    low_traffic_stints['linear_deg_rate_s_per_lap']
)

print(f"Traffic vs Clean Air Degradation")
print(f"High traffic: {high_traffic_stints['linear_deg_rate'].mean():.3f} s/lap")
print(f"Low traffic: {low_traffic_stints['linear_deg_rate'].mean():.3f} s/lap")
print(f"Difference: {p_value < 0.05} (p={p_value:.4f})")
```

**Hypothesis**: Traffic increases degradation by 10-20%

---

### 3.2 Overtaking Analysis

#### Analysis 3: Overtake Feasibility Factors
**Research Question**: What conditions enable overtakes?

**Method**: Logistic regression on overtake attempts

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Features for "overtake attempt" (close following laps)
attempts = laps_df[laps_df['traffic_regime'] == 'CLOSE_FOLLOWING'].copy()

# Label: did position improve next lap?
attempts['overtake_success'] = (attempts['position_change'] < 0).astype(int)

# Feature matrix
X = attempts[['gap_to_ahead_s', 'tyre_age_delta', 'drs_available', 'compound_advantage']]
y = attempts['overtake_success']

# Fit model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# Coefficients (odds ratios)
odds_ratios = np.exp(model.coef_[0])
print("Overtake Success Odds Ratios:")
for feature, odds in zip(X.columns, odds_ratios):
    print(f"  {feature}: {odds:.2f}x")
```

**Expected Predictors**:
- Gap < 0.8s: 3x odds
- Fresh tyres (delta > 5 laps): 2.5x odds
- DRS available: 4x odds
- Compound advantage (softer): 1.8x odds

---

### 3.3 Spatial Analysis

#### Analysis 4: Track-Specific Overtaking Profiles
**Research Question**: Where can you actually overtake on each track?

**Method**: Overtake heatmap per circuit

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_overtake_heatmap(track_name, overtakes_with_pos, circuit_outline):
    """
    Scatter plot of overtake locations on track map.
    """
    track_overtakes = overtakes_with_pos[overtakes_with_pos['gp_name'] == track_name]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot circuit outline
    ax.plot(circuit_outline['x'], circuit_outline['y'], 'k-', linewidth=2, alpha=0.3)
    
    # Plot overtakes (size = frequency, color = success rate)
    scatter = ax.scatter(
        track_overtakes['x'],
        track_overtakes['y'],
        s=track_overtakes['zone_count'] * 10,
        c=track_overtakes['success_rate'],
        cmap='RdYlGn',
        alpha=0.7,
        vmin=0, vmax=1
    )
    
    plt.colorbar(scatter, label='Overtake Success Rate')
    plt.title(f"{track_name} - Overtaking Zones")
    plt.axis('equal')
    plt.show()
```

---

### 3.4 Visualisation Deliverables

#### Viz 1: Traffic Penalty by Regime
**Type**: Box plot with swarm overlay

```python
sns.boxplot(
    data=laps_df,
    x='traffic_regime',
    y='time_loss_ms',
    order=['LIGHT_TRAFFIC', 'MODERATE_TRAFFIC', 'HEAVY_TRAFFIC', 'CLOSE_FOLLOWING']
)
sns.swarmplot(
    data=laps_df.sample(1000),  # Subsample for visibility
    x='traffic_regime',
    y='time_loss_ms',
    color='black',
    alpha=0.3,
    size=2
)
plt.axhline(y=0, color='red', linestyle='--', label='Clean Air Baseline')
plt.ylabel("Time Loss (ms)")
plt.title("Traffic-Induced Lap Time Penalties")
```

---

#### Viz 2: Overtake Success Probability Surface
**Type**: 2D contour plot

```python
from scipy.interpolate import griddata

# Mesh grid of (gap, tyre_delta)
gap_range = np.linspace(0.5, 2.0, 50)
tyre_delta_range = np.linspace(-10, 20, 50)
gap_grid, tyre_grid = np.meshgrid(gap_range, tyre_delta_range)

# Predict overtake probability
X_pred = np.column_stack([
    gap_grid.ravel(),
    tyre_grid.ravel(),
    np.ones(gap_grid.size),  # DRS available
    np.zeros(gap_grid.size)  # Equal compound
])
X_pred_scaled = scaler.transform(X_pred)
probs = model.predict_proba(X_pred_scaled)[:, 1].reshape(gap_grid.shape)

# Plot
plt.contourf(gap_grid, tyre_grid, probs, levels=20, cmap='RdYlGn')
plt.colorbar(label='Overtake Success Probability')
plt.xlabel("Gap to Car Ahead (s)")
plt.ylabel("Tyre Age Advantage (laps)")
plt.title("Overtake Feasibility Surface (DRS Available)")
```

---

#### Viz 3: Track Bottleneck Maps
**Type**: Track layout with density overlay

```python
def plot_bottleneck_map(track_name, bottleneck_data, circuit_coords):
    """
    Show where cars bunch up most.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Circuit outline
    ax.plot(circuit_coords['x'], circuit_coords['y'], 'k-', linewidth=3, alpha=0.5)
    
    # Bottleneck zones (color = density)
    for idx, zone in bottleneck_data.iterrows():
        circle = Circle(
            (zone['center_x'], zone['center_y']),
            radius=zone['avg_density'] * 10,
            color='red',
            alpha=0.3
        )
        ax.add_patch(circle)
    
    plt.title(f"{track_name} - Traffic Bottlenecks")
    plt.axis('equal')
```

---

## Phase 4: Modeling (Optional)

### Model 1: Traffic Impact Predictor
**Type**: Random Forest Regressor

**Target**: `time_loss_ms` (per lap)

**Features**:
- `gap_to_ahead_s`
- `proximity_duration_s`
- `tyre_age`
- `track_regime` (one-hot)
- `drs_available`

**Use Case**: Real-time race simulation

---

### Model 2: Overtake Probability Estimator
**Type**: XGBoost Classifier

**Target**: `overtake_success` (0/1)

**Features**: (as in Analysis 3 + more)

**Use Case**: Strategy tool - "can driver X pass driver Y?"

---

## Phase 5: Results & Documentation

### 5.1 Notebook Deliverables

1. **`domain2_traffic_eda.ipynb`**
   - Traffic regime distributions
   - Performance loss statistics

2. **`domain2_overtake_analysis.ipynb`**
   - Overtake detection validation
   - Success factor analysis

3. **`domain2_spatial_analysis.ipynb`**
   - Overtake zone identification
   - Track bottleneck maps

4. **`domain2_visualizations.ipynb`**
   - All publication plots

### 5.2 Key Insights Document

**Template**:
```markdown
# Traffic & Overtaking Analysis Results

## Traffic Impact
- Average time loss in CLOSE_FOLLOWING: [X.X]s/lap
- Tracks most sensitive to traffic: [list]
- Traffic-induced degradation: [X%] higher than clean air

## Overtaking Patterns
- Total overtakes detected: [N] across [M] races
- Primary overtake zones per track: [visualizations]
- Key success factors:
  1. Gap < [X]s increases odds by [Y]x
  2. Tyre advantage > [Z] laps increases odds by [W]x

## Spatial Findings
- DRS effectiveness varies [X]% between tracks
- Bottleneck sectors identified: [list]
```

---

## Dependencies & Prerequisites

### Critical Blocker
**⚠️ MUST COMPLETE FIRST**: Position data ingestion (`ingest_positions`)

### Software Stack
```txt
# Add to requirements.txt:
scipy>=1.10.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
xgboost>=1.7.0  # For advanced models
```

### Data Dependencies
```
positions_raw (NEW - P0)
  ↓
lap_positions (Job 1)
  ↓
proximity_events (Job 2) ────┐
  ↓                           │
traffic_regimes (Job 3)       │
  ↓                           │
traffic_impact_summary ←──────┘
overtakes_featured
```

### Estimated Timeline
- **Phase 0 (Position Ingestion)**: 1-2 weeks ⚠️
- **Phase 1 (ETL)**: 2-3 weeks
- **Phase 2 (Features)**: 2-3 weeks
- **Phase 3 (Analysis)**: 3-4 weeks
- **Phase 4 (Modeling)**: 2-3 weeks (optional)
- **Phase 5 (Documentation)**: 1 week

**Total**: 11-16 weeks

---

## Success Metrics

1. **Data Completeness**:
   - ≥90% of laps classified into traffic regimes
   - ≥100 overtakes detected per season
   - Position data for ≥95% of race laps

2. **Analysis Validity**:
   - Traffic penalty coefficients statistically significant (p<0.05)
   - Overtake model accuracy ≥70%
   - Overtake zones align with domain knowledge (DRS zones, Turn 1s)

3. **Insights Quality**:
   - Clear track-specific overtaking profiles
   - Quantified traffic penalties per regime
   - Validated traffic-degradation relationship
