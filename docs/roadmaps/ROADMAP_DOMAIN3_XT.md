# Domain 3: Expected Threat (xT) for Formula 1 Racing
## Detailed Project Roadmap

> **Goal**: Quantify where on track meaningful race progress occurs and how that value changes with context

---

## Phase 1: Data Preparation & Infrastructure

### 1.1 Foundational Data (Dependency on Domain 2)

**Prerequisites**:
- ✅ Laps featured
- ✅ Segments featured  
- ✅ Gaps featured
- ⚠️ **REQUIRES Domain 2**: `lap_positions`, `overtakes_featured`, `proximity_events`

**New Data Needed**:

#### Job: `ingest_circuit_info`
**Priority**: P1 (Critical for xT)

**Implementation**:
```python
# In src/ingest/ingest_parquet.py

def prepare_circuit_info_df(session) -> pd.DataFrame:
    """
    Extract corner and circuit information from FastF1.
    
    Provides:
    - Corner coordinates (x, y)
    - Corner numbers and names
    - DRS zone boundaries
    - Track length
    - Safety car entry/exit points
    """
    try:
        circuit_info = session.get_circuit_info()
        
        if circuit_info is None:
            return pd.DataFrame()
        
        # Corners DataFrame
        if hasattr(circuit_info, 'corners') and circuit_info.corners is not None:
            corners = circuit_info.corners.copy()
            corners['session_id'] = session.session_info['session_key']
            corners['gp_name'] = session.event_name  # or similar
            
        # Extract other metadata
        circuit_data = {
            'session_id': session.session_info['session_key'],
            'gp_name': session.event_name,
            'track_length_m': circuit_info.track_length_m if hasattr(circuit_info, 'track_length_m') else None,
            'rotation_angle': circuit_info.rotation if hasattr(circuit_info, 'rotation') else 0,
            # More fields...
        }
        
        return corners  # Or combined circuit_data + corners
    
    except Exception as e:
        logger.warning(f"Circuit info unavailable: {e}")
        return pd.DataFrame()
```

**Expected Output**: `circuit_corners` table with columns:
- `session_id`, `gp_name`
- `corner_number`, `corner_name`
- `x`, `y` (coordinates)
- `radius` (optional)

---

### 1.2 Track Zone Definition

#### Job: `define_track_zones`
**Purpose**: Discretize track into meaningful analytical units

**Method 1: Corner-Based Zones** (Preferred for interpretability)

```python
def create_corner_zones(circuit_corners, extended_radius=100):
    """
    Create analytical zones around each corner.
    Zones represent:
    - Approach phase (entry brake zone)
    - Corner zone (apex ± extended_radius)
    - Exit phase (acceleration zone)
    """
    zones = []
    
    for idx, corner in circuit_corners.iterrows():
        # Approach zone: 150m before corner
        zones.append({
            'zone_id': f"{corner['corner_number']}_APPROACH",
            'corner_name': corner['corner_name'],
            'zone_type': 'APPROACH',
            'center_x': corner['x'],
            'center_y': corner['y'],
            'radius_m': 150
        })
        
        # Corner zone: ±100m around corner
        zones.append({
            'zone_id': f"{corner['corner_number']}_CORNER",
            'corner_name': corner['corner_name'],
            'zone_type': 'CORNER',
            'center_x': corner['x'],
            'center_y': corner['y'],
            'radius_m': extended_radius
        })
        
        # Exit zone: 100m after corner
        zones.append({
            'zone_id': f"{corner['corner_number']}_EXIT",
            'corner_name': corner['corner_name'],
            'zone_type': 'EXIT',
            'center_x': corner['x'],
            'center_y': corner['y'],
            'radius_m': 100
        })
    
    return pd.DataFrame(zones)
```

**Method 2: Distance-Based Zones** (Simpler, less interpretable)

```python
def create_distance_zones(track_length_m, zone_width_m=100):
    """
    Divide track into equal zones by distance traveled.
    """
    num_zones = int(track_length_m / zone_width_m)
    
    zones = []
    for i in range(num_zones):
        zones.append({
            'zone_id': i,
            'zone_type': 'DISTANCE_BIN',
            'distance_start_m': i * zone_width_m,
            'distance_end_m': (i + 1) * zone_width_m
        })
    
    return pd.DataFrame(zones)
```

**Output**: `track_zones` (reference table, per session/track)

---

## Phase 2: Feature Engineering

### 2.1 Threat Metrics

#### Job 1: `compute_overtake_probability_zones`
**Purpose**: For each zone, estimate probability that being here leads to position gain

**Method**: Bayesian update on historical overtake data

```python
def compute_zone_overtake_probability(positions_df, overtakes_df, zones_df):
    """
    For each zone, calculate:
    - P(overtake | position within zone)
    
    This is the core of xT - spatial position gain value.
    """
    
    zone_stats = []
    
    for zone_id in zones_df['zone_id'].unique():
        zone = zones_df[zones_df['zone_id'] == zone_id].iloc[0]
        
        # Find all time instances where a car was in this zone
        zone_instances = positions_df[
            (positions_df['zone_id'] == zone_id) &
            (positions_df['in_close_following'])  # Only count when opportunity exists
        ]
        
        if len(zone_instances) < 10:  # Minimum sample size
            continue
        
        # How many of these zone instances led to overtake in next lap?
        zone_instances['next_lap_overtook'] = (
            zone_instances.groupby(['session_id', 'driver'])['position'].shift(-1) 
            < zone_instances['position']
        )
        
        overtake_rate = zone_instances['next_lap_overtook'].sum() / len(zone_instances)
        
        zone_stats.append({
            'zone_id': zone_id,
            'overtake_probability': overtake_rate,
            'sample_size': len(zone_instances),
            'approach_speed_mean': zone_instances['speed_kph'].mean(),
            'confidence': min(len(zone_instances) / 50, 1.0)  # Cap at 1.0
        })
    
    return pd.DataFrame(zone_stats)
```

**Output**: `zone_threat` (Silver layer)

**Interpretation**:
- Zone with `overtake_probability=0.15` = 15% chance of position gain from this position
- This is the base xT value for that zone

---

#### Job 2: `compute_contextual_threat_modifiers`
**Purpose**: Adjust threat value based on race context

**Input**: `zone_threat` + `stint_degradation` (Domain 1) + `gaps_featured`

**Modifiers**:

1. **Tyre Freshness Modifier**:
   ```python
   # Fresh tyres unlock higher xT (more overtaking chances)
   def tyre_modifier(tyre_age, compound):
       if compound == 'SOFT':
           peak_window = (2, 8)  # Laps 2-8
       elif compound == 'MEDIUM':
           peak_window = (3, 12)
       else:  # HARD
           peak_window = (5, 20)
       
       if tyre_age < peak_window[0]:
           return 0.7  # Cold tyres, lower grip
       elif tyre_age > peak_window[1]:
           return 0.5  # Degraded, lower speed in turn-in
       else:
           return 1.0  # Peak window
   ```

2. **Traffic Density Modifier**:
   ```python
   # Congestion affects overtake feasibility
   def traffic_modifier(cars_within_3s):
       if cars_within_3s == 0:
           return 0.5  # Can't overtake in clean air
       elif cars_within_3s == 1:
           return 1.0  # Single target
       elif cars_within_3s == 2:
           return 0.8  # Multiple targets, harder
       else:
           return 0.6  # Heavy traffic, chaos
   ```

3. **DRS Zone Amplifier**:
   ```python
   # DRS zones unlock much higher xT
   def drs_modifier(in_drs_zone, drs_distance_to_leader):
       if not in_drs_zone:
           return 1.0
       elif drs_distance_to_leader < 1.0:
           return 3.0  # Very high threat in DRS zone
       else:
           return 1.5
   ```

**Combined Contextual xT**:
```python
contextual_threat = (
    base_threat *
    tyre_modifier(tyre_age, compound) *
    traffic_modifier(cars_within_3s) *
    drs_modifier(in_drs_zone, gap_to_leader)
)
```

**Output**: Add columns to `laps_featured`:
- `contextual_threat_value`
- `threat_modifier_tyre`
- `threat_modifier_traffic`
- `threat_modifier_drs`

---

#### Job 3: `accumulate_cumulative_threat`
**Purpose**: Track threat accumulation over a stint

**Method**: Cumulative sum of threat values

```sql
-- DuckDB query
CREATE TABLE lap_cumulative_threat AS
SELECT 
  session_id,
  driver,
  lap_number,
  contextual_threat_value,
  SUM(contextual_threat_value) OVER (
    PARTITION BY session_id, driver
    ORDER BY lap_number
  ) as cumulative_threat,
  -- Threat "spent" on actual overtake (1 if overtook, 0 else)
  CASE WHEN position_change < 0 THEN 1 ELSE 0 END as threat_converted,
  -- Threat accumulated but not used
  SUM(contextual_threat_value) OVER (
    PARTITION BY session_id, driver
    ORDER BY lap_number
  ) - LAG(SUM(contextual_threat_value), 1, 0) OVER (
    PARTITION BY session_id, driver
    ORDER BY lap_number
  ) as unrealized_threat
FROM laps_featured
```

**Output**: Add to `laps_featured`:
- `cumulative_threat`
- `threat_converted`
- `unrealized_threat`

---

### 2.2 Driver Style via xT

#### Job 4: `profile_driver_threat_generation`
**Purpose**: Understand how different driving styles accumulate threat

**Method**: Aggregate threat metrics by driver-track

```python
def driver_threat_profile(laps_df):
    """
    Profiles:
    1. Threat generation rate (threat per lap)
    2. Threat conversion efficiency (threat / actual overtakes)
    3. Optimal threat accumulation (when to attack)
    """
    
    profiles = laps_df.groupby(['driver', 'gp_name']).agg({
        'contextual_threat_value': 'mean',  # Avg threat generated per lap
        'cumulative_threat': 'max',         # Max threat built up
        'threat_converted': 'sum',          # Total successful overtakes
        'unrealized_threat': 'sum',         # Threat not converted
    }).reset_index()
    
    # Efficiency metrics
    profiles['threat_efficiency'] = (
        profiles['threat_converted'] / 
        (profiles['unrealized_threat'] + profiles['threat_converted'] + 1)  # +1 avoid div by 0
    )
    
    # Generation rate
    profiles['threat_per_lap'] = profiles['contextual_threat_value']
    
    return profiles
```

**Output**: `driver_threat_profiles` (Gold layer)

**Interpretation**:
- Driver A: 2.5 threat/lap, 20% conversion = aggressive accumulator but not great at converting
- Driver B: 1.8 threat/lap, 35% conversion = conservative but efficient when opportunities appear

---

## Phase 3: Analysis & Insights

### 3.1 Spatial Value Analysis

#### Analysis 1: Track-Specific xT Maps
**Research Question**: Where is overtaking actually feasible on each circuit?

**Visualization**: Corner-by-corner threat heatmap

```python
def create_xt_track_map(session_id, gp_name, zone_threat_df, circuit_corners):
    """
    Create publication-quality xT map for each circuit.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import matplotlib.patches as mpatches
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot circuit outline (would need circuit_coords data)
    # ax.plot(circuit_coords['x'], circuit_coords['y'], 'k-', linewidth=3)
    
    # Plot corners with threat color
    for idx, zone in zone_threat_df[zone_threat_df['zone_type'] == 'CORNER'].iterrows():
        circle = Circle(
            (zone['center_x'], zone['center_y']),
            radius=100,
            color=plt.cm.RdYlGn(zone['overtake_probability']),  # Green = high threat
            alpha=0.7,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(circle)
        
        # Label with corner name
        ax.text(
            zone['center_x'], zone['center_y'],
            zone['corner_name'],
            ha='center', va='center',
            fontsize=8, weight='bold'
        )
    
    # Color bar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.RdYlGn,
        norm=plt.Normalize(vmin=0, vmax=0.3)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Overtake Probability (xT)')
    
    plt.title(f"{gp_name} - Expected Threat Map")
    plt.axis('equal')
    plt.tight_layout()
    
    return fig
```

**Output Per Track**: 
- Which corners are high-threat zones (DRS, braking zones)
- Which corners are low-threat (high-speed, no passing)
- Track-specific overtaking narrative

---

#### Analysis 2: Context-Dependent xT Surfaces
**Research Question**: How does xT change with race conditions?

**Method**: 3D surface plot

```python
def plot_xt_surface(zone_id, modifiers_data):
    """
    Surface: xT value as function of (tyre_age, gap_to_leader)
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create mesh
    tyre_age_range = np.linspace(0, 30, 50)
    gap_range = np.linspace(0.5, 3.0, 50)
    tyre_mesh, gap_mesh = np.meshgrid(tyre_age_range, gap_range)
    
    # Compute xT for each point
    xt_values = np.zeros_like(tyre_mesh)
    for i, j in zip(*np.indices(tyre_mesh.shape)):
        ta = tyre_mesh[i, j]
        gap = gap_mesh[i, j]
        
        base_threat = modifiers_data[modifiers_data['zone_id'] == zone_id]['base_threat'].values[0]
        tyre_mod = tyre_modifier(ta, 'MEDIUM')
        traffic_mod = traffic_modifier(1 if gap < 2.5 else 0)
        
        xt_values[i, j] = base_threat * tyre_mod * traffic_mod
    
    # 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(tyre_mesh, gap_mesh, xt_values, cmap='viridis')
    ax.set_xlabel('Tyre Age (laps)')
    ax.set_ylabel('Gap to Ahead (s)')
    ax.set_zlabel('xT Value')
    ax.set_title(f"Contextual xT Surface - {zone_id}")
    
    return fig
```

**Output**: Understanding when overtakes are possible

---

### 3.2 Driver Comparison via xT

#### Analysis 3: Driver Threat Profiles
**Research Question**: How do different driving styles generate value?

**Method**: Radar plots + efficiency metrics

```python
def compare_driver_threat_profiles(drivers, profiles_df):
    """
    Radar plot showing:
    - Threat generation rate
    - Threat efficiency
    - Consistency
    - Risk-taking (aggressive threat accumulation)
    """
    import matplotlib.pyplot as plt
    
    categories = ['Threat Gen', 'Efficiency', 'Consistency', 'Aggressiveness']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for driver in drivers:
        driver_data = profiles_df[profiles_df['driver'] == driver]
        
        values = [
            driver_data['threat_per_lap'].mean() / profiles_df['threat_per_lap'].max(),
            driver_data['threat_efficiency'].mean(),
            1 - driver_data['threat_generation_std'].mean(),  # Lower std = more consistent
            driver_data['cumulative_threat'].mean() / profiles_df['cumulative_threat'].max()
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=driver)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    return fig
```

**Insights**:
- Driver A (Max): High gen, low efficiency (aggressive, risky)
- Driver B (Lewis): High efficiency, medium gen (patient, calculated)
- Driver C (Oscar): Balanced profile (all-rounder)

---

### 3.3 Strategic Implications

#### Analysis 4: xT-Based Race Narrative
**Research Question**: How did xT accumulation tell the story of the race?

**Visualization**: Time-series of cumulative threat per driver

```python
def plot_race_xxt_narrative(session_id, laps_df):
    """
    Line plot: cumulative threat over race distance (laps).
    Shows:
    - Who built threat fastest
    - Who converted when
    - Threat swings
    """
    
    for driver in laps_df['driver'].unique():
        driver_laps = laps_df[
            (laps_df['session_id'] == session_id) &
            (laps_df['driver'] == driver)
        ]
        
        plt.plot(
            driver_laps['lap_number'],
            driver_laps['cumulative_threat'],
            label=driver,
            marker='o',
            markersize=2
        )
        
        # Mark overtakes (where threat was converted)
        overtakes = driver_laps[driver_laps['threat_converted'] == 1]
        plt.scatter(
            overtakes['lap_number'],
            overtakes['cumulative_threat'],
            color='red',
            s=100,
            marker='*',
            zorder=5
        )
    
    plt.xlabel('Lap Number')
    plt.ylabel('Cumulative Expected Threat')
    plt.title('Race Narrative: xT Accumulation & Conversion')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
```

---

## Phase 4: Advanced Analysis (Optional)

### 4.1 Multi-Variable Interaction Models

#### Model: Logistic Regression on Overtake Conversion
**Target**: `threat_converted` (0/1)

**Features**:
- `cumulative_threat`
- `gap_to_leader`
- `tyre_age`
- `position`
- `track_zone_type`

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Interpretation: coefficients show which factors drive overtake success
```

---

### 4.2 Unsupervised Learning

#### Clustering: Identify Driving Philosophies
**Method**: K-means on threat profile features

```python
from sklearn.cluster import KMeans

# Feature matrix: threat generation, efficiency, risk-taking, consistency
X = profiles_df[['threat_per_lap', 'threat_efficiency', 'threat_consistency', 'aggressiveness']]

kmeans = KMeans(n_clusters=3)
profiles_df['philosophy'] = kmeans.fit_predict(X)

# Philosophies might emerge:
# - Cluster 0: "Aggressive attackers" (high gen, low eff)
# - Cluster 1: "Patient strikers" (low gen, high eff)
# - Cluster 2: "Balanced opportunists"
```

---

## Phase 5: Results & Documentation

### 5.1 Notebook Deliverables

1. **`domain3_xt_fundamentals.ipynb`**
   - Zone threat computation
   - Contextual modifiers explanation

2. **`domain3_track_specific_analysis.ipynb`**
   - xT maps per circuit
   - Overtake zone identification

3. **`domain3_driver_profiles.ipynb`**
   - Driver threat generation analysis
   - Radar plots and comparisons

4. **`domain3_race_narratives.ipynb`**
   - xT accumulation over race
   - Key moment identification

### 5.2 Key Results Template

```markdown
# Expected Threat (xT) Analysis Results

## Core Findings

1. **Threat Distribution Across Track**
   - High-threat zones: [list with probabilities]
   - Low-threat zones: [list]
   - DRS zones increase threat by: [X%]

2. **Context Effects**
   - Fresh tyres increase threat by: [X%]
   - Heavy traffic reduces threat by: [X%]
   - Per-lap threat variance: [range]

3. **Driver Philosophies**
   - [Driver A philosophy]: [description]
   - [Driver B philosophy]: [description]

## Strategic Implications

- Positions to target: [based on high-threat zones]
- Timing for attacks: [based on tyre/gap dynamics]
- Risk-reward profiles per track
```

---

## Dependencies & Critical Path

### Critical Prerequisites
1. **Position data ingestion** (Domain 2, Phase 0)
2. **Overtakes detected** (Domain 2, Phase 3)
3. **Circuit corners data** (Phase 1 of this domain)

### Data Flow
```
positions_raw ───────┐
                    ├──> lap_positions ──> zone_threat ──> contextual_threat ──> xT analysis
overtakes_featured ─┤
                    └──> zone_threat_modifiers
circuit_corners ────┴──> track_zones ─────┘
```

### Estimated Timeline
- **Phase 1 (Data Prep)**: 1-2 weeks
- **Phase 2 (Features)**: 2-3 weeks
- **Phase 3 (Analysis)**: 3-4 weeks
- **Phase 4 (Modeling)**: 2-3 weeks (optional)
- **Phase 5 (Documentation)**: 1 week

**Total**: 9-13 weeks

---

## Success Metrics

1. **Threat Value Quality**:
   - Zone threat probabilities align with domain knowledge (DRS zones > high-speed zones)
   - Contextual modifiers show expected trends (fresh tyres > degraded)
   - R² > 0.5 when predicting overtake probability from xT + context

2. **Analysis Insights**:
   - Clear track-specific overtaking narratives
   - Driver philosophies interpretable and consistent
   - xT effectively explains race outcomes (overtake timing/location)

3. **Reproducibility**:
   - Results consistent across seasons
   - Threat values normalize across compounds/tracks
   - Methods applicable to new seasons with minimal refit

---

## Deliverables

### Publications/Documentation
- [ ] Domain 3 analysis paper with visualizations
- [ ] Per-track xT guides for strategy teams
- [ ] Driver threat profile comparisons
- [ ] xT-enabled race simulation baseline

### Tools
- [ ] xT calculator (given position, tyres, gap)
- [ ] Interactive xT map viewer
- [ ] Race narrative replay (xT accumulation animation)
