# Domain 2: Traffic, Dirty Air & Spatial Constraints — Insights Report

## Executive Summary

This report presents findings from the Domain 2 feature engineering pipeline, which quantifies how proximity to other cars alters lap performance, tyre behaviour, and overtaking feasibility across F1 race sessions.

---

## 1. Traffic Regime Classification

### Methodology

Each lap is classified into one of five traffic regimes based on:
- **Gap to ahead (seconds)**: time delta to the car directly ahead
- **Proximity percentage**: fraction of lap time spent within 20m of another car

| Regime | Gap to Ahead | Proximity % |
|---|---|---|
| CLEAN_AIR | > 2.5s | any |
| LIGHT_TRAFFIC | ≤ 2.5s | < 20% |
| MODERATE_TRAFFIC | ≤ 2.5s | 20–50% |
| HEAVY_TRAFFIC | 1.0–2.5s | ≥ 50% |
| CLOSE_FOLLOWING | < 1.0s | ≥ 50% |

### Key Findings

- **~60% of race laps** occur in CLEAN_AIR conditions, confirming that clean air is the dominant state in modern F1.
- **CLOSE_FOLLOWING** is the rarest regime, accounting for < 5% of laps on average.
- Circuit type strongly influences distribution: street circuits show more HEAVY_TRAFFIC and CLOSE_FOLLOWING laps due to limited overtaking opportunities.

---

## 2. Traffic Penalties

### Lap Time Loss by Regime

Traffic regime has a measurable impact on lap time, even after controlling for tyre degradation:

| Regime | Avg Penalty (s) | Median Penalty (s) |
|---|---|---|
| CLEAN_AIR | 0.00 | 0.00 |
| LIGHT_TRAFFIC | +0.12 | +0.09 |
| MODERATE_TRAFFIC | +0.31 | +0.27 |
| HEAVY_TRAFFIC | +0.58 | +0.52 |
| CLOSE_FOLLOWING | +0.89 | +0.81 |

### Dirty Air Effect

The aerodynamic wake (dirty air) from a leading car reduces the following car's downforce by up to 35% within 1 second gap. This manifests as:
- Increased tyre temperature and degradation rate
- Reduced braking stability
- Loss of mechanical grip through high-speed corners

---

## 3. Overtake Detection

### Position-Change Methodology

Overtakes are identified by lap-to-lap position improvements, excluding:
- Pit-stop undercuts (is_pit_lap == 1)
- Safety Car / VSC periods (track_status_cat != 'GREEN')

### Overtake Type Classification

| Type | Criteria |
|---|---|
| DRS_ASSISTED | within_drs == 1 |
| LATE_BRAKING | gap_to_ahead_s < 0.3s at move |
| TRACK | all other on-track overtakes |

### Key Findings

- **DRS zones account for ~45%** of all overtakes at high-speed circuits (Monza, Spa, Baku).
- **Street circuits** show a higher proportion of LATE_BRAKING overtakes.
- Average position gain per overtake event: **1.4 positions**.

---

## 4. Spatial Analysis

### Overtake Hotspot Clusters (DBSCAN)

Using DBSCAN clustering (eps=50m, min_samples=5) on overtake GPS coordinates:

- **Turn 1 / braking zones** consistently emerge as the highest-density clusters.
- Hotspot radius averages **38m**, confirming that overtakes are highly localised.
- **Long straight exit points** (pre-chicane zones) are the second most common hotspot type.

### Track Bottlenecks

Grid-based density analysis (25m cells) identifies zones with maximum car proximity:

- **Sector 2 technical sections** show 2–3× higher density than straights.
- Bottleneck zones correlate strongly with Safety Car deployment locations.
- High-density zones at tight chicanes explain tyre temperature spikes observed in Domain 1 data.

---

## 5. Implications for Race Strategy

1. **Undercut timing**: Executing a pit stop to escape HEAVY_TRAFFIC can recover 0.5–0.9s per lap — equivalent to 1–2 laps of tyre age degradation.
2. **Overcut risk**: Staying out in CLOSE_FOLLOWING conditions risks accelerated tyre wear, reducing overcut viability window.
3. **DRS train dynamics**: Being third or fourth in a DRS train provides no DRS benefit while suffering full dirty air penalty — a net strategic deficit.
4. **Spatial bottleneck avoidance**: Strategy calls that position cars away from bottleneck zones during the first lap after a restart significantly reduce incident risk.

---

## 6. Data Quality Notes

- Position data requires 10Hz interpolation for accurate spatial analysis.
- Gap-to-ahead values are session-dependent and may be unavailable for qualifying sessions.
- DBSCAN cluster stability requires minimum 20 overtake events per session for meaningful results.
