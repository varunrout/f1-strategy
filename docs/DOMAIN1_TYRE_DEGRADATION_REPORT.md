# Domain 1: Tyre Degradation Analysis Report

**F1 Race Strategy Project**  
**Analysis Period:** 2022-2024 F1 Seasons  
**Report Date:** December 2024

---

## Executive Summary

This report presents a comprehensive analysis of tyre degradation patterns in Formula 1 racing, examining 28,819 clean-air laps across 893 high-quality stints from 68 races over three seasons. The primary objective was to identify factors that predict tyre degradation and develop actionable insights for race strategy optimization.

### Key Findings

| Metric | Value |
|--------|-------|
| Total clean-air laps analyzed | 28,819 |
| Total stints extracted | 1,905 |
| High-quality stints (R² > 0.3) | 893 |
| Unique drivers | 27 |
| Unique circuits | 25 |
| Best predictive model R² | **56.6%** |

### Critical Discovery

**Teammate comparison is the single most powerful predictor of tyre degradation**, explaining 45.1% of variance alone. This dramatically outperforms all real-time sensor data combined (4.2%).

---

## 1. Methodology

### 1.1 Data Pipeline

```
FastF1 API → Bronze Parquet Files → Clean-Air Extraction → Degradation Fitting → Factor Analysis
```

### 1.2 Clean-Air Definition

A lap is classified as "clean air" when:
- Gap to car ahead > **2.5 seconds**, OR
- Driver is in **P1 position**

This filter removes laps affected by dirty air turbulence, ensuring we measure true tyre degradation.

### 1.3 Degradation Calculation

For each stint, we fit a linear regression:

$$\text{LapTime} = \text{Intercept} + (\text{DegradationRate} \times \text{TyreAge})$$

Where:
- **Degradation Rate** = slope (ms/lap)
- **Quality Filter**: Only stints with R² > 0.3 are retained

### 1.4 Fuel Correction

Raw lap times decrease during a stint due to fuel burn (~1.5-2kg/lap). We apply a correction of **+70 ms/lap** to isolate true tyre wear from fuel mass reduction.

---

## 2. Degradation by Compound

| Compound | Avg Degradation (ms/lap) | Median | Std Dev | Sample Size |
|----------|--------------------------|--------|---------|-------------|
| **SOFT** | +5.8 | +32.9 | 142.3 | 298 |
| **MEDIUM** | +16.6 | +39.8 | 156.7 | 342 |
| **HARD** | +9.6 | +35.7 | 138.2 | 253 |

### Insight
Soft compound shows lower average degradation because teams run shorter stints on softs, which don't accumulate significant wear. Medium compounds, used for longer stints, show the highest measured degradation.

---

## 3. Circuit Analysis

### 3.1 High Degradation Circuits

| Rank | Circuit | Avg Degradation (ms/lap) | Characteristics |
|------|---------|--------------------------|-----------------|
| 1 | **Spa-Francorchamps** | +139 | High-speed corners, long track, temperature variation |
| 2 | **Bahrain** | +96 | Abrasive surface, desert heat, sand particles |
| 3 | **Suzuka** | +70 | High-speed esses, sustained lateral load |
| 4 | **COTA** | +65 | Bumpy surface, temperature extremes |
| 5 | **Silverstone** | +58 | High-speed corners, variable weather |

### 3.2 Low/Negative Degradation Circuits

| Rank | Circuit | Avg Degradation (ms/lap) | Characteristics |
|------|---------|--------------------------|-----------------|
| 1 | **Monaco** | -335 | Track evolution dominates, rubber buildup |
| 2 | **Singapore** | -182 | Street circuit, track grip improves |
| 3 | **Melbourne** | -168 | Low abrasion, temperature stability |
| 4 | **Jeddah** | -97 | Street circuit, grip evolution |
| 5 | **Baku** | -82 | Long straights, limited corner stress |

### Insight
Street circuits often show **negative apparent degradation** because track surface grip improves through the race faster than tyres wear. This is a critical consideration for strategy - at Monaco, you may gain time staying out longer.

---

## 4. Year-Over-Year Trends

| Season | Races Analyzed | Stints | Avg Degradation (ms/lap) |
|--------|----------------|--------|--------------------------|
| **2022** | 22 | 287 | +16.1 |
| **2023** | 22 | 311 | +4.8 |
| **2024** | 24 | 295 | +14.4 |

### Insight
2023 showed notably lower degradation - teams had optimized setups for the ground-effect regulations after the 2022 learning year. 2024 saw a return to higher degradation with new tyre constructions.

---

## 5. Driver Analysis

### 5.1 Best Tyre Managers

| Rank | Driver | Avg Degradation (ms/lap) | Stints | Team(s) |
|------|--------|--------------------------|--------|---------|
| 1 | **Tsunoda (TSU)** | -52.4 | 32 | AlphaTauri/RB |
| 2 | **Leclerc (LEC)** | -28.1 | 45 | Ferrari |
| 3 | **Gasly (GAS)** | -25.7 | 38 | Alpine |
| 4 | **Alonso (ALO)** | -19.2 | 41 | Aston Martin |
| 5 | **Russell (RUS)** | -15.8 | 44 | Mercedes |

### 5.2 Highest Degradation Drivers

| Rank | Driver | Avg Degradation (ms/lap) | Stints | Notes |
|------|--------|--------------------------|--------|-------|
| 1 | **de Vries (DEV)** | +146.5 | 8 | Limited sample |
| 2 | **Drugovich (DOO)** | +121.3 | 5 | Test driver only |
| 3 | **Latifi (LAT)** | +111.0 | 18 | Williams struggles |
| 4 | **Colapinto (COL)** | +98.7 | 12 | Rookie learning |
| 5 | **Zhou (ZHO)** | +78.5 | 28 | Alfa Romeo |

### Insight
There is a **160+ ms/lap spread** between the best and worst tyre managers. This translates to significant strategic implications - over a 20-lap stint, that's over 3 seconds of race time.

---

## 6. Team Analysis

| Team | Avg Degradation (ms/lap) | Best Driver | Interpretation |
|------|--------------------------|-------------|----------------|
| **Alfa Romeo** | -49.6 | Bottas | Strong tyre philosophy |
| **AlphaTauri** | -43.1 | Tsunoda | Good car balance |
| **Mercedes** | -11.8 | Russell | Premium engineering |
| **Red Bull** | -8.2 | Verstappen | Race-winning setup |
| **Ferrari** | +12.4 | Leclerc | Historically tyre-hungry |
| **Kick Sauber** | +73.7 | - | Setup struggles |

### Insight
Team rankings are confounded by car characteristics - we cannot fully separate car setup effects from driver technique using aggregate data.

---

## 7. Factor Analysis: What Causes Degradation?

### 7.1 Original Factors Tested

We tested 18 factors across three categories:

#### Weather Factors
| Factor | Correlation (r) | R² Contribution |
|--------|-----------------|-----------------|
| Track Temperature | +0.117 | ~1.4% |
| Air Temperature | +0.111 | ~1.2% |
| Humidity | -0.115 | ~1.3% |
| **Combined** | - | **~2.0%** |

#### Telemetry / Driving Style
| Factor | Correlation (r) | R² Contribution |
|--------|-----------------|-----------------|
| Average Speed | -0.044 | <0.2% |
| Speed Variance | +0.019 | <0.1% |
| Throttle % | +0.005 | ~0% |
| Brake % | +0.038 | ~0.1% |
| **Combined** | - | **~0.7%** |

#### Position Data (G-Forces)
| Factor | Correlation (r) | R² Contribution |
|--------|-----------------|-----------------|
| Average G-Force | -0.080 | <0.7% |
| Peak G (P95) | -0.066 | <0.5% |
| Z Variance (kerbs) | +0.012 | ~0% |
| **Combined** | - | **~0.5%** |

### 7.2 Original Model Result

**Combined R² = 4.2%** (Weather + Telemetry + Position)

This means 95.8% of degradation variance was unexplained by real-time sensor data.

---

## 8. Derived Features: The Breakthrough

### 8.1 Computed Features

| Feature | Description | Correlation (r) |
|---------|-------------|-----------------|
| **Teammate Delta** | Driver deg vs teammate in same car | **0.671** |
| Circuit Baseline | Historical avg per circuit | 0.299 |
| Driver Baseline | Historical avg per driver | 0.167 |
| Team Baseline | Historical avg per team | 0.126 |
| Downforce Level | Circuit classification | ~0.08 |

### 8.2 Model Comparison

| Model | R² | Improvement vs Baseline |
|-------|-----|-------------------------|
| Original (Weather+Telemetry+Position) | **4.2%** | - |
| Derived Features (no teammate) | **12.3%** | 3x better |
| Teammate Delta Only | **45.1%** | 11x better |
| **Derived + Teammate Delta** | **56.6%** | **13.5x better** |

### 8.3 Why Teammate Delta Works

The teammate comparison isolates **pure driver skill** from car setup:

```
Same Team = Same Car + Same Setup + Same Tyres + Same Conditions
Different Degradation = PURE DRIVER TECHNIQUE

This single feature captures:
├─ Steering smoothness
├─ Trail braking precision
├─ Throttle application
├─ Racing line choices
└─ Tyre management mindset
```

---

## 9. The Unexplained Variance

### 9.1 What's Missing (43.4% unexplained)

Even our best model leaves 43% unexplained, likely due to:

#### Not in Public Data
- **Car Setup**: Downforce levels, suspension stiffness, camber angles
- **Tyre Pressures**: Starting pressures, pressure buildup
- **Tyre Temperatures**: Carcass (internal) vs surface temperature

#### Not Measurable
- **Track Surface**: Micro-abrasiveness, rubber buildup patterns
- **Driver Micro-Techniques**: Corner-by-corner adjustments
- **Strategic Decisions**: Intentional tyre saving modes

---

## 10. Strategic Recommendations

### 10.1 For Pre-Race Strategy

| Decision | Use This | Don't Use |
|----------|----------|-----------|
| Compound choice | Historical circuit degradation | Weather forecast |
| Expected stint length | Driver-specific patterns | Generic models |
| Relative performance | Teammate comparison data | Raw telemetry |

### 10.2 For In-Race Decisions

| Situation | Recommendation |
|-----------|----------------|
| Undercut timing | Base on FP2 long-run data + historical circuit character |
| Overcut potential | Higher on street circuits (negative deg tendency) |
| Driver instructions | Adjust based on teammate gap trends |

### 10.3 Prediction Hierarchy

1. **If teammate data available**: Use teammate comparison (45%+ accuracy)
2. **Circuit + historical baselines**: ~12% accuracy
3. **Real-time telemetry only**: ~4% accuracy (not recommended as sole input)

---

## 11. Data Quality Notes

### 11.1 Filters Applied

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| Clean air gap | > 2.5 seconds | Remove dirty air effects |
| R² quality | > 0.3 | Ensure reliable curve fits |
| Degradation bounds | |deg| < 500 ms/lap | Remove outliers |
| Minimum stint length | ≥ 5 laps | Statistical significance |

### 11.2 Data Sources

| Dataset | Records | Source |
|---------|---------|--------|
| Lap Times | 28,819 clean-air laps | FastF1 API |
| Telemetry | ~149 million points | FastF1 API |
| Position | ~152 million points | FastF1 API |
| Weather | ~33,000 records | FastF1 API |

---

## 12. Conclusions

### 12.1 Primary Findings

1. **Historical patterns dramatically outperform real-time data** for degradation prediction
2. **Teammate comparison is the most powerful single predictor** (r=0.671, 45% R² alone)
3. **Circuit character is consistent** across years - use historical baselines
4. **Weather and telemetry have minimal predictive value** individually (<2% each)

### 12.2 Practical Implications

- **Don't over-rely on real-time sensor data** for tyre strategy
- **Historical driver-circuit combinations** are your best predictive tool
- **FP2 long runs remain critical** - they're the best race-day proxy
- **Track evolution at street circuits** can flip conventional strategy

### 12.3 Future Work

1. **Corner-by-corner analysis**: Break down degradation by track section
2. **Tyre surface temperature modeling**: Better thermal prediction
3. **Machine learning ensemble**: Combine features with non-linear models
4. **Real-time updating**: Bayesian approach to update predictions during race

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Degradation Rate** | Rate at which lap times increase due to tyre wear (ms/lap) |
| **Clean Air** | Driving without aerodynamic interference from car ahead |
| **Stint** | Continuous run on the same set of tyres |
| **R²** | Coefficient of determination - variance explained by model |
| **Teammate Delta** | Difference in degradation between teammates at same race |
| **Fuel Correction** | Adjustment to lap times to account for decreasing car mass |

---

## Appendix B: Model Coefficients (Best Model)

| Feature | Coefficient (ms/lap per std) | Interpretation |
|---------|------------------------------|----------------|
| Teammate Delta | +156.2 | Strongest predictor |
| Circuit Baseline | +72.4 | Track character matters |
| Team Baseline | +45.1 | Car setup effect |
| Downforce Level | +18.3 | Aero configuration |
| Driver Baseline | -8.7 | Individual skill |

---

*Report generated from analysis in `notebooks/domain1_analysis.ipynb`*  
*Data source: FastF1 API covering 2022-2024 F1 seasons*
