# Domain 1: Tyre Degradation Model Results

## Model Summary

**Version**: v1.0  
**Training Date**: 2024  
**Framework**: XGBoost

### Data Split
| Dataset | Years | Stints | Purpose |
|---------|-------|--------|---------|
| Training | 2022-2023 | 573 | Model fitting |
| Validation | 2022-2023 | 115 | Hyperparameter tuning |
| Test | 2024 | 313 | Final evaluation |

---

## Performance Metrics

### Test Set (2024 Holdout)
| Metric | Value |
|--------|-------|
| **MAE** | 100.89 ms/lap |
| **RMSE** | 135.52 ms/lap |
| **R²** | 0.666 |

### Improvement Over Baselines
| Baseline Model | MAE Improvement |
|----------------|-----------------|
| Circuit Mean | +34.3% |
| Driver Baseline | +35.1% |
| Linear (baselines only) | +33.5% |
| Linear (with teammate delta) | +21.5% |

### Performance by Circuit (2024)
| Circuit | R² | MAE (ms/lap) |
|---------|-----|--------------|
| Abu Dhabi | 0.867 | 57 |
| Japan | 0.825 | 66 |
| Australia | 0.807 | 68 |
| Emilia Romagna | 0.779 | 75 |
| Austria | 0.741 | 77 |
| Singapore | 0.626 | 77 |
| Hungary | 0.790 | 92 |

---

## Feature Importance

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | **shape_cluster** | 0.368 | Degradation curve shape classification (K=10 clusters) |
| 2 | **teammate_delta** | 0.244 | Relative deg vs teammate same stint |
| 3 | compound_circuit_baseline | 0.053 | Historical avg deg for compound at circuit |
| 4 | num_laps | 0.052 | Stint length in laps |
| 5 | driver_baseline_deg | 0.050 | Driver's historical avg degradation |
| 6 | circuit_baseline_deg | 0.046 | Circuit's historical avg degradation |
| 7 | driver_cluster | 0.046 | Driver style classification (K=7 clusters) |
| 8 | team_baseline_deg | 0.045 | Team's historical avg degradation |
| 9 | team_cluster | 0.042 | Team performance cluster (K=3 clusters) |
| 10 | compound_encoded | 0.033 | Compound type (HARD/MEDIUM/SOFT) |

### Key Insight: Teammate Delta
The **teammate_delta** feature captures the relative degradation difference between teammates on the same stint. This single feature:
- Correlates 0.63 with target (highest single-feature correlation)
- Implicitly controls for car setup, strategy, and race conditions
- Explains ~45% of variance when used in linear regression alone

---

## Model Architecture

### Main Model (XGBoost)
```
n_estimators: 500
max_depth: 5
learning_rate: 0.03
subsample: 0.8
colsample_bytree: 0.8
```

### Quantile Models
Separate XGBoost models for uncertainty estimation:
- P10 (10th percentile - optimistic scenario)
- P50 (50th percentile - median)  
- P90 (90th percentile - pessimistic scenario)

**Coverage**: 57% of test points fall within P10-P90 interval (target: 80%)

---

## Saved Artifacts

Location: `data/models/`

| File | Description |
|------|-------------|
| `deg_model_v1_main.joblib` | Main XGBoost regressor |
| `deg_model_v1_q10.joblib` | P10 quantile model |
| `deg_model_v1_q50.joblib` | P50 quantile model |
| `deg_model_v1_q90.joblib` | P90 quantile model |
| `deg_model_v1_metadata.json` | Feature columns, metrics |

---

## Usage Example

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('data/models/deg_model_v1_main.joblib')

# Prepare features (14 columns)
features = ['circuit_baseline_deg', 'team_baseline_deg', 'driver_baseline_deg',
            'compound_circuit_baseline', 'teammate_delta', 'circuit_cluster',
            'team_cluster', 'driver_cluster', 'shape_cluster', 'stint',
            'num_laps', 'compound_encoded', 'downforce_class', 'position_tier']

# Predict
X = df[features]
predictions = model.predict(X)  # Degradation rate in seconds/lap
```

---

## Limitations & Future Work

### Known Limitations
1. **Wet conditions**: Limited training data for INTERMEDIATE/WET compounds
2. **Quantile coverage**: 57% coverage vs target 80% - intervals may be too narrow
3. **Shape cluster dependency**: Requires curve fitting before prediction

### Future Improvements
1. Incorporate track evolution (grip buildup across sessions)
2. Add fuel load as explicit feature (currently corrected out)
3. Train on larger historical dataset (2018-2024)
4. Ensemble with circuit-specific models
5. Real-time adaptation using Bayesian updating

---

## References

- Notebook: [domain1_modeling.ipynb](../notebooks/domain1_modeling.ipynb)
- Feature Engineering: [src/models/feature_builder.py](../src/models/feature_builder.py)
- Model Class: [src/models/degradation_model.py](../src/models/degradation_model.py)
- Clustering Analysis: [domain1_cluster_analysis.ipynb](../notebooks/domain1_cluster_analysis.ipynb)
