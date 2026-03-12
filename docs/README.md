# Documentation

This directory contains all project documentation, organised into subdirectories by purpose.

## Structure

```
docs/
├── README.md                  ← You are here
├── roadmaps/                  # Project planning & research roadmaps
│   ├── README_ROADMAP.md          Executive summary of all three domains
│   ├── ROADMAP_MASTER_INTEGRATION.md  Master timeline & critical path
│   ├── ROADMAP_DOMAIN1_DEGRADATION.md Tyre degradation analysis plan
│   ├── ROADMAP_DOMAIN2_TRAFFIC.md     Traffic & spatial constraints plan
│   └── ROADMAP_DOMAIN3_XT.md         Expected Threat (xT) framework plan
├── reports/                   # Analysis results & findings
│   ├── DOMAIN1_TYRE_DEGRADATION_REPORT.md   Full degradation analysis report
│   ├── DOMAIN1_TYRE_DEGRADATION_REPORT.pdf  PDF export of the above
│   └── DOMAIN1_MODEL_RESULTS.md             XGBoost model performance summary
└── figures/                   # Publication-quality visualisations
    └── domain1_modeling/          Figures from the degradation model
        ├── 01_predicted_vs_actual_scatter.png
        ├── 02_residual_analysis.png
        ├── 03_circuit_comparison.png
        ├── ...
        └── 10_model_comparison.png
```

## Subdirectory Guide

### `roadmaps/`
Long-form research plans for each of the three analysis domains plus the master
integration timeline. Start with **README_ROADMAP.md** for a high-level overview,
then dive into individual domain roadmaps for detailed task breakdowns.

### `reports/`
Completed analysis write-ups with methodology, results, and conclusions.
New reports should follow the naming convention `DOMAIN<N>_<TOPIC>.md`.

### `figures/`
Output figures from notebooks and scripts, grouped by analysis area
(e.g. `domain1_modeling/`, `domain1_clustering/`). Referenced by reports
and notebooks. Keep one subfolder per analysis to avoid clutter.

## Adding New Documentation

| Type | Where to put it |
|------|----------------|
| Research / project plan | `roadmaps/` |
| Analysis results or write-up | `reports/` |
| Generated plot or diagram | `figures/<analysis_name>/` |
| Architecture or design doc | Project root (`ARCHITECTURE.md`) |
