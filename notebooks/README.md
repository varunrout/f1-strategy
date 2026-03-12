# F1 Strategy Analysis Notebooks

## Notebooks (run in order)

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_data_exploration.ipynb` | Explore bronze/silver data, schema checks, basic distributions |
| 02 | `02_domain1_analysis.ipynb` | Tyre degradation EDA — stint extraction, curve fitting, factor analysis |
| 03 | `03_domain1_cluster_analysis.ipynb` | Cluster stints by driving style & track regime |
| 04 | `04_domain1_modeling.ipynb` | XGBoost degradation prediction model — train, evaluate, visualise |

## Prerequisites

```bash
# Install project deps (from repo root)
pip install -r requirements.txt

# Ingest at least one season
python -m src.ingest.ingest_parquet --year 2023 --gp "Monaco" --session R

# Build feature tables
python -m src.features.duckdb_features all

# Build degradation stints (needed for notebooks 02-04)
python -m src.features.domain1_degradation build
```

## Figure Output Convention

Notebooks save publication-quality figures to `docs/figures/<analysis_name>/`, not into this directory. Each notebook defines a `FIGURES_DIR` or `plots_dir` constant pointing there:

| Notebook | Figures saved to |
|----------|-----------------|
| 01 | *(inline only — no saved figures)* |
| 02 | *(inline only — no saved figures)* |
| 03 | `docs/figures/domain1_clustering/` |
| 04 | `docs/figures/domain1_modeling/` |

**Rule**: No `.png`, `.csv`, or `.parquet` files should live in `notebooks/`.

## Getting Started

```python
import duckdb
import pandas as pd

# Query silver-layer Parquet directly
con = duckdb.connect()
laps = con.execute("""
    SELECT * FROM read_parquet('data/lake/silver/laps_featured/**/*.parquet')
    LIMIT 1000
""").df()

laps.head()
```
