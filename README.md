# F1 Strategy: Tyre Degradation Analytics Pipeline

A data pipeline for Formula 1 analytics using FastF1, Parquet, and DuckDB — from raw telemetry ingestion to XGBoost-based tyre degradation prediction.

## 🏎️ Overview

This project provides an end-to-end F1 analytics pipeline:

1. **Data Ingestion**: Downloads F1 timing and telemetry data using FastF1
2. **Layered Storage**: Stores data in Parquet files with bronze/silver architecture
3. **Feature Engineering**: Derives analytics-ready features using DuckDB SQL
4. **Domain Analysis**: Tyre degradation curve fitting and clean-air stint extraction
5. **ML Modeling**: XGBoost degradation predictions with quantile uncertainty

### Architecture

```
data/lake/
├── bronze/              # Raw FastF1 data (Parquet, hive-partitioned)
│   ├── laps_raw/
│   ├── telemetry_raw/
│   ├── weather_raw/
│   ├── positions_raw/
│   ├── race_control_raw/
│   ├── circuit_info/
│   └── results_raw/
└── silver/              # Derived features
    ├── laps_featured/
    ├── gaps_featured/
    ├── segments_featured/
    └── domain1/         # Tyre degradation analysis
```

## 📦 Installation

### Requirements

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/varunrout/f1-strategy.git
cd f1-strategy

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## 🚀 Quick Start

### 1. Ingest a Single Session

```bash
# Ingest the 2023 Monaco Grand Prix race
python -m src.ingest.ingest_parquet --year 2023 --gp "Monaco" --session R

# Ingest qualifying session
python -m src.ingest.ingest_parquet --year 2023 --gp "Monaco" --session Q
```

### 2. Batch Ingest a Season

```bash
# Ingest all 2023 races (background, 4 parallel workers)
python -m src.ingest.background_runner start --year 2023 --sessions "Q,R"

# Ingest specific GPs in foreground
python -m src.ingest.background_runner start --year 2023 --gps "Monza,Silverstone,Spa" --foreground

# Check status of background jobs
python -m src.ingest.background_runner status
```

### 3. Build Feature Tables

After ingesting raw data, build derived feature tables using DuckDB:

```bash
# Build all features at once (laps, gaps, segments)
python -m src.features.duckdb_features all

# Or build individually
python -m src.features.duckdb_features laps
python -m src.features.duckdb_features gaps
python -m src.features.duckdb_features segments
```

### 4. Build Domain 1: Tyre Degradation

```bash
# Extract clean-air stints and fit degradation curves
python -m src.features.domain1_degradation build

# Show stats about existing degradation data
python -m src.features.domain1_degradation stats
```

## 📊 Data Schema

### Bronze Layer (`data/lake/bronze/`)

Raw FastF1 data stored as hive-partitioned Parquet files (`year=YYYY/gp=GP_Name/session=R/`):

| Table | Contents |
|-------|----------|
| `laps_raw` | Lap-level timing data (lap times, sectors, compounds, stints) |
| `telemetry_raw` | High-frequency telemetry (speed, throttle, brake, gear, DRS) |
| `positions_raw` | Car position data (x, y, z coordinates) |
| `weather_raw` | Weather time series (temperature, humidity, wind, rainfall) |
| `race_control_raw` | Race control messages and flags |
| `circuit_info` | Corner locations with angles and distances |
| `results_raw` | Driver/team results, grid and finish positions |

### Silver Layer (`data/lake/silver/`)

Enriched, analytics-ready features:

| Table | Key Features |
|-------|-------------|
| `laps_featured` | Tyre age, fuel proxy, track status categorization, weather joins, deltas to best/PB |
| `gaps_featured` | Gap to ahead/behind, clean air flag, DRS range, under pressure flag |
| `segments_featured` | Per-segment speed stats, throttle/brake usage, entry/exit speeds |
| `domain1/` | Clean-air stints, degradation curves, stint summaries |

## 🔧 Command Reference

### Ingestion

```bash
# Single session
python -m src.ingest.ingest_parquet \
  --year 2023 --gp "Monza" --session R [--force] [--lake path/to/lake]

# Batch (background)
python -m src.ingest.background_runner start \
  --year 2023 [--gps "Monza,Spa"] [--sessions "Q,R"] [--workers 4]

# Job management
python -m src.ingest.background_runner status [--job-id JOB_ID]
python -m src.ingest.background_runner logs --job-id JOB_ID
python -m src.ingest.background_runner cancel --job-id JOB_ID
```

### Feature Engineering

```bash
# DuckDB-based features (fast, SQL-driven)
python -m src.features.duckdb_features all
python -m src.features.duckdb_features laps [--session SESSION_ID]
python -m src.features.duckdb_features gaps [--session SESSION_ID]
python -m src.features.duckdb_features segments [--session SESSION_ID] [--segments 20]

# Ad-hoc SQL query on the data lake
python -m src.features.duckdb_features query --sql "SELECT * FROM laps_raw LIMIT 10"
```

### Domain 1: Tyre Degradation

```bash
python -m src.features.domain1_degradation build [--year 2023] [--no-save]
python -m src.features.domain1_degradation stats
```

## 📝 Usage Examples

### Example 1: Complete Season Analysis

```bash
# Step 1: Ingest all 2023 race and qualifying data
python -m src.ingest.background_runner start --year 2023 --sessions "Q,R"

# Step 2: Build all feature tables
python -m src.features.duckdb_features all

# Step 3: Build degradation analysis
python -m src.features.domain1_degradation build --year 2023
```

### Example 2: Query with DuckDB

```python
import duckdb
conn = duckdb.connect()

# Read directly from Parquet
df = conn.execute("""
    SELECT driver, compound, AVG(lap_time_ms/1000.0) as avg_lap
    FROM read_parquet('data/lake/bronze/laps_raw/**/data.parquet', hive_partitioning=true)
    WHERE year = 2023 AND gp = 'Monaco_Grand_Prix' AND session = 'R'
    GROUP BY driver, compound
    ORDER BY avg_lap
""").fetchdf()
print(df)
```

### Example 3: Query Silver Layer

```python
import pandas as pd

# Read featured laps
laps = pd.read_parquet('data/lake/silver/laps_featured/')
print(laps.head())

# Read degradation stints
stints = pd.read_parquet('data/lake/silver/domain1/stints_degradation.parquet')
print(stints.describe())
```

## 🏗️ Project Structure

```
f1-strategy/
├── data/
│   ├── lake/                  # Parquet data lake (gitignored)
│   │   ├── bronze/            # Raw ingested data
│   │   └── silver/            # Derived features
│   ├── models/                # Trained ML models
│   └── jobs/                  # Background job tracking
├── src/
│   ├── ingest/                # Data ingestion
│   │   ├── ingest_parquet.py  # Single session → Parquet
│   │   └── background_runner.py # Batch/parallel ingestion
│   ├── features/              # Feature engineering
│   │   ├── duckdb_features.py # DuckDB SQL feature builder
│   │   └── domain1_degradation.py # Tyre degradation pipeline
│   ├── models/                # ML modeling
│   │   ├── degradation_model.py # XGBoost degradation model
│   │   └── feature_builder.py # Feature construction for ML
│   └── utils/                 # Shared utilities
│       ├── db.py              # Database connection helpers
│       ├── fastf1_utils.py    # FastF1 API wrappers
│       ├── logging_utils.py   # Logging setup
│       ├── parquet_writer.py  # Parquet read/write
│       └── schemas.py         # SQLite DDL definitions
├── notebooks/                 # Jupyter analysis notebooks
├── tests/                     # Unit tests
├── docs/                      # Documentation (see docs/README.md)
│   ├── roadmaps/              #   Project plans & research roadmaps
│   ├── reports/               #   Analysis results & write-ups
│   └── figures/               #   Generated plots & diagrams
├── cache/                     # FastF1 cache (gitignored)
├── scripts/                   # Utility scripts
│   └── init_databases.py      # SQLite DB initialization
├── requirements.txt
├── pyproject.toml
└── ARCHITECTURE.md
```

## 🎯 Design Principles

1. **Idempotent**: Re-run scripts safely — existing data is skipped unless `--force`
2. **Modular**: Clear separation: ingestion → features → domain analysis → ML
3. **Fast**: DuckDB SQL on Parquet for feature engineering (~100x vs Python loops)
4. **Layered**: Bronze (raw) → Silver (features) data lake architecture
5. **Extensible**: Easy to add new domains (traffic, strategy, etc.)

## 📚 Key Concepts

### Tyre Age
```
tyre_age_laps = lap_number - first_lap_of_stint + 1
```

### Fuel Proxy
```
fuel_proxy = lap_number / total_race_laps  (0.0 = full tank, 1.0 = empty)
```

### Track Status Categories
- **GREEN**: Normal racing (status '1')
- **YELLOW**: Yellow flags (status '2')
- **SC**: Safety Car (status '4')
- **VSC**: Virtual Safety Car (status '6')
- **RED**: Red flag (status '5')

### Clean Air & DRS
- **Clean air**: gap to car ahead > 2.5s
- **DRS range**: gap to car ahead < 1.0s

### Degradation Rate
Fuel-corrected linear slope of lap_time vs tyre_age in clean-air stints.

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Domain 2: Traffic impact analysis
- Domain 3: Race strategy simulation
- Data validation and quality checks
- Visualization dashboards

## 📄 License

MIT License — see LICENSE file for details.

## 🙏 Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) — Excellent F1 data access library
- Formula 1 — For making timing data available
