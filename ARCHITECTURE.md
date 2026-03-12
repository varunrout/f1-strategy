# F1 Strategy Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                    FastF1 API                             │      │
│  │  (Official F1 Timing & Telemetry Data)                   │      │
│  └──────────────────────────────────────────────────────────┘      │
│                            │                                         │
└────────────────────────────┼─────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                                   │
│                                                                      │
│  ┌─────────────────────────┐   ┌──────────────────────────┐       │
│  │  ingest_parquet.py      │   │  background_runner.py     │       │
│  │  • Single-session CLI   │   │  • Batch async ingestion  │       │
│  │  • FastF1 → Parquet     │   │  • Job tracking (JSON)    │       │
│  │  • Data validation      │   │  • Progress UI (rich)     │       │
│  │  • Idempotent (skip/    │   │  • Multi-GP support       │       │
│  │    force)               │   │                           │       │
│  └─────────────────────────┘   └──────────────────────────┘       │
│                             │                                        │
│                             ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │            Shared Utilities                              │       │
│  │  • fastf1_utils.py  – cache management, API wrappers    │       │
│  │  • parquet_writer.py – hive-partitioned Parquet I/O     │       │
│  │  • logging_utils.py – consistent rich logging           │       │
│  └─────────────────────────────────────────────────────────┘       │
└────────────────────────────┼─────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              BRONZE LAYER  (data/lake/bronze/)                       │
│              Hive-partitioned Parquet files                          │
│              Partitioned by: year / gp_name / session_type          │
│                                                                      │
│  ┌────────────────┬────────────────┬────────────────┐              │
│  │   sessions     │   laps_raw     │ telemetry_raw  │              │
│  ├────────────────┼────────────────┼────────────────┤              │
│  │ • session_id   │ • driver       │ • speed        │              │
│  │ • year         │ • lap_number   │ • throttle     │              │
│  │ • gp_name      │ • lap_time     │ • brake        │              │
│  │ • track_name   │ • sectors      │ • gear         │              │
│  │ • session_type │ • compound     │ • drs          │              │
│  └────────────────┴────────────────┴────────────────┘              │
│                                                                      │
│  ┌────────────────┬────────────────┐                               │
│  │  weather_raw   │race_control_raw│                               │
│  ├────────────────┼────────────────┤                               │
│  │ • air_temp     │ • message      │                               │
│  │ • track_temp   │ • flag         │                               │
│  │ • humidity     │ • track_status │                               │
│  └────────────────┴────────────────┘                               │
└────────────────────────────┼─────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING LAYER                            │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │     duckdb_features.py  (DuckDB SQL engine)              │       │
│  │                                                          │       │
│  │  Subcommands: laps | gaps | segments | all               │       │
│  │                                                          │       │
│  │  laps_featured:                                          │       │
│  │    • Tyre age, fuel proxy, track status categorisation   │       │
│  │    • In/out lap flags, weather join                      │       │
│  │                                                          │       │
│  │  gaps_featured:                                          │       │
│  │    • Cumulative time, gap to ahead/behind                │       │
│  │    • Clean-air detection (>2.5 s), DRS range (<1.0 s)   │       │
│  │                                                          │       │
│  │  segments_featured:                                      │       │
│  │    • Lap segmentation (distance-based, default 20)       │       │
│  │    • Speed / throttle / brake stats per segment          │       │
│  │    • Entry/exit speeds, time in segment                  │       │
│  └─────────────────────────────────────────────────────────┘       │
└────────────────────────────┼─────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              SILVER LAYER  (data/lake/silver/)                       │
│              Hive-partitioned Parquet files                          │
│                                                                      │
│  ┌────────────────────────────────────────────────────────┐        │
│  │              laps_featured                              │        │
│  │  • All lap data + derived features                      │        │
│  │  • tyre_age_laps, fuel_proxy, track_status_cat         │        │
│  │  • is_inlap, is_outlap, weather conditions             │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                      │
│  ┌────────────────────────────────────────────────────────┐        │
│  │              gaps_featured                              │        │
│  │  • gap_to_ahead_s, gap_to_behind_s                     │        │
│  │  • in_clean_air, within_drs                            │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                      │
│  ┌────────────────────────────────────────────────────────┐        │
│  │           segments_featured                             │        │
│  │  • Track divided into segments (default: 20)           │        │
│  │  • Speed, throttle, brake stats per segment            │        │
│  │  • Entry/exit speeds, time in segment                  │        │
│  └────────────────────────────────────────────────────────┘        │
└────────────────────────────┼─────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 DOMAIN ANALYSIS LAYER                                │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  domain1_degradation.py                                  │       │
│  │  • Clean-air stint extraction                            │       │
│  │  • Degradation curve fitting (linear + polynomial)       │       │
│  │  • Stint quality scoring                                 │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  ML Models  (src/models/)                                │       │
│  │  • degradation_model.py – XGBoost quantile regression    │       │
│  │  • feature_builder.py   – ML feature construction        │       │
│  │  • Trained artefacts in data/models/                     │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  Notebooks  (notebooks/)                                 │       │
│  │  • data_exploration.ipynb                                │       │
│  │  • domain1_analysis.ipynb                                │       │
│  │  • domain1_cluster_analysis.ipynb                        │       │
│  │  • domain1_modeling.ipynb                                │       │
│  └─────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Ingestion**: FastF1 → Bronze Layer (Parquet)
   - Downloads session data via FastF1 API
   - Writes hive-partitioned Parquet to `data/lake/bronze/`
   - Cached API responses in `cache/`

2. **Feature Engineering**: Bronze → Silver Layer (Parquet via DuckDB)
   - DuckDB reads Bronze Parquet files
   - Computes derived features in SQL
   - Writes Silver Parquet to `data/lake/silver/`

3. **Analysis**: Query Silver Layer
   - DuckDB queries on Parquet files
   - Domain-specific Python pipelines
   - Jupyter notebook analysis
   - XGBoost ML models

## CLI Commands

```bash
# Ingest a single session
python -m src.ingest.ingest_parquet --year 2023 --gp "Monaco" --session R

# Batch ingest a full season
python -m src.ingest.background_runner --year 2023 --sessions "Q,R"

# Build all feature tables
python -m src.features.duckdb_features all

# Build individual feature tables
python -m src.features.duckdb_features laps
python -m src.features.duckdb_features gaps
python -m src.features.duckdb_features segments
```

## Key Properties

- **Idempotent**: Re-run safely without duplicates (skip existing, `--force` to overwrite)
- **Modular**: Clean separation of ingestion → features → analysis
- **Performant**: Parquet columnar storage, DuckDB analytical queries
- **Extensible**: Add new feature tables or domain analyses independently
- **Auditable**: Structured logging with rich console output
