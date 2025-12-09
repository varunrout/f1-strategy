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
│  │  ingest_session.py      │   │  ingest_season.py         │       │
│  │  • Single session CLI   │   │  • Batch ingestion        │       │
│  │  • Data validation      │   │  • Progress tracking      │       │
│  │  • Error handling       │   │  • Multi-GP support       │       │
│  └─────────────────────────┘   └──────────────────────────┘       │
│                             │                                        │
│                             ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │            FastF1 Utilities & Logging                    │       │
│  │  • Cache management                                      │       │
│  │  • Session normalization                                 │       │
│  │  • Error handling wrappers                               │       │
│  └─────────────────────────────────────────────────────────┘       │
└────────────────────────────┼─────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   BRONZE LAYER (raw.db)                              │
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
│  ┌────────────────┬────────────────┬────────────────┐              │
│  │ positions_raw  │  weather_raw   │race_control_raw│              │
│  ├────────────────┼────────────────┼────────────────┤              │
│  │ • x, y, z      │ • air_temp     │ • message      │              │
│  │ • status       │ • track_temp   │ • flag         │              │
│  │                │ • humidity     │ • track_status │              │
│  └────────────────┴────────────────┴────────────────┘              │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │                  ingestion_log                           │       │
│  │  • Audit trail for all ingestion operations              │       │
│  └─────────────────────────────────────────────────────────┘       │
└────────────────────────────┼─────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING LAYER                            │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │        build_laps_featured.py                            │       │
│  │  • Tyre age calculation                                  │       │
│  │  • Fuel proxy estimation                                 │       │
│  │  • Track status categorization                           │       │
│  │  • In/out lap identification                             │       │
│  │  • Weather data joining                                  │       │
│  └─────────────────────────────────────────────────────────┘       │
│                             │                                        │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │        build_gaps_featured.py                            │       │
│  │  • Cumulative time calculation                           │       │
│  │  • Gap to ahead/behind                                   │       │
│  │  • Clean air detection (>2.5s)                           │       │
│  │  • DRS range detection (<1.0s)                           │       │
│  └─────────────────────────────────────────────────────────┘       │
│                             │                                        │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │      build_segments_featured.py                          │       │
│  │  • Lap segmentation (distance-based)                     │       │
│  │  • Speed aggregation per segment                         │       │
│  │  • Throttle/brake analysis                               │       │
│  │  • Entry/exit speed tracking                             │       │
│  └─────────────────────────────────────────────────────────┘       │
└────────────────────────────┼─────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│               SILVER LAYER (features_core.db)                        │
│                                                                      │
│  ┌────────────────────────────────────────────────────────┐        │
│  │              laps_featured                              │        │
│  │  • All lap data + derived features                      │        │
│  │  • tyre_age_laps, fuel_proxy, track_status_cat         │        │
│  │  • is_inlap, is_outlap, weather conditions             │        │
│  └────────────────────────────────────────────────────────┘        │
│                             │                                        │
│  ┌────────────────────────────────────────────────────────┐        │
│  │              gaps_featured                              │        │
│  │  • gap_to_ahead_s, gap_to_behind_s                     │        │
│  │  • in_clean_air, within_drs                            │        │
│  └────────────────────────────────────────────────────────┘        │
│                             │                                        │
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
│                 DOMAIN-SPECIFIC LAYERS (Future)                      │
│                                                                      │
│  ┌──────────────────┬──────────────────┬───────────────────┐       │
│  │ features_tyre.db │features_traffic.db│ features_xt.db   │       │
│  │                  │                    │                  │       │
│  │ • Degradation    │ • Overtakes       │ • Strategy       │       │
│  │ • Compounds      │ • DRS zones       │ • Pit stops      │       │
│  │ • Wear patterns  │ • Traffic impact  │ • Race pace      │       │
│  └──────────────────┴──────────────────┴───────────────────┘       │
└────────────────────────────┼─────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ANALYSIS & ML LAYER                               │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  • Jupyter Notebooks                                     │       │
│  │  • Python Analysis Scripts                               │       │
│  │  • SQL Queries                                           │       │
│  │  • Visualization & Dashboards                            │       │
│  │  • Machine Learning Models                               │       │
│  └─────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘

UTILITIES (Supporting All Layers):
┌─────────────────────────────────────────────────────────────────────┐
│  • db.py: Database connection & operations                           │
│  • logging_utils.py: Consistent logging                             │
│  • fastf1_utils.py: FastF1 API wrappers                            │
│  • schemas.py: Database DDL definitions                             │
└─────────────────────────────────────────────────────────────────────┘

KEY FEATURES:
• Idempotent: Re-run safely without duplicates
• Modular: Clean separation of concerns
• Performant: Bulk inserts, proper indexes, WAL mode
• Auditable: Complete ingestion logs
• Extensible: Easy to add new features
• Secure: Parameterized queries, no vulnerabilities
```

## Data Flow

1. **Ingestion**: FastF1 → Bronze Layer (raw.db)
   - Downloads session data
   - Stores in raw tables
   - Logs all operations

2. **Feature Engineering**: Bronze → Silver Layer (features_core.db)
   - Reads from raw.db
   - Computes derived features
   - Stores in feature tables

3. **Analysis**: Query Silver Layer
   - SQL queries
   - Python/Pandas analysis
   - Visualization
   - ML modeling

## CLI Commands

```bash
# Ingestion
python -m src.ingest.ingest_session --year 2023 --gp "Monaco" --session R
python -m src.ingest.ingest_season --year 2023 --sessions "Q,R"

# Feature Building
python -m src.features.build_laps_featured
python -m src.features.build_gaps_featured
python -m src.features.build_segments_featured
```
