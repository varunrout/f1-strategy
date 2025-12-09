# F1 Strategy: FastF1 → SQLite Data Pipeline

A robust, script-driven data ingestion and feature engineering pipeline for Formula 1 analytics using FastF1 and SQLite.

## 🏎️ Overview

This project provides a complete data pipeline for F1 analytics:

1. **Data Ingestion**: Downloads F1 timing and telemetry data using FastF1
2. **Layered Storage**: Stores data in SQLite databases with bronze/silver architecture
3. **Feature Engineering**: Derives analytics-ready features from raw data

### Architecture

```
data/
├── raw.db              # Bronze layer: Raw FastF1 data
├── features_core.db    # Silver layer: Core reusable features
├── features_tyre.db    # (Reserved for domain-specific features)
├── features_traffic.db # (Reserved for domain-specific features)
└── features_xt.db      # (Reserved for domain-specific features)
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

# Or with pip in development mode
pip install -e .
```

## 🚀 Quick Start

### 1. Ingest a Single Session

```bash
# Ingest the 2023 Monaco Grand Prix race
python -m src.ingest.ingest_session --year 2023 --gp "Monaco" --session R

# Ingest qualifying session
python -m src.ingest.ingest_session --year 2023 --gp "Monaco" --session Q
```

### 2. Ingest an Entire Season

```bash
# Ingest all races and qualifying sessions from 2023
python -m src.ingest.ingest_season --year 2023 --sessions "Q,R"

# Ingest all session types (FP1, FP2, FP3, Q, R) for 2023
python -m src.ingest.ingest_season --year 2023

# Ingest specific GPs only
python -m src.ingest.ingest_season --year 2023 --gps "Monza,Silverstone,Spa"
```

### 3. Build Feature Tables

After ingesting raw data, build derived feature tables:

```bash
# Build laps with features (tyre age, fuel proxy, track status, etc.)
python -m src.features.build_laps_featured

# Build gap analysis (gaps to cars ahead/behind, DRS range, clean air)
python -m src.features.build_gaps_featured

# Build segment analysis (telemetry aggregated by track segments)
python -m src.features.build_segments_featured
```

## 📊 Database Schema

### Raw Database (`raw.db`)

**Bronze layer** containing raw FastF1 data:

- **sessions**: Session metadata (year, GP, track, session type)
- **laps_raw**: Lap-level timing data (lap times, sectors, compounds, stints)
- **telemetry_raw**: High-frequency telemetry (speed, throttle, brake, gear, DRS)
- **positions_raw**: Car position data (x, y, z coordinates)
- **weather_raw**: Weather time series (temperature, humidity, wind, rainfall)
- **race_control_raw**: Race control messages and flags
- **ingestion_log**: Audit trail of ingestion operations

### Features Database (`features_core.db`)

**Silver layer** with enriched, analytics-ready features:

- **laps_featured**: Enhanced lap data with derived features
  - Tyre age calculation
  - Fuel proxy (race completion fraction)
  - Track status categorization (GREEN, YELLOW, SC, VSC, RED)
  - In-lap/out-lap identification
  - Weather data joined by time
  
- **gaps_featured**: Gap analysis between cars
  - Gap to car ahead/behind (seconds)
  - Clean air flag (gap > 2.5s)
  - DRS range flag (gap < 1.0s)
  
- **segments_featured**: Track segment-level telemetry analysis
  - Speed statistics per segment (mean, max, min, entry, exit)
  - Throttle and brake usage
  - Time in segment
  - Enriched with tyre and fuel data

## 🔧 Command Reference

### Ingestion Scripts

#### `ingest_session.py`

Ingest a single F1 session.

```bash
python -m src.ingest.ingest_session \
  --year 2023 \
  --gp "Monza" \
  --session R \
  [--force] \
  [--db path/to/raw.db]
```

**Options:**
- `--year, -y`: Season year (required)
- `--gp, -g`: Grand Prix name (required)
- `--session, -s`: Session type - FP1, FP2, FP3, Q, S, R (required)
- `--force, -f`: Force re-ingestion if session exists
- `--db`: Custom database path

#### `ingest_season.py`

Batch ingest multiple sessions for a season.

```bash
python -m src.ingest.ingest_season \
  --year 2023 \
  [--gps "Monza,Silverstone"] \
  [--sessions "Q,R"] \
  [--force] \
  [--db path/to/raw.db]
```

**Options:**
- `--year, -y`: Season year (required)
- `--gps, -g`: Comma-separated GP names (omit for all GPs)
- `--sessions, -s`: Comma-separated session types (omit for FP1,FP2,FP3,Q,R)
- `--force, -f`: Force re-ingestion
- `--db`: Custom database path

### Feature Building Scripts

#### `build_laps_featured.py`

Build enhanced lap features.

```bash
python -m src.features.build_laps_featured \
  [--session-id 1] \
  [--raw-db path/to/raw.db] \
  [--features-db path/to/features_core.db]
```

**Options:**
- `--session-id, -s`: Process specific session only
- `--raw-db`: Path to raw database
- `--features-db`: Path to features database

#### `build_gaps_featured.py`

Build gap analysis features.

```bash
python -m src.features.build_gaps_featured \
  [--session-id 1] \
  [--raw-db path/to/raw.db] \
  [--features-db path/to/features_core.db]
```

#### `build_segments_featured.py`

Build segment-level telemetry features.

```bash
python -m src.features.build_segments_featured \
  [--session-id 1] \
  [--segments 20] \
  [--raw-db path/to/raw.db] \
  [--features-db path/to/features_core.db]
```

**Options:**
- `--segments, -n`: Number of segments per lap (default: 20)

## 📝 Usage Examples

### Example 1: Complete 2023 Season Analysis

```bash
# Step 1: Ingest all 2023 race and qualifying data
python -m src.ingest.ingest_season --year 2023 --sessions "Q,R"

# Step 2: Build all feature tables
python -m src.features.build_laps_featured
python -m src.features.build_gaps_featured
python -m src.features.build_segments_featured

# Now your databases are ready for analysis!
```

### Example 2: Quick Single Race Analysis

```bash
# Ingest Monaco 2023 race
python -m src.ingest.ingest_session --year 2023 --gp "Monaco" --session R

# Build features for this session only
python -m src.features.build_laps_featured --session-id 1
python -m src.features.build_gaps_featured --session-id 1
python -m src.features.build_segments_featured --session-id 1
```

### Example 3: Analyze Specific Tracks

```bash
# Compare street circuits from 2023
python -m src.ingest.ingest_season \
  --year 2023 \
  --gps "Monaco,Singapore,Las Vegas" \
  --sessions "Q,R"

python -m src.features.build_laps_featured
python -m src.features.build_gaps_featured
```

## 🔍 Querying the Data

### SQLite CLI

```bash
# Open raw database
sqlite3 data/raw.db

# Example queries
sqlite> SELECT year, gp_name, session_type, COUNT(*) as lap_count 
        FROM sessions s 
        JOIN laps_raw l ON s.session_id = l.session_id 
        GROUP BY s.session_id;

# Open features database
sqlite3 data/features_core.db

# Example: Find laps in clean air with soft tyres
sqlite> SELECT driver, lap_number, lap_time_s, compound, tyre_age_laps
        FROM laps_featured
        WHERE compound = 'SOFT' 
        AND session_id IN (
            SELECT session_id FROM gaps_featured 
            WHERE in_clean_air = 1
        );
```

### Python

```python
import sqlite3

# Connect to databases
raw_conn = sqlite3.connect('data/raw.db')
features_conn = sqlite3.connect('data/features_core.db')

# Query featured laps
query = """
SELECT driver, lap_number, lap_time_s, compound, tyre_age_laps, fuel_proxy
FROM laps_featured
WHERE session_id = 1
ORDER BY lap_time_s
LIMIT 10
"""

import pandas as pd
df = pd.read_sql_query(query, features_conn)
print(df)
```

### Attaching Multiple Databases

```python
import sqlite3

conn = sqlite3.connect('data/features_core.db')
conn.execute("ATTACH 'data/raw.db' AS raw;")

# Now you can query both databases
query = """
SELECT 
    f.driver,
    f.lap_number,
    f.lap_time_s,
    f.compound,
    r.track_status
FROM laps_featured f
JOIN raw.laps_raw r 
    ON f.session_id = r.session_id 
    AND f.driver = r.driver 
    AND f.lap_number = r.lap_number
WHERE f.session_id = 1
"""
```

## 🏗️ Project Structure

```
f1-strategy/
├── data/                      # SQLite databases (gitignored)
│   ├── raw.db
│   ├── features_core.db
│   ├── features_tyre.db
│   ├── features_traffic.db
│   └── features_xt.db
├── src/
│   ├── ingest/               # Data ingestion modules
│   │   ├── ingest_session.py
│   │   └── ingest_season.py
│   ├── features/             # Feature engineering modules
│   │   ├── build_laps_featured.py
│   │   ├── build_gaps_featured.py
│   │   └── build_segments_featured.py
│   └── utils/                # Shared utilities
│       ├── db.py
│       ├── logging_utils.py
│       ├── fastf1_utils.py
│       └── schemas.py
├── notebooks/                 # (Reserved for analysis notebooks)
├── cache/                     # FastF1 cache (gitignored)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 🎯 Design Principles

1. **Idempotent**: Re-run scripts safely without duplicating data
2. **Modular**: Clear separation between ingestion and feature engineering
3. **Efficient**: Chunked inserts, proper indexing, optimized SQLite settings
4. **Layered**: Bronze (raw) → Silver (features) architecture
5. **Auditable**: Ingestion logs track all operations
6. **Extensible**: Easy to add new feature tables and domain-specific databases

## 📚 Key Concepts

### Tyre Age Calculation

Tyre age is computed as the number of laps since the start of the current stint:

```python
tyre_age_laps = lap_number - first_lap_of_stint + 1
```

### Fuel Proxy

Fuel proxy estimates fuel load as a fraction of race completion:

```python
fuel_proxy = lap_number / total_race_laps
```

Lower values = heavier car (more fuel), higher values = lighter car (less fuel)

### Track Status Categories

Raw FastF1 track status codes are categorized:

- **GREEN**: Normal racing (status '1')
- **YELLOW**: Yellow flags (status '2')
- **SC**: Safety Car (status '4')
- **VSC**: Virtual Safety Car (status '6')
- **RED**: Red flag (status '5')
- **UNKNOWN**: Other statuses

### Gap Computation

Gaps between cars are computed using cumulative lap times and race positions:

```python
gap_to_ahead = current_driver_cumulative_time - ahead_driver_cumulative_time
```

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- Add more feature engineering modules
- Implement domain-specific feature databases (tyre, traffic, etc.)
- Add data validation and quality checks
- Optimize telemetry segment matching
- Add visualization utilities

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) - Excellent F1 data access library
- Formula 1 - For making timing data available