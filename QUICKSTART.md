# Quick Start Guide

This guide will help you get started with the F1 Strategy pipeline.

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Internet connection (for downloading F1 data via FastF1)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/varunrout/f1-strategy.git
   cd f1-strategy
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize databases**:
   ```bash
   python init_databases.py
   ```

   This creates all necessary SQLite database files with proper schemas.

## Your First Data Ingestion

### Example 1: Ingest a Single Race

Let's ingest the 2023 Monaco Grand Prix:

```bash
# Ingest the race session
python -m src.ingest.ingest_session --year 2023 --gp "Monaco" --session R
```

This will:
- Download data from FastF1
- Store it in `data/raw.db`
- Create session metadata, laps, telemetry, weather, and race control data

### Example 2: Build Features

After ingesting data, build feature tables:

```bash
# Build enhanced laps with derived features
python -m src.features.build_laps_featured

# Build gap analysis
python -m src.features.build_gaps_featured

# Build segment analysis (telemetry-based)
python -m src.features.build_segments_featured
```

### Example 3: Query the Data

Use SQLite or Python to query the data:

```bash
sqlite3 data/features_core.db
```

```sql
-- Find fastest laps per driver
SELECT driver, MIN(lap_time_s) as fastest_lap, compound
FROM laps_featured
WHERE session_id = 1
GROUP BY driver
ORDER BY fastest_lap
LIMIT 10;

-- Analyze tyre performance
SELECT compound, 
       AVG(lap_time_s) as avg_lap_time,
       AVG(tyre_age_laps) as avg_tyre_age
FROM laps_featured
WHERE session_id = 1 AND lap_time_s IS NOT NULL
GROUP BY compound;
```

## Next Steps

### Ingest More Data

```bash
# Ingest entire 2023 season (races and qualifying)
python -m src.ingest.ingest_season --year 2023 --sessions "Q,R"

# Ingest specific tracks
python -m src.ingest.ingest_season --year 2023 --gps "Monza,Spa,Silverstone"
```

### Analyze in Python

```python
import sqlite3
import pandas as pd

# Connect to databases
conn = sqlite3.connect('data/features_core.db')

# Load laps data
df = pd.read_sql_query("""
    SELECT driver, lap_number, lap_time_s, compound, tyre_age_laps
    FROM laps_featured
    WHERE session_id = 1
""", conn)

# Analyze
print(df.describe())
```

### Create Visualizations

Use matplotlib or plotly to visualize:

```python
import matplotlib.pyplot as plt

# Plot lap times by tyre compound
for compound in df['compound'].unique():
    compound_data = df[df['compound'] == compound]
    plt.scatter(compound_data['tyre_age_laps'], 
               compound_data['lap_time_s'], 
               label=compound, alpha=0.6)

plt.xlabel('Tyre Age (laps)')
plt.ylabel('Lap Time (s)')
plt.legend()
plt.title('Tyre Degradation Analysis')
plt.show()
```

## Common Commands

### Ingestion

```bash
# Single session
python -m src.ingest.ingest_session --year 2023 --gp "Monza" --session R

# Season batch
python -m src.ingest.ingest_season --year 2023 --sessions "Q,R"

# Force re-ingest
python -m src.ingest.ingest_session --year 2023 --gp "Monza" --session R --force
```

### Feature Building

```bash
# All sessions
python -m src.features.build_laps_featured
python -m src.features.build_gaps_featured
python -m src.features.build_segments_featured

# Specific session
python -m src.features.build_laps_featured --session-id 1
```

## Troubleshooting

### FastF1 Cache Issues

If you encounter FastF1 cache issues, clear the cache:

```bash
rm -rf cache/
```

### Database Locked

If you get "database is locked" errors, ensure no other processes are accessing the database:

```bash
# Check for sqlite3 processes
ps aux | grep sqlite3

# If needed, kill the process
kill <PID>
```

### Memory Issues with Large Datasets

For large multi-season ingestions, process one season at a time:

```bash
for year in {2020..2023}; do
    python -m src.ingest.ingest_season --year $year --sessions "R"
    python -m src.features.build_laps_featured
done
```

## Getting Help

- Check the [README.md](README.md) for detailed documentation
- Review the code in `src/` directories for implementation details
- Open an issue on GitHub for bugs or feature requests

## What's Next?

- Build custom analysis notebooks in `notebooks/`
- Add domain-specific feature databases (tyre, traffic, etc.)
- Create ML models using the feature data
- Build visualizations and dashboards

Happy analyzing! 🏎️
