# F1 Strategy Analysis Notebooks

This directory is reserved for Jupyter notebooks for data analysis and visualization.

## Suggested Notebooks

- `01_data_exploration.ipynb` - Explore raw and featured data
- `02_lap_time_analysis.ipynb` - Analyze lap time distributions
- `03_tyre_degradation.ipynb` - Study tyre wear patterns
- `04_gap_analysis.ipynb` - Analyze gaps and overtaking opportunities
- `05_segment_analysis.ipynb` - Track segment performance comparison

## Getting Started

```python
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to databases
raw_conn = sqlite3.connect('../data/raw.db')
features_conn = sqlite3.connect('../data/features_core.db')

# Example: Load featured laps
laps_df = pd.read_sql_query("""
    SELECT * FROM laps_featured
    WHERE session_id = 1
""", features_conn)

# Analyze
print(laps_df.head())
```
