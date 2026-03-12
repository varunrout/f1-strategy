#!/usr/bin/env python
"""Initialize SQLite database files with proper schemas.

Note: The primary data pipeline uses Parquet (data/lake/), not SQLite.
This script is only needed if you want to use the legacy SQLite path.

Run from project root: python scripts/init_databases.py
"""
from pathlib import Path

from src.utils.schemas import (
    initialize_raw_db,
    initialize_features_core_db,
)

# Database paths (relative to project root, not scripts/)
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_DB = DATA_DIR / "raw.db"
FEATURES_CORE_DB = DATA_DIR / "features_core.db"


def main():
    """Initialize SQLite databases."""
    print("Initializing F1 Strategy SQLite databases...")
    
    print(f"Creating {RAW_DB}...")
    initialize_raw_db(str(RAW_DB))
    
    print(f"Creating {FEATURES_CORE_DB}...")
    initialize_features_core_db(str(FEATURES_CORE_DB))
    
    print("\n✅ Databases initialized successfully!")
    print(f"\nDatabase files created in: {DATA_DIR}")
    print("  - raw.db (bronze layer)")
    print("  - features_core.db (silver layer)")


if __name__ == "__main__":
    main()
