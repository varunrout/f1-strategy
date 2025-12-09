#!/usr/bin/env python
"""Initialize all database files with proper schemas."""
from pathlib import Path

from src.utils.schemas import (
    initialize_raw_db,
    initialize_features_core_db,
    initialize_placeholder_db
)

# Database paths
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_DB = DATA_DIR / "raw.db"
FEATURES_CORE_DB = DATA_DIR / "features_core.db"
FEATURES_TYRE_DB = DATA_DIR / "features_tyre.db"
FEATURES_TRAFFIC_DB = DATA_DIR / "features_traffic.db"
FEATURES_XT_DB = DATA_DIR / "features_xt.db"


def main():
    """Initialize all databases."""
    print("Initializing F1 Strategy databases...")
    
    print(f"Creating {RAW_DB}...")
    initialize_raw_db(str(RAW_DB))
    
    print(f"Creating {FEATURES_CORE_DB}...")
    initialize_features_core_db(str(FEATURES_CORE_DB))
    
    print(f"Creating {FEATURES_TYRE_DB} (placeholder)...")
    initialize_placeholder_db(str(FEATURES_TYRE_DB))
    
    print(f"Creating {FEATURES_TRAFFIC_DB} (placeholder)...")
    initialize_placeholder_db(str(FEATURES_TRAFFIC_DB))
    
    print(f"Creating {FEATURES_XT_DB} (placeholder)...")
    initialize_placeholder_db(str(FEATURES_XT_DB))
    
    print("\n✅ All databases initialized successfully!")
    print(f"\nDatabase files created in: {DATA_DIR}")
    print("  - raw.db (bronze layer)")
    print("  - features_core.db (silver layer)")
    print("  - features_tyre.db (placeholder)")
    print("  - features_traffic.db (placeholder)")
    print("  - features_xt.db (placeholder)")


if __name__ == "__main__":
    main()
