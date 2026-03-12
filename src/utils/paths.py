"""
Utility path constants for the F1 Strategy Analytics project.

All modules should import from here to ensure consistent path resolution
regardless of the working directory from which scripts are executed.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# Data lake layers
BRONZE = DATA_ROOT / "lake" / "bronze"
SILVER = DATA_ROOT / "lake" / "silver"
MODELS = DATA_ROOT / "models"
CACHE = DATA_ROOT / "cache" / "fastf1"

# Domain-specific silver paths
DOMAIN1_SILVER = SILVER / "domain1"

# Bronze sub-directories
BRONZE_LAPS = BRONZE / "laps_raw"
BRONZE_POSITIONS = BRONZE / "positions_raw"
BRONZE_RACE_CONTROL = BRONZE / "race_control_raw"

# Model sub-directories
DOMAIN1_MODELS = MODELS / "domain1"


def ensure_dirs() -> None:
    """Create all required data directories if they do not exist."""
    for path in (
        BRONZE_LAPS,
        BRONZE_POSITIONS,
        BRONZE_RACE_CONTROL,
        DOMAIN1_SILVER,
        DOMAIN1_MODELS,
        CACHE,
    ):
        path.mkdir(parents=True, exist_ok=True)
