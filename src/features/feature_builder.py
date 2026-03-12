"""
Domain 1 – Tyre Degradation: ML feature builder.

Transforms silver stint-degradation data into a flat, ML-ready feature matrix
and saves it to ``data/lake/silver/domain1/ml_features.parquet``.

Usage
-----
    python -m src.features.feature_builder
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils.paths import DOMAIN1_SILVER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger(__name__)

COMPOUND_ENCODING: dict[str, int] = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "INTERMEDIATE": 3,
    "WET": 4,
}

# Approximate calendar order (2023 season rounds 1-22) used as track-temp proxy.
# The round number is normalised to [0, 1] so models don't learn season ordering.
MAX_ROUNDS = 23  # 1-based → normalise by 22


# ---------------------------------------------------------------------------
# 1. Load silver stints
# ---------------------------------------------------------------------------

def load_stints_degradation() -> pd.DataFrame:
    """Load the stints_degradation Parquet produced by domain1_degradation.

    Returns
    -------
    pd.DataFrame
        Stint-level degradation metrics.

    Raises
    ------
    FileNotFoundError
        If the silver file is not yet present.
    """
    path = DOMAIN1_SILVER / "stints_degradation.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Silver file not found: {path}\n"
            "Run `python -m src.features.domain1_degradation` first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    log.info("Loaded stints_degradation: %d rows × %d cols", *df.shape)
    return df


def load_clean_air_laps() -> pd.DataFrame:
    """Load the clean_air_laps Parquet for driver consistency metrics."""
    path = DOMAIN1_SILVER / "clean_air_laps.parquet"
    if not path.exists():
        log.warning("clean_air_laps.parquet not found – consistency features will be NaN.")
        return pd.DataFrame()
    return pd.read_parquet(path, engine="pyarrow")


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def _driver_consistency(laps_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-driver-per-event coefficient of variation of clean lap times.

    A lower CV indicates a more consistent driver (less lap-to-lap variation
    unrelated to tyre deg).
    """
    if laps_df.empty:
        return pd.DataFrame(columns=["driver", "event", "consistency_cv"])

    group_cols = [c for c in ["Driver", "EventName"] if c in laps_df.columns]
    if not group_cols:
        return pd.DataFrame(columns=["driver", "event", "consistency_cv"])

    stats = (
        laps_df.groupby(group_cols)["LapTime"]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats["consistency_cv"] = stats["std"] / stats["mean"]
    rename = {}
    if "Driver" in stats.columns:
        rename["Driver"] = "driver"
    if "EventName" in stats.columns:
        rename["EventName"] = "event"
    return stats.rename(columns=rename)[["driver", "event", "consistency_cv"]]


def build_ml_features(
    stints_df: pd.DataFrame,
    clean_air_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Engineer ML-ready features from the stint-level degradation table.

    Parameters
    ----------
    stints_df:
        Output of :func:`~src.features.domain1_degradation.fit_degradation_curves`.
    clean_air_df:
        Optional clean-air laps for computing driver consistency metrics.

    Returns
    -------
    pd.DataFrame
        Feature matrix with one row per (driver, event, stint) and columns:

        - ``compound_enc``         – ordinal compound encoding
        - ``stint_length``         – total clean-air laps in stint
        - ``track_temp_proxy``     – normalised round number
        - ``consistency_cv``       – CV of lap times (lower = more consistent)
        - ``deg_rate_linear``      – linear degradation slope (s/lap)
        - ``deg_rate_poly``        – mid-stint polynomial derivative (s/lap)
        - ``r2_linear``, ``r2_poly``
        - ``compound_x_temp``      – compound × track_temp interaction
        - ``compound_x_stint``     – compound × stint_length interaction
        - ``target_deg_rate``      – target variable (same as deg_rate_linear)
    """
    if stints_df.empty:
        log.error("stints_df is empty – cannot build features.")
        return pd.DataFrame()

    feat = stints_df.copy()

    # Compound encoding
    compound_col = "compound" if "compound" in feat.columns else "tyre_compound"
    feat["compound_enc"] = (
        feat[compound_col].str.upper().map(COMPOUND_ENCODING).fillna(-1).astype(int)
    )

    # Track temperature proxy (normalised round number)
    if "round_number" in feat.columns:
        feat["track_temp_proxy"] = (feat["round_number"] - 1) / (MAX_ROUNDS - 2)
    else:
        feat["track_temp_proxy"] = np.nan

    # Driver consistency
    if clean_air_df is not None and not clean_air_df.empty:
        consistency = _driver_consistency(clean_air_df)
        merge_cols = [c for c in ["driver", "event"] if c in feat.columns and c in consistency.columns]
        if merge_cols:
            feat = feat.merge(consistency, on=merge_cols, how="left")
        else:
            feat["consistency_cv"] = np.nan
    else:
        feat["consistency_cv"] = np.nan

    # Interaction features
    feat["compound_x_temp"] = feat["compound_enc"] * feat["track_temp_proxy"]
    feat["compound_x_stint"] = feat["compound_enc"] * feat.get("stint_length", np.nan)

    # Rolling average degradation rate (per driver across season)
    if "driver" in feat.columns and "round_number" in feat.columns:
        feat = feat.sort_values(["driver", "round_number", "stint_id"] if "stint_id" in feat.columns else ["driver", "round_number"])
        feat["rolling_deg_rate"] = (
            feat.groupby("driver")["deg_rate_linear"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )
    else:
        feat["rolling_deg_rate"] = np.nan

    # Target variable
    feat["target_deg_rate"] = feat["deg_rate_linear"]

    output_cols = [
        c for c in [
            "driver", "event", "year", "round_number", "stint_id",
            "compound", "compound_enc", "stint_length",
            "track_temp_proxy", "consistency_cv",
            "deg_rate_linear", "deg_rate_poly", "r2_linear", "r2_poly",
            "rolling_deg_rate",
            "compound_x_temp", "compound_x_stint",
            "peak_lap",
            "target_deg_rate",
        ]
        if c in feat.columns
    ]
    feat = feat[output_cols]

    log.info("build_ml_features: %d rows × %d feature columns", *feat.shape)
    return feat.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Save features
# ---------------------------------------------------------------------------

def save_features(features_df: pd.DataFrame) -> None:
    """Write the ML feature matrix to the silver layer."""
    DOMAIN1_SILVER.mkdir(parents=True, exist_ok=True)
    out = DOMAIN1_SILVER / "ml_features.parquet"
    features_df.to_parquet(out, index=False, engine="pyarrow")
    log.info("Saved ml_features → %s (%d rows)", out, len(features_df))


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Feature builder starting ===")
    stints = load_stints_degradation()
    clean_air = load_clean_air_laps()
    features = build_ml_features(stints, clean_air if not clean_air.empty else None)
    if features.empty:
        log.error("No features produced. Exiting.")
        return
    save_features(features)
    log.info("=== Feature builder complete ===")


if __name__ == "__main__":
    main()
