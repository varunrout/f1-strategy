"""
Domain 1 – Tyre Degradation: core feature-engineering module.

Pipeline
--------
1. ``load_bronze_laps()``       – Read all raw Parquet lap files from the bronze layer.
2. ``extract_clean_air_stints()`` – Isolate clean-air, undisturbed stints.
3. ``fit_degradation_curves()`` – Fit linear / polynomial / exponential models per stint.
4. ``save_silver_domain1()``    – Write results to the silver layer.

Usage
-----
    python -m src.features.domain1_degradation
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from src.utils.paths import BRONZE_LAPS, DOMAIN1_SILVER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger(__name__)

# Compounds recognised as slick tyres (exclude wet-weather)
SLICK_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}

# Minimum clean-air laps required to include a stint
MIN_STINT_LAPS = 4


# ---------------------------------------------------------------------------
# 1. Load bronze laps
# ---------------------------------------------------------------------------

def load_bronze_laps(year: int | None = None) -> pd.DataFrame:
    """Load all raw lap Parquet files from the bronze layer.

    Parameters
    ----------
    year:
        When provided, restrict to a single championship year.

    Returns
    -------
    pd.DataFrame
        Concatenated and lightly cleaned lap DataFrame.
    """
    pattern = f"year={year}/**/*.parquet" if year else "**/*.parquet"
    files = sorted(BRONZE_LAPS.glob(pattern))
    if not files:
        log.warning("No bronze lap files found at %s", BRONZE_LAPS)
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f, engine="pyarrow"))
        except Exception:
            log.exception("Failed to read %s", f)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    log.info("Loaded %d lap rows from %d files.", len(df), len(frames))

    # Basic cleaning --------------------------------------------------------
    # Filter to race sessions only (degradation is a race concept)
    df = df[df["session_type"] == "R"].copy()

    # Ensure numeric lap time
    df["LapTime"] = pd.to_numeric(df["LapTime"], errors="coerce")
    df = df.dropna(subset=["LapTime"])

    # Remove obvious outliers (pit stop laps have very long times)
    q_low = df["LapTime"].quantile(0.005)
    q_high = df["LapTime"].quantile(0.995)
    df = df[(df["LapTime"] >= q_low) & (df["LapTime"] <= q_high)]

    # Normalise compound to upper case
    if "Compound" in df.columns:
        df["Compound"] = df["Compound"].str.upper().fillna("UNKNOWN")

    log.info("After cleaning: %d race lap rows.", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Extract clean-air stints
# ---------------------------------------------------------------------------

def _assign_stint_ids_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a monotonic stint_id to each row using a fully vectorized approach.

    Works with any pandas version including 3.0+.
    """
    df = df.sort_values(["year", "round_number", "Driver", "EventName", "LapNumber"]).copy()
    if "Stint" in df.columns:
        df["stint_id"] = df["Stint"].ffill().astype(int)
    else:
        # Fallback: detect stint changes via PitInTime / PitOutTime
        pit_out_laps = df["PitOutTime"].notna() if "PitOutTime" in df.columns else pd.Series(False, index=df.index)
        # Group-aware cumsum: within each driver/event, cumsum of pit-out flags
        group_cols = [c for c in ["year", "round_number", "Driver", "EventName"] if c in df.columns]
        df["stint_id"] = df.groupby(group_cols, sort=False)[pit_out_laps.name if hasattr(pit_out_laps, "name") else "PitOutTime"].cumsum() if group_cols else pit_out_laps.cumsum()
    return df


def extract_clean_air_stints(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to clean-air, undisturbed stints and assign stint metadata.

    Clean-air criteria applied in order:
    1. Remove Lap 1 (formation / first-lap incidents).
    2. Remove in-lap and out-lap (pit entry / exit laps).
    3. Remove laps where TrackStatus indicates safety car / red flag.
    4. Keep only slick-compound laps.
    5. Require at least ``MIN_STINT_LAPS`` valid laps per stint.

    The function does *not* filter on inter-car gap because position-gap data
    is not always available in the laps DataFrame.  Use enriched stints for
    gap-based filtering (see ``enrich_stint_context.py``).

    Parameters
    ----------
    df:
        Raw laps DataFrame as returned by :func:`load_bronze_laps`.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with additional columns:
        ``stint_id``, ``stint_lap``, ``tyre_compound``.
    """
    if df.empty:
        return df.copy()

    clean = df.copy()

    # Step 1 – remove lap 1
    clean = clean[clean["LapNumber"] > 1]

    # Step 2 – remove in/out laps (have finite pit times)
    if "PitInTime" in clean.columns:
        clean = clean[clean["PitInTime"].isna()]
    if "PitOutTime" in clean.columns:
        clean = clean[clean["PitOutTime"].isna()]

    # Step 3 – remove disturbed-track status laps
    # TrackStatus codes: 1=Clear, 2=Yellow, 4=Safety Car, 6=VSC, 7=Red Flag
    if "TrackStatus" in clean.columns:
        clean["TrackStatus"] = clean["TrackStatus"].astype(str)
        clean = clean[clean["TrackStatus"].isin(["1", "nan", ""])]

    # Step 4 – slick compounds only
    if "Compound" in clean.columns:
        clean = clean[clean["Compound"].isin(SLICK_COMPOUNDS)]

    # Step 5 – assign stint metadata per driver × event
    group_cols = ["year", "round_number", "Driver", "EventName"]
    available_group = [c for c in group_cols if c in clean.columns]

    clean = _assign_stint_ids_vectorized(clean)

    # Compute lap-within-stint (stint_lap)
    stint_group_cols = available_group + ["stint_id"]
    clean = clean.sort_values(stint_group_cols + ["LapNumber"]).copy()
    clean["stint_lap"] = (
        clean
        .groupby(stint_group_cols, sort=False)
        .cumcount() + 1
    )

    # Alias compound column
    clean["tyre_compound"] = clean.get("Compound", pd.Series("UNKNOWN", index=clean.index))

    # Step 6 – drop stints shorter than minimum
    stint_len = clean.groupby(stint_group_cols)["stint_lap"].transform("max")
    clean = clean[stint_len >= MIN_STINT_LAPS]

    log.info(
        "extract_clean_air_stints: %d clean-air lap rows, %d unique stints.",
        len(clean),
        clean.groupby(stint_group_cols).ngroups if not clean.empty else 0,
    )
    return clean.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Fit degradation curves
# ---------------------------------------------------------------------------

def _linear_model(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def _poly2_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * x**2 + b * x + c


def _exp_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(b * x) + c


def _safe_curve_fit(fn, x, y, p0, **kwargs) -> tuple[np.ndarray | None, bool]:
    """Wrapper around scipy curve_fit that returns (params, success)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(fn, x, y, p0=p0, maxfev=5000, **kwargs)
        return popt, True
    except Exception:
        return None, False


def fit_degradation_curves(stint_df: pd.DataFrame) -> pd.DataFrame:
    """Fit linear, polynomial and exponential degradation curves per stint.

    Parameters
    ----------
    stint_df:
        Clean-air stints DataFrame as returned by :func:`extract_clean_air_stints`.

    Returns
    -------
    pd.DataFrame
        One row per unique stint with fitted model parameters and quality metrics:

        ``driver``, ``event``, ``year``, ``round_number``, ``compound``,
        ``stint_id``, ``stint_length``,
        ``deg_rate_linear``    – seconds per lap from linear fit (slope *a*),
        ``intercept_linear``   – intercept *b* from linear fit,
        ``deg_rate_poly``      – mean derivative of polynomial fit at mid-stint,
        ``r2_linear``, ``r2_poly``,
        ``exp_a``, ``exp_b``, ``exp_c`` – exponential fit coefficients,
        ``peak_lap``           – stint_lap at which predicted pace is best (lap 1
                                 of a polynomial/exp fit may differ from linear).
    """
    if stint_df.empty:
        return pd.DataFrame()

    group_keys = [
        c for c in
        ["year", "round_number", "EventName", "Driver", "stint_id", "tyre_compound"]
        if c in stint_df.columns
    ]

    records: list[dict[str, Any]] = []

    for keys, group in stint_df.groupby(group_keys):
        group = group.sort_values("stint_lap")
        x = group["stint_lap"].to_numpy(dtype=float)
        y = group["LapTime"].to_numpy(dtype=float)

        if len(x) < MIN_STINT_LAPS:
            continue

        row: dict[str, Any] = dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,)))
        row["stint_length"] = len(x)

        # -- Linear fit -------------------------------------------------------
        popt_lin, ok_lin = _safe_curve_fit(_linear_model, x, y, p0=[0.05, y[0]])
        if ok_lin and popt_lin is not None:
            y_pred_lin = _linear_model(x, *popt_lin)
            row["deg_rate_linear"] = float(popt_lin[0])
            row["intercept_linear"] = float(popt_lin[1])
            row["r2_linear"] = float(r2_score(y, y_pred_lin))
        else:
            row["deg_rate_linear"] = np.nan
            row["intercept_linear"] = np.nan
            row["r2_linear"] = np.nan

        # -- Polynomial degree-2 fit ------------------------------------------
        popt_poly, ok_poly = _safe_curve_fit(
            _poly2_model, x, y, p0=[0.001, 0.05, y[0]]
        )
        if ok_poly and popt_poly is not None:
            y_pred_poly = _poly2_model(x, *popt_poly)
            # Instantaneous slope at mid-stint
            mid = float(np.median(x))
            deg_poly = 2 * popt_poly[0] * mid + popt_poly[1]
            row["deg_rate_poly"] = float(deg_poly)
            row["poly_a"] = float(popt_poly[0])
            row["poly_b"] = float(popt_poly[1])
            row["poly_c"] = float(popt_poly[2])
            row["r2_poly"] = float(r2_score(y, y_pred_poly))
        else:
            row["deg_rate_poly"] = np.nan
            row["poly_a"] = np.nan
            row["poly_b"] = np.nan
            row["poly_c"] = np.nan
            row["r2_poly"] = np.nan

        # -- Exponential fit --------------------------------------------------
        popt_exp, ok_exp = _safe_curve_fit(
            _exp_model, x, y, p0=[y[0] * 0.1, 0.01, y[0] * 0.9]
        )
        if ok_exp and popt_exp is not None:
            row["exp_a"] = float(popt_exp[0])
            row["exp_b"] = float(popt_exp[1])
            row["exp_c"] = float(popt_exp[2])
        else:
            row["exp_a"] = np.nan
            row["exp_b"] = np.nan
            row["exp_c"] = np.nan

        # -- Peak lap (lap with lowest predicted lap time in polynomial fit) --
        if ok_poly and popt_poly is not None and popt_poly[0] > 0:
            # Minimum of parabola: x = -b / (2a)
            peak_lap = -popt_poly[1] / (2 * popt_poly[0])
            row["peak_lap"] = float(np.clip(peak_lap, x[0], x[-1]))
        else:
            row["peak_lap"] = float(x[0])

        records.append(row)

    result = pd.DataFrame(records)
    # Friendly column aliases
    if "Driver" in result.columns:
        result = result.rename(columns={"Driver": "driver"})
    if "EventName" in result.columns:
        result = result.rename(columns={"EventName": "event"})
    if "tyre_compound" in result.columns:
        result = result.rename(columns={"tyre_compound": "compound"})

    log.info("fit_degradation_curves: %d stint records fitted.", len(result))
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Save silver layer
# ---------------------------------------------------------------------------

def save_silver_domain1(
    clean_air_df: pd.DataFrame,
    stints_df: pd.DataFrame,
) -> None:
    """Persist clean-air laps and stint degradation metrics to the silver layer.

    Parameters
    ----------
    clean_air_df:
        Output of :func:`extract_clean_air_stints`.
    stints_df:
        Output of :func:`fit_degradation_curves`.
    """
    DOMAIN1_SILVER.mkdir(parents=True, exist_ok=True)

    if not clean_air_df.empty:
        out = DOMAIN1_SILVER / "clean_air_laps.parquet"
        clean_air_df.to_parquet(out, index=False, engine="pyarrow")
        log.info("Saved clean_air_laps → %s (%d rows)", out, len(clean_air_df))

    if not stints_df.empty:
        out = DOMAIN1_SILVER / "stints_degradation.parquet"
        stints_df.to_parquet(out, index=False, engine="pyarrow")
        log.info("Saved stints_degradation → %s (%d rows)", out, len(stints_df))


# ---------------------------------------------------------------------------
# 5. Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Domain 1 – Tyre Degradation pipeline starting ===")

    bronze_laps = load_bronze_laps()
    if bronze_laps.empty:
        log.error("No bronze laps data available. Run ingest_fastf1 first.")
        return

    clean_air = extract_clean_air_stints(bronze_laps)
    if clean_air.empty:
        log.error("No clean-air stints extracted. Check bronze data quality.")
        return

    stints = fit_degradation_curves(clean_air)
    save_silver_domain1(clean_air, stints)

    log.info("=== Domain 1 pipeline complete ===")


if __name__ == "__main__":
    main()
