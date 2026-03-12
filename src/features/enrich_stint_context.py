"""
Domain 1 – Tyre Degradation: stint context enrichment.

Enriches silver stint data with contextual signals that are not captured
in the lap time alone:

- Safety car proximity
- Track position at stint start / mean gap to leader
- Weather context (air temp, track temp, humidity)
- Estimated fuel load and fuel-corrected pace

Usage
-----
    python -m src.features.enrich_stint_context
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import BRONZE_RACE_CONTROL, BRONZE_LAPS, DOMAIN1_SILVER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger(__name__)

# F1 fuel model constants (approximate)
FUEL_START_KG = 110.0      # starting fuel load in kg
FUEL_BURN_RATE = 1.5       # kg per racing lap
FUEL_TIME_EFFECT = 0.03    # seconds per kg (heavier car = slower pace)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load silver stints and bronze race control messages.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (stints_df, race_control_df)
    """
    stints_path = DOMAIN1_SILVER / "stints_degradation.parquet"
    if not stints_path.exists():
        raise FileNotFoundError(
            f"Silver stints not found: {stints_path}\n"
            "Run `python -m src.features.domain1_degradation` first."
        )
    stints_df = pd.read_parquet(stints_path, engine="pyarrow")
    log.info("Loaded stints_degradation: %d rows", len(stints_df))

    # Collect all race-control Parquet files
    rc_files = sorted(BRONZE_RACE_CONTROL.glob("**/*.parquet"))
    if rc_files:
        rc_frames = [pd.read_parquet(f, engine="pyarrow") for f in rc_files]
        race_control_df = pd.concat(rc_frames, ignore_index=True)
        log.info("Loaded race_control: %d rows from %d files", len(race_control_df), len(rc_files))
    else:
        log.warning("No race control data found at %s", BRONZE_RACE_CONTROL)
        race_control_df = pd.DataFrame()

    return stints_df, race_control_df


# ---------------------------------------------------------------------------
# 2. Safety car proximity
# ---------------------------------------------------------------------------

def enrich_with_safety_car_proximity(
    stints_df: pd.DataFrame,
    race_control_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add safety car proximity features to the stints DataFrame.

    Added columns
    -------------
    ``pre_sc_laps``:
        Number of laps between the *end* of the most recent safety car period
        and the *start* of this stint.  ``-1`` if no SC occurred before this
        stint or race control data is unavailable.
    ``post_sc_tyre_age``:
        Tyre age (in laps) when the safety car was *deployed* during this
        stint.  ``-1`` if no SC occurred within the stint boundaries.
    """
    result = stints_df.copy()
    result["pre_sc_laps"] = -1
    result["post_sc_tyre_age"] = -1

    if race_control_df.empty or "Message" not in race_control_df.columns:
        return result

    # Identify SC deployment and end messages
    sc_mask = race_control_df["Message"].str.contains(
        r"SAFETY CAR DEPLOYED|VIRTUAL SAFETY CAR DEPLOYED", case=False, na=False
    )
    sc_end_mask = race_control_df["Message"].str.contains(
        r"SAFETY CAR IN THIS LAP|VIRTUAL SAFETY CAR ENDING", case=False, na=False
    )

    # Work per-event
    event_col = next((c for c in ["event", "EventName"] if c in result.columns), None)
    if event_col is None:
        return result

    for event, stints_event in result.groupby(event_col):
        rc_event = race_control_df[race_control_df.get("EventName", race_control_df.get("event", pd.Series(dtype=str))) == event]
        if rc_event.empty:
            continue

        sc_laps = rc_event.loc[sc_mask.reindex(rc_event.index, fill_value=False), "Lap"].dropna().astype(float)
        sc_end_laps = rc_event.loc[sc_end_mask.reindex(rc_event.index, fill_value=False), "Lap"].dropna().astype(float)

        for idx in stints_event.index:
            stint_start_lap = stints_event.at[idx, "stint_id"] if "stint_id" in stints_event else np.nan
            stint_len = stints_event.at[idx, "stint_length"] if "stint_length" in stints_event else np.nan

            if pd.isna(stint_start_lap):
                continue

            # pre_sc_laps: last SC end before stint start
            sc_ends_before = sc_end_laps[sc_end_laps < stint_start_lap]
            if not sc_ends_before.empty:
                result.at[idx, "pre_sc_laps"] = int(stint_start_lap - sc_ends_before.max())

            # post_sc_tyre_age: SC deployed within stint window
            if not pd.isna(stint_len):
                stint_end_lap = stint_start_lap + stint_len
                sc_in_stint = sc_laps[(sc_laps >= stint_start_lap) & (sc_laps <= stint_end_lap)]
                if not sc_in_stint.empty:
                    result.at[idx, "post_sc_tyre_age"] = int(sc_in_stint.min() - stint_start_lap)

    return result


# ---------------------------------------------------------------------------
# 3. Track position
# ---------------------------------------------------------------------------

def enrich_with_track_position(stints_df: pd.DataFrame) -> pd.DataFrame:
    """Add track-position features derived from clean-air lap data.

    Added columns
    -------------
    ``stint_start_position``:
        Driver's on-track position on the first clean-air lap of the stint.
    ``gap_to_leader_mean``:
        Mean gap to race leader across all clean-air laps in the stint (seconds).
        Both fields are ``NaN`` when position data is unavailable in the laps
        table (standard case without the enriched positions parquet).
    """
    result = stints_df.copy()
    result["stint_start_position"] = np.nan
    result["gap_to_leader_mean"] = np.nan

    clean_air_path = DOMAIN1_SILVER / "clean_air_laps.parquet"
    if not clean_air_path.exists():
        log.warning("clean_air_laps.parquet not found – track position features remain NaN.")
        return result

    laps = pd.read_parquet(clean_air_path, engine="pyarrow")
    if "Position" not in laps.columns:
        return result

    driver_col = next((c for c in ["driver", "Driver"] if c in result.columns), None)
    event_col = next((c for c in ["event", "EventName"] if c in result.columns), None)
    if driver_col is None or event_col is None:
        return result

    laps_driver = "Driver" if "Driver" in laps.columns else "driver"
    laps_event = "EventName" if "EventName" in laps.columns else "event"

    for idx, row in result.iterrows():
        driver = row.get(driver_col)
        event = row.get(event_col)
        stint_id = row.get("stint_id", np.nan)

        mask = (laps[laps_driver] == driver) & (laps[laps_event] == event)
        if "stint_id" in laps.columns and not pd.isna(stint_id):
            mask &= laps["stint_id"] == stint_id

        stint_laps = laps[mask].sort_values("LapNumber")
        if stint_laps.empty:
            continue

        result.at[idx, "stint_start_position"] = float(stint_laps.iloc[0]["Position"])

        # Gap to leader approximation: leader is pos=1; not always available
        result.at[idx, "gap_to_leader_mean"] = float(
            (stint_laps["Position"] - 1).mean()
        )

    return result


# ---------------------------------------------------------------------------
# 4. Weather context
# ---------------------------------------------------------------------------

def enrich_with_weather_context(stints_df: pd.DataFrame) -> pd.DataFrame:
    """Join weather summary statistics onto the stints DataFrame.

    FastF1 weather data is stored per session but not yet loaded here;
    this function populates the columns with ``NaN`` unless a
    ``weather_summary.parquet`` is found in the silver layer.

    Added columns: ``air_temp_mean``, ``track_temp_mean``, ``humidity_mean``.
    """
    result = stints_df.copy()
    for col in ("air_temp_mean", "track_temp_mean", "humidity_mean"):
        result[col] = np.nan

    weather_path = DOMAIN1_SILVER / "weather_summary.parquet"
    if not weather_path.exists():
        log.warning(
            "weather_summary.parquet not found – weather features remain NaN. "
            "Run weather enrichment to populate."
        )
        return result

    weather = pd.read_parquet(weather_path, engine="pyarrow")
    event_col = next((c for c in ["event", "EventName"] if c in result.columns), None)
    if event_col is None or event_col not in weather.columns:
        return result

    result = result.merge(
        weather[[event_col, "air_temp_mean", "track_temp_mean", "humidity_mean"]],
        on=event_col,
        how="left",
        suffixes=("", "_weather"),
    )
    # Prefer merged columns
    for col in ("air_temp_mean", "track_temp_mean", "humidity_mean"):
        if col + "_weather" in result.columns:
            result[col] = result[col + "_weather"].combine_first(result[col])
            result.drop(columns=[col + "_weather"], inplace=True)

    return result


# ---------------------------------------------------------------------------
# 5. Fuel load
# ---------------------------------------------------------------------------

def enrich_with_fuel_load(stints_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate fuel load and fuel-corrected pace per stint.

    F1 cars start with approximately 110 kg of fuel and burn ~1.5 kg/lap.
    Each kilogram of fuel slows the car by ~0.03 s/lap (team-average figure).

    Added columns
    -------------
    ``estimated_fuel_kg``:
        Estimated fuel at the *start* of the stint.
    ``fuel_corrected_pace``:
        ``intercept_linear`` adjusted for estimated fuel weight delta
        relative to an empty-tank reference.
    """
    result = stints_df.copy()

    # Determine stint start lap from LapNumber when available; fall back to a
    # round_number-based estimate when lap numbers are absent.
    if "LapNumber" in result.columns:
        # Use the minimum LapNumber per stint as the race lap at stint start.
        start_lap_col = "stint_start_lap"
        group_cols = [c for c in ["driver", "Driver", "event", "EventName", "round_number", "stint_id"]
                      if c in result.columns]
        if group_cols:
            result[start_lap_col] = result["LapNumber"]
        else:
            result[start_lap_col] = result["LapNumber"]
    elif "round_number" in result.columns:
        # Very rough proxy: assume stint starts at lap ~1 for first stint, etc.
        log.warning(
            "LapNumber not available in stints_df – fuel estimates use stint_id as lap proxy."
        )
        start_lap_col = "stint_id"
        if start_lap_col not in result.columns:
            result["estimated_fuel_kg"] = FUEL_START_KG
            result["fuel_corrected_pace"] = result.get("intercept_linear", np.nan)
            return result
    else:
        log.warning("No lap-number column available – fuel estimates assume full tank.")
        result["estimated_fuel_kg"] = FUEL_START_KG
        result["fuel_corrected_pace"] = result.get("intercept_linear", np.nan)
        return result

    result["estimated_fuel_kg"] = np.maximum(
        0.0,
        FUEL_START_KG - result[start_lap_col].astype(float) * FUEL_BURN_RATE,
    )

    # Fuel-corrected pace: remove fuel weight contribution from intercept
    if "intercept_linear" in result.columns:
        fuel_delta = result["estimated_fuel_kg"] * FUEL_TIME_EFFECT
        result["fuel_corrected_pace"] = result["intercept_linear"] - fuel_delta
    else:
        result["fuel_corrected_pace"] = np.nan

    return result


# ---------------------------------------------------------------------------
# 6. Save enriched stints
# ---------------------------------------------------------------------------

def save_enriched_stints(enriched_df: pd.DataFrame) -> None:
    """Write the enriched stints DataFrame to the silver layer."""
    DOMAIN1_SILVER.mkdir(parents=True, exist_ok=True)
    out = DOMAIN1_SILVER / "stints_enriched.parquet"
    enriched_df.to_parquet(out, index=False, engine="pyarrow")
    log.info("Saved stints_enriched → %s (%d rows)", out, len(enriched_df))


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Stint context enrichment starting ===")

    stints_df, race_control_df = load_data()

    stints_df = enrich_with_safety_car_proximity(stints_df, race_control_df)
    stints_df = enrich_with_track_position(stints_df)
    stints_df = enrich_with_weather_context(stints_df)
    stints_df = enrich_with_fuel_load(stints_df)

    save_enriched_stints(stints_df)
    log.info("=== Stint context enrichment complete ===")


if __name__ == "__main__":
    main()
