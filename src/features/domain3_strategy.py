"""
domain3_strategy.py
===================
Cross-domain (XT) feature engineering for Domain 3: Race Strategy &
Pit-Stop Optimisation.

Inputs
------
* features_core.db  → laps_featured, gaps_featured
* features_tyre.db  → stints_degradation, clean_air_laps
* features_traffic.db → lap_positions, proximity_events, traffic_regimes

Outputs
-------
Populates four tables in features_xt.db:
  pit_window_features  – per-lap pit-stop candidate features
  undercut_scores      – per-lap undercut opportunity scores
  sc_delta_features    – safety-car delta features per stint
  strategy_scores      – composite strategy score per lap
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attach(conn: sqlite3.Connection, path: Path, alias: str) -> None:
    """Attach an external SQLite database."""
    conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(path),))


def _safe_read(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute a query and return a DataFrame; empty DF on failure."""
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception as exc:
        logger.warning("Query failed (%s): %s", exc, sql[:120])
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 3A-i  Pit-window features
# ---------------------------------------------------------------------------

def build_pit_window_features(
    laps: pd.DataFrame,
    stints: pd.DataFrame,
    deg_per_lap: float = 0.08,
) -> pd.DataFrame:
    """
    Compute per-lap pit-stop candidate features.

    Parameters
    ----------
    laps:
        laps_featured rows with columns:
        session_id, driver, lap_number, lap_time_ms, tyre_age,
        compound, fuel_proxy, total_laps.
    stints:
        stints_degradation rows with columns:
        session_id, driver, stint, compound,
        deg_rate_ms_per_lap (optional, synthesised if absent).
    deg_per_lap:
        Fallback degradation rate (seconds/lap) when stints table is
        missing or has no rate column.

    Returns
    -------
    DataFrame with one row per (session_id, driver, lap_number).
    """
    if laps.empty:
        return pd.DataFrame()

    df = laps.copy()

    # Ensure numeric types
    for col in ("lap_time_ms", "tyre_age", "fuel_proxy"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Merge degradation rate from stints table if available
    if not stints.empty and "deg_rate_ms_per_lap" in stints.columns:
        merge_cols = ["session_id", "driver"]
        if "stint" in stints.columns and "stint" in df.columns:
            merge_cols.append("stint")
        rate_df = stints[merge_cols + ["deg_rate_ms_per_lap"]].drop_duplicates(
            subset=merge_cols
        )
        df = df.merge(rate_df, on=merge_cols, how="left")
    else:
        df["deg_rate_ms_per_lap"] = deg_per_lap * 1000  # convert to ms

    df["deg_rate_ms_per_lap"] = df["deg_rate_ms_per_lap"].fillna(
        deg_per_lap * 1000
    )

    # Laps remaining in race
    if "total_laps" in df.columns:
        df["laps_to_go"] = (df["total_laps"] - df["lap_number"]).clip(lower=0)
    else:
        # Estimate from max lap_number per session
        max_lap = df.groupby("session_id")["lap_number"].transform("max")
        df["laps_to_go"] = (max_lap - df["lap_number"]).clip(lower=0)

    # Projected tyre-deg loss over remaining laps (ms)
    df["projected_deg_loss_ms"] = df["tyre_age"] * df["deg_rate_ms_per_lap"]

    # Pit-urgency score: high when tyre is old AND many laps remain
    # Normalise to [0, 1] range per session
    raw_urgency = df["tyre_age"] * df["laps_to_go"]
    session_max = raw_urgency.groupby(df["session_id"]).transform(
        lambda x: x.max() if x.max() > 0 else 1
    )
    df["pit_urgency_score"] = (raw_urgency / session_max).clip(0, 1)

    # Ideal pit window flag: urgency in top 30 % and at least 5 laps to go
    threshold = df.groupby("session_id")["pit_urgency_score"].transform(
        lambda x: x.quantile(0.70)
    )
    df["in_pit_window"] = (
        (df["pit_urgency_score"] >= threshold) & (df["laps_to_go"] >= 5)
    ).astype(int)

    keep = [
        "session_id",
        "driver",
        "lap_number",
        "compound",
        "tyre_age",
        "laps_to_go",
        "deg_rate_ms_per_lap",
        "projected_deg_loss_ms",
        "pit_urgency_score",
        "in_pit_window",
    ]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3A-ii  Undercut opportunity scores
# ---------------------------------------------------------------------------

def build_undercut_scores(
    gaps: pd.DataFrame,
    pit_window: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate undercut opportunity per lap.

    An undercut is attractive when:
    * the car ahead is close (gap < 2.5 s)
    * the attacking car's tyre age is higher (more degradation)
    * the difference in projected deg loss favours the attacker

    Parameters
    ----------
    gaps:
        gaps_featured rows: session_id, driver, lap_number,
        gap_ahead_s, tyre_age (optional), compound.
    pit_window:
        Output of build_pit_window_features.

    Returns
    -------
    DataFrame with undercut_score per (session_id, driver, lap_number).
    """
    if gaps.empty or pit_window.empty:
        return pd.DataFrame()

    merge_keys = ["session_id", "driver", "lap_number"]
    df = gaps.merge(
        pit_window[merge_keys + ["tyre_age", "projected_deg_loss_ms", "in_pit_window"]],
        on=merge_keys,
        how="left",
    )

    # Gap component: smaller gap → higher undercut potential (inverse)
    if "gap_ahead_s" in df.columns:
        df["gap_ahead_s"] = pd.to_numeric(df["gap_ahead_s"], errors="coerce")
        # Normalise: gap=0 → score 1, gap≥5 → score 0
        df["gap_score"] = (1 - df["gap_ahead_s"].clip(0, 5) / 5).clip(0, 1)
    else:
        df["gap_score"] = 0.0

    # Deg component: higher projected loss → more benefit from pitting now
    if "projected_deg_loss_ms" in df.columns:
        session_max = df.groupby("session_id")["projected_deg_loss_ms"].transform(
            lambda x: x.max() if x.max() > 0 else 1
        )
        df["deg_score"] = (df["projected_deg_loss_ms"] / session_max).clip(0, 1)
    else:
        df["deg_score"] = 0.0

    # Composite undercut score (equal weights)
    df["undercut_score"] = 0.5 * df["gap_score"] + 0.5 * df["deg_score"]

    # Flag: genuine undercut opportunity
    df["undercut_opportunity"] = (
        (df["undercut_score"] >= 0.6) & (df.get("in_pit_window", 0) == 1)
    ).astype(int)

    keep = [
        "session_id",
        "driver",
        "lap_number",
        "gap_score",
        "deg_score",
        "undercut_score",
        "undercut_opportunity",
    ]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3A-iii  Safety-car delta features
# ---------------------------------------------------------------------------

def build_sc_delta_features(
    laps: pd.DataFrame,
    pit_window: pd.DataFrame,
    pit_lane_time_s: float = 22.0,
) -> pd.DataFrame:
    """
    Estimate the time saved (or lost) by pitting under Safety Car vs green.

    Under a Safety Car the field bunches up, so the pit-lane time cost
    (in terms of positions/seconds *relative to the field*) is much lower.

    Parameters
    ----------
    laps:
        laps_featured rows with track_status column.
    pit_window:
        Output of build_pit_window_features.
    pit_lane_time_s:
        Typical pit-lane time loss in seconds under green flag.

    Returns
    -------
    DataFrame with sc_delta_s per (session_id, driver, lap_number).
    """
    if laps.empty:
        return pd.DataFrame()

    df = laps.copy()

    # Identify SC/VSC laps
    if "track_status" in df.columns:
        df["is_sc"] = df["track_status"].str.contains(
            "SC|VSC|SAFETY", case=False, na=False
        ).astype(int)
    else:
        df["is_sc"] = 0

    # Average lap time under SC (field bunching reduces effective pit cost)
    sc_lap_avg = df[df["is_sc"] == 1]["lap_time_ms"].mean()
    green_lap_avg = df[df["is_sc"] == 0]["lap_time_ms"].mean()

    # If no SC laps, use a synthetic estimate
    if np.isnan(sc_lap_avg) or sc_lap_avg == 0:
        sc_lap_avg = green_lap_avg * 1.3 if not np.isnan(green_lap_avg) else 90_000

    # SC pit-lane cost: field moves slowly, so cost is lower
    sc_pit_fraction = (pit_lane_time_s * 1000) / sc_lap_avg if sc_lap_avg > 0 else 0.3
    sc_pit_cost_s = sc_pit_fraction * sc_lap_avg / 1000

    # Delta = normal pit cost - SC pit cost
    sc_delta_s = pit_lane_time_s - sc_pit_cost_s

    # Merge pit_window flag
    merge_keys = ["session_id", "driver", "lap_number"]
    pw_subset = pit_window[merge_keys + ["in_pit_window"]].drop_duplicates(
        merge_keys
    )
    df = df.merge(pw_subset, on=merge_keys, how="left")

    df["sc_delta_s"] = sc_delta_s
    df["sc_pit_recommended"] = (
        (df["is_sc"] == 1) & (df.get("in_pit_window", 0) == 1)
    ).astype(int)

    keep = [
        "session_id",
        "driver",
        "lap_number",
        "is_sc",
        "sc_delta_s",
        "sc_pit_recommended",
    ]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3A-iv  Composite strategy scores
# ---------------------------------------------------------------------------

def build_strategy_scores(
    pit_window: pd.DataFrame,
    undercut: pd.DataFrame,
    sc_delta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine pit-window, undercut and SC-delta features into a single
    per-lap strategy score.

    Parameters
    ----------
    pit_window, undercut, sc_delta:
        Outputs of the corresponding build_* functions.

    Returns
    -------
    DataFrame with strategy_score and strategy_action columns.
    """
    if pit_window.empty:
        return pd.DataFrame()

    merge_keys = ["session_id", "driver", "lap_number"]
    df = pit_window[merge_keys + ["pit_urgency_score", "in_pit_window"]].copy()

    # Merge undercut score
    if not undercut.empty:
        df = df.merge(
            undercut[merge_keys + ["undercut_score", "undercut_opportunity"]],
            on=merge_keys,
            how="left",
        )
    else:
        df["undercut_score"] = 0.0
        df["undercut_opportunity"] = 0

    # Merge SC delta
    if not sc_delta.empty:
        df = df.merge(
            sc_delta[merge_keys + ["is_sc", "sc_delta_s", "sc_pit_recommended"]],
            on=merge_keys,
            how="left",
        )
    else:
        df["is_sc"] = 0
        df["sc_delta_s"] = 0.0
        df["sc_pit_recommended"] = 0

    # Fill nulls
    df["undercut_score"] = df["undercut_score"].fillna(0.0)
    df["undercut_opportunity"] = df["undercut_opportunity"].fillna(0).astype(int)
    df["is_sc"] = df["is_sc"].fillna(0).astype(int)
    df["sc_delta_s"] = df["sc_delta_s"].fillna(0.0)
    df["sc_pit_recommended"] = df["sc_pit_recommended"].fillna(0).astype(int)

    # Normalise SC delta to [0, 1]
    max_delta = df["sc_delta_s"].max()
    df["sc_score"] = (
        df["sc_delta_s"] / max_delta if max_delta > 0 else 0.0
    )
    df["sc_score"] = df["sc_score"].clip(0, 1)

    # Composite (weighted sum)
    df["strategy_score"] = (
        0.40 * df["pit_urgency_score"]
        + 0.35 * df["undercut_score"]
        + 0.25 * df["sc_score"]
    ).clip(0, 1)

    # Recommended action
    conditions = [
        df["sc_pit_recommended"] == 1,
        df["undercut_opportunity"] == 1,
        df["in_pit_window"] == 1,
    ]
    choices = ["pit_sc", "pit_undercut", "pit_normal"]
    df["strategy_action"] = np.select(conditions, choices, default="stay_out")

    keep = [
        "session_id",
        "driver",
        "lap_number",
        "pit_urgency_score",
        "undercut_score",
        "sc_score",
        "strategy_score",
        "strategy_action",
    ]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_all_features(
    db_core: Path,
    db_tyre: Path,
    db_traffic: Path,
) -> dict[str, pd.DataFrame]:
    """
    Load raw data from all domain databases and compute the four XT feature
    tables.

    Parameters
    ----------
    db_core:      Path to features_core.db
    db_tyre:      Path to features_tyre.db
    db_traffic:   Path to features_traffic.db

    Returns
    -------
    dict with keys: pit_window, undercut, sc_delta, strategy
    """
    # ---- Load inputs ---------------------------------------------------
    def _open_or_empty(path: Path) -> sqlite3.Connection:
        """Return a connection to *path* if it exists, else an in-memory DB."""
        if path.exists():
            return sqlite3.connect(str(path))
        logger.warning("Database not found: %s — using empty data frames", path)
        return sqlite3.connect(":memory:")

    conn_core = _open_or_empty(db_core)
    conn_tyre = _open_or_empty(db_tyre)
    conn_traffic = _open_or_empty(db_traffic)

    try:
        laps = _safe_read(conn_core, "SELECT * FROM laps_featured")
        gaps = _safe_read(conn_core, "SELECT * FROM gaps_featured")
        stints = _safe_read(
            conn_tyre,
            "SELECT * FROM stints_degradation"
        )
        # traffic regimes used for track status if available
        regimes = _safe_read(
            conn_traffic,
            "SELECT session_id, driver, lap_number, regime AS track_status "
            "FROM traffic_regimes"
        )
    finally:
        conn_core.close()
        conn_tyre.close()
        conn_traffic.close()

    # Merge traffic regime as track_status if laps doesn't have it
    if not regimes.empty and "track_status" not in laps.columns:
        laps = laps.merge(
            regimes,
            on=["session_id", "driver", "lap_number"],
            how="left",
        )

    # ---- Compute -------------------------------------------------------
    logger.info("Building pit-window features …")
    pit_window = build_pit_window_features(laps, stints)

    logger.info("Building undercut scores …")
    undercut = build_undercut_scores(gaps, pit_window)

    logger.info("Building SC-delta features …")
    sc_delta = build_sc_delta_features(laps, pit_window)

    logger.info("Building composite strategy scores …")
    strategy = build_strategy_scores(pit_window, undercut, sc_delta)

    return {
        "pit_window": pit_window,
        "undercut": undercut,
        "sc_delta": sc_delta,
        "strategy": strategy,
    }
