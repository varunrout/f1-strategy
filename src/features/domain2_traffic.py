"""Domain 2: Traffic, Dirty Air & Spatial Constraints Feature Engineering.

This module implements the full Domain 2 pipeline for quantifying how proximity
to other cars alters performance, tyre behaviour, and overtaking feasibility.

Pipeline:
  Job 1: build_lap_positions       — Match position coords to lap numbers
  Job 2: detect_proximity_events   — KDTree spatial search (within 20m)
  Job 3: classify_traffic_regimes  — Per-lap traffic regime labels
  Job 4: quantify_traffic_penalties— Baseline-adjusted lap time loss per regime
  Job 5: detect_overtakes          — Position-change-based overtake detection
  Job 6: detect_overtake_hotspots  — DBSCAN clustering of (x, y) overtake coords
  Job 7: map_track_bottlenecks     — Density-based bottleneck zone detection
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLEAN_AIR_GAP_S = 2.5          # seconds gap = clean air
DRS_RANGE_S = 1.0              # seconds gap = DRS activation range
PROXIMITY_THRESHOLD_M = 20.0   # metres radius for proximity events

# Traffic regime thresholds
LIGHT_TRAFFIC_PROXIMITY_PCT = 0.20
MODERATE_TRAFFIC_PROXIMITY_PCT = 0.50
HEAVY_TRAFFIC_GAP_S = 1.0

TRAFFIC_REGIMES = [
    "CLEAN_AIR",
    "LIGHT_TRAFFIC",
    "MODERATE_TRAFFIC",
    "HEAVY_TRAFFIC",
    "CLOSE_FOLLOWING",
]


# ---------------------------------------------------------------------------
# Job 1 — Build lap positions
# ---------------------------------------------------------------------------

def build_lap_positions(
    positions_df: pd.DataFrame,
    laps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Match raw position coordinates to lap numbers via time-based join."""
    if positions_df.empty or laps_df.empty:
        logger.warning("build_lap_positions: empty input, returning empty DataFrame")
        return pd.DataFrame()

    laps = laps_df.copy()

    if "lap_start_time_s" not in laps.columns or "lap_end_time_s" not in laps.columns:
        laps = _derive_lap_times(laps)

    laps = laps[["session_id", "driver", "lap_number",
                  "lap_start_time_s", "lap_end_time_s"]].dropna()

    results: list[pd.DataFrame] = []

    for (session_id, driver), pos_grp in positions_df.groupby(
        ["session_id", "driver"]
    ):
        lap_grp = laps[
            (laps["session_id"] == session_id) & (laps["driver"] == driver)
        ].sort_values("lap_number")

        if lap_grp.empty:
            continue

        pos_grp = pos_grp.sort_values("time_s").copy()

        starts = lap_grp["lap_start_time_s"].values
        ends = lap_grp["lap_end_time_s"].values
        lap_numbers = lap_grp["lap_number"].values
        times = pos_grp["time_s"].values

        lap_idx = np.searchsorted(starts, times, side="right") - 1
        lap_idx = np.clip(lap_idx, 0, len(lap_numbers) - 1)

        valid = times <= ends[lap_idx]

        pos_grp = pos_grp[valid].copy()
        pos_grp["lap_number"] = lap_numbers[lap_idx[valid]]
        results.append(pos_grp)

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)
    logger.info(
        "build_lap_positions: matched %d position rows across %d sessions",
        len(out),
        out["session_id"].nunique(),
    )
    return out


def _derive_lap_times(laps: pd.DataFrame) -> pd.DataFrame:
    """Derive lap_start_time_s and lap_end_time_s from cumulative lap times."""
    laps = laps.copy().sort_values(["session_id", "driver", "lap_number"])

    lap_time_col = None
    for col in ("lap_time_s", "lap_time_ms"):
        if col in laps.columns:
            lap_time_col = col
            break

    if lap_time_col is None:
        logger.warning("No lap_time column found; cannot derive lap boundaries")
        laps["lap_start_time_s"] = np.nan
        laps["lap_end_time_s"] = np.nan
        return laps

    multiplier = 0.001 if "ms" in lap_time_col else 1.0

    laps["_lap_s"] = laps[lap_time_col] * multiplier
    laps["lap_end_time_s"] = laps.groupby(["session_id", "driver"])[
        "_lap_s"
    ].cumsum()
    laps["lap_start_time_s"] = laps["lap_end_time_s"] - laps["_lap_s"]
    laps.drop(columns=["_lap_s"], inplace=True)
    return laps


# ---------------------------------------------------------------------------
# Job 2 — Detect proximity events
# ---------------------------------------------------------------------------

def detect_proximity_events(
    lap_positions: pd.DataFrame,
    distance_threshold_m: float = PROXIMITY_THRESHOLD_M,
    time_resolution_s: float = 0.5,
) -> pd.DataFrame:
    """Detect car-pairs within distance_threshold_m at each time snapshot."""
    if lap_positions.empty:
        return pd.DataFrame()

    required = {"session_id", "driver", "time_s", "x", "y", "z", "lap_number"}
    missing = required - set(lap_positions.columns)
    if missing:
        raise ValueError(f"detect_proximity_events: missing columns {missing}")

    records: list[dict] = []

    for (session_id, lap_number), frame in lap_positions.groupby(
        ["session_id", "lap_number"]
    ):
        frame = frame.copy()
        frame["time_bin"] = (
            (frame["time_s"] / time_resolution_s).round() * time_resolution_s
        )

        # Keep one row per driver per time bin (nearest to bin centre)
        binned_rows = []
        for (t_bin, driver), grp in frame.groupby(["time_bin", "driver"]):
            best = grp.iloc[(grp["time_s"] - t_bin).abs().argsort()[:1]]
            binned_rows.append(best)

        if not binned_rows:
            continue

        frame = pd.concat(binned_rows, ignore_index=True)

        for t_bin, t_frame in frame.groupby("time_bin"):
            if len(t_frame) < 2:
                continue

            coords = t_frame[["x", "y", "z"]].values.astype(float)
            drivers = t_frame["driver"].values

            tree = KDTree(coords)
            pairs = tree.query_pairs(r=distance_threshold_m)

            for i, j in pairs:
                dist = float(np.linalg.norm(coords[i] - coords[j]))
                # Use y-coordinate as a proxy for track progression (higher y = further ahead).
                # This is a simplification; for production use track distance / lap progress.
                y_i, y_j = coords[i][1], coords[j][1]
                if y_i >= y_j:
                    ahead_idx, behind_idx = i, j
                else:
                    ahead_idx, behind_idx = j, i

                records.append(
                    {
                        "session_id": session_id,
                        "lap_number": lap_number,
                        "time_s": float(t_bin),
                        "driver_ahead": drivers[ahead_idx],
                        "driver_behind": drivers[behind_idx],
                        "distance_m": dist,
                    }
                )

    if not records:
        return pd.DataFrame(
            columns=[
                "session_id", "lap_number", "time_s",
                "driver_ahead", "driver_behind", "distance_m",
            ]
        )

    result = pd.DataFrame(records)
    logger.info(
        "detect_proximity_events: found %d proximity events", len(result)
    )
    return result


# ---------------------------------------------------------------------------
# Job 3 — Classify traffic regimes per lap
# ---------------------------------------------------------------------------

def classify_traffic_regimes(
    laps_df: pd.DataFrame,
    proximity_events: pd.DataFrame,
    time_resolution_s: float = 0.5,
) -> pd.DataFrame:
    """Classify each lap into a traffic regime."""
    out = laps_df.copy()

    if proximity_events.empty:
        out["traffic_regime"] = "CLEAN_AIR"
        out["proximity_pct"] = 0.0
        return out

    prox = proximity_events.copy()
    prox["duration_s"] = time_resolution_s

    proximity_time = (
        prox.groupby(["session_id", "driver_behind", "lap_number"])["duration_s"]
        .sum()
        .reset_index()
        .rename(columns={"driver_behind": "driver", "duration_s": "proximity_time_s"})
    )

    out = out.merge(proximity_time, on=["session_id", "driver", "lap_number"], how="left")
    out["proximity_time_s"] = out["proximity_time_s"].fillna(0.0)

    if "lap_time_s" in out.columns:
        lap_time_s = out["lap_time_s"].clip(lower=1.0)
    elif "lap_time_ms" in out.columns:
        lap_time_s = (out["lap_time_ms"] / 1000.0).clip(lower=1.0)
    else:
        lap_time_s = pd.Series(90.0, index=out.index)

    out["proximity_pct"] = out["proximity_time_s"] / lap_time_s

    def _regime(row):
        gap = row.get("gap_to_ahead_s", np.nan)
        pct = row["proximity_pct"]

        if pd.isna(gap) or gap > CLEAN_AIR_GAP_S:
            return "CLEAN_AIR"
        if pct < LIGHT_TRAFFIC_PROXIMITY_PCT:
            return "LIGHT_TRAFFIC"
        if pct < MODERATE_TRAFFIC_PROXIMITY_PCT:
            return "MODERATE_TRAFFIC"
        if gap > HEAVY_TRAFFIC_GAP_S:
            return "HEAVY_TRAFFIC"
        return "CLOSE_FOLLOWING"

    out["traffic_regime"] = out.apply(_regime, axis=1)
    logger.info(
        "classify_traffic_regimes: distribution\n%s",
        out["traffic_regime"].value_counts(normalize=True).to_string(),
    )
    return out


# ---------------------------------------------------------------------------
# Job 4 — Quantify traffic penalties
# ---------------------------------------------------------------------------

def quantify_traffic_penalties(
    laps_df: pd.DataFrame,
    deg_rates: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Measure lap-time loss attributable to traffic."""
    required = {"session_id", "driver", "lap_number", "traffic_regime"}
    missing = required - set(laps_df.columns)
    if missing:
        raise ValueError(f"quantify_traffic_penalties: missing columns {missing}")

    lap_time_col = "lap_time_s" if "lap_time_s" in laps_df.columns else "lap_time_ms"
    multiplier = 1.0 if lap_time_col == "lap_time_s" else 0.001

    out = laps_df.copy()
    out["_lap_s"] = out[lap_time_col] * multiplier

    tyre_age_col = "tyre_age_laps" if "tyre_age_laps" in out.columns else None
    stint_col = "stint" if "stint" in out.columns else None

    group_keys = ["session_id", "driver"]
    if stint_col:
        group_keys.append(stint_col)

    baselines: list[pd.DataFrame] = []
    for keys, grp in out.groupby(group_keys):
        clean = grp[grp["traffic_regime"] == "CLEAN_AIR"]
        if clean.empty:
            baseline = grp["_lap_s"].median()
            deg_rate = 0.0
        else:
            baseline = clean["_lap_s"].median()
            if tyre_age_col and tyre_age_col in clean.columns and len(clean) >= 3:
                x = clean[tyre_age_col].values.astype(float)
                y = clean["_lap_s"].values.astype(float)
                coeffs = np.polyfit(x, y, 1)
                deg_rate = float(coeffs[0])
            else:
                deg_rate = 0.0

        grp = grp.copy()
        if tyre_age_col and tyre_age_col in grp.columns:
            expected = baseline + deg_rate * grp[tyre_age_col].fillna(0)
        else:
            expected = baseline

        grp["traffic_penalty_s"] = grp["_lap_s"] - expected
        baselines.append(grp)

    out = pd.concat(baselines, ignore_index=True)
    out.drop(columns=["_lap_s"], inplace=True, errors="ignore")

    logger.info(
        "quantify_traffic_penalties: summary by regime\n%s",
        out.groupby("traffic_regime")["traffic_penalty_s"]
        .agg(["mean", "median", "count"])
        .round(3)
        .to_string(),
    )
    return out


def summarise_traffic_penalties(laps_with_penalties: pd.DataFrame) -> pd.DataFrame:
    """Aggregate traffic penalties into a summary table."""
    return (
        laps_with_penalties.groupby("traffic_regime")["traffic_penalty_s"]
        .agg(
            avg_penalty_s="mean",
            median_penalty_s="median",
            std_penalty_s="std",
            n_laps="count",
        )
        .reindex(TRAFFIC_REGIMES)
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Job 5 — Detect overtakes
# ---------------------------------------------------------------------------

def detect_overtakes(
    laps_df: pd.DataFrame,
    exclude_sc_vsc: bool = True,
) -> pd.DataFrame:
    """Detect lap-to-lap position improvements that represent real overtakes."""
    if laps_df.empty:
        return pd.DataFrame()

    required = {"session_id", "driver", "lap_number", "position"}
    missing = required - set(laps_df.columns)
    if missing:
        raise ValueError(f"detect_overtakes: missing columns {missing}")

    df = laps_df.sort_values(["session_id", "driver", "lap_number"]).copy()
    df["position"] = pd.to_numeric(df["position"], errors="coerce")

    df["position_before"] = df.groupby(["session_id", "driver"])["position"].shift(1)
    df["position_change"] = df["position_before"] - df["position"]

    overtakes = df[df["position_change"] >= 1.0].copy()

    if "is_pit_lap" in df.columns:
        overtakes = overtakes[overtakes["is_pit_lap"] == 0]

    if exclude_sc_vsc and "track_status_cat" in df.columns:
        overtakes = overtakes[overtakes["track_status_cat"] == "GREEN"]

    overtakes["overtake_type"] = "TRACK"
    if "within_drs" in overtakes.columns:
        overtakes.loc[
            overtakes["within_drs"] == 1, "overtake_type"
        ] = "DRS_ASSISTED"
    if "gap_to_ahead_s" in overtakes.columns:
        overtakes.loc[
            (overtakes["gap_to_ahead_s"] < 0.3) &
            (overtakes["overtake_type"] == "TRACK"),
            "overtake_type",
        ] = "LATE_BRAKING"

    cols = [
        "session_id", "driver", "lap_number",
        "position_before", "position", "position_change",
    ]
    optional_cols = ["track_status_cat", "overtake_type", "gap_to_ahead_s",
                     "within_drs", "gp_name", "year"]
    cols += [c for c in optional_cols if c in overtakes.columns]

    result = overtakes[cols].rename(columns={"position": "position_after"})
    logger.info("detect_overtakes: found %d overtake events", len(result))
    return result


# ---------------------------------------------------------------------------
# Job 6 — Detect overtake hotspots (DBSCAN clustering)
# ---------------------------------------------------------------------------

def detect_overtake_hotspots(
    overtakes: pd.DataFrame,
    lap_positions: pd.DataFrame,
    eps_m: float = 50.0,
    min_samples: int = 5,
) -> pd.DataFrame:
    """Cluster overtake x,y locations using DBSCAN to find hotspot zones."""
    from sklearn.cluster import DBSCAN

    if overtakes.empty or lap_positions.empty:
        return pd.DataFrame()

    overtake_positions = overtakes.merge(
        lap_positions[["session_id", "driver", "lap_number", "x", "y"]],
        on=["session_id", "driver", "lap_number"],
        how="left",
    )
    overtake_positions = overtake_positions.dropna(subset=["x", "y"])

    event_coords = (
        overtake_positions.groupby(["session_id", "driver", "lap_number"])[["x", "y"]]
        .mean()
        .reset_index()
    )

    coords = event_coords[["x", "y"]].values

    if len(coords) < min_samples:
        logger.warning(
            "detect_overtake_hotspots: too few points (%d) for DBSCAN", len(coords)
        )
        return pd.DataFrame()

    db = DBSCAN(eps=eps_m, min_samples=min_samples).fit(coords)
    event_coords = event_coords.copy()
    event_coords["cluster_id"] = db.labels_

    clusters = (
        event_coords[event_coords["cluster_id"] >= 0]
        .groupby("cluster_id")
        .agg(
            center_x=("x", "mean"),
            center_y=("y", "mean"),
            n_overtakes=("x", "count"),
        )
        .reset_index()
    )

    def _radius(grp):
        centre = grp[["x", "y"]].mean().values
        dists = np.linalg.norm(grp[["x", "y"]].values - centre, axis=1)
        return float(dists.std()) if len(dists) > 1 else 0.0

    filtered = event_coords[event_coords["cluster_id"] >= 0]
    radii_list = []
    for cid, grp in filtered.groupby("cluster_id"):
        radii_list.append({"cluster_id": cid, "radius_m": _radius(grp)})
    radii = pd.DataFrame(radii_list)

    if not radii.empty:
        clusters = clusters.merge(radii, on="cluster_id")

    logger.info(
        "detect_overtake_hotspots: found %d hotspot clusters", len(clusters)
    )
    return clusters


# ---------------------------------------------------------------------------
# Job 7 — Map track bottlenecks (density analysis)
# ---------------------------------------------------------------------------

def map_track_bottlenecks(
    lap_positions: pd.DataFrame,
    proximity_events: pd.DataFrame,
    grid_size_m: float = 25.0,
    top_n: int = 10,
) -> pd.DataFrame:
    """Identify track locations with highest car density (bottlenecks)."""
    if lap_positions.empty or proximity_events.empty:
        return pd.DataFrame()

    pos_snap = lap_positions[["session_id", "driver", "time_s", "x", "y"]].copy()
    pos_snap = pos_snap.rename(columns={"driver": "driver_behind"})

    merged = proximity_events.merge(
        pos_snap,
        on=["session_id", "driver_behind"],
        how="left",
        suffixes=("", "_pos"),
    )
    merged["time_diff"] = (merged["time_s"] - merged["time_s_pos"]).abs()
    merged = (
        merged.sort_values("time_diff")
        .groupby(["session_id", "driver_behind", "time_s"])
        .first()
        .reset_index()
    )
    merged = merged.dropna(subset=["x", "y"])

    if merged.empty:
        return pd.DataFrame()

    merged["grid_x"] = (merged["x"] / grid_size_m).round().astype(int)
    merged["grid_y"] = (merged["y"] / grid_size_m).round().astype(int)

    density = (
        merged.groupby(["grid_x", "grid_y"])
        .agg(
            proximity_count=("time_s", "count"),
            center_x=("x", "mean"),
            center_y=("y", "mean"),
        )
        .reset_index()
        .sort_values("proximity_count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    density["density_rank"] = density.index + 1

    logger.info(
        "map_track_bottlenecks: top %d bottleneck zones identified", len(density)
    )
    return density


# ---------------------------------------------------------------------------
# High-level pipeline runner
# ---------------------------------------------------------------------------

def run_domain2_pipeline(
    positions_df: pd.DataFrame,
    laps_df: pd.DataFrame,
    proximity_threshold_m: float = PROXIMITY_THRESHOLD_M,
    time_resolution_s: float = 0.5,
) -> dict:
    """Run the full Domain 2 pipeline end-to-end."""
    logger.info("=== Domain 2 Pipeline Start ===")

    lap_positions = build_lap_positions(positions_df, laps_df)

    proximity_events = detect_proximity_events(
        lap_positions,
        distance_threshold_m=proximity_threshold_m,
        time_resolution_s=time_resolution_s,
    )

    laps_with_regimes = classify_traffic_regimes(laps_df, proximity_events)

    laps_with_penalties = quantify_traffic_penalties(laps_with_regimes)

    overtakes = detect_overtakes(laps_with_regimes)

    hotspots = detect_overtake_hotspots(overtakes, lap_positions)

    bottlenecks = map_track_bottlenecks(lap_positions, proximity_events)

    logger.info("=== Domain 2 Pipeline Complete ===")

    return {
        "lap_positions": lap_positions,
        "proximity_events": proximity_events,
        "laps_with_regimes": laps_with_regimes,
        "laps_with_penalties": laps_with_penalties,
        "overtakes": overtakes,
        "hotspots": hotspots,
        "bottlenecks": bottlenecks,
    }
