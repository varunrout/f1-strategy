"""
Domain 1 – Tyre Degradation: driving style signals.

Extracts quantitative driving-style signals from clean-air telemetry and
classifies each driver into one of three archetypes:

- **Preserver**  – conserves tyres, lower early pace, flatter deg curve
- **Balanced**   – median across all signals
- **Aggressor**  – pushes hard early, higher deg rate

Usage
-----
    python -m src.features.driving_style_signals
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils.paths import DOMAIN1_SILVER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger(__name__)

STYLE_LABELS = {0: "Preserver", 1: "Balanced", 2: "Aggressor"}
N_CLUSTERS = 3
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_clean_air_laps() -> pd.DataFrame:
    """Load clean-air laps from the silver domain1 layer."""
    path = DOMAIN1_SILVER / "clean_air_laps.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"clean_air_laps.parquet not found at {path}.\n"
            "Run `python -m src.features.domain1_degradation` first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    log.info("Loaded clean_air_laps: %d rows", len(df))
    return df


def load_stints_degradation() -> pd.DataFrame:
    """Load stints_degradation from the silver domain1 layer."""
    path = DOMAIN1_SILVER / "stints_degradation.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"stints_degradation.parquet not found at {path}.\n"
            "Run `python -m src.features.domain1_degradation` first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    log.info("Loaded stints_degradation: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 2. Consistency score
# ---------------------------------------------------------------------------

def compute_consistency_score(driver_laps: pd.DataFrame) -> pd.DataFrame:
    """Compute coefficient of variation of clean-air lap times per driver/event.

    Parameters
    ----------
    driver_laps:
        Clean-air laps DataFrame with at least ``Driver``/``driver``,
        ``EventName``/``event``, and ``LapTime`` columns.

    Returns
    -------
    pd.DataFrame
        Columns: ``driver``, ``event``, ``consistency_cv``.
        Lower CV → more consistent driver.
    """
    driver_col = next((c for c in ["Driver", "driver"] if c in driver_laps.columns), None)
    event_col = next((c for c in ["EventName", "event"] if c in driver_laps.columns), None)

    if driver_col is None or event_col is None or "LapTime" not in driver_laps.columns:
        log.warning("Required columns missing for consistency score.")
        return pd.DataFrame(columns=["driver", "event", "consistency_cv"])

    stats = (
        driver_laps
        .groupby([driver_col, event_col])["LapTime"]
        .agg(mean_lap="mean", std_lap="std")
        .reset_index()
    )
    stats["consistency_cv"] = stats["std_lap"] / stats["mean_lap"]
    stats = stats.rename(columns={driver_col: "driver", event_col: "event"})
    return stats[["driver", "event", "consistency_cv"]]


# ---------------------------------------------------------------------------
# 3. Degradation management score
# ---------------------------------------------------------------------------

def compute_degradation_management_score(stints_df: pd.DataFrame) -> pd.DataFrame:
    """Compare each driver's deg rate vs the field average per compound/event.

    A negative score means the driver degrades tyres *less* than average
    (better tyre management). A positive score means the driver is harder
    on tyres than average.

    Returns
    -------
    pd.DataFrame
        Columns: ``driver``, ``event``, ``compound``,
        ``deg_mgmt_score`` (mean over all stints in that event).
    """
    required = {"deg_rate_linear", "compound"}
    driver_col = next((c for c in ["driver", "Driver"] if c in stints_df.columns), None)
    event_col = next((c for c in ["event", "EventName"] if c in stints_df.columns), None)
    missing = required - set(stints_df.columns)

    if missing or driver_col is None or event_col is None:
        log.warning("Missing columns for deg_mgmt_score: %s", missing)
        return pd.DataFrame(columns=["driver", "event", "compound", "deg_mgmt_score"])

    df = stints_df.rename(columns={driver_col: "driver", event_col: "event"})

    # Field average per compound × event
    field_avg = (
        df.groupby(["event", "compound"])["deg_rate_linear"]
        .mean()
        .rename("field_avg_deg")
        .reset_index()
    )
    df = df.merge(field_avg, on=["event", "compound"], how="left")
    df["deg_mgmt_score"] = df["deg_rate_linear"] - df["field_avg_deg"]

    result = (
        df.groupby(["driver", "event", "compound"])["deg_mgmt_score"]
        .mean()
        .reset_index()
    )
    return result


# ---------------------------------------------------------------------------
# 4. Tyre exploitation score
# ---------------------------------------------------------------------------

def compute_tyre_exploitation_score(stints_df: pd.DataFrame) -> pd.DataFrame:
    """Measure how aggressively a driver pushes in early stint laps vs baseline.

    The exploitation score is defined as:

        exploitation = (pace_lap1..3 - stint_median_pace) / stint_median_pace

    A positive score means the driver is faster early (aggressive push),
    a negative score means they build gradually (conservative).

    Returns
    -------
    pd.DataFrame
        Columns: ``driver``, ``event``, ``exploit_score``.
    """
    clean_air_path = DOMAIN1_SILVER / "clean_air_laps.parquet"
    if not clean_air_path.exists():
        log.warning("clean_air_laps not found – exploitation score unavailable.")
        return pd.DataFrame(columns=["driver", "event", "exploit_score"])

    laps = pd.read_parquet(clean_air_path, engine="pyarrow")
    driver_col = next((c for c in ["Driver", "driver"] if c in laps.columns), None)
    event_col = next((c for c in ["EventName", "event"] if c in laps.columns), None)

    if driver_col is None or event_col is None or "LapTime" not in laps.columns:
        return pd.DataFrame(columns=["driver", "event", "exploit_score"])

    stint_group = [driver_col, event_col]
    if "stint_id" in laps.columns:
        stint_group.append("stint_id")

    scores: list[dict] = []
    for keys, group in laps.groupby(stint_group):
        group = group.sort_values("stint_lap" if "stint_lap" in group.columns else "LapNumber")
        if len(group) < 4:
            continue
        early = group.iloc[:3]["LapTime"].mean()
        median = group["LapTime"].median()
        if median == 0:
            continue
        exploit = (median - early) / median  # positive → early laps faster
        # Normalise keys to a tuple regardless of the number of group columns
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        key_dict = dict(zip(stint_group, keys_tuple))
        scores.append({
            "driver": key_dict.get(driver_col),
            "event": key_dict.get(event_col),
            "exploit_score": exploit,
        })

    if not scores:
        return pd.DataFrame(columns=["driver", "event", "exploit_score"])

    return (
        pd.DataFrame(scores)
        .groupby(["driver", "event"])["exploit_score"]
        .mean()
        .reset_index()
    )


# ---------------------------------------------------------------------------
# 5. Classify driving style
# ---------------------------------------------------------------------------

def classify_driving_style(driver_signals_df: pd.DataFrame) -> pd.DataFrame:
    """Classify drivers into Preserver / Balanced / Aggressor using KMeans.

    Parameters
    ----------
    driver_signals_df:
        DataFrame with at least ``driver``, and at least one of
        ``consistency_cv``, ``deg_mgmt_score``, ``exploit_score``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with appended ``driving_style`` column (string label)
        and ``style_cluster`` (integer 0-2).
    """
    feature_cols = [
        c for c in ["consistency_cv", "deg_mgmt_score", "exploit_score"]
        if c in driver_signals_df.columns
    ]
    if not feature_cols:
        log.error("No signal columns available for clustering.")
        driver_signals_df["driving_style"] = "Unknown"
        driver_signals_df["style_cluster"] = -1
        return driver_signals_df

    df = driver_signals_df.copy()
    X = df[feature_cols].fillna(0).to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

    df["style_cluster"] = labels

    # Map cluster centres to semantic labels.
    # The cluster with the highest exploit_score → Aggressor,
    # the cluster with the lowest (most negative) deg_mgmt_score → Preserver.
    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=feature_cols,
    )

    if "exploit_score" in centers.columns:
        aggressor_cluster = int(centers["exploit_score"].idxmax())
    else:
        aggressor_cluster = int(centers.iloc[:, -1].idxmax())

    if "deg_mgmt_score" in centers.columns:
        preserver_cluster = int(centers["deg_mgmt_score"].idxmin())
    else:
        remaining = [i for i in range(N_CLUSTERS) if i != aggressor_cluster]
        preserver_cluster = remaining[0]

    balanced_cluster = next(
        i for i in range(N_CLUSTERS)
        if i not in (aggressor_cluster, preserver_cluster)
    )

    cluster_map = {
        aggressor_cluster: "Aggressor",
        preserver_cluster: "Preserver",
        balanced_cluster: "Balanced",
    }
    df["driving_style"] = df["style_cluster"].map(cluster_map)

    log.info(
        "Driving style distribution: %s",
        df["driving_style"].value_counts().to_dict(),
    )
    return df


# ---------------------------------------------------------------------------
# 6. Save driving style signals
# ---------------------------------------------------------------------------

def save_driving_style_signals(signals_df: pd.DataFrame) -> None:
    """Write driving style signals to the silver domain1 layer."""
    DOMAIN1_SILVER.mkdir(parents=True, exist_ok=True)
    out = DOMAIN1_SILVER / "driving_style_signals.parquet"
    signals_df.to_parquet(out, index=False, engine="pyarrow")
    log.info("Saved driving_style_signals → %s (%d rows)", out, len(signals_df))


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Driving style signals pipeline starting ===")

    laps = load_clean_air_laps()
    stints = load_stints_degradation()

    consistency = compute_consistency_score(laps)
    deg_mgmt = compute_degradation_management_score(stints)
    exploitation = compute_tyre_exploitation_score(stints)

    # Aggregate deg_mgmt to driver/event level
    if not deg_mgmt.empty:
        deg_mgmt_agg = (
            deg_mgmt.groupby(["driver", "event"])["deg_mgmt_score"].mean().reset_index()
        )
    else:
        deg_mgmt_agg = pd.DataFrame(columns=["driver", "event", "deg_mgmt_score"])

    # Merge all signals on driver × event
    signals = consistency
    if not deg_mgmt_agg.empty:
        signals = signals.merge(deg_mgmt_agg, on=["driver", "event"], how="outer")
    if not exploitation.empty:
        signals = signals.merge(exploitation, on=["driver", "event"], how="outer")

    if signals.empty:
        log.error("No signals computed. Exiting.")
        return

    # Aggregate to driver level (mean across events) for classification
    signal_cols = [c for c in ["consistency_cv", "deg_mgmt_score", "exploit_score"] if c in signals.columns]
    driver_signals = signals.groupby("driver")[signal_cols].mean().reset_index()
    driver_signals = classify_driving_style(driver_signals)

    save_driving_style_signals(driver_signals)
    log.info("=== Driving style signals pipeline complete ===")


if __name__ == "__main__":
    main()
