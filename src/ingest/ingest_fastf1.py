"""
Ingest F1 lap data using FastF1 and persist to the bronze data lake layer.

Bronze layer path: data/lake/bronze/laps_raw/year=YYYY/round=RR/
Output: Parquet files with raw lap telemetry keyed by session type.

Usage
-----
    python -m src.ingest.ingest_fastf1               # defaults: 2023, rounds 1-22, R+Q
    python -m src.ingest.ingest_fastf1 --year 2024 --rounds 1 2 3 --sessions R
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import fastf1
import pandas as pd

from src.utils.paths import BRONZE_LAPS, CACHE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger(__name__)

# Columns to keep from the raw laps DataFrame (subset to reduce storage)
_LAP_COLUMNS = [
    "Driver",
    "DriverNumber",
    "LapTime",
    "LapNumber",
    "Stint",
    "PitOutTime",
    "PitInTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "LapStartTime",
    "Team",
    "TrackStatus",
    "Position",
    "IsPersonalBest",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
]


def _enable_cache() -> None:
    """Enable FastF1 file-based cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))
    log.debug("FastF1 cache enabled at %s", CACHE)


def _output_path(year: int, round_number: int, session_type: str) -> Path:
    return (
        BRONZE_LAPS
        / f"year={year}"
        / f"round={round_number:02d}"
        / f"{session_type}.parquet"
    )


def ingest_session(
    year: int,
    round_number: int,
    session_type: str,
    *,
    overwrite: bool = False,
) -> Path | None:
    """Ingest a single session and write a Parquet file to the bronze layer.

    Parameters
    ----------
    year:
        Championship year (e.g. 2023).
    round_number:
        Round number within the championship (1-based).
    session_type:
        FastF1 session identifier, e.g. ``'R'`` (race) or ``'Q'`` (qualifying).
    overwrite:
        When *False* (default) skip rounds that have already been ingested.

    Returns
    -------
    Path | None
        Path of the written Parquet file, or *None* when skipped.
    """
    out = _output_path(year, round_number, session_type)

    if out.exists() and not overwrite:
        log.info("Skipping %d round=%02d %s (already exists)", year, round_number, session_type)
        return None

    log.info("Loading %d round=%02d session=%s …", year, round_number, session_type)
    try:
        session = fastf1.get_session(year, round_number, session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception:
        log.exception(
            "Failed to load %d round=%02d %s", year, round_number, session_type
        )
        return None

    laps: pd.DataFrame = session.laps.copy()
    if laps.empty:
        log.warning("No lap data for %d round=%02d %s", year, round_number, session_type)
        return None

    # Keep only available columns to guard against API changes
    available = [c for c in _LAP_COLUMNS if c in laps.columns]
    laps = laps[available].copy()

    # Enrich with session-level metadata
    laps["year"] = year
    laps["round_number"] = round_number
    laps["session_type"] = session_type
    laps["EventName"] = session.event["EventName"]
    laps["CircuitKey"] = session.event.get("CircuitKey", session.event["Location"])

    # Convert timedelta columns to float seconds for Parquet compatibility
    for col in laps.select_dtypes(include=["timedelta64[ns]"]).columns:
        laps[col] = laps[col].dt.total_seconds()

    out.parent.mkdir(parents=True, exist_ok=True)
    laps.to_parquet(out, index=False, engine="pyarrow")
    log.info("Wrote %d rows → %s", len(laps), out)
    return out


def ingest_rounds(
    year: int,
    rounds: Sequence[int],
    sessions: Sequence[str],
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Ingest multiple rounds and session types for a given year.

    Parameters
    ----------
    year:
        Championship year.
    rounds:
        Iterable of round numbers to ingest.
    sessions:
        List of session type codes, e.g. ``['R', 'Q']``.
    overwrite:
        Passed through to :func:`ingest_session`.

    Returns
    -------
    list[Path]
        Paths of all successfully written Parquet files.
    """
    _enable_cache()
    written: list[Path] = []
    for rnd in rounds:
        for sess in sessions:
            path = ingest_session(year, rnd, sess, overwrite=overwrite)
            if path is not None:
                written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest FastF1 lap data to bronze layer.")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--rounds", type=int, nargs="+", default=list(range(1, 23)))
    parser.add_argument("--sessions", nargs="+", default=["R", "Q"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    written = ingest_rounds(args.year, args.rounds, args.sessions, overwrite=args.overwrite)
    log.info("Ingestion complete. %d file(s) written.", len(written))


if __name__ == "__main__":
    main()
