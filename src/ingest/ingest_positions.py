"""
Ingest per-driver position/telemetry data using FastF1 and persist to the
bronze data lake layer.

Bronze layer path: data/lake/bronze/positions_raw/year=YYYY/round=RR/
Output: Parquet files with position data concatenated across all drivers.

Usage
-----
    python -m src.ingest.ingest_positions               # defaults: 2023, rounds 1-22, R
    python -m src.ingest.ingest_positions --year 2024 --rounds 1 2 --sessions R
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import fastf1
import pandas as pd

from src.utils.paths import BRONZE_POSITIONS, CACHE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger(__name__)


def _enable_cache() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))
    log.debug("FastF1 cache enabled at %s", CACHE)


def _output_path(year: int, round_number: int, session_type: str) -> Path:
    return (
        BRONZE_POSITIONS
        / f"year={year}"
        / f"round={round_number:02d}"
        / f"{session_type}.parquet"
    )


def ingest_session_positions(
    year: int,
    round_number: int,
    session_type: str,
    *,
    overwrite: bool = False,
) -> Path | None:
    """Load and persist per-driver position data for a single session.

    Position data records the on-track X/Y/Z coordinates sampled at ~10 Hz.
    All drivers are concatenated into one Parquet file per session.

    Parameters
    ----------
    year, round_number, session_type:
        Identify the session uniquely.
    overwrite:
        Skip existing files when *False* (default).

    Returns
    -------
    Path | None
        Path of the written file, or *None* when skipped / no data available.
    """
    out = _output_path(year, round_number, session_type)

    if out.exists() and not overwrite:
        log.info(
            "Skipping %d round=%02d %s positions (already exists)",
            year, round_number, session_type,
        )
        return None

    log.info(
        "Loading positions %d round=%02d session=%s …", year, round_number, session_type
    )
    try:
        session = fastf1.get_session(year, round_number, session_type)
        # pos_data requires telemetry=True
        session.load(laps=True, telemetry=True, weather=False, messages=False)
    except Exception:
        log.exception(
            "Failed to load session %d round=%02d %s", year, round_number, session_type
        )
        return None

    frames: list[pd.DataFrame] = []
    for driver in session.drivers:
        try:
            pos = session.pos_data[driver]
            if pos is None or pos.empty:
                continue
            pos = pos.copy()
            pos["Driver"] = driver
            frames.append(pos)
        except (KeyError, TypeError):
            log.debug("No position data for driver %s in round %02d", driver, round_number)

    if not frames:
        log.warning(
            "No position data for %d round=%02d %s", year, round_number, session_type
        )
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined["year"] = year
    combined["round_number"] = round_number
    combined["session_type"] = session_type
    combined["EventName"] = session.event["EventName"]
    combined["CircuitKey"] = session.event.get("CircuitKey", session.event["Location"])

    # Normalise timedelta columns → float seconds
    for col in combined.select_dtypes(include=["timedelta64[ns]"]).columns:
        combined[col] = combined[col].dt.total_seconds()

    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, index=False, engine="pyarrow")
    log.info("Wrote %d rows → %s", len(combined), out)
    return out


def ingest_rounds(
    year: int,
    rounds: Sequence[int],
    sessions: Sequence[str],
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Ingest position data for multiple rounds and session types."""
    _enable_cache()
    written: list[Path] = []
    for rnd in rounds:
        for sess in sessions:
            path = ingest_session_positions(year, rnd, sess, overwrite=overwrite)
            if path is not None:
                written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest FastF1 position data to bronze layer."
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--rounds", type=int, nargs="+", default=list(range(1, 23)))
    parser.add_argument("--sessions", nargs="+", default=["R"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    written = ingest_rounds(args.year, args.rounds, args.sessions, overwrite=args.overwrite)
    log.info("Position ingestion complete. %d file(s) written.", len(written))


if __name__ == "__main__":
    main()
