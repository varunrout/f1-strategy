"""
Ingest race control messages using FastF1 and persist to the bronze layer.

Race control messages include safety car deployments, VSC periods, red flags,
DRS enabled/disabled events and penalty notifications.

Bronze layer path: data/lake/bronze/race_control_raw/year=YYYY/round=RR/
Output: Parquet files with one row per message.

Usage
-----
    python -m src.ingest.ingest_race_control               # defaults: 2023, rounds 1-22
    python -m src.ingest.ingest_race_control --year 2024 --rounds 1 5 10
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import fastf1
import pandas as pd

from src.utils.paths import BRONZE_RACE_CONTROL, CACHE

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
        BRONZE_RACE_CONTROL
        / f"year={year}"
        / f"round={round_number:02d}"
        / f"{session_type}.parquet"
    )


def ingest_session_race_control(
    year: int,
    round_number: int,
    session_type: str = "R",
    *,
    overwrite: bool = False,
) -> Path | None:
    """Load and persist race control messages for a single session.

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
            "Skipping %d round=%02d %s race_control (already exists)",
            year, round_number, session_type,
        )
        return None

    log.info(
        "Loading race control %d round=%02d session=%s …", year, round_number, session_type
    )
    try:
        session = fastf1.get_session(year, round_number, session_type)
        session.load(laps=False, telemetry=False, weather=False, messages=True)
    except Exception:
        log.exception(
            "Failed to load session %d round=%02d %s", year, round_number, session_type
        )
        return None

    try:
        messages: pd.DataFrame = session.race_control_messages
    except AttributeError:
        log.warning(
            "race_control_messages not available for %d round=%02d %s",
            year, round_number, session_type,
        )
        return None

    if messages is None or messages.empty:
        log.warning(
            "No race control messages for %d round=%02d %s",
            year, round_number, session_type,
        )
        return None

    messages = messages.copy()
    messages["year"] = year
    messages["round_number"] = round_number
    messages["session_type"] = session_type
    messages["EventName"] = session.event["EventName"]
    messages["CircuitKey"] = session.event.get("CircuitKey", session.event["Location"])

    # Normalise timedelta columns
    for col in messages.select_dtypes(include=["timedelta64[ns]"]).columns:
        messages[col] = messages[col].dt.total_seconds()

    out.parent.mkdir(parents=True, exist_ok=True)
    messages.to_parquet(out, index=False, engine="pyarrow")
    log.info("Wrote %d rows → %s", len(messages), out)
    return out


def ingest_rounds(
    year: int,
    rounds: Sequence[int],
    sessions: Sequence[str] = ("R",),
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Ingest race control data for multiple rounds and session types."""
    _enable_cache()
    written: list[Path] = []
    for rnd in rounds:
        for sess in sessions:
            path = ingest_session_race_control(year, rnd, sess, overwrite=overwrite)
            if path is not None:
                written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest FastF1 race control messages to bronze layer."
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--rounds", type=int, nargs="+", default=list(range(1, 23)))
    parser.add_argument("--sessions", nargs="+", default=["R"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    written = ingest_rounds(args.year, args.rounds, args.sessions, overwrite=args.overwrite)
    log.info("Race control ingestion complete. %d file(s) written.", len(written))


if __name__ == "__main__":
    main()
