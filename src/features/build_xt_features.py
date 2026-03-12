"""
build_xt_features.py
====================
Pipeline script: compute Domain 3 cross-domain (XT) features and write
them to data/features_xt.db.

Usage
-----
    python -m src.features.build_xt_features [--data-dir DATA_DIR] [--verbose]

Arguments
---------
--data-dir   Directory that contains the SQLite databases.
             Defaults to ``data/`` relative to the repository root.
--verbose    Enable DEBUG logging.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from src.features.domain3_strategy import compute_all_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = {
    "pit_window_features": """
        CREATE TABLE IF NOT EXISTS pit_window_features (
            session_id            INTEGER NOT NULL,
            driver                TEXT    NOT NULL,
            lap_number            INTEGER NOT NULL,
            compound              TEXT,
            tyre_age              REAL,
            laps_to_go            REAL,
            deg_rate_ms_per_lap   REAL,
            projected_deg_loss_ms REAL,
            pit_urgency_score     REAL,
            in_pit_window         INTEGER,
            PRIMARY KEY (session_id, driver, lap_number)
        )
    """,
    "undercut_scores": """
        CREATE TABLE IF NOT EXISTS undercut_scores (
            session_id           INTEGER NOT NULL,
            driver               TEXT    NOT NULL,
            lap_number           INTEGER NOT NULL,
            gap_score            REAL,
            deg_score            REAL,
            undercut_score       REAL,
            undercut_opportunity INTEGER,
            PRIMARY KEY (session_id, driver, lap_number)
        )
    """,
    "sc_delta_features": """
        CREATE TABLE IF NOT EXISTS sc_delta_features (
            session_id        INTEGER NOT NULL,
            driver            TEXT    NOT NULL,
            lap_number        INTEGER NOT NULL,
            is_sc             INTEGER,
            sc_delta_s        REAL,
            sc_pit_recommended INTEGER,
            PRIMARY KEY (session_id, driver, lap_number)
        )
    """,
    "strategy_scores": """
        CREATE TABLE IF NOT EXISTS strategy_scores (
            session_id         INTEGER NOT NULL,
            driver             TEXT    NOT NULL,
            lap_number         INTEGER NOT NULL,
            pit_urgency_score  REAL,
            undercut_score     REAL,
            sc_score           REAL,
            strategy_score     REAL,
            strategy_action    TEXT,
            PRIMARY KEY (session_id, driver, lap_number)
        )
    """,
}

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pw_session ON pit_window_features(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_pw_driver  ON pit_window_features(driver)",
    "CREATE INDEX IF NOT EXISTS idx_uc_session ON undercut_scores(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_sc_session ON sc_delta_features(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_ss_session ON strategy_scores(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_ss_score   ON strategy_scores(strategy_score)",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_db(path: Path) -> sqlite3.Connection:
    """Open (or create) features_xt.db and set up all tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA cache_size=-65536")  # 64 MB
    for ddl in _DDL.values():
        conn.execute(ddl)
    for idx in _INDEXES:
        conn.execute(idx)
    conn.commit()
    return conn


def _upsert(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    """Insert or replace rows into *table*. Returns row count written."""
    if df.empty:
        return 0
    cols = ", ".join(df.columns)
    placeholders = ", ".join(["?"] * len(df.columns))
    sql = f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})"
    data = [tuple(row) for row in df.itertuples(index=False, name=None)]
    conn.executemany(sql, data)
    conn.commit()
    return len(data)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(data_dir: Path) -> None:
    db_core = data_dir / "features_core.db"
    db_tyre = data_dir / "features_tyre.db"
    db_traffic = data_dir / "features_traffic.db"
    db_xt = data_dir / "features_xt.db"

    # Validate required DBs exist (warn but continue with empty frames if not)
    for db in (db_core, db_tyre, db_traffic):
        if not db.exists():
            logger.warning("Database not found: %s — using empty data frames", db)

    logger.info("Computing XT features …")
    features = compute_all_features(db_core, db_tyre, db_traffic)

    logger.info("Writing to %s …", db_xt)
    conn = _init_db(db_xt)
    try:
        mapping = {
            "pit_window_features": features["pit_window"],
            "undercut_scores": features["undercut"],
            "sc_delta_features": features["sc_delta"],
            "strategy_scores": features["strategy"],
        }
        for table, df in mapping.items():
            n = _upsert(conn, table, df)
            logger.info("  %-28s → %d rows written", table, n)
    finally:
        conn.close()

    logger.info("Done — features_xt.db updated.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Domain 3 cross-domain (XT) strategy features."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the SQLite databases (default: data/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
    )
    run_pipeline(args.data_dir)


if __name__ == "__main__":
    main()
