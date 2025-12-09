"""Build gaps_featured table with gap analysis between cars."""
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Tuple
import typer
from rich.console import Console

from src.utils.db import DatabaseManager, attach_database, detach_database
from src.utils.logging_utils import setup_logger
from src.utils.schemas import initialize_features_core_db

app = typer.Typer()
console = Console()
logger = setup_logger(__name__)

# Default database paths
RAW_DB_PATH = Path(__file__).parent.parent.parent / "data" / "raw.db"
FEATURES_DB_PATH = Path(__file__).parent.parent.parent / "data" / "features_core.db"

# Gap thresholds
CLEAN_AIR_THRESHOLD = 2.5  # seconds
DRS_THRESHOLD = 1.0  # seconds


def build_gaps_featured(session_id: Optional[int] = None,
                       raw_db_path: Path = RAW_DB_PATH,
                       features_db_path: Path = FEATURES_DB_PATH) -> int:
    """Build gaps_featured table from lap data.
    
    This computes gaps between cars based on cumulative lap times and positions.
    
    Args:
        session_id: Process specific session (None for all)
        raw_db_path: Path to raw database
        features_db_path: Path to features database
        
    Returns:
        Number of rows inserted
    """
    # Initialize features database
    initialize_features_core_db(str(features_db_path))
    
    # Connect to features database and attach raw/self
    features_db = DatabaseManager(str(features_db_path))
    conn = features_db.get_connection()
    attach_database(conn, str(raw_db_path), 'raw')
    
    try:
        # Get list of sessions to process (only race sessions for meaningful gaps)
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute("""
                SELECT s.session_id 
                FROM raw.sessions s
                WHERE s.session_type IN ('R', 'S')
                AND s.session_id = ?
            """, (session_id,))
        else:
            cursor.execute("""
                SELECT s.session_id 
                FROM raw.sessions s
                WHERE s.session_type IN ('R', 'S')
            """)
        sessions = [row[0] for row in cursor.fetchall()]
        
        total_rows = 0
        
        for sid in sessions:
            logger.info(f"Processing gaps for session {sid}")
            
            # Get all laps with cumulative time
            cursor.execute("""
                SELECT 
                    driver,
                    lap_number,
                    position,
                    lap_time_ms
                FROM raw.laps_raw
                WHERE session_id = ?
                AND lap_time_ms IS NOT NULL
                AND deleted = 0
                ORDER BY driver, lap_number
            """, (sid,))
            
            all_laps = cursor.fetchall()
            
            # Build cumulative times for each driver
            driver_cumulative = {}  # {driver: {lap_number: cumulative_time_s}}
            
            for driver, lap_number, position, lap_time_ms in all_laps:
                if driver not in driver_cumulative:
                    driver_cumulative[driver] = {}
                
                lap_time_s = lap_time_ms / 1000.0
                
                # Calculate cumulative time
                if lap_number == 1:
                    cumulative = lap_time_s
                else:
                    prev_cumulative = driver_cumulative[driver].get(lap_number - 1, 0)
                    cumulative = prev_cumulative + lap_time_s
                
                driver_cumulative[driver][lap_number] = cumulative
            
            # Now compute gaps for each lap
            gaps_data = []
            
            # Get lap-by-lap positions
            cursor.execute("""
                SELECT 
                    lap_number,
                    driver,
                    position
                FROM raw.laps_raw
                WHERE session_id = ?
                AND position IS NOT NULL
                AND deleted = 0
                ORDER BY lap_number, position
            """, (sid,))
            
            # Group by lap number
            laps_by_number = {}
            for lap_number, driver, position in cursor.fetchall():
                if lap_number not in laps_by_number:
                    laps_by_number[lap_number] = []
                laps_by_number[lap_number].append((driver, position))
            
            # Process each lap
            for lap_number, drivers_positions in laps_by_number.items():
                # Sort by position
                drivers_positions.sort(key=lambda x: x[1])
                
                for i, (driver, position) in enumerate(drivers_positions):
                    # Get cumulative time for this driver at this lap
                    driver_time = driver_cumulative.get(driver, {}).get(lap_number)
                    if driver_time is None:
                        continue
                    
                    # Gap to car ahead
                    gap_to_ahead_s = None
                    if i > 0:
                        ahead_driver, ahead_pos = drivers_positions[i - 1]
                        ahead_time = driver_cumulative.get(ahead_driver, {}).get(lap_number)
                        if ahead_time is not None:
                            gap_to_ahead_s = driver_time - ahead_time
                    
                    # Gap to car behind
                    gap_to_behind_s = None
                    if i < len(drivers_positions) - 1:
                        behind_driver, behind_pos = drivers_positions[i + 1]
                        behind_time = driver_cumulative.get(behind_driver, {}).get(lap_number)
                        if behind_time is not None:
                            gap_to_behind_s = behind_time - driver_time
                    
                    # Compute flags
                    in_clean_air = 1 if (gap_to_ahead_s is None or gap_to_ahead_s > CLEAN_AIR_THRESHOLD) else 0
                    within_drs = 1 if (gap_to_ahead_s is not None and gap_to_ahead_s < DRS_THRESHOLD) else 0
                    
                    gaps_data.append((
                        sid,
                        driver,
                        lap_number,
                        gap_to_ahead_s,
                        gap_to_behind_s,
                        in_clean_air,
                        within_drs
                    ))
            
            # Insert gaps
            if gaps_data:
                conn.executemany("""
                    INSERT OR REPLACE INTO gaps_featured
                    (session_id, driver, lap_number, gap_to_ahead_s, gap_to_behind_s,
                     in_clean_air, within_drs)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, gaps_data)
                conn.commit()
                total_rows += len(gaps_data)
                logger.info(f"Inserted {len(gaps_data)} gap records for session {sid}")
        
        logger.info(f"Total gap records created: {total_rows}")
        return total_rows
        
    finally:
        detach_database(conn, 'raw')
        conn.close()


@app.command()
def main(
    session_id: Optional[int] = typer.Option(None, "--session-id", "-s",
                                             help="Process specific session ID (omit for all)"),
    raw_db: str = typer.Option(str(RAW_DB_PATH), "--raw-db", help="Path to raw database"),
    features_db: str = typer.Option(str(FEATURES_DB_PATH), "--features-db",
                                   help="Path to features database"),
):
    """Build gaps_featured table with gap analysis between cars.
    
    This script computes:
    - Gap to car ahead (in seconds)
    - Gap to car behind (in seconds)
    - Clean air flag (gap > 2.5s)
    - DRS range flag (gap < 1.0s)
    
    Gaps are computed using cumulative lap times and race positions.
    Only race (R) and sprint (S) sessions are processed.
    
    Examples:
        # Process all race/sprint sessions
        python -m src.features.build_gaps_featured
        
        # Process specific session
        python -m src.features.build_gaps_featured --session-id 5
    """
    console.print("[bold blue]Building gaps_featured table...[/bold blue]")
    rows = build_gaps_featured(session_id, Path(raw_db), Path(features_db))
    console.print(f"[bold green]Done! Created {rows} gap records.[/bold green]")


if __name__ == "__main__":
    app()
