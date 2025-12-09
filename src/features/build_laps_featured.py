"""Build laps_featured table with derived features."""
import sqlite3
from pathlib import Path
from typing import Optional
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


def categorize_track_status(track_status: str) -> str:
    """Categorize track status into simplified categories.
    
    Args:
        track_status: Raw track status string from FastF1
        
    Returns:
        Categorized status: 'GREEN', 'YELLOW', 'SC', 'VSC', 'RED', or 'UNKNOWN'
    """
    if not track_status or track_status == '1':
        return 'GREEN'
    elif '2' in track_status:
        return 'YELLOW'
    elif '4' in track_status:
        return 'SC'
    elif '6' in track_status:
        return 'VSC'
    elif '5' in track_status:
        return 'RED'
    else:
        return 'UNKNOWN'


def build_laps_featured(session_id: Optional[int] = None, 
                       raw_db_path: Path = RAW_DB_PATH,
                       features_db_path: Path = FEATURES_DB_PATH) -> int:
    """Build laps_featured table from raw data.
    
    Args:
        session_id: Process specific session (None for all)
        raw_db_path: Path to raw database
        features_db_path: Path to features database
        
    Returns:
        Number of rows inserted
    """
    # Initialize features database
    initialize_features_core_db(str(features_db_path))
    
    # Connect to features database and attach raw
    features_db = DatabaseManager(str(features_db_path))
    conn = features_db.get_connection()
    attach_database(conn, str(raw_db_path), 'raw')
    
    try:
        # Get list of sessions to process
        if session_id:
            session_filter = f"WHERE session_id = {session_id}"
        else:
            session_filter = ""
        
        cursor = conn.cursor()
        cursor.execute(f"SELECT session_id FROM raw.sessions {session_filter}")
        sessions = [row[0] for row in cursor.fetchall()]
        
        total_rows = 0
        
        for sid in sessions:
            logger.info(f"Processing laps for session {sid}")
            
            # First, compute stint starts to calculate tyre age
            cursor.execute("""
                SELECT driver, stint, MIN(lap_number) as first_lap
                FROM raw.laps_raw
                WHERE session_id = ?
                GROUP BY driver, stint
            """, (sid,))
            stint_starts = {(row[0], row[1]): row[2] for row in cursor.fetchall()}
            
            # Get total laps for fuel proxy calculation
            cursor.execute("""
                SELECT MAX(lap_number) as max_lap
                FROM raw.laps_raw
                WHERE session_id = ?
            """, (sid,))
            max_lap = cursor.fetchone()[0] or 1
            
            # Build featured laps
            cursor.execute("""
                SELECT 
                    l.session_id,
                    l.driver,
                    l.lap_number,
                    l.lap_time_ms,
                    l.sector1_ms,
                    l.sector2_ms,
                    l.sector3_ms,
                    l.position,
                    l.compound,
                    l.stint,
                    l.track_status,
                    l.is_pit_lap,
                    l.team,
                    l.is_personal_best
                FROM raw.laps_raw l
                WHERE l.session_id = ?
                ORDER BY l.driver, l.lap_number
            """, (sid,))
            
            laps = cursor.fetchall()
            featured_data = []
            
            for lap in laps:
                (session_id, driver, lap_number, lap_time_ms, 
                 sector1_ms, sector2_ms, sector3_ms, position,
                 compound, stint, track_status, is_pit_lap, team, is_personal_best) = lap
                
                # Convert times from ms to seconds
                lap_time_s = lap_time_ms / 1000.0 if lap_time_ms else None
                sector1_s = sector1_ms / 1000.0 if sector1_ms else None
                sector2_s = sector2_ms / 1000.0 if sector2_ms else None
                sector3_s = sector3_ms / 1000.0 if sector3_ms else None
                
                # Calculate tyre age
                stint_key = (driver, stint)
                first_lap_of_stint = stint_starts.get(stint_key, lap_number)
                tyre_age_laps = lap_number - first_lap_of_stint + 1 if stint else None
                
                # Calculate fuel proxy (fraction of race completed)
                fuel_proxy = lap_number / max_lap if max_lap > 0 else None
                
                # Categorize track status
                track_status_cat = categorize_track_status(track_status)
                
                # Determine in/out lap flags
                # Simple heuristic: pit lap is both in and out lap for simplicity
                is_inlap = is_pit_lap
                is_outlap = is_pit_lap
                
                # Get weather data (nearest in time)
                # Approximate lap time as lap_number * 90 seconds (rough estimate)
                approx_time = lap_number * 90
                cursor.execute("""
                    SELECT air_temp_c, track_temp_c
                    FROM raw.weather_raw
                    WHERE session_id = ?
                    ORDER BY ABS(time_s - ?)
                    LIMIT 1
                """, (sid, approx_time))
                weather = cursor.fetchone()
                air_temp = weather[0] if weather else None
                track_temp = weather[1] if weather else None
                
                featured_data.append((
                    session_id, driver, lap_number, lap_time_s,
                    sector1_s, sector2_s, sector3_s, position,
                    compound, stint, tyre_age_laps, fuel_proxy,
                    track_status_cat, is_inlap, is_outlap, is_pit_lap,
                    air_temp, track_temp, team, is_personal_best
                ))
            
            # Insert featured laps
            if featured_data:
                conn.executemany("""
                    INSERT OR REPLACE INTO laps_featured
                    (session_id, driver, lap_number, lap_time_s,
                     sector1_s, sector2_s, sector3_s, position,
                     compound, stint, tyre_age_laps, fuel_proxy,
                     track_status_cat, is_inlap, is_outlap, is_pit_lap,
                     air_temp_c, track_temp_c, team, is_personal_best)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, featured_data)
                conn.commit()
                total_rows += len(featured_data)
                logger.info(f"Inserted {len(featured_data)} featured laps for session {sid}")
        
        logger.info(f"Total featured laps created: {total_rows}")
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
    """Build laps_featured table with derived features.
    
    This script reads from raw.db and creates enriched lap data in features_core.db,
    including:
    - Tyre age calculation
    - Fuel proxy estimation
    - Track status categorization
    - In/out lap identification
    - Weather data joining
    
    Examples:
        # Process all sessions
        python -m src.features.build_laps_featured
        
        # Process specific session
        python -m src.features.build_laps_featured --session-id 1
    """
    console.print("[bold blue]Building laps_featured table...[/bold blue]")
    rows = build_laps_featured(session_id, Path(raw_db), Path(features_db))
    console.print(f"[bold green]Done! Created {rows} featured laps.[/bold green]")


if __name__ == "__main__":
    app()
