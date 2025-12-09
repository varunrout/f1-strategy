"""Ingest a single F1 session into raw.db."""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
import typer
from rich.console import Console

from src.utils.db import DatabaseManager, attach_database
from src.utils.logging_utils import setup_logger, log_ingestion_status
from src.utils.fastf1_utils import get_session_safely, load_session_safely, normalize_session_type
from src.utils.schemas import initialize_raw_db

app = typer.Typer()
console = Console()
logger = setup_logger(__name__)

# Default database path
DB_PATH = Path(__file__).parent.parent.parent / "data" / "raw.db"


def session_exists(conn: sqlite3.Connection, year: int, gp_name: str, 
                   session_type: str) -> Optional[int]:
    """Check if session already exists in database.
    
    Args:
        conn: Database connection
        year: Season year
        gp_name: Grand Prix name
        session_type: Session type
        
    Returns:
        session_id if exists, None otherwise
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT session_id FROM sessions WHERE year = ? AND gp_name = ? AND session_type = ?",
        (year, gp_name, session_type)
    )
    result = cursor.fetchone()
    return result[0] if result else None


def insert_session_metadata(conn: sqlite3.Connection, session, year: int, 
                           gp_name: str, session_type: str) -> int:
    """Insert session metadata and return session_id.
    
    Args:
        conn: Database connection
        session: FastF1 session object
        year: Season year
        gp_name: Grand Prix name
        session_type: Session type
        
    Returns:
        session_id of inserted session
    """
    cursor = conn.cursor()
    
    # Extract session metadata
    track_name = getattr(session.event, 'Location', None) or gp_name
    session_name = getattr(session, 'name', session_type)
    start_time = session.session_start_time.isoformat() if hasattr(session, 'session_start_time') and session.session_start_time is not None else None
    end_time = None  # Not always available
    fastf1_key = f"{year}_{gp_name}_{session_type}"
    
    cursor.execute("""
        INSERT INTO sessions (year, gp_name, track_name, session_type, session_name, 
                             start_time_utc, end_time_utc, fastf1_key)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (year, gp_name, track_name, session_type, session_name, start_time, end_time, fastf1_key))
    
    conn.commit()
    return cursor.lastrowid


def ingest_laps(conn: sqlite3.Connection, session_id: int, session) -> int:
    """Ingest lap data from session.
    
    Args:
        conn: Database connection
        session_id: Session ID
        session: FastF1 session object
        
    Returns:
        Number of laps inserted
    """
    laps = session.laps
    if laps.empty:
        logger.warning("No laps data available")
        return 0
    
    # Prepare lap data
    lap_data = []
    for idx, lap in laps.iterrows():
        lap_data.append((
            session_id,
            str(lap.get('Driver', '')),
            str(lap.get('DriverNumber', '')),
            int(lap.get('LapNumber', 0)) if pd.notna(lap.get('LapNumber')) else 0,
            float(lap['LapTime'].total_seconds() * 1000) if pd.notna(lap.get('LapTime')) else None,
            float(lap['Sector1Time'].total_seconds() * 1000) if pd.notna(lap.get('Sector1Time')) else None,
            float(lap['Sector2Time'].total_seconds() * 1000) if pd.notna(lap.get('Sector2Time')) else None,
            float(lap['Sector3Time'].total_seconds() * 1000) if pd.notna(lap.get('Sector3Time')) else None,
            str(lap.get('Compound', '')) if pd.notna(lap.get('Compound')) else None,
            int(lap.get('Stint', 0)) if pd.notna(lap.get('Stint')) else None,
            float(lap.get('TyreLife', 0)) if pd.notna(lap.get('TyreLife')) else None,
            int(lap.get('FreshTyre', 0)) if pd.notna(lap.get('FreshTyre')) else 0,
            str(lap.get('Team', '')) if pd.notna(lap.get('Team')) else None,
            str(lap.get('TrackStatus', '')) if pd.notna(lap.get('TrackStatus')) else None,
            int(lap.get('PitInTime', False) is not None or lap.get('PitOutTime', False) is not None),
            int(lap.get('IsAccurate', 1)) if pd.notna(lap.get('IsAccurate')) else 1,
            float(lap.get('Position', 0)) if pd.notna(lap.get('Position')) else None,
            int(lap.get('Deleted', 0)) if pd.notna(lap.get('Deleted')) else 0,
            str(lap.get('DeletedReason', '')) if pd.notna(lap.get('DeletedReason')) else None,
            int(lap.get('FastF1Generated', 0)) if pd.notna(lap.get('FastF1Generated')) else 0,
            int(lap.get('IsPersonalBest', 0)) if pd.notna(lap.get('IsPersonalBest')) else 0,
        ))
    
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR REPLACE INTO laps_raw 
        (session_id, driver, driver_number, lap_number, lap_time_ms, 
         sector1_ms, sector2_ms, sector3_ms, compound, stint, tyre_life, fresh_tyre,
         team, track_status, is_pit_lap, is_accurate, position, deleted, 
         deleted_reason, fast_f1_generated, is_personal_best)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, lap_data)
    conn.commit()
    
    return len(lap_data)


def ingest_telemetry(conn: sqlite3.Connection, session_id: int, session) -> int:
    """Ingest telemetry data from session.
    
    Args:
        conn: Database connection
        session_id: Session ID
        session: FastF1 session object
        
    Returns:
        Number of telemetry rows inserted
    """
    # Get telemetry for all drivers
    total_rows = 0
    
    try:
        # Try to get car data which includes telemetry
        car_data = session.car_data
        if car_data is None or car_data.empty:
            logger.warning("No telemetry data available")
            return 0
        
        # Group by driver and process
        for driver in car_data['Driver'].unique():
            driver_data = car_data[car_data['Driver'] == driver]
            
            if driver_data.empty:
                continue
            
            telemetry_data = []
            for idx, row in driver_data.iterrows():
                # Convert time to seconds from session start
                time_s = row['Time'].total_seconds() if pd.notna(row.get('Time')) else None
                if time_s is None:
                    continue
                
                telemetry_data.append((
                    session_id,
                    str(driver),
                    None,  # lap_number - would need to match with laps
                    float(time_s),
                    float(row.get('Distance', 0)) if pd.notna(row.get('Distance')) else None,
                    float(row.get('Speed', 0)) if pd.notna(row.get('Speed')) else None,
                    float(row.get('Throttle', 0)) if pd.notna(row.get('Throttle')) else None,
                    int(row.get('Brake', 0)) if pd.notna(row.get('Brake')) else None,
                    int(row.get('nGear', 0)) if pd.notna(row.get('nGear')) else None,
                    int(row.get('DRS', 0)) if pd.notna(row.get('DRS')) else None,
                    float(row.get('RPM', 0)) if pd.notna(row.get('RPM')) else None,
                ))
            
            if telemetry_data:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR REPLACE INTO telemetry_raw 
                    (session_id, driver, lap_number, time_s, distance_m, speed_kph, 
                     throttle, brake, gear, drs, rpm)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, telemetry_data)
                conn.commit()
                total_rows += len(telemetry_data)
                
    except Exception as e:
        logger.warning(f"Could not ingest telemetry: {e}")
    
    return total_rows


def ingest_weather(conn: sqlite3.Connection, session_id: int, session) -> int:
    """Ingest weather data from session.
    
    Args:
        conn: Database connection
        session_id: Session ID
        session: FastF1 session object
        
    Returns:
        Number of weather rows inserted
    """
    try:
        weather = session.weather_data
        if weather is None or weather.empty:
            logger.warning("No weather data available")
            return 0
        
        weather_data = []
        for idx, row in weather.iterrows():
            time_s = row['Time'].total_seconds() if pd.notna(row.get('Time')) else None
            if time_s is None:
                continue
            
            weather_data.append((
                session_id,
                float(time_s),
                float(row.get('AirTemp', 0)) if pd.notna(row.get('AirTemp')) else None,
                float(row.get('TrackTemp', 0)) if pd.notna(row.get('TrackTemp')) else None,
                float(row.get('Humidity', 0)) if pd.notna(row.get('Humidity')) else None,
                float(row.get('Pressure', 0)) if pd.notna(row.get('Pressure')) else None,
                float(row.get('WindSpeed', 0)) if pd.notna(row.get('WindSpeed')) else None,
                float(row.get('WindDirection', 0)) if pd.notna(row.get('WindDirection')) else None,
                int(row.get('Rainfall', 0)) if pd.notna(row.get('Rainfall')) else 0,
            ))
        
        if weather_data:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO weather_raw 
                (session_id, time_s, air_temp_c, track_temp_c, humidity_pct, 
                 pressure_hpa, wind_speed_mps, wind_dir_deg, rainfall_flag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, weather_data)
            conn.commit()
            return len(weather_data)
    
    except Exception as e:
        logger.warning(f"Could not ingest weather data: {e}")
    
    return 0


def ingest_race_control(conn: sqlite3.Connection, session_id: int, session) -> int:
    """Ingest race control messages.
    
    Args:
        conn: Database connection
        session_id: Session ID
        session: FastF1 session object
        
    Returns:
        Number of race control rows inserted
    """
    try:
        race_control = session.race_control_messages
        if race_control is None or race_control.empty:
            logger.info("No race control messages available")
            return 0
        
        rc_data = []
        for idx, row in race_control.iterrows():
            time_s = row['Time'].total_seconds() if pd.notna(row.get('Time')) else None
            
            rc_data.append((
                session_id,
                float(time_s) if time_s is not None else None,
                str(row.get('Category', '')) if pd.notna(row.get('Category')) else None,
                str(row.get('Message', '')) if pd.notna(row.get('Message')) else None,
                str(row.get('Flag', '')) if pd.notna(row.get('Flag')) else None,
                int(row.get('LapNumber', 0)) if pd.notna(row.get('LapNumber')) else None,
                str(row.get('DriverNumber', '')) if pd.notna(row.get('DriverNumber')) else None,
                str(row.get('Scope', '')) if pd.notna(row.get('Scope')) else None,
                int(row.get('Sector', 0)) if pd.notna(row.get('Sector')) else None,
            ))
        
        if rc_data:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO race_control_raw 
                (session_id, time_s, category, message, flag, lap_number, 
                 driver_number, scope, sector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rc_data)
            conn.commit()
            return len(rc_data)
    
    except Exception as e:
        logger.info(f"No race control data available: {e}")
    
    return 0


def log_ingestion(conn: sqlite3.Connection, session_id: int, status: str, message: str) -> None:
    """Log ingestion attempt to database.
    
    Args:
        conn: Database connection
        session_id: Session ID
        status: Status string
        message: Log message
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ingestion_log (session_id, status, message, ingested_at)
        VALUES (?, ?, ?, ?)
    """, (session_id, status, message, datetime.now().isoformat()))
    conn.commit()


def ingest_session(year: int, gp_name: str, session_type: str, force: bool = False,
                  db_path: Path = DB_PATH) -> None:
    """Ingest a single F1 session into raw.db.
    
    Args:
        year: Season year
        gp_name: Grand Prix name
        session_type: Session type (FP1, FP2, FP3, Q, S, R)
        force: Force re-ingestion if session exists
        db_path: Path to raw database
    """
    session_info = {'year': year, 'gp_name': gp_name, 'session_type': session_type}
    
    # Initialize database if needed
    initialize_raw_db(str(db_path))
    
    # Connect to database
    db_manager = DatabaseManager(str(db_path))
    conn = db_manager.get_connection()
    
    try:
        # Check if session already exists
        existing_session_id = session_exists(conn, year, gp_name, session_type)
        if existing_session_id and not force:
            log_ingestion_status(logger, session_info, 'skipped', 
                               f"Session already exists (id={existing_session_id})")
            log_ingestion(conn, existing_session_id, 'skipped', 
                         'Session already ingested')
            return
        
        # Get session from FastF1
        logger.info(f"Loading session: {year} {gp_name} {session_type}")
        session = get_session_safely(year, gp_name, session_type)
        if session is None:
            log_ingestion_status(logger, session_info, 'error', 
                               'Failed to get session from FastF1')
            return
        
        # Load session data
        if not load_session_safely(session):
            log_ingestion_status(logger, session_info, 'error', 
                               'Failed to load session data')
            return
        
        # Insert or update session metadata
        if existing_session_id and force:
            session_id = existing_session_id
            logger.info(f"Re-ingesting session (id={session_id})")
        else:
            session_id = insert_session_metadata(conn, session, year, gp_name, session_type)
            logger.info(f"Created session record (id={session_id})")
        
        # Ingest data
        laps_count = ingest_laps(conn, session_id, session)
        logger.info(f"Ingested {laps_count} laps")
        
        telemetry_count = ingest_telemetry(conn, session_id, session)
        logger.info(f"Ingested {telemetry_count} telemetry rows")
        
        weather_count = ingest_weather(conn, session_id, session)
        logger.info(f"Ingested {weather_count} weather rows")
        
        rc_count = ingest_race_control(conn, session_id, session)
        logger.info(f"Ingested {rc_count} race control messages")
        
        # Log success
        message = f"Laps: {laps_count}, Telemetry: {telemetry_count}, Weather: {weather_count}, RC: {rc_count}"
        log_ingestion_status(logger, session_info, 'success', message)
        log_ingestion(conn, session_id, 'success', message)
        
    except Exception as e:
        logger.error(f"Error ingesting session: {e}", exc_info=True)
        log_ingestion_status(logger, session_info, 'error', str(e))
        if 'session_id' in locals():
            log_ingestion(conn, session_id, 'error', str(e))
    finally:
        conn.close()


@app.command()
def main(
    year: int = typer.Option(..., "--year", "-y", help="Season year"),
    gp_name: str = typer.Option(..., "--gp", "-g", help="Grand Prix name"),
    session_type: str = typer.Option(..., "--session", "-s", help="Session type (FP1, FP2, FP3, Q, S, R)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-ingestion if session exists"),
    db_path: str = typer.Option(str(DB_PATH), "--db", help="Path to raw database"),
):
    """Ingest a single F1 session into raw.db."""
    console.print(f"[bold blue]Ingesting {year} {gp_name} {session_type}[/bold blue]")
    session_type = normalize_session_type(session_type)
    ingest_session(year, gp_name, session_type, force, Path(db_path))
    console.print("[bold green]Done![/bold green]")


if __name__ == "__main__":
    app()
