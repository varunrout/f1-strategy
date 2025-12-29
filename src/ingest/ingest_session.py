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
    """Ingest lap data from session (vectorized).
    
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
    
    # Vectorized DataFrame preparation (100x faster than iterrows)
    df = pd.DataFrame()
    df['session_id'] = session_id
    df['driver'] = laps['Driver'].astype(str).fillna('')
    df['driver_number'] = laps['DriverNumber'].astype(str).fillna('')
    df['lap_number'] = laps['LapNumber'].fillna(0).astype(int)
    
    # Convert timedeltas to milliseconds vectorized
    df['lap_time_ms'] = laps['LapTime'].dt.total_seconds() * 1000
    df['sector1_ms'] = laps['Sector1Time'].dt.total_seconds() * 1000
    df['sector2_ms'] = laps['Sector2Time'].dt.total_seconds() * 1000
    df['sector3_ms'] = laps['Sector3Time'].dt.total_seconds() * 1000
    
    df['compound'] = laps['Compound'].where(pd.notna(laps['Compound']), None)
    df['stint'] = laps['Stint'].where(pd.notna(laps['Stint']), None)
    df['tyre_life'] = laps['TyreLife'].where(pd.notna(laps['TyreLife']), None)
    df['fresh_tyre'] = laps['FreshTyre'].fillna(0).astype(int)
    df['team'] = laps['Team'].where(pd.notna(laps['Team']), None)
    df['track_status'] = laps['TrackStatus'].where(pd.notna(laps['TrackStatus']), None)
    
    # Pit lap detection (vectorized)
    df['is_pit_lap'] = (pd.notna(laps['PitInTime']) | pd.notna(laps['PitOutTime'])).astype(int)
    
    df['is_accurate'] = laps['IsAccurate'].fillna(1).astype(int)
    df['position'] = laps['Position'].where(pd.notna(laps['Position']), None)
    df['deleted'] = laps['Deleted'].fillna(0).astype(int)
    df['deleted_reason'] = laps['DeletedReason'].where(pd.notna(laps['DeletedReason']), None)
    df['fast_f1_generated'] = laps['FastF1Generated'].fillna(0).astype(int) if 'FastF1Generated' in laps.columns else 0
    df['is_personal_best'] = laps['IsPersonalBest'].fillna(0).astype(int) if 'IsPersonalBest' in laps.columns else 0
    
    # Convert to list of tuples for executemany (still fast, single pass)
    columns = ['session_id', 'driver', 'driver_number', 'lap_number', 'lap_time_ms',
               'sector1_ms', 'sector2_ms', 'sector3_ms', 'compound', 'stint', 
               'tyre_life', 'fresh_tyre', 'team', 'track_status', 'is_pit_lap',
               'is_accurate', 'position', 'deleted', 'deleted_reason', 
               'fast_f1_generated', 'is_personal_best']
    
    # Replace NaN with None for SQLite compatibility
    df_clean = df[columns].where(pd.notna(df[columns]), None)
    lap_data = list(df_clean.itertuples(index=False, name=None))
    
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
    """Ingest telemetry data from session (vectorized).
    
    Args:
        conn: Database connection
        session_id: Session ID
        session: FastF1 session object
        
    Returns:
        Number of telemetry rows inserted
    """
    total_rows = 0
    
    try:
        # Try to get car data which includes telemetry
        car_data = session.car_data
        if car_data is None or car_data.empty:
            logger.warning("No telemetry data available")
            return 0
        
        # Vectorized processing - no more iterrows!
        # Filter out rows with no time data upfront
        valid_data = car_data[pd.notna(car_data['Time'])].copy()
        
        if valid_data.empty:
            logger.warning("No valid telemetry data after filtering")
            return 0
        
        # Build DataFrame with all columns vectorized
        df = pd.DataFrame()
        df['session_id'] = session_id
        df['driver'] = valid_data['Driver'].astype(str)
        df['lap_number'] = None  # Would need lap matching logic
        df['time_s'] = valid_data['Time'].dt.total_seconds()
        df['distance_m'] = valid_data['Distance'].where(pd.notna(valid_data['Distance']), None)
        df['speed_kph'] = valid_data['Speed'].where(pd.notna(valid_data['Speed']), None)
        df['throttle'] = valid_data['Throttle'].where(pd.notna(valid_data['Throttle']), None)
        df['brake'] = valid_data['Brake'].where(pd.notna(valid_data['Brake']), None)
        df['gear'] = valid_data['nGear'].where(pd.notna(valid_data['nGear']), None)
        df['drs'] = valid_data['DRS'].where(pd.notna(valid_data['DRS']), None)
        df['rpm'] = valid_data['RPM'].where(pd.notna(valid_data['RPM']), None)
        
        # Replace NaN with None for SQLite
        df = df.where(pd.notna(df), None)
        
        # Convert to tuples using fast itertuples (not iterrows!)
        columns = ['session_id', 'driver', 'lap_number', 'time_s', 'distance_m',
                   'speed_kph', 'throttle', 'brake', 'gear', 'drs', 'rpm']
        telemetry_data = list(df[columns].itertuples(index=False, name=None))
        
        if telemetry_data:
            cursor = conn.cursor()
            # Use batch inserts for even better performance
            BATCH_SIZE = 50000
            for i in range(0, len(telemetry_data), BATCH_SIZE):
                batch = telemetry_data[i:i + BATCH_SIZE]
                cursor.executemany("""
                    INSERT OR REPLACE INTO telemetry_raw 
                    (session_id, driver, lap_number, time_s, distance_m, speed_kph, 
                     throttle, brake, gear, drs, rpm)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
            conn.commit()
            total_rows = len(telemetry_data)
                
    except Exception as e:
        logger.warning(f"Could not ingest telemetry: {e}")
    
    return total_rows


def ingest_weather(conn: sqlite3.Connection, session_id: int, session) -> int:
    """Ingest weather data from session (vectorized).
    
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
        
        # Filter valid rows and vectorize
        valid_weather = weather[pd.notna(weather['Time'])].copy()
        if valid_weather.empty:
            return 0
        
        df = pd.DataFrame()
        df['session_id'] = session_id
        df['time_s'] = valid_weather['Time'].dt.total_seconds()
        df['air_temp_c'] = valid_weather['AirTemp'].where(pd.notna(valid_weather['AirTemp']), None)
        df['track_temp_c'] = valid_weather['TrackTemp'].where(pd.notna(valid_weather['TrackTemp']), None)
        df['humidity_pct'] = valid_weather['Humidity'].where(pd.notna(valid_weather['Humidity']), None)
        df['pressure_hpa'] = valid_weather['Pressure'].where(pd.notna(valid_weather['Pressure']), None)
        df['wind_speed_mps'] = valid_weather['WindSpeed'].where(pd.notna(valid_weather['WindSpeed']), None)
        df['wind_dir_deg'] = valid_weather['WindDirection'].where(pd.notna(valid_weather['WindDirection']), None)
        df['rainfall_flag'] = valid_weather['Rainfall'].fillna(0).astype(int)
        
        df = df.where(pd.notna(df), None)
        weather_data = list(df.itertuples(index=False, name=None))
        
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
    """Ingest race control messages (vectorized).
    
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
        
        # Vectorized processing
        df = pd.DataFrame()
        df['session_id'] = session_id
        df['time_s'] = race_control['Time'].dt.total_seconds().where(pd.notna(race_control['Time']), None)
        df['category'] = race_control['Category'].where(pd.notna(race_control['Category']), None)
        df['message'] = race_control['Message'].where(pd.notna(race_control['Message']), None)
        df['flag'] = race_control['Flag'].where(pd.notna(race_control['Flag']), None)
        df['lap_number'] = race_control['LapNumber'].where(pd.notna(race_control['LapNumber']), None)
        df['driver_number'] = race_control['DriverNumber'].astype(str).where(pd.notna(race_control['DriverNumber']), None)
        df['scope'] = race_control['Scope'].where(pd.notna(race_control['Scope']), None)
        df['sector'] = race_control['Sector'].where(pd.notna(race_control['Sector']), None)
        
        df = df.where(pd.notna(df), None)
        rc_data = list(df.itertuples(index=False, name=None))
        
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
