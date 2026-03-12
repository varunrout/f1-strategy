"""Ingest F1 sessions to Parquet (Bronze layer) with optional SQLite.

This module provides a modern ingestion pipeline that writes to Parquet files
for the data lake architecture, with optional SQLite for backward compatibility.
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import typer
from rich.console import Console

from src.utils.logging_utils import setup_logger, log_ingestion_status
from src.utils.fastf1_utils import get_session_safely, load_session_safely, normalize_session_type
from src.utils.parquet_writer import ParquetWriter, DATA_LAKE_PATH

app = typer.Typer()
console = Console()
logger = setup_logger(__name__)


def prepare_laps_df(session, session_id: int, year: int, gp_name: str, session_type: str) -> pd.DataFrame:
    """Prepare laps DataFrame from FastF1 session (vectorized).
    
    Args:
        session: FastF1 session object
        session_id: Session ID for reference
        year: Season year
        gp_name: Grand Prix name
        session_type: Session type
        
    Returns:
        Prepared DataFrame ready for storage
    """
    laps = session.laps
    if laps.empty:
        logger.warning("No laps data available")
        return pd.DataFrame()
    
    # Build DataFrame from laps first to establish row count
    n_rows = len(laps)
    df = pd.DataFrame({
        'session_id': [session_id] * n_rows,  # Must broadcast to all rows
        'year': [year] * n_rows,
        'gp_name': [gp_name] * n_rows,
        'session_type': [session_type] * n_rows,
    })
    df['driver'] = laps['Driver'].astype(str).fillna('').values
    df['driver_number'] = laps['DriverNumber'].astype(str).fillna('').values
    df['lap_number'] = laps['LapNumber'].fillna(0).astype(int).values
    
    # Convert timedeltas to milliseconds vectorized
    df['lap_time_ms'] = (laps['LapTime'].dt.total_seconds() * 1000).values
    df['sector1_ms'] = (laps['Sector1Time'].dt.total_seconds() * 1000).values
    df['sector2_ms'] = (laps['Sector2Time'].dt.total_seconds() * 1000).values
    df['sector3_ms'] = (laps['Sector3Time'].dt.total_seconds() * 1000).values
    
    df['compound'] = laps['Compound'].where(pd.notna(laps['Compound']), None).values
    df['stint'] = laps['Stint'].where(pd.notna(laps['Stint']), None).values
    df['tyre_life'] = laps['TyreLife'].where(pd.notna(laps['TyreLife']), None).values
    df['fresh_tyre'] = laps['FreshTyre'].fillna(0).astype(int).values
    df['team'] = laps['Team'].where(pd.notna(laps['Team']), None).values
    df['track_status'] = laps['TrackStatus'].where(pd.notna(laps['TrackStatus']), None).values
    df['is_pit_lap'] = (pd.notna(laps['PitInTime']) | pd.notna(laps['PitOutTime'])).astype(int).values
    df['is_accurate'] = laps['IsAccurate'].fillna(1).astype(int).values
    df['position'] = laps['Position'].where(pd.notna(laps['Position']), None).values
    df['deleted'] = laps['Deleted'].fillna(0).astype(int).values
    df['deleted_reason'] = laps['DeletedReason'].where(pd.notna(laps['DeletedReason']), None).values
    df['fast_f1_generated'] = (laps['FastF1Generated'].fillna(0).astype(int).values if 'FastF1Generated' in laps.columns else 0)
    df['is_personal_best'] = laps['IsPersonalBest'].fillna(0).astype(int) if 'IsPersonalBest' in laps.columns else 0
    
    return df


def prepare_positions_df(session, session_id: int, year: int, gp_name: str, session_type: str) -> pd.DataFrame:
    """Prepare positions DataFrame from FastF1 session (X, Y, Z coordinates).
    
    FastF1's session.pos_data is a DICTIONARY keyed by driver number (as string),
    where each value is a DataFrame with position data (X, Y, Z, Status, Time).
    
    Args:
        session: FastF1 session object
        session_id: Session ID
        year: Season year
        gp_name: Grand Prix name
        session_type: Session type
        
    Returns:
        Prepared DataFrame ready for storage with columns:
        session_id, year, gp_name, session_type, driver_number, time_s, x, y, z, status
    """
    try:
        pos_data = session.pos_data
        
        if pos_data is None:
            logger.warning("No position data available (None)")
            return pd.DataFrame()
        
        if not isinstance(pos_data, dict):
            logger.warning(f"Unexpected pos_data type: {type(pos_data)}")
            return pd.DataFrame()
        
        if len(pos_data) == 0:
            logger.warning("No position data available (empty dict)")
            return pd.DataFrame()
        
        all_driver_dfs = []
        total_rows = 0
        
        for driver_num, driver_pos in pos_data.items():
            if driver_pos is None or not isinstance(driver_pos, pd.DataFrame):
                continue
            if driver_pos.empty:
                continue
            
            # Filter valid rows (with Time)
            if 'Time' not in driver_pos.columns:
                continue
            
            valid_data = driver_pos[pd.notna(driver_pos['Time'])].copy()
            if valid_data.empty:
                continue
            
            n_rows = len(valid_data)
            
            df = pd.DataFrame({
                'session_id': [session_id] * n_rows,
                'year': [year] * n_rows,
                'gp_name': [gp_name] * n_rows,
                'session_type': [session_type] * n_rows,
                'driver_number': [driver_num] * n_rows,
            })
            
            # Time column
            if hasattr(valid_data['Time'].iloc[0], 'total_seconds'):
                df['time_s'] = valid_data['Time'].dt.total_seconds().values
            else:
                df['time_s'] = valid_data['Time'].values
            
            # Position coordinates
            df['x'] = valid_data['X'].values if 'X' in valid_data.columns else None
            df['y'] = valid_data['Y'].values if 'Y' in valid_data.columns else None
            df['z'] = valid_data['Z'].values if 'Z' in valid_data.columns else None
            df['status'] = valid_data['Status'].values if 'Status' in valid_data.columns else None
            
            all_driver_dfs.append(df)
            total_rows += n_rows
        
        if not all_driver_dfs:
            logger.warning("No valid position data for any driver")
            return pd.DataFrame()
        
        result = pd.concat(all_driver_dfs, ignore_index=True)
        logger.info(f"Prepared {total_rows} position rows from {len(all_driver_dfs)} drivers")
        return result
        
    except Exception as e:
        logger.warning(f"Could not prepare positions: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return pd.DataFrame()


def prepare_circuit_info_df(session, session_id: int, year: int, gp_name: str, session_type: str) -> pd.DataFrame:
    """Prepare circuit info DataFrame from FastF1 session (corners and track layout).
    
    Args:
        session: FastF1 session object
        session_id: Session ID
        year: Season year
        gp_name: Grand Prix name
        session_type: Session type
        
    Returns:
        Prepared DataFrame with corner locations (X, Y, Number, Angle, Distance)
    """
    try:
        circuit_info = session.get_circuit_info()
        if circuit_info is None:
            logger.warning("No circuit info available")
            return pd.DataFrame()
        
        corners = circuit_info.corners
        if corners is None or corners.empty:
            logger.warning("No corners data available")
            return pd.DataFrame()
        
        n_rows = len(corners)
        df = pd.DataFrame({
            'session_id': [session_id] * n_rows,
            'year': [year] * n_rows,
            'gp_name': [gp_name] * n_rows,
            'session_type': [session_type] * n_rows,
            'corner_number': corners['Number'].values,
            'x': corners['X'].values,
            'y': corners['Y'].values,
            'angle': corners['Angle'].values,
            'distance': corners['Distance'].values,
        })
        
        # Add rotation (track map rotation)
        df['rotation'] = circuit_info.rotation
        
        logger.info(f"Prepared {n_rows} corner entries for circuit")
        return df
        
    except Exception as e:
        logger.warning(f"Could not prepare circuit info: {e}")
        return pd.DataFrame()


def prepare_results_df(session, session_id: int, year: int, gp_name: str, session_type: str) -> pd.DataFrame:
    """Prepare session results DataFrame (driver/team mapping, grid, finish positions).
    
    Args:
        session: FastF1 session object
        session_id: Session ID
        year: Season year
        gp_name: Grand Prix name
        session_type: Session type
        
    Returns:
        Prepared DataFrame with driver info and results
    """
    try:
        results = session.results
        if results is None or results.empty:
            logger.warning("No results data available")
            return pd.DataFrame()
        
        n_rows = len(results)
        df = pd.DataFrame({
            'session_id': [session_id] * n_rows,
            'year': [year] * n_rows,
            'gp_name': [gp_name] * n_rows,
            'session_type': [session_type] * n_rows,
            'driver_number': results['DriverNumber'].astype(str).values,
            'driver_abbrev': results['Abbreviation'].values if 'Abbreviation' in results.columns else None,
            'team_name': results['TeamName'].values if 'TeamName' in results.columns else None,
            'grid_position': results['GridPosition'].values if 'GridPosition' in results.columns else None,
            'finish_position': results['Position'].values if 'Position' in results.columns else None,
            'status': results['Status'].values if 'Status' in results.columns else None,
            'points': results['Points'].values if 'Points' in results.columns else None,
        })
        
        logger.info(f"Prepared {n_rows} results entries")
        return df
        
    except Exception as e:
        logger.warning(f"Could not prepare results: {e}")
        return pd.DataFrame()


def prepare_telemetry_df(session, session_id: int, year: int, gp_name: str, session_type: str) -> pd.DataFrame:
    """Prepare telemetry DataFrame from FastF1 session (vectorized).
    
    FastF1's session.car_data is a DICTIONARY keyed by driver number (as string),
    where each value is a Telemetry DataFrame for that driver.
    
    Args:
        session: FastF1 session object
        session_id: Session ID
        year: Season year
        gp_name: Grand Prix name
        session_type: Session type
        
    Returns:
        Prepared DataFrame ready for storage
    """
    try:
        # car_data is a dict: {driver_number_str: Telemetry DataFrame}
        car_data = session.car_data
        
        if car_data is None:
            logger.warning("No telemetry data available (None)")
            return pd.DataFrame()
        
        # car_data SHOULD be a dict - iterate over all drivers
        if not isinstance(car_data, dict):
            logger.warning(f"Unexpected car_data type: {type(car_data)}")
            return pd.DataFrame()
        
        if len(car_data) == 0:
            logger.warning("No telemetry data available (empty dict)")
            return pd.DataFrame()
        
        # Collect telemetry from all drivers
        all_driver_dfs = []
        total_rows = 0
        
        for driver_num, driver_telemetry in car_data.items():
            if driver_telemetry is None or not isinstance(driver_telemetry, pd.DataFrame):
                continue
            if driver_telemetry.empty:
                continue
            
            # Filter valid rows (with Time)
            if 'Time' not in driver_telemetry.columns:
                continue
            
            valid_data = driver_telemetry[pd.notna(driver_telemetry['Time'])].copy()
            if valid_data.empty:
                continue
            
            n_rows = len(valid_data)
            
            # Build DataFrame for this driver with proper length
            df = pd.DataFrame({
                'session_id': [session_id] * n_rows,
                'year': [year] * n_rows,
                'gp_name': [gp_name] * n_rows,
                'session_type': [session_type] * n_rows,
                'driver_number': [driver_num] * n_rows,
            })
            
            # Time columns - handle timedelta conversion
            if hasattr(valid_data['Time'].iloc[0], 'total_seconds'):
                df['time_s'] = valid_data['Time'].dt.total_seconds().values
            else:
                df['time_s'] = valid_data['Time'].values
            
            # SessionTime for absolute positioning
            if 'SessionTime' in valid_data.columns:
                if hasattr(valid_data['SessionTime'].iloc[0], 'total_seconds'):
                    df['session_time_s'] = valid_data['SessionTime'].dt.total_seconds().values
                else:
                    df['session_time_s'] = valid_data['SessionTime'].values
            
            # Car data channels
            df['speed_kph'] = valid_data['Speed'].values if 'Speed' in valid_data.columns else None
            df['rpm'] = valid_data['RPM'].values if 'RPM' in valid_data.columns else None
            df['gear'] = valid_data['nGear'].values if 'nGear' in valid_data.columns else None
            df['throttle'] = valid_data['Throttle'].values if 'Throttle' in valid_data.columns else None
            df['brake'] = valid_data['Brake'].values if 'Brake' in valid_data.columns else None
            df['drs'] = valid_data['DRS'].values if 'DRS' in valid_data.columns else None
            
            all_driver_dfs.append(df)
            total_rows += n_rows
        
        if not all_driver_dfs:
            logger.warning("No valid telemetry data for any driver")
            return pd.DataFrame()
        
        # Concatenate all drivers
        result = pd.concat(all_driver_dfs, ignore_index=True)
        logger.info(f"Prepared {total_rows} telemetry rows from {len(all_driver_dfs)} drivers")
        return result
        
    except Exception as e:
        logger.warning(f"Could not prepare telemetry: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return pd.DataFrame()


def prepare_weather_df(session, session_id: int, year: int, gp_name: str, session_type: str) -> pd.DataFrame:
    """Prepare weather DataFrame from FastF1 session."""
    try:
        weather = session.weather_data
        if weather is None or weather.empty:
            return pd.DataFrame()
        
        valid_weather = weather[pd.notna(weather['Time'])].copy()
        if valid_weather.empty:
            return pd.DataFrame()
        
        df = pd.DataFrame()
        df['session_id'] = session_id
        df['year'] = year
        df['gp_name'] = gp_name
        df['session_type'] = session_type
        df['time_s'] = valid_weather['Time'].dt.total_seconds()
        df['air_temp_c'] = valid_weather['AirTemp'].where(pd.notna(valid_weather['AirTemp']), None)
        df['track_temp_c'] = valid_weather['TrackTemp'].where(pd.notna(valid_weather['TrackTemp']), None)
        df['humidity_pct'] = valid_weather['Humidity'].where(pd.notna(valid_weather['Humidity']), None)
        df['pressure_hpa'] = valid_weather['Pressure'].where(pd.notna(valid_weather['Pressure']), None)
        df['wind_speed_mps'] = valid_weather['WindSpeed'].where(pd.notna(valid_weather['WindSpeed']), None)
        df['wind_dir_deg'] = valid_weather['WindDirection'].where(pd.notna(valid_weather['WindDirection']), None)
        df['rainfall_flag'] = valid_weather['Rainfall'].fillna(0).astype(int)
        
        return df
        
    except Exception as e:
        logger.warning(f"Could not prepare weather: {e}")
        return pd.DataFrame()


def prepare_race_control_df(session, session_id: int, year: int, gp_name: str, session_type: str) -> pd.DataFrame:
    """Prepare race control DataFrame from FastF1 session."""
    try:
        race_control = session.race_control_messages
        if race_control is None or race_control.empty:
            return pd.DataFrame()
        
        df = pd.DataFrame()
        df['session_id'] = session_id
        df['year'] = year
        df['gp_name'] = gp_name
        df['session_type'] = session_type
        df['time_s'] = race_control['Time'].dt.total_seconds().where(pd.notna(race_control['Time']), None)
        df['category'] = race_control['Category'].where(pd.notna(race_control['Category']), None)
        df['message'] = race_control['Message'].where(pd.notna(race_control['Message']), None)
        df['flag'] = race_control['Flag'].where(pd.notna(race_control['Flag']), None)
        df['lap_number'] = race_control['LapNumber'].where(pd.notna(race_control['LapNumber']), None)
        df['driver_number'] = race_control['DriverNumber'].astype(str).where(pd.notna(race_control['DriverNumber']), None)
        df['scope'] = race_control['Scope'].where(pd.notna(race_control['Scope']), None)
        df['sector'] = race_control['Sector'].where(pd.notna(race_control['Sector']), None)
        
        return df
        
    except Exception as e:
        logger.info(f"No race control data: {e}")
        return pd.DataFrame()


def ingest_session_parquet(
    year: int,
    gp_name: str,
    session_type: str,
    force: bool = False,
    lake_path: Path = DATA_LAKE_PATH,
) -> Tuple[bool, str]:
    """Ingest a single F1 session to Parquet Bronze layer.
    
    Args:
        year: Season year
        gp_name: Grand Prix name
        session_type: Session type (FP1, FP2, FP3, Q, S, R)
        force: Force re-ingestion if exists
        lake_path: Path to data lake
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    session_info = {'year': year, 'gp_name': gp_name, 'session_type': session_type}
    writer = ParquetWriter(lake_path)
    
    # Check if already exists (simple file check)
    gp_safe = gp_name.replace(" ", "_").replace("/", "_")
    check_path = lake_path / "bronze" / "laps_raw" / f"year={year}" / f"gp={gp_safe}" / f"session={session_type}"
    if check_path.exists() and not force:
        msg = f"Session already exists at {check_path}"
        logger.info(msg)
        return True, msg
    
    # Get session from FastF1
    logger.info(f"Loading session: {year} {gp_name} {session_type}")
    session = get_session_safely(year, gp_name, session_type)
    if session is None:
        return False, "Failed to get session from FastF1"
    
    # Load session data
    if not load_session_safely(session):
        return False, "Failed to load session data"
    
    # Generate a session_id (hash-based for consistency)
    session_id = hash(f"{year}_{gp_name}_{session_type}") % (10**9)
    
    # Prepare all DataFrames
    laps_df = prepare_laps_df(session, session_id, year, gp_name, session_type)
    telemetry_df = prepare_telemetry_df(session, session_id, year, gp_name, session_type)
    weather_df = prepare_weather_df(session, session_id, year, gp_name, session_type)
    rc_df = prepare_race_control_df(session, session_id, year, gp_name, session_type)
    positions_df = prepare_positions_df(session, session_id, year, gp_name, session_type)
    circuit_df = prepare_circuit_info_df(session, session_id, year, gp_name, session_type)
    results_df = prepare_results_df(session, session_id, year, gp_name, session_type)
    
    # Write to Parquet Bronze layer
    counts = {}
    
    if not laps_df.empty:
        writer.write_bronze(laps_df, "laps_raw", year, gp_name, session_type)
        counts['laps'] = len(laps_df)
    
    if not telemetry_df.empty:
        writer.write_bronze(telemetry_df, "telemetry_raw", year, gp_name, session_type)
        counts['telemetry'] = len(telemetry_df)
    
    if not weather_df.empty:
        writer.write_bronze(weather_df, "weather_raw", year, gp_name, session_type)
        counts['weather'] = len(weather_df)
    
    if not rc_df.empty:
        writer.write_bronze(rc_df, "race_control_raw", year, gp_name, session_type)
        counts['race_control'] = len(rc_df)
    
    if not positions_df.empty:
        writer.write_bronze(positions_df, "positions_raw", year, gp_name, session_type)
        counts['positions'] = len(positions_df)
    
    if not circuit_df.empty:
        writer.write_bronze(circuit_df, "circuit_info", year, gp_name, session_type)
        counts['circuit'] = len(circuit_df)
    
    if not results_df.empty:
        writer.write_bronze(results_df, "results_raw", year, gp_name, session_type)
        counts['results'] = len(results_df)
    
    message = f"Ingested: {counts}"
    log_ingestion_status(logger, session_info, 'success', message)
    return True, message


@app.command()
def main(
    year: int = typer.Option(..., "--year", "-y", help="Season year"),
    gp_name: str = typer.Option(..., "--gp", "-g", help="Grand Prix name"),
    session_type: str = typer.Option(..., "--session", "-s", help="Session type"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-ingestion"),
    lake_path: str = typer.Option(str(DATA_LAKE_PATH), "--lake", help="Data lake path"),
):
    """Ingest a single F1 session to Parquet Bronze layer."""
    console.print(f"[bold blue]Ingesting {year} {gp_name} {session_type} to Parquet[/bold blue]")
    session_type = normalize_session_type(session_type)
    
    success, message = ingest_session_parquet(
        year, gp_name, session_type, force, Path(lake_path)
    )
    
    if success:
        console.print(f"[bold green]Done! {message}[/bold green]")
    else:
        console.print(f"[bold red]Failed: {message}[/bold red]")


if __name__ == "__main__":
    app()
