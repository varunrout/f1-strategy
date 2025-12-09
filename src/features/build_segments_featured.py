"""Build segments_featured table with telemetry-based segment analysis."""
import sqlite3
from pathlib import Path
from typing import Optional, List, Tuple
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

# Segmentation configuration
NUM_SEGMENTS = 20  # Divide each lap into N segments


def aggregate_segment(telemetry_rows: List[Tuple]) -> Optional[Tuple]:
    """Aggregate telemetry data for a segment.
    
    Args:
        telemetry_rows: List of telemetry rows for the segment
        
    Returns:
        Aggregated segment data tuple or None
    """
    if not telemetry_rows:
        return None
    
    speeds = [row[0] for row in telemetry_rows if row[0] is not None]
    throttles = [row[1] for row in telemetry_rows if row[1] is not None]
    brakes = [row[2] for row in telemetry_rows if row[2] is not None]
    times = [row[3] for row in telemetry_rows if row[3] is not None]
    
    if not speeds:
        return None
    
    mean_speed = sum(speeds) / len(speeds)
    max_speed = max(speeds)
    min_speed = min(speeds)
    avg_throttle = sum(throttles) / len(throttles) if throttles else None
    max_brake = max(brakes) if brakes else None
    time_in_segment = max(times) - min(times) if len(times) >= 2 else 0
    entry_speed = speeds[0] if speeds else None
    exit_speed = speeds[-1] if speeds else None
    
    return (mean_speed, max_speed, min_speed, avg_throttle, max_brake, 
            time_in_segment, entry_speed, exit_speed)


def build_segments_featured(session_id: Optional[int] = None,
                           raw_db_path: Path = RAW_DB_PATH,
                           features_db_path: Path = FEATURES_DB_PATH,
                           num_segments: int = NUM_SEGMENTS) -> int:
    """Build segments_featured table from telemetry data.
    
    Args:
        session_id: Process specific session (None for all)
        raw_db_path: Path to raw database
        features_db_path: Path to features database
        num_segments: Number of segments to divide each lap into
        
    Returns:
        Number of rows inserted
    """
    # Initialize features database
    initialize_features_core_db(str(features_db_path))
    
    # Connect to features database and attach raw/features
    features_db = DatabaseManager(str(features_db_path))
    conn = features_db.get_connection()
    attach_database(conn, str(raw_db_path), 'raw')
    
    try:
        # Get list of sessions to process
        cursor = conn.cursor()
        
        if session_id:
            session_filter = f"WHERE session_id = {session_id}"
        else:
            session_filter = ""
        
        cursor.execute(f"SELECT session_id FROM raw.sessions {session_filter}")
        sessions = [row[0] for row in cursor.fetchall()]
        
        total_rows = 0
        
        for sid in sessions:
            logger.info(f"Processing segments for session {sid}")
            
            # Get list of driver-lap combinations
            cursor.execute("""
                SELECT DISTINCT driver, lap_number
                FROM raw.laps_raw
                WHERE session_id = ?
                AND lap_time_ms IS NOT NULL
                ORDER BY driver, lap_number
            """, (sid,))
            
            driver_laps = cursor.fetchall()
            
            segments_data = []
            
            for driver, lap_number in driver_laps:
                # Get telemetry for this driver-lap
                # We'll approximate lap boundaries using distance
                cursor.execute("""
                    SELECT distance_m
                    FROM raw.telemetry_raw
                    WHERE session_id = ? AND driver = ?
                    ORDER BY distance_m
                """, (sid, driver))
                
                distances = [row[0] for row in cursor.fetchall() if row[0] is not None]
                
                if not distances or len(distances) < 2:
                    continue
                
                # Estimate lap distance (use max distance / number of laps, or approximate)
                # For simplicity, we'll use distance ranges per lap
                # This is a heuristic - in production, you'd want better lap matching
                
                # Get total distance range for this driver
                min_dist = min(distances)
                max_dist = max(distances)
                total_distance = max_dist - min_dist
                
                # Estimate distance per lap (typical F1 circuit is 5-7km)
                # We'll get telemetry for an approximate lap
                cursor.execute("""
                    SELECT MIN(distance_m), MAX(distance_m)
                    FROM raw.telemetry_raw
                    WHERE session_id = ? AND driver = ?
                """, (sid, driver))
                
                dist_range = cursor.fetchone()
                if not dist_range or dist_range[0] is None:
                    continue
                
                min_dist, max_dist = dist_range
                lap_distance = (max_dist - min_dist) / max(lap_number, 1)
                
                # Approximate distance range for this lap
                lap_start_dist = min_dist + (lap_number - 1) * lap_distance
                lap_end_dist = lap_start_dist + lap_distance
                
                # Get telemetry for this approximate lap
                cursor.execute("""
                    SELECT speed_kph, throttle, brake, time_s, distance_m
                    FROM raw.telemetry_raw
                    WHERE session_id = ? AND driver = ?
                    AND distance_m BETWEEN ? AND ?
                    ORDER BY distance_m
                """, (sid, driver, lap_start_dist, lap_end_dist))
                
                telemetry = cursor.fetchall()
                
                if len(telemetry) < num_segments:
                    continue
                
                # Divide into segments
                segment_size = len(telemetry) // num_segments
                
                # Get compound and tyre age from laps_featured if available
                cursor.execute("""
                    SELECT compound, tyre_age_laps, fuel_proxy
                    FROM laps_featured
                    WHERE session_id = ? AND driver = ? AND lap_number = ?
                """, (sid, driver, lap_number))
                
                lap_features = cursor.fetchone()
                if lap_features:
                    compound, tyre_age_laps, fuel_proxy = lap_features
                else:
                    # Fallback to raw laps
                    cursor.execute("""
                        SELECT compound
                        FROM raw.laps_raw
                        WHERE session_id = ? AND driver = ? AND lap_number = ?
                    """, (sid, driver, lap_number))
                    lap_raw = cursor.fetchone()
                    compound = lap_raw[0] if lap_raw else None
                    tyre_age_laps = None
                    fuel_proxy = None
                
                # Process each segment
                for seg_id in range(num_segments):
                    start_idx = seg_id * segment_size
                    end_idx = start_idx + segment_size if seg_id < num_segments - 1 else len(telemetry)
                    
                    segment_telemetry = telemetry[start_idx:end_idx]
                    
                    if not segment_telemetry:
                        continue
                    
                    # Get distance boundaries
                    distance_start_m = segment_telemetry[0][4] if segment_telemetry[0][4] else None
                    distance_end_m = segment_telemetry[-1][4] if segment_telemetry[-1][4] else None
                    
                    # Aggregate telemetry
                    agg_data = aggregate_segment([(row[0], row[1], row[2], row[3]) 
                                                  for row in segment_telemetry])
                    
                    if agg_data is None:
                        continue
                    
                    (mean_speed, max_speed, min_speed, avg_throttle, max_brake,
                     time_in_segment, entry_speed, exit_speed) = agg_data
                    
                    segments_data.append((
                        sid,
                        driver,
                        lap_number,
                        seg_id,
                        distance_start_m,
                        distance_end_m,
                        mean_speed,
                        max_speed,
                        min_speed,
                        avg_throttle,
                        max_brake,
                        time_in_segment,
                        entry_speed,
                        exit_speed,
                        compound,
                        tyre_age_laps,
                        fuel_proxy
                    ))
            
            # Insert segments
            if segments_data:
                conn.executemany("""
                    INSERT OR REPLACE INTO segments_featured
                    (session_id, driver, lap_number, segment_id,
                     distance_start_m, distance_end_m, mean_speed_kph, max_speed_kph,
                     min_speed_kph, avg_throttle, max_brake, time_in_segment_s,
                     entry_speed_kph, exit_speed_kph, compound, tyre_age_laps, fuel_proxy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, segments_data)
                conn.commit()
                total_rows += len(segments_data)
                logger.info(f"Inserted {len(segments_data)} segment records for session {sid}")
        
        logger.info(f"Total segment records created: {total_rows}")
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
    num_segments: int = typer.Option(NUM_SEGMENTS, "--segments", "-n",
                                    help="Number of segments per lap"),
):
    """Build segments_featured table with telemetry-based segment analysis.
    
    This script divides each lap into fixed-size segments and aggregates
    telemetry data for each segment, including:
    - Speed statistics (mean, max, min, entry, exit)
    - Throttle and brake usage
    - Time spent in segment
    - Tyre compound and age
    - Fuel proxy
    
    Examples:
        # Process all sessions with 20 segments per lap
        python -m src.features.build_segments_featured
        
        # Process specific session with 30 segments
        python -m src.features.build_segments_featured --session-id 1 --segments 30
    """
    console.print("[bold blue]Building segments_featured table...[/bold blue]")
    rows = build_segments_featured(session_id, Path(raw_db), Path(features_db), num_segments)
    console.print(f"[bold green]Done! Created {rows} segment records.[/bold green]")


if __name__ == "__main__":
    app()
