"""DuckDB-based feature engineering for F1 data.

This module provides fast, SQL-based feature engineering using DuckDB
to process Parquet Bronze layer data into Silver layer features.
"""
import duckdb
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console

from src.utils.logging_utils import setup_logger
from src.utils.parquet_writer import ParquetWriter, DATA_LAKE_PATH

app = typer.Typer()
console = Console()
logger = setup_logger(__name__)


class DuckDBFeatureBuilder:
    """Build features using DuckDB SQL on Parquet files."""
    
    def __init__(self, lake_path: Path = DATA_LAKE_PATH):
        """Initialize DuckDB feature builder.
        
        Args:
            lake_path: Path to data lake root
        """
        self.lake_path = Path(lake_path)
        self.conn = duckdb.connect()  # In-memory DuckDB
        
        # Register Parquet paths as views for easy querying
        self._register_bronze_views()
    
    def _register_bronze_views(self) -> None:
        """Register Bronze layer Parquet files as DuckDB views."""
        bronze_tables = ['laps_raw', 'telemetry_raw', 'weather_raw', 'race_control_raw']
        
        for table in bronze_tables:
            table_path = self.lake_path / "bronze" / table
            if table_path.exists():
                glob_pattern = str(table_path / "**" / "*.parquet")
                try:
                    self.conn.execute(f"""
                        CREATE OR REPLACE VIEW {table} AS 
                        SELECT * FROM read_parquet('{glob_pattern}', hive_partitioning=true)
                    """)
                    logger.debug(f"Registered view: {table}")
                except Exception as e:
                    logger.warning(f"Could not register {table}: {e}")
    
    def build_laps_featured(self, session_id: Optional[int] = None) -> int:
        """Build laps_featured Silver table using DuckDB SQL.
        
        This is ~100x faster than the Python loop version.
        
        Args:
            session_id: Process specific session (None for all)
            
        Returns:
            Number of rows created
        """
        logger.info("Building laps_featured with DuckDB...")
        
        # SQL-based feature engineering (replaces Python loops)
        query = """
        WITH stint_info AS (
            -- Calculate first lap of each stint for tyre age
            SELECT 
                session_id,
                driver,
                stint,
                MIN(lap_number) as stint_first_lap
            FROM laps_raw
            WHERE stint IS NOT NULL
            GROUP BY session_id, driver, stint
        ),
        session_max_laps AS (
            -- Get max lap per session for fuel proxy
            SELECT 
                session_id,
                MAX(lap_number) as max_lap
            FROM laps_raw
            GROUP BY session_id
        ),
        weather_avg AS (
            -- Average weather per session (simplified join)
            SELECT 
                session_id,
                AVG(air_temp_c) as avg_air_temp,
                AVG(track_temp_c) as avg_track_temp,
                AVG(humidity_pct) as avg_humidity,
                MAX(rainfall_flag) as had_rain
            FROM weather_raw
            GROUP BY session_id
        )
        SELECT 
            l.session_id,
            l.year,
            l.gp_name,
            l.session_type,
            l.driver,
            l.driver_number,
            l.lap_number,
            l.lap_time_ms / 1000.0 as lap_time_s,
            l.sector1_ms / 1000.0 as sector1_s,
            l.sector2_ms / 1000.0 as sector2_s,
            l.sector3_ms / 1000.0 as sector3_s,
            l.position,
            l.compound,
            l.stint,
            l.team,
            l.is_personal_best,
            
            -- Tyre age calculation (vectorized via window)
            l.lap_number - COALESCE(si.stint_first_lap, l.lap_number) + 1 as tyre_age_laps,
            
            -- Fuel proxy (fraction of race completed)
            CAST(l.lap_number AS DOUBLE) / NULLIF(sm.max_lap, 0) as fuel_proxy,
            
            -- Track status categorization
            CASE 
                WHEN l.track_status IS NULL OR l.track_status = '1' THEN 'GREEN'
                WHEN l.track_status LIKE '%2%' THEN 'YELLOW'
                WHEN l.track_status LIKE '%4%' THEN 'SC'
                WHEN l.track_status LIKE '%6%' THEN 'VSC'
                WHEN l.track_status LIKE '%5%' THEN 'RED'
                ELSE 'UNKNOWN'
            END as track_status_cat,
            
            -- In/out lap flags
            l.is_pit_lap as is_inlap,
            l.is_pit_lap as is_outlap,
            
            -- Weather data
            w.avg_air_temp,
            w.avg_track_temp,
            w.avg_humidity,
            w.had_rain,
            
            -- Lap delta to session best (window function)
            l.lap_time_ms - MIN(l.lap_time_ms) OVER (
                PARTITION BY l.session_id 
            ) as delta_to_best_ms,
            
            -- Personal best delta
            l.lap_time_ms - MIN(l.lap_time_ms) OVER (
                PARTITION BY l.session_id, l.driver
            ) as delta_to_pb_ms
            
        FROM laps_raw l
        LEFT JOIN stint_info si 
            ON l.session_id = si.session_id 
            AND l.driver = si.driver 
            AND l.stint = si.stint
        LEFT JOIN session_max_laps sm 
            ON l.session_id = sm.session_id
        LEFT JOIN weather_avg w 
            ON l.session_id = w.session_id
        WHERE l.lap_time_ms IS NOT NULL
        """
        
        if session_id:
            query += f" AND l.session_id = {session_id}"
        
        # Execute and get result
        result_df = self.conn.execute(query).fetchdf()
        
        if result_df.empty:
            logger.warning("No data to process for laps_featured")
            return 0
        
        # Write to Silver layer
        writer = ParquetWriter(self.lake_path)
        writer.write_silver(result_df, "laps_featured", partition_cols=['year', 'gp_name'])
        
        logger.info(f"Built laps_featured: {len(result_df)} rows")
        return len(result_df)
    
    def build_gaps_featured(self, session_id: Optional[int] = None) -> int:
        """Build gaps_featured Silver table using DuckDB SQL.
        
        Computes gap to car ahead/behind using window functions.
        
        Args:
            session_id: Process specific session (None for all)
            
        Returns:
            Number of rows created
        """
        logger.info("Building gaps_featured with DuckDB...")
        
        # SQL-based gap calculation (replaces Python loops!)
        query = """
        WITH cumulative_times AS (
            -- Calculate cumulative lap time per driver
            SELECT 
                session_id,
                year,
                gp_name,
                session_type,
                driver,
                lap_number,
                position,
                lap_time_ms / 1000.0 as lap_time_s,
                SUM(lap_time_ms / 1000.0) OVER (
                    PARTITION BY session_id, driver 
                    ORDER BY lap_number
                ) as cumulative_time_s
            FROM laps_raw
            WHERE lap_time_ms IS NOT NULL
            AND deleted = 0
            AND session_type IN ('R', 'S')  -- Only race/sprint for gaps
        ),
        ranked_by_position AS (
            -- Rank drivers by position each lap
            SELECT 
                *,
                LAG(cumulative_time_s) OVER (
                    PARTITION BY session_id, lap_number 
                    ORDER BY position
                ) as ahead_cumulative,
                LAG(driver) OVER (
                    PARTITION BY session_id, lap_number 
                    ORDER BY position
                ) as driver_ahead,
                LEAD(cumulative_time_s) OVER (
                    PARTITION BY session_id, lap_number 
                    ORDER BY position
                ) as behind_cumulative,
                LEAD(driver) OVER (
                    PARTITION BY session_id, lap_number 
                    ORDER BY position
                ) as driver_behind
            FROM cumulative_times
            WHERE position IS NOT NULL
        )
        SELECT 
            session_id,
            year,
            gp_name,
            session_type,
            driver,
            lap_number,
            position,
            lap_time_s,
            cumulative_time_s,
            driver_ahead,
            driver_behind,
            
            -- Gap to car ahead (positive = behind)
            cumulative_time_s - ahead_cumulative as gap_to_ahead_s,
            
            -- Gap to car behind (positive = ahead)
            behind_cumulative - cumulative_time_s as gap_to_behind_s,
            
            -- Clean air flag (>2.5s to car ahead)
            CASE 
                WHEN ahead_cumulative IS NULL THEN true  -- Leader
                WHEN (cumulative_time_s - ahead_cumulative) > 2.5 THEN true
                ELSE false
            END as in_clean_air,
            
            -- DRS range flag (<1.0s to car ahead)
            CASE 
                WHEN ahead_cumulative IS NULL THEN false  -- Leader
                WHEN (cumulative_time_s - ahead_cumulative) <= 1.0 THEN true
                ELSE false
            END as in_drs_range,
            
            -- Under pressure flag (<1.0s to car behind)
            CASE 
                WHEN behind_cumulative IS NULL THEN false  -- Last place
                WHEN (behind_cumulative - cumulative_time_s) <= 1.0 THEN true
                ELSE false
            END as under_pressure
            
        FROM ranked_by_position
        ORDER BY session_id, lap_number, position
        """
        
        if session_id:
            query = query.replace(
                "WHERE position IS NOT NULL",
                f"WHERE position IS NOT NULL AND session_id = {session_id}"
            )
        
        result_df = self.conn.execute(query).fetchdf()
        
        if result_df.empty:
            logger.warning("No data to process for gaps_featured")
            return 0
        
        # Write to Silver layer
        writer = ParquetWriter(self.lake_path)
        writer.write_silver(result_df, "gaps_featured", partition_cols=['year', 'gp_name'])
        
        logger.info(f"Built gaps_featured: {len(result_df)} rows")
        return len(result_df)
    
    def build_segments_featured(
        self, 
        session_id: Optional[int] = None,
        num_segments: int = 20
    ) -> int:
        """Build segments_featured Silver table using DuckDB SQL.
        
        Divides each lap into segments and aggregates telemetry.
        
        Args:
            session_id: Process specific session (None for all)
            num_segments: Number of segments per lap
            
        Returns:
            Number of rows created
        """
        logger.info("Building segments_featured with DuckDB...")
        
        # SQL-based segment aggregation
        query = f"""
        WITH telemetry_with_segments AS (
            SELECT 
                t.session_id,
                t.year,
                t.gp_name,
                t.session_type,
                t.driver,
                t.time_s,
                t.distance_m,
                t.speed_kph,
                t.throttle,
                t.brake,
                t.gear,
                -- Assign segment based on distance (assuming ~5km lap)
                CAST(FLOOR((t.distance_m % 5500) / (5500.0 / {num_segments})) AS INTEGER) + 1 as segment_num
            FROM telemetry_raw t
            WHERE t.speed_kph IS NOT NULL
        )
        SELECT 
            session_id,
            year,
            gp_name,
            session_type,
            driver,
            segment_num,
            
            -- Speed aggregations
            AVG(speed_kph) as mean_speed_kph,
            MAX(speed_kph) as max_speed_kph,
            MIN(speed_kph) as min_speed_kph,
            STDDEV(speed_kph) as std_speed_kph,
            
            -- Throttle/brake aggregations
            AVG(throttle) as avg_throttle,
            MAX(brake) as max_brake,
            AVG(CASE WHEN throttle > 90 THEN 1.0 ELSE 0.0 END) as full_throttle_pct,
            AVG(CASE WHEN brake > 0 THEN 1.0 ELSE 0.0 END) as braking_pct,
            
            -- Entry/exit speeds (approximation)
            FIRST(speed_kph) as entry_speed_kph,
            LAST(speed_kph) as exit_speed_kph,
            
            -- Sample count
            COUNT(*) as sample_count
            
        FROM telemetry_with_segments
        GROUP BY session_id, year, gp_name, session_type, driver, segment_num
        HAVING COUNT(*) > 10  -- Filter noise
        ORDER BY session_id, driver, segment_num
        """
        
        result_df = self.conn.execute(query).fetchdf()
        
        if result_df.empty:
            logger.warning("No data to process for segments_featured")
            return 0
        
        # Write to Silver layer
        writer = ParquetWriter(self.lake_path)
        writer.write_silver(result_df, "segments_featured", partition_cols=['year', 'gp_name'])
        
        logger.info(f"Built segments_featured: {len(result_df)} rows")
        return len(result_df)
    
    def build_all_features(self, session_id: Optional[int] = None) -> dict:
        """Build all feature tables.
        
        Args:
            session_id: Process specific session (None for all)
            
        Returns:
            Dict with row counts per table
        """
        results = {}
        
        results['laps_featured'] = self.build_laps_featured(session_id)
        results['gaps_featured'] = self.build_gaps_featured(session_id)
        results['segments_featured'] = self.build_segments_featured(session_id)
        
        return results
    
    def query(self, sql: str):
        """Execute arbitrary SQL query on the data lake.
        
        Args:
            sql: SQL query string
            
        Returns:
            Query result as DataFrame
        """
        return self.conn.execute(sql).fetchdf()
    
    def close(self):
        """Close DuckDB connection."""
        self.conn.close()


@app.command("laps")
def build_laps_cmd(
    session_id: Optional[int] = typer.Option(None, "--session", "-s"),
    lake_path: str = typer.Option(str(DATA_LAKE_PATH), "--lake"),
):
    """Build laps_featured Silver table."""
    builder = DuckDBFeatureBuilder(Path(lake_path))
    count = builder.build_laps_featured(session_id)
    console.print(f"[green]Built laps_featured: {count} rows[/green]")


@app.command("gaps")
def build_gaps_cmd(
    session_id: Optional[int] = typer.Option(None, "--session", "-s"),
    lake_path: str = typer.Option(str(DATA_LAKE_PATH), "--lake"),
):
    """Build gaps_featured Silver table."""
    builder = DuckDBFeatureBuilder(Path(lake_path))
    count = builder.build_gaps_featured(session_id)
    console.print(f"[green]Built gaps_featured: {count} rows[/green]")


@app.command("segments")
def build_segments_cmd(
    session_id: Optional[int] = typer.Option(None, "--session", "-s"),
    lake_path: str = typer.Option(str(DATA_LAKE_PATH), "--lake"),
    num_segments: int = typer.Option(20, "--segments", "-n"),
):
    """Build segments_featured Silver table."""
    builder = DuckDBFeatureBuilder(Path(lake_path))
    count = builder.build_segments_featured(session_id, num_segments)
    console.print(f"[green]Built segments_featured: {count} rows[/green]")


@app.command("all")
def build_all_cmd(
    session_id: Optional[int] = typer.Option(None, "--session", "-s"),
    lake_path: str = typer.Option(str(DATA_LAKE_PATH), "--lake"),
):
    """Build all feature tables."""
    builder = DuckDBFeatureBuilder(Path(lake_path))
    results = builder.build_all_features(session_id)
    
    console.print("[bold green]Feature build complete![/bold green]")
    for table, count in results.items():
        console.print(f"  {table}: {count} rows")


@app.command("query")
def query_cmd(
    sql: str = typer.Option(..., "--sql", "-q", help="SQL query to execute"),
    lake_path: str = typer.Option(str(DATA_LAKE_PATH), "--lake"),
):
    """Execute a SQL query on the data lake."""
    builder = DuckDBFeatureBuilder(Path(lake_path))
    result = builder.query(sql)
    console.print(result.to_string())


if __name__ == "__main__":
    app()
