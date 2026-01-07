"""Domain 1: Tyre Degradation Feature Engineering.

This module extracts clean-air stints and fits degradation curves for
understanding how tyres degrade in real race conditions.

Pipeline:
1. Extract clean_air_stints (laps with gap_to_ahead > threshold)
2. Fit degradation curves per stint (lap_time vs tyre_life)
3. Cluster behaviour patterns (optional)
4. Cluster track regimes (optional)
"""
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from scipy import stats
from scipy.optimize import curve_fit
import typer
from rich.console import Console
from rich.progress import Progress

from src.utils.logging_utils import setup_logger
from src.utils.parquet_writer import ParquetWriter, DATA_LAKE_PATH

app = typer.Typer()
console = Console()
logger = setup_logger(__name__)

# Thresholds for clean air detection
CLEAN_AIR_GAP_THRESHOLD = 2.5  # seconds ahead to consider "clean air"
MIN_STINT_LENGTH = 5  # minimum laps for a stint to be analyzed
MAX_STINT_LENGTH = 50  # maximum realistic stint length

# Fuel correction factor
# F1 cars use ~1.5-2kg fuel per lap, losing ~0.03-0.04s per kg
# This translates to approximately 0.06-0.08s per lap of fuel effect
FUEL_EFFECT_PER_LAP = 0.07  # seconds - car gets faster as fuel burns


class DegradationFeatureBuilder:
    """Build tyre degradation features using DuckDB + Python analysis."""
    
    def __init__(self, lake_path: Path = DATA_LAKE_PATH):
        """Initialize degradation feature builder.
        
        Args:
            lake_path: Path to data lake root
        """
        self.lake_path = Path(lake_path)
        self.conn = duckdb.connect()
        self.writer = ParquetWriter(lake_path)
        self._register_views()
    
    def _register_views(self) -> None:
        """Register Bronze and Silver layer views."""
        # Bronze tables
        bronze_tables = ['laps_raw', 'weather_raw']
        for table in bronze_tables:
            table_path = self.lake_path / "bronze" / table
            if table_path.exists():
                glob_pattern = str(table_path / "**" / "*.parquet")
                self.conn.execute(f"""
                    CREATE OR REPLACE VIEW {table} AS 
                    SELECT * FROM read_parquet('{glob_pattern}', hive_partitioning=true)
                """)
        
        # Check for positions_raw (new)
        positions_path = self.lake_path / "bronze" / "positions_raw"
        if positions_path.exists():
            glob_pattern = str(positions_path / "**" / "*.parquet")
            self.conn.execute(f"""
                CREATE OR REPLACE VIEW positions_raw AS 
                SELECT * FROM read_parquet('{glob_pattern}', hive_partitioning=true)
            """)
            logger.info("Registered positions_raw view")
    
    def extract_clean_air_stints(self, 
                                  year: Optional[int] = None,
                                  session_types: list = ['R']) -> pd.DataFrame:
        """Extract clean-air stint data from races.
        
        A "clean air stint" is a continuous sequence of laps where:
        - Driver has gap_to_ahead > CLEAN_AIR_GAP_THRESHOLD
        - Same compound throughout
        - No pit stops or safety cars interrupting
        - Track status is GREEN
        
        Args:
            year: Filter to specific year (None for all)
            session_types: Session types to include (default: races only)
            
        Returns:
            DataFrame with clean_air_stints data
        """
        logger.info("Extracting clean-air stints...")
        
        year_filter = f"AND year = {year}" if year else ""
        session_filter = f"AND session_type IN ({','.join(repr(s) for s in session_types)})"
        
        # Step 1: Compute gaps between cars using position and cumulative time
        query = f"""
        WITH cumulative_times AS (
            -- Calculate cumulative race time per driver
            SELECT 
                session_id,
                year,
                gp_name,
                session_type,
                driver,
                driver_number,
                lap_number,
                lap_time_ms,
                position,
                compound,
                stint,
                tyre_life,
                is_pit_lap,
                is_accurate,
                track_status,
                team,
                SUM(lap_time_ms) OVER (
                    PARTITION BY session_id, driver 
                    ORDER BY lap_number
                ) as cumulative_time_ms
            FROM laps_raw
            WHERE lap_time_ms IS NOT NULL 
              AND lap_time_ms > 0
              AND is_accurate = TRUE
              {year_filter}
              {session_filter}
        ),
        gaps AS (
            -- Calculate gap to car ahead (by position)
            SELECT 
                c.*,
                LAG(c.cumulative_time_ms) OVER (
                    PARTITION BY c.session_id, c.lap_number 
                    ORDER BY c.position
                ) as ahead_cumulative_ms,
                c.cumulative_time_ms - LAG(c.cumulative_time_ms) OVER (
                    PARTITION BY c.session_id, c.lap_number 
                    ORDER BY c.position
                ) as gap_to_ahead_ms
            FROM cumulative_times c
        ),
        clean_air_laps AS (
            -- Filter to clean air laps
            SELECT *,
                -- Mark as clean air if gap > threshold or in P1
                CASE 
                    WHEN position = 1 THEN TRUE
                    WHEN gap_to_ahead_ms / 1000.0 > {CLEAN_AIR_GAP_THRESHOLD} THEN TRUE
                    ELSE FALSE
                END as is_clean_air,
                -- Track status check
                CASE 
                    WHEN track_status IS NULL OR track_status = '1' THEN TRUE
                    ELSE FALSE
                END as is_green_flag
            FROM gaps
        ),
        clean_air_with_prev AS (
            -- Add previous lap info for segment detection
            SELECT *,
                LAG(is_clean_air) OVER (
                    PARTITION BY session_id, driver ORDER BY lap_number
                ) as prev_is_clean_air,
                LAG(stint) OVER (
                    PARTITION BY session_id, driver ORDER BY lap_number
                ) as prev_stint
            FROM clean_air_laps
            WHERE is_green_flag = TRUE
              AND is_pit_lap = FALSE
        ),
        stint_segments AS (
            -- Identify continuous clean-air segments within stints
            SELECT *,
                -- Create segment ID: changes when clean_air status changes or stint changes
                SUM(CASE 
                    WHEN is_clean_air != prev_is_clean_air OR stint != prev_stint THEN 1 
                    ELSE 0 
                END) OVER (
                    PARTITION BY session_id, driver ORDER BY lap_number
                ) as segment_id
            FROM clean_air_with_prev
        )
        SELECT 
            session_id,
            year,
            gp_name,
            session_type,
            driver,
            driver_number,
            team,
            compound,
            stint,
            segment_id,
            lap_number,
            lap_time_ms / 1000.0 as lap_time_s,
            tyre_life,
            position,
            gap_to_ahead_ms / 1000.0 as gap_to_ahead_s,
            is_clean_air
        FROM stint_segments
        WHERE is_clean_air = TRUE
        ORDER BY session_id, driver, lap_number
        """
        
        df = self.conn.execute(query).fetchdf()
        logger.info(f"Found {len(df):,} clean-air laps")
        
        return df
    
    def aggregate_clean_air_stints(self, clean_air_laps: pd.DataFrame) -> pd.DataFrame:
        """Aggregate clean-air laps into stint-level summaries.
        
        Args:
            clean_air_laps: DataFrame from extract_clean_air_stints
            
        Returns:
            DataFrame with one row per clean-air stint segment
        """
        logger.info("Aggregating into stint summaries...")
        
        # Group by stint segment
        stints = clean_air_laps.groupby([
            'session_id', 'year', 'gp_name', 'session_type', 
            'driver', 'driver_number', 'team', 'compound', 'stint', 'segment_id'
        ]).agg({
            'lap_number': ['min', 'max', 'count'],
            'lap_time_s': ['mean', 'std', 'min', 'max'],
            'tyre_life': ['min', 'max'],
            'position': ['mean', 'min', 'max'],
            'gap_to_ahead_s': ['mean', 'min']
        }).reset_index()
        
        # Flatten column names
        stints.columns = [
            'session_id', 'year', 'gp_name', 'session_type',
            'driver', 'driver_number', 'team', 'compound', 'stint', 'segment_id',
            'first_lap', 'last_lap', 'num_laps',
            'avg_lap_time_s', 'std_lap_time_s', 'best_lap_time_s', 'worst_lap_time_s',
            'tyre_life_start', 'tyre_life_end',
            'avg_position', 'best_position', 'worst_position',
            'avg_gap_ahead_s', 'min_gap_ahead_s'
        ]
        
        # Filter to meaningful stints
        stints = stints[stints['num_laps'] >= MIN_STINT_LENGTH].copy()
        
        # Calculate stint-level features
        stints['stint_length'] = stints['last_lap'] - stints['first_lap'] + 1
        stints['lap_time_range_s'] = stints['worst_lap_time_s'] - stints['best_lap_time_s']
        stints['tyre_age_span'] = stints['tyre_life_end'] - stints['tyre_life_start']
        
        logger.info(f"Created {len(stints):,} clean-air stint summaries")
        
        return stints
    
    def fit_degradation_curve(self, 
                               lap_times: np.ndarray, 
                               tyre_ages: np.ndarray) -> Dict[str, Any]:
        """Fit a degradation curve to stint data.
        
        Tries multiple models:
        1. Linear: lap_time = a + b * tyre_age
        2. Quadratic: lap_time = a + b * tyre_age + c * tyre_age^2
        3. Log: lap_time = a + b * log(tyre_age)
        
        Args:
            lap_times: Array of lap times (seconds)
            tyre_ages: Array of tyre ages (laps on tyres)
            
        Returns:
            Dict with best fit parameters and metrics
        """
        if len(lap_times) < MIN_STINT_LENGTH:
            return {'model': 'insufficient_data', 'r_squared': 0.0}
        
        results = {}
        
        # Linear fit
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(tyre_ages, lap_times)
            results['linear'] = {
                'model': 'linear',
                'intercept': intercept,
                'slope': slope,  # This is the degradation rate (s/lap)
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_err': std_err,
                'degradation_rate': slope  # seconds per lap of tyre age
            }
        except Exception:
            results['linear'] = {'model': 'linear', 'r_squared': 0.0}
        
        # Quadratic fit
        try:
            coeffs = np.polyfit(tyre_ages, lap_times, 2)
            predicted = np.polyval(coeffs, tyre_ages)
            ss_res = np.sum((lap_times - predicted) ** 2)
            ss_tot = np.sum((lap_times - np.mean(lap_times)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            results['quadratic'] = {
                'model': 'quadratic',
                'a': coeffs[0],  # coefficient for age^2
                'b': coeffs[1],  # coefficient for age
                'c': coeffs[2],  # intercept
                'r_squared': r_squared,
                'degradation_rate_linear': coeffs[1],
                'degradation_rate_accel': coeffs[0]  # acceleration of degradation
            }
        except Exception:
            results['quadratic'] = {'model': 'quadratic', 'r_squared': 0.0}
        
        # Select best model by R²
        best_model = max(results.values(), key=lambda x: x.get('r_squared', 0))
        best_model['all_fits'] = results
        
        return best_model
    
    def compute_stint_degradation(self, 
                                   clean_air_laps: pd.DataFrame,
                                   stints_summary: pd.DataFrame) -> pd.DataFrame:
        """Compute degradation curves for each stint.
        
        Args:
            clean_air_laps: Raw clean-air lap data
            stints_summary: Aggregated stint summaries
            
        Returns:
            stints_summary with degradation parameters added
        """
        logger.info("Fitting degradation curves...")
        
        degradation_results = []
        
        with Progress() as progress:
            task = progress.add_task("Fitting curves...", total=len(stints_summary))
            
            for idx, stint in stints_summary.iterrows():
                # Get laps for this stint
                mask = (
                    (clean_air_laps['session_id'] == stint['session_id']) &
                    (clean_air_laps['driver'] == stint['driver']) &
                    (clean_air_laps['stint'] == stint['stint']) &
                    (clean_air_laps['segment_id'] == stint['segment_id'])
                )
                stint_laps = clean_air_laps[mask].sort_values('lap_number')
                
                if len(stint_laps) >= MIN_STINT_LENGTH:
                    lap_times_raw = stint_laps['lap_time_s'].values
                    tyre_ages = stint_laps['tyre_life'].values
                    lap_numbers = stint_laps['lap_number'].values
                    
                    # Apply fuel correction: add back the time "lost" to fuel burn
                    # As fuel burns, car gets faster - we need to reverse this effect
                    # The fuel effect accumulates through the RACE (lap_number), not stint
                    # We correct relative to lap 1 of the race
                    fuel_correction = FUEL_EFFECT_PER_LAP * lap_numbers
                    lap_times_corrected = lap_times_raw + fuel_correction
                    
                    # Fit on CORRECTED lap times (raw degradation)
                    fit_result_corrected = self.fit_degradation_curve(lap_times_corrected, tyre_ages)
                    
                    # Also fit on raw for comparison
                    fit_result_raw = self.fit_degradation_curve(lap_times_raw, tyre_ages)
                    
                    degradation_results.append({
                        'session_id': stint['session_id'],
                        'driver': stint['driver'],
                        'stint': stint['stint'],
                        'segment_id': stint['segment_id'],
                        'deg_model': fit_result_corrected.get('model', 'unknown'),
                        'deg_r_squared': fit_result_corrected.get('r_squared', 0.0),
                        'deg_rate_per_lap': fit_result_corrected.get('degradation_rate', 
                                                           fit_result_corrected.get('degradation_rate_linear', 0.0)),
                        'deg_intercept': fit_result_corrected.get('intercept', fit_result_corrected.get('c', 0.0)),
                        # Also store raw (uncorrected) values for comparison
                        'deg_rate_raw': fit_result_raw.get('degradation_rate', 
                                                           fit_result_raw.get('degradation_rate_linear', 0.0)),
                        'deg_r_squared_raw': fit_result_raw.get('r_squared', 0.0),
                    })
                
                progress.advance(task)
        
        # Merge degradation results back to stints
        deg_df = pd.DataFrame(degradation_results)
        
        if not deg_df.empty:
            stints_with_deg = stints_summary.merge(
                deg_df,
                on=['session_id', 'driver', 'stint', 'segment_id'],
                how='left'
            )
        else:
            stints_with_deg = stints_summary.copy()
            stints_with_deg['deg_model'] = 'unknown'
            stints_with_deg['deg_r_squared'] = 0.0
            stints_with_deg['deg_rate_per_lap'] = 0.0
            stints_with_deg['deg_intercept'] = 0.0
            stints_with_deg['deg_rate_raw'] = 0.0
            stints_with_deg['deg_r_squared_raw'] = 0.0
        
        logger.info(f"Fitted {len(deg_df)} degradation curves")
        
        return stints_with_deg
    
    def build_all(self, 
                  year: Optional[int] = None,
                  save_to_lake: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run full Domain 1 ETL pipeline.
        
        Args:
            year: Filter to specific year
            save_to_lake: Whether to save results to data lake
            
        Returns:
            Tuple of (clean_air_laps, stints_with_degradation)
        """
        console.print("[bold blue]Domain 1: Tyre Degradation ETL[/bold blue]")
        
        # Step 1: Extract clean-air laps
        console.print("Step 1: Extracting clean-air laps...")
        clean_air_laps = self.extract_clean_air_stints(year=year)
        
        if clean_air_laps.empty:
            console.print("[yellow]No clean-air laps found[/yellow]")
            return pd.DataFrame(), pd.DataFrame()
        
        # Step 2: Aggregate into stint summaries
        console.print("Step 2: Aggregating into stints...")
        stints = self.aggregate_clean_air_stints(clean_air_laps)
        
        if stints.empty:
            console.print("[yellow]No valid stints found[/yellow]")
            return clean_air_laps, pd.DataFrame()
        
        # Step 3: Fit degradation curves
        console.print("Step 3: Fitting degradation curves...")
        stints_with_deg = self.compute_stint_degradation(clean_air_laps, stints)
        
        # Step 4: Save to data lake (Silver layer)
        if save_to_lake:
            console.print("Step 4: Saving to data lake...")
            
            # Save clean-air laps
            silver_path = self.lake_path / "silver" / "domain1"
            silver_path.mkdir(parents=True, exist_ok=True)
            
            clean_air_laps.to_parquet(
                silver_path / "clean_air_laps.parquet",
                index=False
            )
            
            stints_with_deg.to_parquet(
                silver_path / "stints_degradation.parquet",
                index=False
            )
            
            console.print(f"[green]Saved to {silver_path}[/green]")
        
        # Summary stats
        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  Clean-air laps: {len(clean_air_laps):,}")
        console.print(f"  Valid stints: {len(stints_with_deg):,}")
        if not stints_with_deg.empty:
            good_fits = (stints_with_deg['deg_r_squared'] > 0.5).sum()
            console.print(f"  Good fits (R² > 0.5): {good_fits:,} ({100*good_fits/len(stints_with_deg):.1f}%)")
            avg_deg = stints_with_deg['deg_rate_per_lap'].mean()
            console.print(f"  Avg degradation: {avg_deg*1000:.1f} ms/lap")
        
        return clean_air_laps, stints_with_deg


@app.command()
def build(
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Filter to specific year"),
    no_save: bool = typer.Option(False, "--no-save", help="Don't save to data lake"),
):
    """Build Domain 1 tyre degradation features."""
    builder = DegradationFeatureBuilder()
    clean_air_laps, stints = builder.build_all(year=year, save_to_lake=not no_save)
    
    if not stints.empty:
        console.print("\n[bold green]Domain 1 ETL complete![/bold green]")
    else:
        console.print("\n[bold yellow]No data produced - check input data[/bold yellow]")


@app.command()
def stats():
    """Show statistics about existing Domain 1 data."""
    lake_path = DATA_LAKE_PATH / "silver" / "domain1"
    
    if not lake_path.exists():
        console.print("[yellow]No Domain 1 data found. Run 'build' first.[/yellow]")
        return
    
    stints_path = lake_path / "stints_degradation.parquet"
    if stints_path.exists():
        df = pd.read_parquet(stints_path)
        console.print(f"\n[bold]Stints Degradation Data:[/bold]")
        console.print(f"  Total stints: {len(df):,}")
        console.print(f"  Years: {sorted(df['year'].unique())}")
        console.print(f"  Compounds: {sorted(df['compound'].dropna().unique())}")
        console.print(f"  Avg stint length: {df['num_laps'].mean():.1f} laps")
        console.print(f"  Avg degradation: {df['deg_rate_per_lap'].mean()*1000:.2f} ms/lap")
        
        # By compound
        console.print("\n[bold]By Compound:[/bold]")
        for compound in sorted(df['compound'].dropna().unique()):
            comp_df = df[df['compound'] == compound]
            avg_deg = comp_df['deg_rate_per_lap'].mean() * 1000
            console.print(f"  {compound}: {len(comp_df)} stints, {avg_deg:.2f} ms/lap avg deg")


if __name__ == "__main__":
    app()
