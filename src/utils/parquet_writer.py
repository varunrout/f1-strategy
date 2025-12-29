"""Parquet writer utilities for Bronze/Silver layer storage."""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Literal
import pandas as pd

logger = logging.getLogger(__name__)

# Compression type
CompressionType = Literal['snappy', 'gzip', 'brotli', 'lz4', 'zstd']

# Default data lake path
DATA_LAKE_PATH = Path(__file__).parent.parent.parent / "data" / "lake"


class ParquetWriter:
    """Write DataFrames to partitioned Parquet files."""
    
    def __init__(self, base_path: Path = DATA_LAKE_PATH):
        """Initialize Parquet writer.
        
        Args:
            base_path: Root path for the data lake
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def write_bronze(
        self,
        df: pd.DataFrame,
        table: str,
        year: int,
        gp_name: str,
        session_type: str,
        compression: CompressionType = "zstd"
    ) -> Path:
        """Write DataFrame to Bronze layer (raw data).
        
        Partitions by year/gp_name/session_type for optimal query performance.
        
        Args:
            df: DataFrame to write
            table: Table name (e.g., 'telemetry_raw', 'laps_raw')
            year: Season year
            gp_name: Grand Prix name
            session_type: Session type (FP1, Q, R, etc.)
            compression: Compression codec (zstd, snappy, gzip)
            
        Returns:
            Path to written file
        """
        # Sanitize gp_name for filesystem
        gp_safe = gp_name.replace(" ", "_").replace("/", "_")
        
        # Build partition path
        partition_path = (
            self.base_path / "bronze" / table /
            f"year={year}" / f"gp={gp_safe}" / f"session={session_type}"
        )
        partition_path.mkdir(parents=True, exist_ok=True)
        
        # Write Parquet file
        file_path = partition_path / "data.parquet"
        df.to_parquet(
            file_path,
            engine="pyarrow",
            compression=compression,
            index=False
        )
        
        logger.info(f"Wrote {len(df)} rows to {file_path}")
        return file_path
    
    def write_silver(
        self,
        df: pd.DataFrame,
        table: str,
        partition_cols: Optional[list] = None,
        compression: CompressionType = "zstd"
    ) -> Path:
        """Write DataFrame to Silver layer (featured data).
        
        Args:
            df: DataFrame to write
            table: Table name (e.g., 'laps_featured', 'gaps_featured')
            partition_cols: Columns to partition by (optional)
            compression: Compression codec
            
        Returns:
            Path to written directory
        """
        table_path = self.base_path / "silver" / table
        table_path.mkdir(parents=True, exist_ok=True)
        
        if partition_cols:
            # Write with partitioning
            df.to_parquet(
                table_path,
                engine="pyarrow",
                compression=compression,
                partition_cols=partition_cols,
                index=False
            )
        else:
            # Single file
            file_path = table_path / "data.parquet"
            df.to_parquet(
                file_path,
                engine="pyarrow",
                compression=compression,
                index=False
            )
        
        logger.info(f"Wrote {len(df)} rows to silver/{table}")
        return table_path
    
    def read_bronze(
        self,
        table: str,
        year: Optional[int] = None,
        gp_name: Optional[str] = None,
        session_type: Optional[str] = None
    ) -> pd.DataFrame:
        """Read from Bronze layer with optional filters.
        
        Args:
            table: Table name
            year: Filter by year (optional)
            gp_name: Filter by GP (optional)
            session_type: Filter by session type (optional)
            
        Returns:
            DataFrame with filtered data
        """
        import pyarrow.parquet as pq
        
        table_path = self.base_path / "bronze" / table
        
        # Build glob pattern based on filters
        if year and gp_name and session_type:
            gp_safe = gp_name.replace(" ", "_").replace("/", "_")
            pattern = f"year={year}/gp={gp_safe}/session={session_type}/*.parquet"
        elif year and gp_name:
            gp_safe = gp_name.replace(" ", "_").replace("/", "_")
            pattern = f"year={year}/gp={gp_safe}/**/*.parquet"
        elif year:
            pattern = f"year={year}/**/*.parquet"
        else:
            pattern = "**/*.parquet"
        
        files = list(table_path.glob(pattern))
        
        if not files:
            logger.warning(f"No files found for {table} with pattern {pattern}")
            return pd.DataFrame()
        
        # Read all matching files
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)
    
    def get_bronze_path(self, table: str) -> Path:
        """Get the base path for a Bronze table."""
        return self.base_path / "bronze" / table
    
    def get_silver_path(self, table: str) -> Path:
        """Get the base path for a Silver table."""
        return self.base_path / "silver" / table


def get_parquet_glob_pattern(base_path: Path, table: str, layer: str = "bronze") -> str:
    """Get glob pattern for reading Parquet files.
    
    Useful for DuckDB's read_parquet() function.
    
    Args:
        base_path: Data lake base path
        table: Table name
        layer: 'bronze' or 'silver'
        
    Returns:
        Glob pattern string for read_parquet()
    """
    return str(base_path / layer / table / "**" / "*.parquet")
