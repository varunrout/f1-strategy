"""Parquet writer utilities for Bronze/Silver layer storage."""
import logging
from pathlib import Path
from typing import Optional, Literal
import pandas as pd

logger = logging.getLogger(__name__)

CompressionType = Literal['snappy', 'gzip', 'brotli', 'lz4', 'zstd']
DATA_LAKE_PATH = Path(__file__).parent.parent.parent / "data" / "lake"


class ParquetWriter:
    """Write DataFrames to partitioned Parquet files."""

    def __init__(self, base_path: Path = DATA_LAKE_PATH):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write_bronze(self, df: pd.DataFrame, table: str, year: int, gp_name: str,
                     session_type: str, compression: CompressionType = "zstd") -> Path:
        """Write DataFrame to Bronze layer (raw data)."""
        gp_safe = gp_name.replace(" ", "_").replace("/", "_")
        partition_path = (
            self.base_path / "bronze" / table /
            f"year={year}" / f"gp={gp_safe}" / f"session={session_type}"
        )
        partition_path.mkdir(parents=True, exist_ok=True)
        file_path = partition_path / "data.parquet"
        df.to_parquet(file_path, engine="pyarrow", compression=compression, index=False)
        logger.info(f"Wrote {len(df)} rows to {file_path}")
        return file_path

    def write_silver(self, df: pd.DataFrame, table: str, partition_cols: Optional[list] = None,
                     compression: CompressionType = "zstd") -> Path:
        """Write DataFrame to Silver layer (featured data)."""
        table_path = self.base_path / "silver" / table
        table_path.mkdir(parents=True, exist_ok=True)
        if partition_cols:
            df.to_parquet(table_path, engine="pyarrow", compression=compression,
                          partition_cols=partition_cols, index=False)
        else:
            file_path = table_path / "data.parquet"
            df.to_parquet(file_path, engine="pyarrow", compression=compression, index=False)
        logger.info(f"Wrote {len(df)} rows to silver/{table}")
        return table_path

    def read_bronze(self, table: str, year: Optional[int] = None,
                    gp_name: Optional[str] = None, session_type: Optional[str] = None) -> pd.DataFrame:
        """Read from Bronze layer with optional filters."""
        table_path = self.base_path / "bronze" / table
        gp_safe = gp_name.replace(" ", "_").replace("/", "_") if gp_name else None
        if year and gp_safe and session_type:
            pattern = f"year={year}/gp={gp_safe}/session={session_type}/*.parquet"
        elif year and gp_safe:
            pattern = f"year={year}/gp={gp_safe}/**/*.parquet"
        elif year:
            pattern = f"year={year}/**/*.parquet"
        else:
            pattern = "**/*.parquet"
        files = list(table_path.glob(pattern))
        if not files:
            logger.warning(f"No files found for {table} with pattern {pattern}")
            return pd.DataFrame()
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    def get_bronze_path(self, table: str) -> Path:
        return self.base_path / "bronze" / table

    def get_silver_path(self, table: str) -> Path:
        return self.base_path / "silver" / table


def get_parquet_glob_pattern(base_path: Path, table: str, layer: str = "bronze") -> str:
    return str(base_path / layer / table / "**" / "*.parquet")
