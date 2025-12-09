"""Database utilities for SQLite operations."""
import sqlite3
from pathlib import Path
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database connections and operations."""
    
    def __init__(self, db_path: str):
        """Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with optimized settings.
        
        Returns:
            SQLite connection object
        """
        conn = sqlite3.connect(self.db_path)
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON;")
        # Use WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL;")
        # Optimize for performance
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA cache_size = -64000;")  # 64MB cache
        return conn
    
    def execute_script(self, conn: sqlite3.Connection, script: str) -> None:
        """Execute a SQL script.
        
        Args:
            conn: Database connection
            script: SQL script to execute
        """
        conn.executescript(script)
        conn.commit()
        
    def bulk_insert(self, conn: sqlite3.Connection, table: str, 
                   columns: List[str], data: List[Tuple]) -> int:
        """Perform bulk insert operation.
        
        Args:
            conn: Database connection
            table: Table name
            columns: List of column names
            data: List of tuples containing row data
            
        Returns:
            Number of rows inserted
        """
        if not data:
            return 0
            
        placeholders = ','.join(['?' for _ in columns])
        column_names = ','.join(columns)
        query = f"INSERT OR REPLACE INTO {table} ({column_names}) VALUES ({placeholders})"
        
        cursor = conn.cursor()
        cursor.executemany(query, data)
        conn.commit()
        return cursor.rowcount


def attach_database(conn: sqlite3.Connection, db_path: str, alias: str) -> None:
    """Attach another database to the connection.
    
    Args:
        conn: Main database connection
        db_path: Path to database to attach
        alias: Alias name for the attached database
    """
    conn.execute(f"ATTACH DATABASE '{db_path}' AS {alias};")
    logger.debug(f"Attached database {db_path} as {alias}")


def detach_database(conn: sqlite3.Connection, alias: str) -> None:
    """Detach a database from the connection.
    
    Args:
        conn: Main database connection
        alias: Alias name of the attached database
    """
    conn.execute(f"DETACH DATABASE {alias};")
    logger.debug(f"Detached database {alias}")
