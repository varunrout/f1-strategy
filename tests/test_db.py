"""Tests for database utilities."""
import sqlite3
import tempfile
from pathlib import Path
import pytest

from src.utils.db import DatabaseManager, attach_database, detach_database


def test_database_manager_creation():
    """Test database manager creates database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db_manager = DatabaseManager(str(db_path))
        
        # Should create parent directory
        assert db_path.parent.exists()


def test_get_connection():
    """Test getting connection with pragmas enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db_manager = DatabaseManager(str(db_path))
        conn = db_manager.get_connection()
        
        # Check pragmas are set
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys")
        assert cursor.fetchone()[0] == 1
        
        cursor.execute("PRAGMA journal_mode")
        assert cursor.fetchone()[0] == "wal"
        
        conn.close()


def test_bulk_insert():
    """Test bulk insert operation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db_manager = DatabaseManager(str(db_path))
        conn = db_manager.get_connection()
        
        # Create test table
        conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        
        # Bulk insert
        data = [(1, "Alice"), (2, "Bob"), (3, "Charlie")]
        rows = db_manager.bulk_insert(conn, "test", ["id", "name"], data)
        
        assert rows == 3
        
        # Verify data
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM test")
        assert cursor.fetchone()[0] == 3
        
        conn.close()


def test_attach_detach_database():
    """Test attaching and detaching databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db1_path = Path(tmpdir) / "db1.db"
        db2_path = Path(tmpdir) / "db2.db"
        
        # Create two databases
        conn1 = sqlite3.connect(db1_path)
        conn1.execute("CREATE TABLE table1 (id INTEGER)")
        conn1.close()
        
        conn2 = sqlite3.connect(db2_path)
        conn2.execute("CREATE TABLE table2 (id INTEGER)")
        conn2.close()
        
        # Attach db2 to db1
        conn = sqlite3.connect(db1_path)
        attach_database(conn, str(db2_path), "db2")
        
        # Should be able to query both
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        assert "table1" in tables
        
        cursor.execute("SELECT name FROM db2.sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        assert "table2" in tables
        
        detach_database(conn, "db2")
        conn.close()
