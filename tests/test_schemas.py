"""Tests for schema initialization."""
import sqlite3
import tempfile
from pathlib import Path
import pytest

from src.utils.schemas import (
    initialize_raw_db,
    initialize_features_core_db,
    initialize_placeholder_db
)


def test_initialize_raw_db():
    """Test raw database initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "raw.db"
        initialize_raw_db(str(db_path))
        
        assert db_path.exists()
        
        # Check tables exist
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            'sessions', 'laps_raw', 'telemetry_raw', 
            'positions_raw', 'weather_raw', 'race_control_raw',
            'ingestion_log'
        ]
        
        for table in expected_tables:
            assert table in tables, f"Table {table} not found"
        
        conn.close()


def test_initialize_features_core_db():
    """Test features core database initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "features_core.db"
        initialize_features_core_db(str(db_path))
        
        assert db_path.exists()
        
        # Check tables exist
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['laps_featured', 'gaps_featured', 'segments_featured']
        
        for table in expected_tables:
            assert table in tables, f"Table {table} not found"
        
        conn.close()


def test_initialize_placeholder_db():
    """Test placeholder database initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "placeholder.db"
        initialize_placeholder_db(str(db_path))
        
        assert db_path.exists()
        
        # Check placeholder table exists
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert '_placeholder' in tables
        
        conn.close()


def test_raw_db_indexes():
    """Test that indexes are created in raw database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "raw.db"
        initialize_raw_db(str(db_path))
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]
        
        expected_indexes = [
            'idx_laps_session_driver_lap',
            'idx_telemetry_session_driver_time',
            'idx_positions_session_driver',
            'idx_weather_session_time',
            'idx_race_control_session'
        ]
        
        for index in expected_indexes:
            assert index in indexes, f"Index {index} not found"
        
        conn.close()


def test_features_core_db_indexes():
    """Test that indexes are created in features core database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "features_core.db"
        initialize_features_core_db(str(db_path))
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]
        
        expected_indexes = [
            'idx_laps_featured_session',
            'idx_gaps_featured_session',
            'idx_segments_featured_session'
        ]
        
        for index in expected_indexes:
            assert index in indexes, f"Index {index} not found"
        
        conn.close()
