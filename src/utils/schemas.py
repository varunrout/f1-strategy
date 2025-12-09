"""Database schema definitions for raw and feature databases."""

RAW_DB_SCHEMA = """
-- Sessions table: One row per F1 session
CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL,
    gp_name TEXT NOT NULL,
    track_name TEXT,
    session_type TEXT NOT NULL,
    session_name TEXT,
    start_time_utc TEXT,
    end_time_utc TEXT,
    fastf1_key TEXT,
    UNIQUE(year, gp_name, session_type)
);

-- Laps raw data
CREATE TABLE IF NOT EXISTS laps_raw (
    session_id INTEGER NOT NULL,
    driver TEXT NOT NULL,
    driver_number TEXT,
    lap_number INTEGER NOT NULL,
    lap_time_ms REAL,
    sector1_ms REAL,
    sector2_ms REAL,
    sector3_ms REAL,
    compound TEXT,
    stint INTEGER,
    tyre_life REAL,
    fresh_tyre INTEGER,
    team TEXT,
    track_status TEXT,
    is_pit_lap INTEGER,
    is_accurate INTEGER,
    position REAL,
    deleted INTEGER,
    deleted_reason TEXT,
    fast_f1_generated INTEGER,
    is_personal_best INTEGER,
    PRIMARY KEY (session_id, driver, lap_number),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_laps_session_driver_lap 
    ON laps_raw(session_id, driver, lap_number);

-- Telemetry raw data
CREATE TABLE IF NOT EXISTS telemetry_raw (
    session_id INTEGER NOT NULL,
    driver TEXT NOT NULL,
    lap_number INTEGER,
    time_s REAL NOT NULL,
    distance_m REAL,
    speed_kph REAL,
    throttle REAL,
    brake INTEGER,
    gear INTEGER,
    drs INTEGER,
    rpm REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_telemetry_session_driver_time 
    ON telemetry_raw(session_id, driver, time_s);

-- Position data
CREATE TABLE IF NOT EXISTS positions_raw (
    session_id INTEGER NOT NULL,
    driver TEXT NOT NULL,
    time_s REAL NOT NULL,
    x REAL,
    y REAL,
    z REAL,
    status TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_positions_session_driver 
    ON positions_raw(session_id, driver);

-- Weather data
CREATE TABLE IF NOT EXISTS weather_raw (
    session_id INTEGER NOT NULL,
    time_s REAL NOT NULL,
    air_temp_c REAL,
    track_temp_c REAL,
    humidity_pct REAL,
    pressure_hpa REAL,
    wind_speed_mps REAL,
    wind_dir_deg REAL,
    rainfall_flag INTEGER,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_weather_session_time 
    ON weather_raw(session_id, time_s);

-- Race control messages
CREATE TABLE IF NOT EXISTS race_control_raw (
    session_id INTEGER NOT NULL,
    time_s REAL,
    category TEXT,
    message TEXT,
    flag TEXT,
    lap_number INTEGER,
    driver_number TEXT,
    scope TEXT,
    sector INTEGER,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_race_control_session 
    ON race_control_raw(session_id);

-- Ingestion log
CREATE TABLE IF NOT EXISTS ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    status TEXT,
    message TEXT,
    ingested_at TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""

FEATURES_CORE_SCHEMA = """
-- Featured laps with derived features
CREATE TABLE IF NOT EXISTS laps_featured (
    session_id INTEGER NOT NULL,
    driver TEXT NOT NULL,
    lap_number INTEGER NOT NULL,
    lap_time_s REAL,
    sector1_s REAL,
    sector2_s REAL,
    sector3_s REAL,
    position INTEGER,
    compound TEXT,
    stint INTEGER,
    tyre_age_laps INTEGER,
    fuel_proxy REAL,
    track_status_cat TEXT,
    is_inlap INTEGER,
    is_outlap INTEGER,
    is_pit_lap INTEGER,
    air_temp_c REAL,
    track_temp_c REAL,
    team TEXT,
    is_personal_best INTEGER,
    PRIMARY KEY (session_id, driver, lap_number)
);

CREATE INDEX IF NOT EXISTS idx_laps_featured_session 
    ON laps_featured(session_id);

-- Gap analysis features
CREATE TABLE IF NOT EXISTS gaps_featured (
    session_id INTEGER NOT NULL,
    driver TEXT NOT NULL,
    lap_number INTEGER NOT NULL,
    gap_to_ahead_s REAL,
    gap_to_behind_s REAL,
    in_clean_air INTEGER,
    within_drs INTEGER,
    PRIMARY KEY (session_id, driver, lap_number)
);

CREATE INDEX IF NOT EXISTS idx_gaps_featured_session 
    ON gaps_featured(session_id);

-- Segment analysis features
CREATE TABLE IF NOT EXISTS segments_featured (
    session_id INTEGER NOT NULL,
    driver TEXT NOT NULL,
    lap_number INTEGER NOT NULL,
    segment_id INTEGER NOT NULL,
    distance_start_m REAL,
    distance_end_m REAL,
    mean_speed_kph REAL,
    max_speed_kph REAL,
    min_speed_kph REAL,
    avg_throttle REAL,
    max_brake REAL,
    time_in_segment_s REAL,
    entry_speed_kph REAL,
    exit_speed_kph REAL,
    compound TEXT,
    tyre_age_laps INTEGER,
    fuel_proxy REAL,
    PRIMARY KEY (session_id, driver, lap_number, segment_id)
);

CREATE INDEX IF NOT EXISTS idx_segments_featured_session 
    ON segments_featured(session_id);
"""


def initialize_raw_db(db_path: str) -> None:
    """Initialize raw database with schema.
    
    Args:
        db_path: Path to raw database file
    """
    import sqlite3
    from pathlib import Path
    
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(RAW_DB_SCHEMA)
    conn.commit()
    conn.close()


def initialize_features_core_db(db_path: str) -> None:
    """Initialize features core database with schema.
    
    Args:
        db_path: Path to features core database file
    """
    import sqlite3
    from pathlib import Path
    
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(FEATURES_CORE_SCHEMA)
    conn.commit()
    conn.close()


def initialize_placeholder_db(db_path: str) -> None:
    """Initialize placeholder database (empty).
    
    Args:
        db_path: Path to database file
    """
    import sqlite3
    from pathlib import Path
    
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS _placeholder (id INTEGER PRIMARY KEY);")
    conn.commit()
    conn.close()
