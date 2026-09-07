"""
SQLite Database Manager for EV Telemetry Readings.
Stores real-time packets, state estimations (SOC, SOH), and computed range.
"""

import sqlite3
import time
from typing import List, Optional, Dict, Any

DEFAULT_DB_PATH = "readings.db"

def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Returns a connection configured with WAL mode and Row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            speed REAL NOT NULL,
            voltage REAL NOT NULL,
            current REAL NOT NULL,
            temperature REAL NOT NULL,
            soc REAL NOT NULL,
            soh REAL NOT NULL,
            range_km REAL NOT NULL
        );
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_readings_timestamp 
        ON readings(timestamp DESC);
    """)
    conn.commit()
    return conn

def init_db(db_path: str = DEFAULT_DB_PATH):
    """Explicitly initializes database schema."""
    conn = get_connection(db_path)
    conn.close()

def insert_reading(data: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Inserts a new telemetry record.
    Returns the inserted row ID.
    """
    ts = data.get("timestamp", time.time())
    with get_connection(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO readings (timestamp, speed, voltage, current, temperature, soc, soh, range_km)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts,
            float(data.get("speed", 0.0)),
            float(data.get("voltage", 0.0)),
            float(data.get("current", 0.0)),
            float(data.get("temperature", 0.0)),
            float(data.get("soc", 100.0)),
            float(data.get("soh", 100.0)),
            float(data.get("range_km", 0.0))
        ))
        conn.commit()
        return cursor.lastrowid

def get_latest_reading(db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Fetches the most recent reading from the database."""
    with get_connection(db_path) as conn:
        cursor = conn.execute("""
            SELECT id, timestamp, speed, voltage, current, temperature, soc, soh, range_km
            FROM readings
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

def get_history(limit: int = 40, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    Fetches the latest N readings returned in chronological order (oldest to newest)
    for pre-populating frontend charts.
    """
    with get_connection(db_path) as conn:
        cursor = conn.execute("""
            SELECT id, timestamp, speed, voltage, current, temperature, soc, soh, range_km
            FROM readings
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        # Reverse to chronological order (oldest to newest)
        results = [dict(r) for r in reversed(rows)]
        return results

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
