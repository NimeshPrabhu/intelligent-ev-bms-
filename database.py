"""
SQLite Database Manager for EV BMS Telemetry & Fault Auditing.
Optimized for Raspberry Pi 3B+ (WAL mode, index optimizations, safe migrations).
"""

import sqlite3
import time
from typing import List, Optional, Dict, Any
from core.config import DB_PATH

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Returns an optimized connection with WAL mode and Row factory."""
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA cache_size=-2000;")  # 2MB cache, conservative for Pi 3B+
    return conn

def init_db(db_path: str = DB_PATH):
    """Initializes tables and performs idempotent schema migrations."""
    with get_connection(db_path) as conn:
        # 1. Readings Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                seq INTEGER DEFAULT 0,
                speed REAL NOT NULL,
                voltage REAL NOT NULL,
                current REAL NOT NULL,
                power REAL DEFAULT 0.0,
                temperature REAL NOT NULL,
                soc REAL NOT NULL,
                soh REAL NOT NULL,
                range_km REAL NOT NULL,
                relay_state INTEGER DEFAULT 1,
                fault_mask INTEGER DEFAULT 0,
                source TEXT DEFAULT 'hw'
            );
        """)
        
        # 2. Schema Migrations (ensure existing tables have newer columns)
        cursor = conn.execute("PRAGMA table_info(readings);")
        columns = [row["name"] for row in cursor.fetchall()]
        
        if "seq" not in columns:
            conn.execute("ALTER TABLE readings ADD COLUMN seq INTEGER DEFAULT 0;")
        if "power" not in columns:
            conn.execute("ALTER TABLE readings ADD COLUMN power REAL DEFAULT 0.0;")
        if "relay_state" not in columns:
            conn.execute("ALTER TABLE readings ADD COLUMN relay_state INTEGER DEFAULT 1;")
        if "fault_mask" not in columns:
            conn.execute("ALTER TABLE readings ADD COLUMN fault_mask INTEGER DEFAULT 0;")
        if "source" not in columns:
            conn.execute("ALTER TABLE readings ADD COLUMN source TEXT DEFAULT 'hw';")

        # 3. Fault Events Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fault_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                fault_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                value REAL NOT NULL,
                threshold REAL NOT NULL,
                resolved INTEGER DEFAULT 0
            );
        """)

        # 4. Indexes for fast dashboard and time-series queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_readings_timestamp 
            ON readings(timestamp DESC);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_faults_timestamp 
            ON fault_events(timestamp DESC);
        """)
        conn.commit()

def insert_reading(data: Dict[str, Any], db_path: str = DB_PATH) -> int:
    """Inserts a new telemetry record and returns the inserted row ID."""
    ts = float(data.get("timestamp", time.time()))
    v = float(data.get("voltage", 0.0))
    i = float(data.get("current", 0.0))
    p = float(data.get("power", abs(v * i)))

    with get_connection(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO readings (
                timestamp, seq, speed, voltage, current, power,
                temperature, soc, soh, range_km, relay_state, fault_mask, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts,
            int(data.get("seq", 0)),
            float(data.get("speed", 0.0)),
            v,
            i,
            round(p, 2),
            float(data.get("temperature", 0.0)),
            float(data.get("soc", 100.0)),
            float(data.get("soh", 100.0)),
            float(data.get("range_km", 0.0)),
            int(data.get("relay_state", 1)),
            int(data.get("fault_mask", 0)),
            str(data.get("source", "hw"))
        ))
        conn.commit()
        return cursor.lastrowid

def get_latest_reading(db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Fetches the most recent reading from the database."""
    with get_connection(db_path) as conn:
        cursor = conn.execute("""
            SELECT id, timestamp, seq, speed, voltage, current, power,
                   temperature, soc, soh, range_km, relay_state, fault_mask, source
            FROM readings
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        return dict(row) if row else None

def get_history(limit: int = 40, page: int = 1, time_range_s: Optional[int] = None, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Fetches paginated telemetry readings.
    If time_range_s is provided, filters for readings within that time window.
    Returns in chronological order (oldest to newest) for plotting.
    """
    offset = max(0, (page - 1) * limit)
    with get_connection(db_path) as conn:
        if time_range_s:
            cutoff_ts = time.time() - time_range_s
            cursor = conn.execute("""
                SELECT id, timestamp, seq, speed, voltage, current, power,
                       temperature, soc, soh, range_km, relay_state, fault_mask, source
                FROM readings
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, (cutoff_ts, limit, offset))
        else:
            cursor = conn.execute("""
                SELECT id, timestamp, seq, speed, voltage, current, power,
                       temperature, soc, soh, range_km, relay_state, fault_mask, source
                FROM readings
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))

        rows = cursor.fetchall()
        return [dict(r) for r in reversed(rows)]

def insert_fault_event(event: Dict[str, Any], db_path: str = DB_PATH) -> int:
    """Logs a battery safety fault event to the auditable log."""
    with get_connection(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO fault_events (timestamp, fault_type, severity, message, value, threshold, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            float(event.get("timestamp", time.time())),
            str(event.get("type", "UNKNOWN")),
            str(event.get("severity", "WARNING")),
            str(event.get("message", "")),
            float(event.get("value", 0.0)),
            float(event.get("threshold", 0.0)),
            int(event.get("resolved", 0))
        ))
        conn.commit()
        return cursor.lastrowid

def get_recent_fault_events(limit: int = 20, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves the most recent fault events."""
    with get_connection(db_path) as conn:
        cursor = conn.execute("""
            SELECT id, timestamp, fault_type, severity, message, value, threshold, resolved
            FROM fault_events
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        return [dict(r) for r in cursor.fetchall()]

def get_telemetry_stats(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Returns database summary statistics for diagnostics."""
    with get_connection(db_path) as conn:
        total_readings = conn.execute("SELECT COUNT(*) FROM readings;").fetchone()[0]
        total_faults = conn.execute("SELECT COUNT(*) FROM fault_events;").fetchone()[0]
        first_ts = conn.execute("SELECT MIN(timestamp) FROM readings;").fetchone()[0]
        last_ts = conn.execute("SELECT MAX(timestamp) FROM readings;").fetchone()[0]

    return {
        "total_readings": total_readings,
        "total_fault_events": total_faults,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "uptime_recorded_s": (last_ts - first_ts) if (first_ts and last_ts) else 0.0
    }

# Auto-initialize default schema on import
try:
    init_db()
except Exception:
    pass

if __name__ == "__main__":
    init_db()
    print("Database schema initialized and verified successfully.")

