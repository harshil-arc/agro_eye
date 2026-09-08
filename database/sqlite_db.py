"""
Database management module for SIH Smart Farming Edge System.
Provides SQLite database initialization, connection management,
and parameterized queries for sensor data, AI detections, and system logs.
"""

import sqlite3
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from config import SQLITE_DB_PATH
from utils.logger import logger

# Default database location
DEFAULT_DB_PATH = SQLITE_DB_PATH


def get_utc_now_iso() -> str:
    """Return the current UTC timestamp formatted as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Establish and return a SQLite database connection with row factory and WAL mode enabled.
    """
    target_path = db_path or DEFAULT_DB_PATH
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    
    conn = sqlite3.connect(target_path, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    
    # Optimize SQLite for edge concurrency: WAL mode & normal sync
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()
    
    return conn


def init_database(db_path: Optional[str] = None) -> None:
    """
    Create tables and indexes if they do not exist, and auto-migrate fields if needed.
    """
    target_path = db_path or DEFAULT_DB_PATH
    conn = get_db_connection(target_path)
    try:
        cursor = conn.cursor()
        
        # 1. sensor_readings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'local_sensor',
                soil_moisture REAL,
                soil_raw INTEGER,
                temperature REAL,
                humidity REAL,
                npk_n REAL,
                npk_p REAL,
                npk_k REAL,
                soil_conductivity REAL,
                rainfall REAL,
                light_intensity REAL,
                air_quality REAL,
                mq135_raw INTEGER,
                mq135_voltage REAL,
                synced INTEGER DEFAULT 0
            );
        """)
        
        # 2. detections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                x1 REAL,
                y1 REAL,
                x2 REAL,
                y2 REAL,
                image_path TEXT,
                firebase_image_url TEXT,
                source TEXT DEFAULT 'camera',
                synced INTEGER DEFAULT 0
            );
        """)
        
        # 3. system_events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'INFO',
                message TEXT NOT NULL
            );
        """)

        # 4. lora_alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lora_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                datetime TEXT,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                sent_status INTEGER DEFAULT 1
            );
        """)
        
        # Indexes for fast querying & time-series operations
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sensor_timestamp ON sensor_readings(timestamp DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sensor_source ON sensor_readings(source);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sensor_synced ON sensor_readings(synced);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_model ON detections(model_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_source ON detections(source);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_synced ON detections(synced);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events(timestamp DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_severity ON system_events(severity);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON system_events(event_type);")
        
        # Auto-migrate columns if table existed from an earlier schema
        cursor.execute("PRAGMA table_info(sensor_readings);")
        columns = [row[1] for row in cursor.fetchall()]
        for col, col_type in [
            ("npk_n", "REAL"),
            ("npk_p", "REAL"),
            ("npk_k", "REAL"),
            ("soil_conductivity", "REAL"),
            ("rainfall", "REAL"),
            ("light_intensity", "REAL"),
            ("air_quality", "REAL"),
            ("soil_raw", "INTEGER"),
            ("mq135_raw", "INTEGER"),
            ("mq135_voltage", "REAL"),
            ("synced", "INTEGER DEFAULT 0")
        ]:
            if col not in columns:
                cursor.execute(f"ALTER TABLE sensor_readings ADD COLUMN {col} {col_type};")

        cursor.execute("PRAGMA table_info(detections);")
        det_columns = [row[1] for row in cursor.fetchall()]
        for col, col_type in [
            ("firebase_image_url", "TEXT"),
            ("synced", "INTEGER DEFAULT 0")
        ]:
            if col not in det_columns:
                cursor.execute(f"ALTER TABLE detections ADD COLUMN {col} {col_type};")

        cursor.execute("PRAGMA table_info(lora_alerts);")
        lora_columns = [row[1] for row in cursor.fetchall()]
        if "datetime" not in lora_columns:
            cursor.execute("ALTER TABLE lora_alerts ADD COLUMN datetime TEXT;")

        conn.commit()
        logger.info(f"SQLite database initialized with WAL mode at: {target_path}")
    finally:
        conn.close()


def insert_sensor_reading(data: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """
    Insert a sensor measurement record into sensor_readings and prune table to maximum 50 newest entries.
    Returns the newly inserted row ID.
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        timestamp = data.get("timestamp") or get_utc_now_iso()
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
            
        source = data.get("source", "esp32_sensor")
        
        cursor.execute("""
            INSERT INTO sensor_readings (
                timestamp, source, soil_moisture, soil_raw, temperature, humidity,
                npk_n, npk_p, npk_k, soil_conductivity, rainfall,
                light_intensity, air_quality, mq135_raw, mq135_voltage, synced
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0);
        """, (
            timestamp,
            source,
            data.get("soil_moisture"),
            data.get("soil_raw"),
            data.get("temperature"),
            data.get("humidity"),
            data.get("npk_n") or data.get("nitrogen"),
            data.get("npk_p") or data.get("phosphorus"),
            data.get("npk_k") or data.get("potassium"),
            data.get("soil_conductivity"),
            data.get("rainfall"),
            data.get("light_intensity"),
            data.get("air_quality"),
            data.get("mq135_raw"),
            data.get("mq135_voltage")
        ))
        inserted_id = cursor.lastrowid

        # Strict FIFO Limit: Keep only the 200 newest records, delete oldest
        cursor.execute("""
            DELETE FROM sensor_readings WHERE id NOT IN (
                SELECT id FROM sensor_readings ORDER BY id DESC LIMIT 200
            );
        """)

        conn.commit()
        return inserted_id
    except Exception as e:
        logger.error(f"Error inserting sensor reading: {e}")
        return -1
    finally:
        conn.close()


def insert_detection(data: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """
    Insert an AI detection result into detections and prune to maximum 200 newest entries.
    Returns the newly inserted row ID.
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        timestamp = data.get("timestamp") or get_utc_now_iso()
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
            
        cursor.execute("""
            INSERT INTO detections (
                timestamp, model_name, model_version, class_name, confidence,
                x1, y1, x2, y2, image_path, firebase_image_url, source, synced
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0);
        """, (
            timestamp,
            data.get("model_name", "detecting-diseases"),
            data.get("model_version", "5"),
            data.get("class_name") or data.get("disease_name", "disease"),
            data.get("confidence", 0.0),
            data.get("x1"),
            data.get("y1"),
            data.get("x2"),
            data.get("y2"),
            data.get("image_path") or data.get("snapshot_path"),
            data.get("firebase_image_url") or data.get("photo_url"),
            data.get("source", "camera")
        ))
        inserted_id = cursor.lastrowid

        # Strict FIFO Limit: Keep only the 200 newest records
        cursor.execute("""
            DELETE FROM detections WHERE id NOT IN (
                SELECT id FROM detections ORDER BY id DESC LIMIT 200
            );
        """)

        conn.commit()
        return inserted_id
    except Exception as e:
        logger.error(f"Error inserting AI detection: {e}")
        return -1
    finally:
        conn.close()


def insert_system_event(data: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """
    Insert a system event/log entry into system_events and prune to maximum 200 newest entries.
    Returns the newly inserted row ID.
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        timestamp = data.get("timestamp") or get_utc_now_iso()
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
            
        cursor.execute("""
            INSERT INTO system_events (
                timestamp, event_type, severity, message
            ) VALUES (?, ?, ?, ?);
        """, (
            timestamp,
            data.get("event_type", "SYSTEM"),
            data.get("severity", "INFO"),
            data.get("message", "")
        ))
        inserted_id = cursor.lastrowid

        cursor.execute("""
            DELETE FROM system_events WHERE id NOT IN (
                SELECT id FROM system_events ORDER BY id DESC LIMIT 200
            );
        """)

        conn.commit()
        return inserted_id
    except Exception as e:
        logger.error(f"Error inserting system event: {e}")
        return -1
    finally:
        conn.close()


def insert_lora_alert(data: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """Insert a dispatched LoRa alert entry and prune to maximum 200 newest entries."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        timestamp = data.get("timestamp") or get_utc_now_iso()
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()

        cursor.execute("""
            INSERT INTO lora_alerts (timestamp, alert_type, message, sent_status)
            VALUES (?, ?, ?, ?);
        """, (
            timestamp,
            data.get("alert_type", "ALERT"),
            data.get("message", ""),
            data.get("sent_status", 1)
        ))
        inserted_id = cursor.lastrowid

        cursor.execute("""
            DELETE FROM lora_alerts WHERE id NOT IN (
                SELECT id FROM lora_alerts ORDER BY id DESC LIMIT 200
            );
        """)


        conn.commit()
        return inserted_id
    except Exception as e:
        logger.error(f"Error inserting LoRa alert: {e}")
        return -1
    finally:
        conn.close()


def get_latest_sensor_readings(source: Optional[str] = None, limit: int = 1, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve the most recent sensor reading(s)."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM sensor_readings"
        params: List[Any] = []
        
        if source:
            query += " WHERE source = ?"
            params.append(source)
            
        query += " ORDER BY timestamp DESC, id DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_sensor_history(
    limit: int = 100,
    hours: Optional[float] = None,
    source: Optional[str] = None,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieve historical sensor readings with optional time filtering (hours) and source filtering."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        conditions = []
        params: List[Any] = []
        
        if hours is not None and hours > 0:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            conditions.append("timestamp >= ?")
            params.append(cutoff)
            
        if source:
            conditions.append("source = ?")
            params.append(source)
            
        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM sensor_readings{where_clause} ORDER BY timestamp DESC, id DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_latest_detections(
    limit: int = 10,
    model_name: Optional[str] = None,
    source: Optional[str] = None,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieve the most recent AI detections."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        conditions = []
        params: List[Any] = []
        
        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)
            
        if source:
            conditions.append("source = ?")
            params.append(source)
            
        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM detections{where_clause} ORDER BY timestamp DESC, id DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_detection_history(
    limit: int = 100,
    hours: Optional[float] = None,
    model_name: Optional[str] = None,
    source: Optional[str] = None,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieve historical AI detections with optional time filtering, model filtering, and source filtering."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        conditions = []
        params: List[Any] = []
        
        if hours is not None and hours > 0:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            conditions.append("timestamp >= ?")
            params.append(cutoff)
            
        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)
            
        if source:
            conditions.append("source = ?")
            params.append(source)
            
        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM detections{where_clause} ORDER BY timestamp DESC, id DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_system_events(
    limit: int = 100,
    severity: Optional[str] = None,
    event_type: Optional[str] = None,
    hours: Optional[float] = None,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieve recent system events/logs."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        conditions = []
        params: List[Any] = []
        
        if hours is not None and hours > 0:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            conditions.append("timestamp >= ?")
            params.append(cutoff)
            
        if severity:
            conditions.append("severity = ?")
            params.append(severity)
            
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
            
        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM system_events{where_clause} ORDER BY timestamp DESC, id DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def check_database_health(db_path: Optional[str] = None) -> bool:
    """Perform a quick health check to verify database readiness."""
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1;")
        cursor.fetchone()
        conn.close()
        return True
    except Exception:
        return False
