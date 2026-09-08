import sqlite3
from typing import Dict, Any, List, Optional
from database.sqlite_db import (
    get_db_connection,
    init_database,
    insert_sensor_reading,
    insert_detection,
    insert_system_event,
    insert_lora_alert,
    get_latest_sensor_readings,
    get_sensor_history,
    get_latest_detections,
    get_detection_history,
    get_system_events,
    check_database_health
)
from utils.logger import logger

class DatabaseRepository:
    """
    Repository wrapper exposing high-level data access, time-series query APIs,
    and cloud sync tracking.
    """
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        init_database(self.db_path)

    def insert_sensor_reading(self, data: Dict[str, Any]) -> int:
        return insert_sensor_reading(data, self.db_path)

    def insert_detection(self, data: Dict[str, Any]) -> int:
        return insert_detection(data, self.db_path)

    def insert_disease_event(
        self,
        timestamp: Any,
        datetime_str: str,
        disease_name: str,
        confidence: float,
        snapshot_path: str,
        firebase_image_url: str = "",
        x1: Optional[float] = None,
        y1: Optional[float] = None,
        x2: Optional[float] = None,
        y2: Optional[float] = None
    ) -> int:
        """Helper to insert plant disease detection matching both old and new schema."""
        payload = {
            "timestamp": timestamp,
            "model_name": "detecting-diseases",
            "model_version": "5",
            "class_name": disease_name,
            "confidence": confidence,
            "image_path": snapshot_path,
            "firebase_image_url": firebase_image_url,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "source": "camera"
        }
        return insert_detection(payload, self.db_path)

    def insert_system_event(self, event_type: str, severity: str, message: str, timestamp: Optional[str] = None) -> int:
        return insert_system_event({
            "timestamp": timestamp,
            "event_type": event_type,
            "severity": severity,
            "message": message
        }, self.db_path)

    def insert_lora_alert(self, timestamp: Any, datetime_str: str, alert_type: str, message: str, sent_status: int = 1) -> int:
        return insert_lora_alert({
            "timestamp": timestamp,
            "alert_type": alert_type,
            "message": message,
            "sent_status": sent_status
        }, self.db_path)

    def get_unsynced_sensor_readings(self, limit: int = 50) -> List[sqlite3.Row]:
        """Fetches pending sensor readings that haven't been pushed to Firebase yet."""
        conn = get_db_connection(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sensor_readings WHERE synced = 0 ORDER BY id ASC LIMIT ?", (limit,))
            return cursor.fetchall()
        finally:
            conn.close()

    def get_unsynced_detections(self, limit: int = 20) -> List[sqlite3.Row]:
        """Fetches pending AI detections that haven't been synced to Firebase."""
        conn = get_db_connection(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM detections WHERE synced = 0 ORDER BY id ASC LIMIT ?", (limit,))
            return cursor.fetchall()
        finally:
            conn.close()

    def mark_sensor_reading_synced(self, record_id: int):
        """Marks a sensor reading record as synced."""
        conn = get_db_connection(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("UPDATE sensor_readings SET synced = 1 WHERE id = ?", (record_id,))
            conn.commit()
        finally:
            conn.close()

    def mark_detection_synced(self, record_id: int, firebase_image_url: Optional[str] = None):
        """Marks an AI detection record as synced."""
        conn = get_db_connection(self.db_path)
        try:
            cursor = conn.cursor()
            if firebase_image_url:
                cursor.execute("UPDATE detections SET synced = 1, firebase_image_url = ? WHERE id = ?", (firebase_image_url, record_id))
            else:
                cursor.execute("UPDATE detections SET synced = 1 WHERE id = ?", (record_id,))
            conn.commit()
        finally:
            conn.close()

    def delete_synced_sensor_reading(self, record_id: int):
        """Deletes/releases a sensor reading from local SQLite once successfully synced to cloud."""
        conn = get_db_connection(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sensor_readings WHERE id = ?", (record_id,))
            conn.commit()
        finally:
            conn.close()

    def delete_synced_detection(self, record_id: int):
        """Deletes/releases a detection record from local SQLite once successfully uploaded to Firebase."""
        conn = get_db_connection(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM detections WHERE id = ?", (record_id,))
            conn.commit()
        finally:
            conn.close()

    def check_health(self) -> bool:
        return check_database_health(self.db_path)

