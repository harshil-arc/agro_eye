import requests
import json
from typing import Dict, Any, Optional
from config import FIREBASE_DATABASE_URL
from firebase.firebase_client import FirebaseClient
from utils.logger import logger

class RealtimeDatabaseManager:
    """
    Dual-mode Firebase Realtime Database Manager:
    1. Direct Firebase Admin SDK (when service_account.json is present)
    2. Zero-config High-Speed Firebase REST API (guarantees real-time cloud sync even
       if service_account.json is missing or running without Google Cloud SDK).
    """
    def __init__(self, db_url: Optional[str] = None):
        self.fb = FirebaseClient()
        self.db_url = (db_url or FIREBASE_DATABASE_URL or "").rstrip("/")
        self.session = requests.Session()

    def _sanitize(self, data: Any) -> Any:
        """Sanitizes payload to ensure strict JSON compatibility."""
        if isinstance(data, dict):
            return {str(k): self._sanitize(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._sanitize(v) for v in data]
        elif isinstance(data, float):
            import math
            if math.isnan(data) or math.isinf(data):
                return None
            return data
        return data

    def _rest_put(self, endpoint: str, data: Dict[str, Any]) -> bool:
        """Pushes data via REST PUT (replaces node)."""
        if not self.db_url:
            return False
        url = f"{self.db_url}/{endpoint.lstrip('/')}.json"
        try:
            sanitized = self._sanitize(data)
            resp = self.session.put(url, json=sanitized, timeout=6.0)
            if resp.status_code in (200, 201, 204):
                return True
            logger.warning(f"Firebase REST PUT to /{endpoint} returned HTTP {resp.status_code}: {resp.text[:120]}")
            return False
        except Exception as e:
            logger.debug(f"Firebase REST PUT error: {e}")
            return False

    def _rest_post(self, endpoint: str, data: Dict[str, Any]) -> bool:
        """Pushes data via REST POST (appends child with unique push ID)."""
        if not self.db_url:
            return False
        url = f"{self.db_url}/{endpoint.lstrip('/')}.json"
        try:
            sanitized = self._sanitize(data)
            resp = self.session.post(url, json=sanitized, timeout=6.0)
            if resp.status_code in (200, 201, 204):
                return True
            logger.warning(f"Firebase REST POST to /{endpoint} returned HTTP {resp.status_code}: {resp.text[:120]}")
            return False
        except Exception as e:
            logger.debug(f"Firebase REST POST error: {e}")
            return False

    def update_live_status(self, data: Dict[str, Any]) -> bool:
        """Pushes current live system status & latest sensor telemetry to /live_status."""
        # 1. Try Firebase Admin SDK if active
        if self.fb.is_ready or self.fb.initialize():
            try:
                from firebase_admin import db
                ref = db.reference("live_status")
                ref.set(data)
                return True
            except Exception as e:
                logger.debug(f"Admin SDK update_live_status failed: {e}. Falling back to REST.")

        # 2. Direct REST API Fallback
        return self._rest_put("live_status", data)

    def push_sensor_reading(self, data: Dict[str, Any]) -> bool:
        """Appends historical sensor reading to /sensor_readings."""
        # 1. Try Firebase Admin SDK if active
        if self.fb.is_ready or self.fb.initialize():
            try:
                from firebase_admin import db
                ref = db.reference("sensor_readings")
                ref.push(data)
                logger.info("Sensor reading pushed to Firebase Realtime Database (Admin SDK).")
                return True
            except Exception as e:
                logger.debug(f"Admin SDK push_sensor_reading failed: {e}. Falling back to REST.")

        # 2. Direct REST API Fallback
        if self._rest_post("sensor_readings", data):
            logger.info("Sensor reading pushed to Firebase Realtime Database (REST).")
            return True
        return False

    def push_disease_event(self, event_data: Dict[str, Any]) -> bool:
        """Appends disease detection alert with image URL to /disease_alerts."""
        # 1. Try Firebase Admin SDK if active
        if self.fb.is_ready or self.fb.initialize():
            try:
                from firebase_admin import db
                ref = db.reference("disease_alerts")
                ref.push(event_data)
                logger.info(f"Disease alert pushed to Firebase Realtime Database (Admin SDK): {event_data.get('disease_name')}")
                return True
            except Exception as e:
                logger.debug(f"Admin SDK push_disease_event failed: {e}. Falling back to REST.")

        # 2. Direct REST API Fallback
        if self._rest_post("disease_alerts", event_data):
            logger.info(f"Disease alert pushed to Firebase Realtime Database (REST): {event_data.get('disease_name')}")
            return True
        return False

