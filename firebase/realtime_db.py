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

    def _rest_delete(self, endpoint: str) -> bool:
        """Deletes node via REST DELETE."""
        if not self.db_url:
            return False
        url = f"{self.db_url}/{endpoint.lstrip('/')}.json"
        try:
            resp = self.session.delete(url, timeout=6.0)
            if resp.status_code in (200, 204):
                return True
            return False
        except Exception as e:
            logger.debug(f"Firebase REST DELETE error for /{endpoint}: {e}")
            return False

    def _trim_fifo_node(self, node_name: str, max_limit: int = 200):
        """
        Enforces a strict FIFO ring-buffer on a Firebase node (e.g., /snapshots, /disease_alerts).
        When items exceed max_limit (200), oldest entries are automatically deleted so newest entries fit.
        """
        try:
            # 1. Admin SDK Pruning
            if self.fb.is_ready or self.fb.initialize():
                try:
                    from firebase_admin import db
                    ref = db.reference(node_name)
                    items = ref.get()
                    if items and isinstance(items, dict) and len(items) > max_limit:
                        sorted_keys = sorted(items.keys())
                        excess = len(sorted_keys) - max_limit
                        for old_k in sorted_keys[:excess]:
                            ref.child(old_k).delete()
                        logger.info(f"Firebase /{node_name} FIFO pruned: removed {excess} oldest records (capped at {max_limit}).")
                    return
                except Exception as e:
                    logger.debug(f"Admin SDK FIFO prune on /{node_name} failed: {e}. Trying REST.")

            # 2. REST API Pruning
            items = self._rest_get(node_name)
            if items and isinstance(items, dict) and len(items) > max_limit:
                sorted_keys = sorted(items.keys())
                excess = len(sorted_keys) - max_limit
                for old_k in sorted_keys[:excess]:
                    self._rest_delete(f"{node_name}/{old_k}")
                logger.info(f"Firebase /{node_name} REST FIFO pruned: removed {excess} oldest records (capped at {max_limit}).")
        except Exception as e:
            logger.debug(f"Error during FIFO prune for /{node_name}: {e}")

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
        """Appends historical sensor reading to /sensor_readings and caps at 200."""
        success = False
        # 1. Try Firebase Admin SDK if active
        if self.fb.is_ready or self.fb.initialize():
            try:
                from firebase_admin import db
                ref = db.reference("sensor_readings")
                ref.push(data)
                logger.info("Sensor reading pushed to Firebase Realtime Database (Admin SDK).")
                success = True
            except Exception as e:
                logger.debug(f"Admin SDK push_sensor_reading failed: {e}. Falling back to REST.")

        # 2. Direct REST API Fallback
        if not success and self._rest_post("sensor_readings", data):
            logger.info("Sensor reading pushed to Firebase Realtime Database (REST).")
            success = True

        if success:
            self._trim_fifo_node("sensor_readings", 200)
        return success

    def push_snapshot(self, snapshot_data: Dict[str, Any], max_limit: int = 200) -> bool:
        """
        Appends snapshot metadata with photo URL to /snapshots and strictly caps at 200 records (FIFO).
        """
        success = False
        # 1. Try Firebase Admin SDK if active
        if self.fb.is_ready or self.fb.initialize():
            try:
                from firebase_admin import db
                ref = db.reference("snapshots")
                ref.push(snapshot_data)
                logger.info("Snapshot record pushed to Firebase Realtime Database /snapshots (Admin SDK).")
                success = True
            except Exception as e:
                logger.debug(f"Admin SDK push_snapshot failed: {e}. Falling back to REST.")

        # 2. Direct REST API Fallback
        if not success and self._rest_post("snapshots", snapshot_data):
            logger.info("Snapshot record pushed to Firebase Realtime Database /snapshots (REST).")
            success = True

        if success:
            self._trim_fifo_node("snapshots", max_limit)
        return success

    def push_disease_event(self, event_data: Dict[str, Any], max_limit: int = 200) -> bool:
        """
        Appends disease detection alert with image URL to /disease_alerts and /snapshots (capped at 200 FIFO).
        """
        success = False
        # 1. Try Firebase Admin SDK if active
        if self.fb.is_ready or self.fb.initialize():
            try:
                from firebase_admin import db
                ref = db.reference("disease_alerts")
                ref.push(event_data)
                logger.info(f"Disease alert pushed to Firebase Realtime Database (Admin SDK): {event_data.get('disease_name')}")
                success = True
            except Exception as e:
                logger.debug(f"Admin SDK push_disease_event failed: {e}. Falling back to REST.")

        # 2. Direct REST API Fallback
        if not success and self._rest_post("disease_alerts", event_data):
            logger.info(f"Disease alert pushed to Firebase Realtime Database (REST): {event_data.get('disease_name')}")
            success = True

        if success:
            self._trim_fifo_node("disease_alerts", max_limit)
            # Also dual-sync to /snapshots for unified photo galleries
            self.push_snapshot(event_data, max_limit=max_limit)

        return success

    def _rest_get(self, endpoint: str) -> Optional[Any]:
        """Fetches data via REST GET."""
        if not self.db_url:
            return None
        url = f"{self.db_url}/{endpoint.lstrip('/')}.json"
        try:
            resp = self.session.get(url, timeout=5.0)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"Firebase REST GET error for /{endpoint}: {e}")
        return None

    def get_camera_control(self) -> Optional[Dict[str, Any]]:
        """Fetches current /camera_control state from Firebase."""
        # 1. Try Firebase Admin SDK if active
        if self.fb.is_ready or self.fb.initialize():
            try:
                from firebase_admin import db
                ref = db.reference("camera_control")
                val = ref.get()
                if val and isinstance(val, dict):
                    return val
            except Exception as e:
                logger.debug(f"Admin SDK get_camera_control failed: {e}. Falling back to REST.")

        # 2. REST API Fallback
        res = self._rest_get("camera_control")
        return res if isinstance(res, dict) else None

    def update_camera_control(self, data: Dict[str, Any]) -> bool:
        """Pushes updated camera / servo state to /camera_control."""
        # 1. Try Firebase Admin SDK if active
        if self.fb.is_ready or self.fb.initialize():
            try:
                from firebase_admin import db
                ref = db.reference("camera_control")
                ref.update(data)
                return True
            except Exception as e:
                logger.debug(f"Admin SDK update_camera_control failed: {e}. Falling back to REST.")

        # 2. REST API Fallback
        return self._rest_put("camera_control", data)

