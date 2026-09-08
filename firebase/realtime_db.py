from typing import Dict, Any
from firebase.firebase_client import FirebaseClient
from utils.logger import logger

class RealtimeDatabaseManager:
    def __init__(self):
        self.fb = FirebaseClient()

    def update_live_status(self, data: Dict[str, Any]) -> bool:
        """Pushes current live system status & latest sensor telemetry to /live_status."""
        if not self.fb.is_ready and not self.fb.initialize():
            return False

        try:
            from firebase_admin import db
            ref = db.reference("live_status")
            ref.set(data)
            return True
        except Exception as e:
            logger.error(f"Failed to update Firebase Realtime live_status: {e}")
            return False

    def push_sensor_reading(self, data: Dict[str, Any]) -> bool:
        """Appends historical sensor reading to /sensor_readings."""
        if not self.fb.is_ready and not self.fb.initialize():
            return False

        try:
            from firebase_admin import db
            ref = db.reference("sensor_readings")
            ref.push(data)
            logger.info("Sensor reading pushed to Firebase Realtime Database.")
            return True
        except Exception as e:
            logger.error(f"Failed to push sensor reading to Firebase: {e}")
            return False

    def push_disease_event(self, event_data: Dict[str, Any]) -> bool:
        """Appends disease detection alert with image URL to /disease_alerts."""
        if not self.fb.is_ready and not self.fb.initialize():
            return False

        try:
            from firebase_admin import db
            ref = db.reference("disease_alerts")
            ref.push(event_data)
            logger.info(f"Disease alert pushed to Firebase Realtime Database: {event_data.get('disease_name')}")
            return True
        except Exception as e:
            logger.error(f"Failed to push disease alert to Firebase: {e}")
            return False
