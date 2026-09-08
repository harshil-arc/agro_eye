import os
import time
import threading
from typing import Optional
from database.repository import DatabaseRepository
from firebase.realtime_db import RealtimeDatabaseManager
from firebase.storage_uploader import StorageUploader
from config import SYNC_INTERVAL
from utils.logger import logger

class DatabaseSyncWorker:
    """
    Background worker that continuously syncs offline SQLite buffered records
    to Firebase Realtime Database and Cloud Storage when internet connectivity is restored.
    Once synced to the cloud, local buffers/snapshots are released.
    """
    def __init__(self, interval: float = SYNC_INTERVAL):
        self.interval = interval
        self.repo = DatabaseRepository()
        self.rtdb = RealtimeDatabaseManager()
        self.storage = StorageUploader()
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("Database sync worker started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Database sync worker stopped.")

    def _run_loop(self):
        while self.running:
            try:
                self._sync_sensor_records()
                self._sync_disease_records()
            except Exception as e:
                logger.error(f"Error in sync worker loop: {e}")
            time.sleep(self.interval)

    def _sync_sensor_records(self):
        records = self.repo.get_unsynced_sensor_readings(limit=30)
        for rec in records:
            payload = {
                "timestamp": rec["timestamp"],
                "datetime": rec["datetime"] if "datetime" in rec.keys() else rec["timestamp"],
                "source": rec["source"] if "source" in rec.keys() else "esp32_sensor",
                "temperature": rec["temperature"],
                "humidity": rec["humidity"],
                "soil_moisture": rec["soil_moisture"],
                "soil_raw": rec["soil_raw"] if "soil_raw" in rec.keys() else None,
                "mq135_raw": rec["mq135_raw"] if "mq135_raw" in rec.keys() else None,
                "mq135_voltage": rec["mq135_voltage"] if "mq135_voltage" in rec.keys() else None
            }
            if self.rtdb.push_sensor_reading(payload):
                self.repo.mark_sensor_reading_synced(rec["id"])
                # Release synced sensor records from local buffer
                self.repo.delete_synced_sensor_reading(rec["id"])
                logger.info(f"Offline sensor record #{rec['id']} synced to Firebase and released from local buffer.")

    def _sync_disease_records(self):
        events = self.repo.get_unsynced_detections(limit=10)
        for ev in events:
            image_url = ev["firebase_image_url"] if "firebase_image_url" in ev.keys() else None
            snapshot_path = ev["image_path"] if "image_path" in ev.keys() else None

            # Upload local snapshot to Firebase Storage if not already uploaded
            if not image_url and snapshot_path and os.path.exists(snapshot_path):
                image_url = self.storage.upload_image(snapshot_path)

            payload = {
                "timestamp": ev["timestamp"],
                "model_name": ev["model_name"] if "model_name" in ev.keys() else "gemini-plant-pathologist",
                "disease_name": ev["class_name"] if "class_name" in ev.keys() else "disease",
                "confidence": ev["confidence"],
                "photo_url": image_url or ""
            }

            if self.rtdb.push_disease_event(payload):
                self.repo.mark_detection_synced(ev["id"], firebase_image_url=image_url)
                # Release local record & cleanup local snapshot to free disk space
                self.repo.delete_synced_detection(ev["id"])
                logger.info(f"Offline disease alert #{ev['id']} ('{payload['disease_name']}') synced to Firebase & released locally.")
