import sys
import os
import unittest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from database import (
    init_db,
    DatabaseRepository,
    get_connection,
    get_latest_sensor_readings,
    get_latest_detections,
    get_system_events
)
from sensors import SensorManager
from lora import LoRaAlertManager

class TestPlantSystem(unittest.TestCase):
    def setUp(self):
        self.test_db_path = "data/test_plant_system.db"
        init_db(self.test_db_path)
        self.repo = DatabaseRepository(db_path=self.test_db_path)

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except Exception:
                pass

    def test_database_insert_and_sync(self):
        sample_sensor_data = {
            "timestamp": "2026-09-06T12:00:00Z",
            "source": "esp32_sensor",
            "temperature": 26.5,
            "humidity": 65.0,
            "soil_moisture": 45.2,
            "soil_raw": 2150,
            "mq135_raw": 160,
            "mq135_voltage": 0.52
        }
        rec_id = self.repo.insert_sensor_reading(sample_sensor_data)
        self.assertGreater(rec_id, 0)

        unsynced = self.repo.get_unsynced_sensor_readings(limit=10)
        self.assertEqual(len(unsynced), 1)
        self.assertEqual(unsynced[0]["temperature"], 26.5)
        self.assertEqual(unsynced[0]["mq135_raw"], 160)

        # Test latest sensor reading helper
        latest = get_latest_sensor_readings(db_path=self.test_db_path)
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["humidity"], 65.0)

        self.repo.mark_sensor_reading_synced(rec_id)
        unsynced_after = self.repo.get_unsynced_sensor_readings(limit=10)
        self.assertEqual(len(unsynced_after), 0)

    def test_detections_and_events(self):
        # Insert detection
        det_id = self.repo.insert_disease_event(
            timestamp="2026-09-06T12:05:00Z",
            datetime_str="2026-09-06 12:05:00",
            disease_name="Powdery_Mildew",
            confidence=0.89,
            snapshot_path="data/snapshots/test.jpg"
        )
        self.assertGreater(det_id, 0)

        detections = get_latest_detections(limit=5, db_path=self.test_db_path)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_name"], "Powdery_Mildew")

        # Insert system event
        ev_id = self.repo.insert_system_event(
            event_type="BOOT",
            severity="INFO",
            message="System initialized successfully"
        )
        self.assertGreater(ev_id, 0)

        events = get_system_events(db_path=self.test_db_path)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "BOOT")

        # Check DB Health
        self.assertTrue(self.repo.check_health())

    def test_sensor_manager_direct_data(self):
        mgr = SensorManager()
        try:
            # Simulate receiving genuine reading from ESP32
            with mgr.esp32_receiver.lock:
                mgr.esp32_receiver.latest_data = {
                    "source": "esp32_serial",
                    "temperature": 27.2,
                    "humidity": 61.5,
                    "soil_moisture": 52.0,
                    "soil_raw": 2040,
                    "mq135_raw": 150,
                    "mq135_voltage": 0.50
                }
            readings = mgr.read_all()
            self.assertIsNotNone(readings)
            self.assertEqual(readings["temperature"], 27.2)
            self.assertEqual(readings["humidity"], 61.5)
            self.assertEqual(readings["soil_moisture"], 52.0)
            self.assertEqual(readings["mq135_raw"], 150)
            self.assertNotIn("nitrogen", readings)
        finally:
            mgr.stop()

    def test_threshold_alerts(self):
        mgr = SensorManager()
        try:
            test_data = {
                "temperature": 25.0,
                "humidity": 50.0,
                "soil_moisture": 15.0,
                "mq135_raw": 750
            }
            alerts = mgr.check_alerts(test_data)
            self.assertIn("SOIL_DRY:15.0%", alerts)
            self.assertIn("POOR_AIR_QUALITY_MQ135:750", alerts)
        finally:
            mgr.stop()

    def test_database_50_entries_fifo_limit(self):
        """Verify that SQLite strictly caps table size at maximum 50 newest entries."""
        for i in range(1, 66):  # Insert 65 records
            data = {
                "timestamp": f"2026-09-07T12:00:{i:02d}Z",
                "source": "esp32_sensor",
                "temperature": float(20.0 + (i * 0.1)),
                "humidity": 60.0,
                "soil_moisture": 50.0,
                "soil_raw": 2000 + i,
                "mq135_raw": 100 + i,
                "mq135_voltage": 0.5
            }
            self.repo.insert_sensor_reading(data)

        # Query all records
        all_records = self.repo.get_unsynced_sensor_readings(limit=100)
        self.assertEqual(len(all_records), 50, f"Expected 50 entries, found {len(all_records)}")

        # Verify the newest entry (iteration 65) is preserved
        latest = get_latest_sensor_readings(db_path=self.test_db_path, limit=1)
        self.assertEqual(latest[0]["soil_raw"], 2065)

    def test_lora_packet_formatting(self):
        alert_mgr = LoRaAlertManager(cooldown=0.1)
        # Should return boolean without crashing
        alert_mgr.trigger_disease_alert("Tomato_Leaf_Mold", 0.88)
        alert_mgr.send_latest_sensor_data()

if __name__ == "__main__":
    unittest.main()
