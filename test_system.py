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

    def test_database_200_entries_fifo_limit(self):
        """Verify that SQLite strictly caps table size at maximum 200 newest entries."""
        for i in range(1, 221):  # Insert 220 records
            data = {
                "timestamp": f"2026-09-07T12:{i // 60:02d}:{i % 60:02d}Z",
                "source": "esp32_sensor",
                "temperature": float(20.0 + (i * 0.05)),
                "humidity": 60.0,
                "soil_moisture": 50.0,
                "soil_raw": 2000 + i,
                "mq135_raw": 100 + i,
                "mq135_voltage": 0.5
            }
            self.repo.insert_sensor_reading(data)

        # Query all records
        all_records = self.repo.get_unsynced_sensor_readings(limit=250)
        self.assertEqual(len(all_records), 200, f"Expected 200 entries, found {len(all_records)}")

        # Verify the newest entry (iteration 220) is preserved
        latest = get_latest_sensor_readings(db_path=self.test_db_path, limit=1)
        self.assertEqual(latest[0]["soil_raw"], 2220)



    def test_lora_packet_formatting(self):
        alert_mgr = LoRaAlertManager(cooldown=0.1)
        # Should return boolean without crashing
        alert_mgr.trigger_disease_alert("Tomato_Leaf_Mold", 0.88)
        alert_mgr.send_latest_sensor_data()

    def test_temporal_confirmation_tracker(self):
        """Tests sliding-window confirmation logic."""
        from main import TemporalConfirmationTracker
        tracker = TemporalConfirmationTracker(required_frames=3, window_seconds=2.0)

        # 1st detection (unconfirmed)
        is_conf, name, conf, status, count = tracker.update([
            {"class_id": 0, "class_name": "Tomato Late Blight", "confidence": 0.91, "box": [10, 10, 100, 100]}
        ])
        self.assertFalse(is_conf)
        self.assertEqual(count, 1)
        self.assertEqual(status, "DETECTING (1/3)")

        # 2nd detection (unconfirmed)
        is_conf, name, conf, status, count = tracker.update([
            {"class_id": 0, "class_name": "Tomato Late Blight", "confidence": 0.89, "box": [10, 10, 100, 100]}
        ])
        self.assertFalse(is_conf)
        self.assertEqual(count, 2)
        self.assertEqual(status, "DETECTING (2/3)")

        # 3rd detection (CONFIRMED)
        is_conf, name, conf, status, count = tracker.update([
            {"class_id": 0, "class_name": "Tomato Late Blight", "confidence": 0.93, "box": [10, 10, 100, 100]}
        ])
        self.assertTrue(is_conf)
        self.assertEqual(count, 3)
        self.assertEqual(status, "CONFIRMED")
        self.assertEqual(name, "Tomato Late Blight")
        self.assertAlmostEqual(conf, 0.91, delta=0.02)

    def test_save_snapshot_and_database(self):
        """Tests snapshot file generation and SQLite disease_detections record insertion."""
        import numpy as np
        from main import save_snapshot, save_detection_to_database
        from database import get_recent_disease_detections

        # Test Frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        snap_path = save_snapshot(
            annotated_frame=dummy_frame,
            disease_name="Tomato Early Blight",
            confidence=0.914,
            snapshot_dir="snapshots"
        )
        self.assertIsNotNone(snap_path)
        self.assertTrue(os.path.exists(snap_path))
        self.assertIn("tomato_early_blight_91", snap_path)

        # Test Database Insertion
        rec_id = save_detection_to_database(
            db_path=self.test_db_path,
            disease_name="Tomato Early Blight",
            confidence=0.914,
            snapshot_path=snap_path,
            camera_id="usb_camera_0",
            status="confirmed"
        )
        self.assertGreater(rec_id, 0)

        # Verify record retrieval
        records = get_recent_disease_detections(limit=5, db_path=self.test_db_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["disease_name"], "Tomato Early Blight")
        self.assertEqual(records[0]["confidence"], 0.914)
        self.assertEqual(records[0]["status"], "confirmed")
        self.assertEqual(records[0]["snapshot_path"], snap_path)

        # Cleanup snapshot file
        if os.path.exists(snap_path):
            try:
                os.remove(snap_path)
            except Exception:
                pass

if __name__ == "__main__":
    unittest.main()

