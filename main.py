import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import sys
import time
import signal
import threading
import json
import re
import cv2
import numpy as np
from typing import Optional


# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, ROI_SIZE,
    SENSOR_POLL_INTERVAL, ENABLE_GUI_DISPLAY,
    GEMINI_API_KEY, GEMINI_MODEL_NAME, GEMINI_AI_INTERVAL, CONFIDENCE_THRESHOLD
)
from utils.logger import logger
from database import init_db, DatabaseRepository, DatabaseSyncWorker
from camera import USBCamera
from ai import DiseaseDetector, DetectionResult
from sensors import SensorManager
from firebase import RealtimeDatabaseManager, StorageUploader
from lora import LoRaAlertManager

class PlantDetectionSystem:
    def __init__(self):
        self.running = False

        # Initialize Subsystems
        logger.info("==========================================")
        logger.info(" Initializing Plant Detection System")
        logger.info("==========================================")

        # 1. Local Database (WAL mode, FIFO 50-record capping)
        init_db()
        self.repo = DatabaseRepository()
        self.sync_worker = DatabaseSyncWorker()

        # 2. Camera & High-Speed Dual Vision AI (Instant Foliage + YOLO + Background Gemini AI)
        self.camera = USBCamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, roi_size=ROI_SIZE)
        self.detector = DiseaseDetector(
            api_key=GEMINI_API_KEY,
            model_name=GEMINI_MODEL_NAME,
            ai_interval=GEMINI_AI_INTERVAL,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )

        # 3. Sensors
        self.sensor_mgr = SensorManager()

        # 4. Cloud & LoRa
        self.rtdb = RealtimeDatabaseManager()
        self.storage = StorageUploader()
        self.lora = LoRaAlertManager()

        # State & Threading
        self.latest_result: DetectionResult = DetectionResult()
        self.last_sensor_time: float = 0.0
        self.latest_sensor_data: dict = {}
        self.hud_expanded: bool = True
        self.last_disease_alert_time: float = 0.0
        self.disease_alert_cooldown: float = 4.0  # seconds between repeated auto-snapshots / alerts

    def _sensor_loop(self):
        """Background loop to periodically read sensors, save to DB, sync to Firebase, and broadcast over LoRa."""
        logger.info("Sensor monitoring thread started.")
        last_wait_log = 0.0
        while self.running:
            try:
                # Read genuine sensors from Arduino/ESP32 via USB
                sensor_data = self.sensor_mgr.read_all()
                if sensor_data is None:
                    now = time.time()
                    if now - last_wait_log >= 6.0:
                        last_wait_log = now
                        logger.info("Awaiting sensor telemetry from Arduino/ESP32...")
                    time.sleep(SENSOR_POLL_INTERVAL)
                    continue

                self.latest_sensor_data = sensor_data
                t_str = f"{sensor_data.get('temperature')}°C" if sensor_data.get('temperature') is not None else "N/C"
                h_str = f"{sensor_data.get('humidity')}%" if sensor_data.get('humidity') is not None else "N/C"
                soil_str = f"{sensor_data.get('soil_moisture')}%" if sensor_data.get('soil_moisture') is not None else "N/C"
                mq_str = f"{sensor_data.get('mq135_raw')} ({sensor_data.get('mq135_voltage')}V)" if sensor_data.get('mq135_raw') is not None else "N/C"
                logger.info(f"Sensors -> Temp: {t_str} | Hum: {h_str} | Soil: {soil_str} | Air Quality: {mq_str}")

                # Save to local SQLite database (strictly buffered offline)
                record_id = self.repo.insert_sensor_reading(sensor_data)

                # 1. Update Firebase Live Status
                live_payload = {
                    "last_updated": sensor_data.get("datetime") or sensor_data.get("timestamp"),
                    "sensors": sensor_data,
                    "latest_disease": self.latest_result.top_class if self.latest_result.has_disease else "None",
                    "disease_confidence": self.latest_result.confidence,
                    "severity": self.latest_result.severity
                }
                self.rtdb.update_live_status(live_payload)

                # 2. Push historical reading to Firebase /sensor_readings (if online)
                if self.rtdb.push_sensor_reading(sensor_data):
                    self.repo.mark_sensor_reading_synced(record_id)
                    # Release local synced sensor record
                    self.repo.delete_synced_sensor_reading(record_id)

                # 3. Broadcast newest sensor reading over LoRa to ESP32 OLED Receiver
                self.lora.send_latest_sensor_data(sensor_data)

                # 4. Check sensor threshold alerts
                alerts = self.sensor_mgr.check_alerts(sensor_data)
                for alert in alerts:
                    logger.warning(f"Sensor threshold alert triggered: {alert}")
                    self.lora.trigger_sensor_alert(alert)

            except Exception as e:
                logger.error(f"Error in sensor monitoring thread: {e}")

            time.sleep(SENSOR_POLL_INTERVAL)

    def _handle_disease_event(self, frame: np.ndarray, result: DetectionResult):
        """
        Triggered when plant disease is detected:
        1. Saves automatic snapshot locally
        2. Transmits instant Emergency Alert over LoRa to ESP32 OLED Receiver
        3. Records event in local SQLite (offline persistence)
        4. Uploads snapshot to Firebase Storage & pushes record to Firebase RTDB (if online)
        5. Once confirmed uploaded to cloud, local record/snapshot is released.
        """
        now = time.time()
        if (now - self.last_disease_alert_time) < self.disease_alert_cooldown:
            return
        self.last_disease_alert_time = now

        logger.warning(f"🚨 DISEASE DETECTED: {result.top_class} (Confidence: {result.confidence:.2f})")

        # 1. Save Automatic Snapshot locally
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', result.top_class)
        snapshot_path = self.camera.save_snapshot(frame, prefix=f"disease_{clean_name}")

        # 2. Transmit over LoRa SX127x to ESP32 OLED receiver
        self.lora.trigger_disease_alert(result.top_class, result.confidence)

        # 3. Log event into local SQLite database (persists in case of no internet)
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        dt_str = time.strftime("%Y-%m-%d %H:%M:%S")
        event_id = self.repo.insert_disease_event(
            timestamp=now_iso,
            datetime_str=dt_str,
            disease_name=result.top_class,
            confidence=result.confidence,
            snapshot_path=snapshot_path
        )

        # 4. Asynchronously upload to Firebase if network is available
        def upload_and_sync():
            try:
                photo_url = self.storage.upload_image(
                    local_image_path=snapshot_path,
                    metadata={
                        "disease_name": result.top_class,
                        "confidence": f"{result.confidence:.2f}",
                        "timestamp": now_iso
                    }
                )

                alert_payload = {
                    "timestamp": now_iso,
                    "datetime": dt_str,
                    "model_name": GEMINI_MODEL_NAME,
                    "disease_name": result.top_class,
                    "confidence": round(result.confidence, 4),
                    "severity": result.severity,
                    "organic_remedy": result.organic_remedy,
                    "chemical_remedy": result.chemical_remedy,
                    "photo_url": photo_url or ""
                }

                if self.rtdb.push_disease_event(alert_payload):
                    self.repo.mark_detection_synced(event_id, firebase_image_url=photo_url or "")
                    # Release local SQLite entry once confirmed uploaded to Firebase
                    self.repo.delete_synced_detection(event_id)
                    logger.info(f"Disease alert & Photo URL for '{result.top_class}' sent to Firebase: {photo_url}")

                # Update live_status on cloud
                self.rtdb.update_live_status({
                    "last_updated": dt_str,
                    "latest_disease": result.top_class,
                    "disease_confidence": result.confidence,
                    "latest_photo_url": photo_url or "",
                    "severity": result.severity,
                    "sensors": self.latest_sensor_data
                })
            except Exception as e:
                logger.error(f"Failed to sync disease event to Firebase (stored locally for offline retry): {e}")

        threading.Thread(target=upload_and_sync, daemon=True).start()

    def start(self):
        """Starts the main system pipeline."""
        self.running = True

        # 1. Start SQLite-to-Firebase offline sync worker
        self.sync_worker.start()

        # 2. Start Sensor polling background thread
        sensor_thread = threading.Thread(target=self._sensor_loop, daemon=True)
        sensor_thread.start()

        # 3. Open USB Camera
        camera_ok = self.camera.open()
        if not camera_ok:
            logger.warning("Camera not detected at startup. System will run sensor monitoring and retry camera connection.")

        logger.info("\n=======================================================")
        logger.info(" System Active - Real-Time Plant Vision & Sensor Bridge")
        logger.info(f" AI Vision Model : {GEMINI_MODEL_NAME}")
        logger.info(f" LoRa Transmitter : 433 MHz / SF7 / BW125")
        logger.info(" Controls: [Q] Quit  |  [SPACE] Instant AI Re-scan  |  [H] Toggle HUD")
        logger.info("=======================================================\n")

        fps_history = []
        prev_time = time.time()
        last_cam_retry = time.time()

        try:
            while self.running:
                # Camera reconnection handler
                if self.camera.cap is None or not self.camera.cap.isOpened():
                    now = time.time()
                    if now - last_cam_retry >= 4.0:
                        last_cam_retry = now
                        self.camera.open(verbose=False)
                    time.sleep(0.5)
                    continue

                ret, frame = self.camera.read_frame()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # Run Multi-Tiered Dual Engine Detection (Instant Foliage Tracker + YOLO + Background Gemini)
                result, local_yolo_boxes, instant_leaf_boxes, gemini_data = self.detector.process_frame(frame)
                self.latest_result = result

                # Check if disease was detected
                if result.has_disease:
                    self._handle_disease_event(frame, result)

                # FPS Calculation
                curr_time = time.time()
                dt = curr_time - prev_time
                prev_time = curr_time
                fps = 1.0 / dt if dt > 0 else 30.0
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)

                # GUI Display Handling
                if ENABLE_GUI_DISPLAY:
                    # Prepare sensor overlay string
                    sensor_text = None
                    if self.latest_sensor_data:
                        t = self.latest_sensor_data.get('temperature', '--')
                        h = self.latest_sensor_data.get('humidity', '--')
                        m = self.latest_sensor_data.get('soil_moisture', '--')
                        mq = self.latest_sensor_data.get('mq135_raw', '--')
                        sensor_text = f"Sensors: T:{t}C  H:{h}%  Soil:{m}%  Air:{mq}"

                    annotated = self.detector.draw_hud(
                        frame=frame,
                        fps=avg_fps,
                        gemini_data=gemini_data,
                        local_yolo_boxes=local_yolo_boxes,
                        instant_leaf_boxes=instant_leaf_boxes,
                        hud_expanded=self.hud_expanded,
                        sensor_overlay_text=sensor_text
                    )

                    cv2.imshow("Plant Disease & Health Monitor", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord('q'), ord('Q'), 27]:
                        logger.info("Quit command received.")
                        break
                    elif key == ord(' '):
                        logger.info("Triggering immediate Gemini AI re-scan...")
                        self.detector.request_instant_scan()
                    elif key in [ord('h'), ord('H')]:
                        self.hud_expanded = not self.hud_expanded
                    elif key in [ord('s'), ord('S')]:
                        self.camera.save_snapshot(annotated, prefix="manual_report")
                else:
                    time.sleep(0.03)

        except KeyboardInterrupt:
            logger.info("Shutdown signal received (KeyboardInterrupt).")
        finally:
            self.stop()

    def stop(self):
        """Gracefully shuts down all hardware, threads, and databases."""
        if not self.running:
            return
        self.running = False
        logger.info("Stopping Plant Detection System...")

        self.camera.release()
        self.sensor_mgr.stop()
        self.detector.stop()
        self.sync_worker.stop()
        self.lora.lora.close()

        if ENABLE_GUI_DISPLAY:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        logger.info("Plant Detection System terminated safely.")

def main():
    app = PlantDetectionSystem()

    def sig_handler(signum, frame):
        logger.info(f"Signal {signum} received. Exiting...")
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    app.start()

if __name__ == "__main__":
    main()
