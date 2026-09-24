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
    YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD,
    ANIMAL_MODEL_PATH, ANIMAL_CONF_THRESHOLD, ENABLE_ANIMAL_DETECTION
)
from utils.logger import logger
from database import init_db, DatabaseRepository, DatabaseSyncWorker
from camera import USBCamera
from ai import DiseaseDetector, DetectionResult, AnimalDetector, AnimalDetectionResult
from sensors import SensorManager
from firebase import RealtimeDatabaseManager, StorageUploader
from lora import LoRaAlertManager
from streaming import WebRTCStreamer

class PlantDetectionSystem:
    def __init__(self):
        self.running = False

        # Initialize Subsystems
        logger.info("==========================================")
        logger.info(" Initializing AgroEye Dual Vision AI System")
        logger.info("==========================================")

        # 1. Local Database (WAL mode, FIFO 50-record capping)
        init_db()
        self.repo = DatabaseRepository()
        self.sync_worker = DatabaseSyncWorker()

        # 2. Camera & 100% Offline AI Vision Models
        self.camera = USBCamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, roi_size=ROI_SIZE)
        
        # Model 1: Plant Pathology Vision AI
        self.detector = DiseaseDetector(
            model_path=YOLO_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )

        # Model 2: Wildlife & Farm Intrusion Vision AI (D:\animal_detection_model)
        self.animal_detector = AnimalDetector(
            model_path=ANIMAL_MODEL_PATH,
            confidence_threshold=ANIMAL_CONF_THRESHOLD
        )
        self.enable_animal_ai = ENABLE_ANIMAL_DETECTION

        # 3. Sensors
        self.sensor_mgr = SensorManager()

        # 4. Cloud, LoRa & WebRTC Live Streaming
        self.rtdb = RealtimeDatabaseManager()
        self.storage = StorageUploader()
        self.lora = LoRaAlertManager()
        self.streamer = WebRTCStreamer()


        # State & Threading
        self.latest_result: DetectionResult = DetectionResult()
        self.latest_animal_result: AnimalDetectionResult = AnimalDetectionResult()
        self.last_sensor_time: float = 0.0
        self.latest_sensor_data: dict = {}
        self.hud_expanded: bool = True
        self.last_disease_alert_time: float = 0.0
        self.disease_alert_cooldown: float = 4.0
        self.consecutive_disease_frames: int = 0

        self.last_animal_alert_time: float = 0.0
        self.animal_alert_cooldown: float = 4.0
        self.consecutive_animal_frames: int = 0

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

        disp_conf = getattr(result, "display_confidence", int(result.confidence * 100))
        logger.warning(f"🚨 DISEASE DETECTED: {result.top_class} (Match: {disp_conf}%)")

        # 1. Save Automatic Snapshot locally
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', result.top_class)
        snapshot_path = self.camera.save_snapshot(frame, prefix=f"disease_{clean_name}")

        # 2. Transmit over LoRa SX127x to ESP32 OLED receiver
        self.lora.update_latest_disease(result.top_class, float(disp_conf))
        self.lora.trigger_disease_alert(result.top_class, float(disp_conf))

        # 3. Log event into local SQLite database (persists in case of no internet)
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        dt_str = time.strftime("%Y-%m-%d %H:%M:%S")
        event_id = self.repo.insert_disease_event(
            timestamp=now_iso,
            datetime_str=dt_str,
            disease_name=result.top_class,
            confidence=float(disp_conf) / 100.0,
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
                    "model_name": "yolov8-disease-model",
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

    def _handle_animal_event(self, frame: np.ndarray, animal_result: AnimalDetectionResult):
        """
        Triggered when wildlife or farm animal intrusion is detected:
        1. Saves automatic snapshot locally
        2. Transmits instant Emergency Intrusion Alert over LoRa
        3. Asynchronously uploads snapshot to Firebase Storage & RTDB
        """
        now = time.time()
        if (now - self.last_animal_alert_time) < self.animal_alert_cooldown:
            return
        self.last_animal_alert_time = now

        top_animal = animal_result.top_animal
        total = animal_result.total_animals
        conf = animal_result.display_confidence
        is_threat = animal_result.has_threat
        threat_level = animal_result.threat_level

        logger.warning(f"🐾 ANIMAL DETECTED: {top_animal} (Count: {total}, Match: {conf}%, Status: {threat_level})")

        # 1. Save Automatic Snapshot locally
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', top_animal)
        snapshot_path = self.camera.save_snapshot(frame, prefix=f"animal_{clean_name}")

        # 2. Transmit over LoRa SX127x
        self.lora.trigger_animal_alert(
            animal_name=top_animal,
            count=total,
            confidence=float(conf),
            is_threat=is_threat
        )

        # 3. Asynchronously upload to Firebase
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        dt_str = time.strftime("%Y-%m-%d %H:%M:%S")

        def upload_animal_sync():
            try:
                photo_url = self.storage.upload_image(
                    local_image_path=snapshot_path,
                    metadata={
                        "animal_name": top_animal,
                        "count": str(total),
                        "confidence": f"{animal_result.top_confidence:.2f}",
                        "timestamp": now_iso
                    }
                )

                animal_payload = {
                    "timestamp": now_iso,
                    "datetime": dt_str,
                    "model_name": animal_result.model_name,
                    "animal_name": top_animal,
                    "total_count": total,
                    "species_breakdown": animal_result.counts,
                    "confidence": round(animal_result.top_confidence, 4),
                    "is_crop_threat": is_threat,
                    "threat_level": threat_level,
                    "photo_url": photo_url or ""
                }

                self.rtdb.push_disease_event(animal_payload)
                logger.info(f"Animal intrusion alert for '{top_animal}' sent to Firebase: {photo_url}")
            except Exception as e:
                logger.error(f"Failed to sync animal intrusion event to Firebase: {e}")

        threading.Thread(target=upload_animal_sync, daemon=True).start()

    def start(self):
        """Starts the main system pipeline."""
        self.running = True

        # 1. Start SQLite-to-Firebase offline sync worker
        self.sync_worker.start()

        # 2. Start Sensor polling background thread
        sensor_thread = threading.Thread(target=self._sensor_loop, daemon=True)
        sensor_thread.start()

        # 3. Start WebRTC Live Video Streaming Engine (Pi -> Internet -> Farmer App)
        self.streamer.start()

        # 4. Open USB Camera
        camera_ok = self.camera.open()
        if not camera_ok:
            logger.warning("Camera not detected at startup. System will run sensor monitoring and retry camera connection.")

        logger.info("\n=======================================================")
        logger.info(" System Active - Dual Vision AI (Plant + Animal Detection)")
        logger.info(" Plant Disease Model : 100% Offline (models/disease_model.pt)")
        logger.info(" Animal AI Model     : Wildlife & Intrusion Guard")
        logger.info(" LoRa Transmitter    : 433 MHz / SF7 / BW125")
        logger.info(" WebRTC Streaming    : Active (Pi -> Internet -> Farmer's App)")
        logger.info(" Controls: [Q] Quit | [C] Cam | [S] Save | [H] HUD | [A] Animal AI | [+/-] Sens")
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

                # 1. Run 100% Offline Plant Pathology Model
                plant_result, plant_boxes, _, det_data = self.detector.process_frame(frame)
                self.latest_result = plant_result

                if plant_result.has_disease:
                    self.consecutive_disease_frames += 1
                    if self.consecutive_disease_frames >= 2:
                        self._handle_disease_event(frame, plant_result)
                else:
                    self.consecutive_disease_frames = 0

                # 2. Run Animal & Wildlife Intrusion Model
                animal_result = AnimalDetectionResult()
                if self.enable_animal_ai and self.animal_detector.model_loaded:
                    animal_result = self.animal_detector.detect(frame)
                    self.latest_animal_result = animal_result

                    if animal_result.has_animals:
                        self.consecutive_animal_frames += 1
                        if self.consecutive_animal_frames >= 2:
                            self._handle_animal_event(frame, animal_result)
                    else:
                        self.consecutive_animal_frames = 0

                # FPS Calculation
                curr_time = time.time()
                dt = curr_time - prev_time
                prev_time = curr_time
                fps = 1.0 / dt if dt > 0 else 30.0
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)

                # Prepare sensor overlay string
                sensor_text = None
                if self.latest_sensor_data:
                    t = self.latest_sensor_data.get('temperature', '--')
                    h = self.latest_sensor_data.get('humidity', '--')
                    m = self.latest_sensor_data.get('soil_moisture', '--')
                    mq = self.latest_sensor_data.get('mq135_raw', '--')
                    sensor_text = f"Sensors: T:{t}C  H:{h}%  Soil:{m}%  Air:{mq}"

                # Generate Annotated Frame with AI Bounding Boxes & Sensor HUD
                annotated = self.detector.draw_hud(
                    frame=frame,
                    fps=avg_fps,
                    detection_data=det_data,
                    local_yolo_boxes=plant_boxes,
                    animal_boxes=animal_result.boxes if self.enable_animal_ai else [],
                    animal_result=animal_result if self.enable_animal_ai else None,
                    animal_thresh=self.animal_detector.conf_thresh if self.enable_animal_ai else 0.35,
                    hud_expanded=self.hud_expanded,
                    sensor_overlay_text=sensor_text
                )

                # Push live frame to WebRTC Streamer buffer for remote farmer app
                self.streamer.update_frame(annotated)

                # GUI Display Handling (if desktop/monitor attached)
                if ENABLE_GUI_DISPLAY:
                    cv2.imshow("AgroEye - Plant Disease & Animal Intrusion Monitor", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord('q'), ord('Q'), 27]:
                        logger.info("Quit command received.")
                        break
                    elif key in [ord('h'), ord('H')]:
                        self.hud_expanded = not self.hud_expanded
                    elif key in [ord('s'), ord('S')]:
                        self.camera.save_snapshot(annotated, prefix="manual_report")
                    elif key in [ord('c'), ord('C')]:
                        self.camera.switch_camera()
                    elif key in [ord('a'), ord('A')]:
                        self.enable_animal_ai = not self.enable_animal_ai
                        status_str = "ENABLED" if self.enable_animal_ai else "DISABLED"
                        logger.info(f"Animal Intrusion AI is now: {status_str}")
                    elif key in [ord('+'), ord('=')]:
                        self.detector.adjust_threshold(+0.02)
                        if self.enable_animal_ai:
                            self.animal_detector.adjust_threshold(+0.05)
                    elif key in [ord('-'), ord('_')]:
                        self.detector.adjust_threshold(-0.02)
                        if self.enable_animal_ai:
                            self.animal_detector.adjust_threshold(-0.05)
                else:
                    time.sleep(0.01)

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

        self.streamer.stop()
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
