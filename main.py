import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

def _setup_linux_display():
    """Auto-configures X11 / Wayland display environment for Raspberry Pi desktop."""
    if os.name == "nt":
        return
    
    # 1. Fallback to default X11 display if not present
    if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"

    # 2. Discover XDG_RUNTIME_DIR for Wayland and Qt
    if "XDG_RUNTIME_DIR" not in os.environ:
        for uid in [os.getuid() if hasattr(os, "getuid") else 1000, 1000, 1001]:
            candidate = f"/run/user/{uid}"
            if os.path.exists(candidate):
                os.environ["XDG_RUNTIME_DIR"] = candidate
                break

    # 3. Discover Wayland socket if active
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir and "WAYLAND_DISPLAY" not in os.environ:
        for sock in ["wayland-0", "wayland-1"]:
            if os.path.exists(os.path.join(runtime_dir, sock)):
                os.environ["WAYLAND_DISPLAY"] = sock
                break

    # 4. Discover Xauthority for sudo / SSH execution
    if "XAUTHORITY" not in os.environ:
        for auth in [
            os.path.expanduser("~/.Xauthority"),
            "/home/pi/.Xauthority",
            "/home/gullu/.Xauthority",
            "/run/user/1000/.Xauthority"
        ]:
            if os.path.exists(auth):
                os.environ["XAUTHORITY"] = auth
                break

_setup_linux_display()

import sys
import time
import signal
import threading
import json
import re
import gc
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from typing import Optional

def _ensure_cv2_shims():
    """Ensures critical OpenCV function stubs exist so dependencies like Ultralytics never crash."""
    if not hasattr(cv2, "imshow"):
        cv2.imshow = lambda winname, mat: None
    if not hasattr(cv2, "namedWindow"):
        cv2.namedWindow = lambda winname, flags=0: None
    if not hasattr(cv2, "resizeWindow"):
        cv2.resizeWindow = lambda winname, width, height: None
    if not hasattr(cv2, "destroyAllWindows"):
        cv2.destroyAllWindows = lambda: None
    if not hasattr(cv2, "waitKey"):
        cv2.waitKey = lambda delay=0: -1

_ensure_cv2_shims()

try:
    import torch
    # Cap PyTorch intra-op threads to 2 to prevent starving OpenCV and WebRTC on Raspberry Pi
    torch.set_num_threads(min(2, os.cpu_count() or 2))
    torch.set_grad_enabled(False)
    TORCH_INFERENCE = torch.inference_mode
except Exception:
    import contextlib
    TORCH_INFERENCE = contextlib.nullcontext

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
from servo import ServoController

class PlantDetectionSystem:
    def __init__(self):
        self.running = False
        self.upload_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="AgroEyeUpload")

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

        # 4. Cloud, LoRa & Outbound WebRTC Live Streaming (Pi <-> Farmer App via Firebase Signaling)
        self.rtdb = RealtimeDatabaseManager()
        self.storage = StorageUploader()
        self.lora = LoRaAlertManager()
        self.streamer = WebRTCStreamer()

        # 5. Dual-Mode PTZ Camera Servo Controller (Driven natively via ESP32)
        self.servo = ServoController(sensor_manager=self.sensor_mgr)


        # State & Threading
        self.latest_result: DetectionResult = DetectionResult()
        self.latest_plant_boxes: list = []
        self.latest_det_data: dict = {
            "has_disease": False,
            "top_class": "None",
            "disease_name": "None",
            "plant_name": "Crop",
            "display_confidence": 0,
            "confidence": 0.0,
            "severity": "None",
            "organic_remedy": "",
            "chemical_remedy": ""
        }
        self.latest_animal_result: AnimalDetectionResult = AnimalDetectionResult()
        self.ai_lock = threading.Lock()

        # Display & Monitor Auto-Configuration for Linux / Raspberry Pi Desktop
        if os.name != "nt" and ENABLE_GUI_DISPLAY:
            if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
                os.environ["DISPLAY"] = ":0"
            if "XDG_RUNTIME_DIR" not in os.environ:
                user_id = os.getuid() if hasattr(os, "getuid") else 1000
                if os.path.exists(f"/run/user/{user_id}"):
                    os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{user_id}"

        self.gui_available = ENABLE_GUI_DISPLAY

        self.last_sensor_time: float = 0.0
        self.latest_sensor_data: dict = {}
        self.hud_expanded: bool = True
        self.last_disease_alert_time: float = 0.0
        self.disease_alert_cooldown: float = 12.0
        self.consecutive_disease_frames: int = 0

        self.last_animal_alert_time: float = 0.0
        self.animal_alert_cooldown: float = 10.0
        self.consecutive_animal_frames: int = 0
        self.last_sensor_alert_times: dict = {}
        self.sensor_alert_cooldown: float = 60.0

        # Elephant-Specific Active-LOW Hooter Relay Control (ESP32 GPIO 26)
        self.hooter_active: bool = False
        self.last_elephant_seen_time: float = 0.0
        self.consecutive_elephant_frames: int = 0
        self.elephant_clear_delay: float = 4.0  # Seconds to keep hooter sounding after elephant exits frame

    def _ai_worker_loop(self):
        """
        High-Speed Decoupled AI Vision Worker Thread.
        Continuously processes full-resolution camera frames with 100% full model precision
        and accuracy in the background, allowing the camera & WebRTC video stream
        to achieve smooth 25-30+ FPS with sub-50ms latency.
        """
        logger.info("Decoupled Vision AI inference worker started.")
        iter_count = 0
        while self.running:
            try:
                ret, frame = self.camera.read_frame()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                with TORCH_INFERENCE():
                    # 1. Run 100% Full-Precision Plant Pathology Model
                    plant_result, plant_boxes, _, det_data = self.detector.process_frame(frame)

                    # 2. Run 100% Full-Precision Animal & Wildlife Intrusion Model
                    animal_result = AnimalDetectionResult()
                    if self.enable_animal_ai and self.animal_detector.model_loaded:
                        animal_result = self.animal_detector.detect(frame)

                # Atomically update latest detection results for the high-speed stream
                with self.ai_lock:
                    self.latest_result = plant_result
                    self.latest_plant_boxes = plant_boxes
                    self.latest_det_data = det_data
                    self.latest_animal_result = animal_result

                # 3. Handle Disease Alerts (Require >= 3 consecutive frames with confidence >= threshold)
                if plant_result.has_disease and plant_result.confidence >= self.detector.confidence_threshold:
                    self.consecutive_disease_frames += 1
                    if self.consecutive_disease_frames >= 3:
                        self._handle_disease_event(frame, plant_result, plant_boxes, det_data)
                else:
                    self.consecutive_disease_frames = 0

                # 4. Handle Wildlife / Animal Intrusion Alerts
                if animal_result.has_animals and animal_result.top_confidence >= self.animal_detector.conf_thresh:
                    self.consecutive_animal_frames += 1
                    if self.consecutive_animal_frames >= 2:
                        self._handle_animal_event(frame, animal_result)
                else:
                    self.consecutive_animal_frames = 0

                # 5. Handle Elephant-Specific Hooter Alarm (ESP32 GPIO 26 -> HIGH)
                is_elephant = False
                if animal_result.has_animals:
                    if "Elephant" in animal_result.counts or any(b.get("clean_name") == "elephant" for b in animal_result.boxes):
                        is_elephant = True

                if is_elephant and animal_result.top_confidence >= self.animal_detector.conf_thresh:
                    self.consecutive_elephant_frames += 1
                    if self.consecutive_elephant_frames >= 2:
                        self.last_elephant_seen_time = time.time()
                        if not self.hooter_active:
                            self.hooter_active = True
                            logger.warning("🐘 ELEPHANT DETECTED! Triggering Hooter Relay on ESP32 (GPIO 26 -> HIGH)...")
                            self.sensor_mgr.set_hooter(True)
                else:
                    self.consecutive_elephant_frames = 0
                    # Auto-turn off hooter once elephant leaves the frame after clear delay
                    if self.hooter_active and (time.time() - self.last_elephant_seen_time >= self.elephant_clear_delay):
                        self.hooter_active = False
                        logger.info("Elephant perimeter cleared. Deactivating Hooter Relay on ESP32 (GPIO 26 -> LOW)...")
                        self.sensor_mgr.set_hooter(False)

                iter_count += 1
                if iter_count % 120 == 0:
                    gc.collect()

                # Adaptive cadence: sleep 60ms between inferences (~12-15 inferences/sec)
                # Prevents CPU saturation while maintaining immediate (<100ms) detection reaction time.
                time.sleep(0.06)

            except Exception as e:
                logger.error(f"Error in Vision AI worker thread: {e}")
                time.sleep(0.05)

    def _sensor_loop(self):
        """Background loop to periodically read sensors, save to DB, sync to Firebase, and broadcast over LoRa."""
        logger.info("Sensor monitoring thread started.")
        while self.running:
            try:
                # Read sensors from Arduino/ESP32 via USB (returns NULL/None for disconnected sensors)
                sensor_data = self.sensor_mgr.read_all()
                self.latest_sensor_data = sensor_data

                t_val = sensor_data.get('temperature')
                h_val = sensor_data.get('humidity')
                soil_val = sensor_data.get('soil_moisture')
                mq_val = sensor_data.get('mq135_raw')
                servo_val = sensor_data.get('servo_angle')

                t_str = f"{t_val}°C" if t_val is not None else "NULL"
                h_str = f"{h_val}%" if h_val is not None else "NULL"
                soil_str = f"{soil_val}%" if soil_val is not None else "NULL"
                mq_str = f"{mq_val} ({sensor_data.get('mq135_voltage')}V)" if mq_val is not None else "NULL"
                servo_mode = str(sensor_data.get('servo_mode', 'auto')).upper()
                servo_str = f"{servo_val}° ({servo_mode})" if servo_val is not None else "NULL"
                hooter_str = str(sensor_data.get('hooter', 'OFF')).upper()
                conn_str = "ESP32 Connected" if sensor_data.get('is_connected') else "ESP32 Disconnected (Standby)"

                logger.info(f"Sensors [{conn_str}] -> Temp: {t_str} | Hum: {h_str} | Soil: {soil_str} | Air: {mq_str} | Servo: {servo_str} | Hooter: {hooter_str}")

                # Save to local SQLite database (buffered offline)
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

                # 3. Broadcast newest sensor reading over LoRa to ESP32 OLED Receiver (sends nulls if disconnected)
                self.lora.send_latest_sensor_data(sensor_data)

                # 4. Check sensor threshold alerts (Debounced to max once per 60s per alert)
                alerts = self.sensor_mgr.check_alerts(sensor_data)
                now_t = time.time()
                for alert in alerts:
                    last_fired = self.last_sensor_alert_times.get(alert, 0.0)
                    if (now_t - last_fired) >= self.sensor_alert_cooldown:
                        self.last_sensor_alert_times[alert] = now_t
                        logger.warning(f"Sensor threshold alert triggered: {alert}")
                        self.lora.trigger_sensor_alert(alert)

            except Exception as e:
                logger.error(f"Error in sensor monitoring thread: {e}")

            time.sleep(SENSOR_POLL_INTERVAL)

    def _handle_disease_event(self, frame: np.ndarray, result: DetectionResult, plant_boxes: list = None, det_data: dict = None):
        """
        Triggered when plant disease is confirmed:
        1. Generates and saves an annotated diagnostic snapshot with disease bounding boxes & remedy tags
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

        # 1. Render clear diagnostic overlay on snapshot for cloud and local archive
        annotated_snapshot = frame.copy()
        if det_data is not None and plant_boxes:
            annotated_snapshot = self.detector.draw_hud(
                frame=annotated_snapshot,
                fps=30.0,
                detection_data=det_data,
                local_yolo_boxes=plant_boxes,
                hud_expanded=True,
                sensor_overlay_text=None
            )

        # 2. Save Automatic Snapshot locally
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', result.top_class)
        snapshot_path = self.camera.save_snapshot(annotated_snapshot, prefix=f"disease_{clean_name}")

        # 3. Transmit over LoRa SX127x to ESP32 OLED receiver
        self.lora.update_latest_disease(result.top_class, float(disp_conf))
        self.lora.trigger_disease_alert(result.top_class, float(disp_conf))

        # 4. Log event into local SQLite database (persists in case of no internet)
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        dt_str = time.strftime("%Y-%m-%d %H:%M:%S")
        event_id = self.repo.insert_disease_event(
            timestamp=now_iso,
            datetime_str=dt_str,
            disease_name=result.top_class,
            confidence=float(disp_conf) / 100.0,
            snapshot_path=snapshot_path
        )

        # 5. Asynchronously upload to Firebase if network is available
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

        self.upload_executor.submit(upload_and_sync)

    def _handle_animal_event(self, frame: np.ndarray, animal_result: AnimalDetectionResult):
        """
        Triggered when wildlife or farm animal intrusion is detected:
        1. Saves annotated snapshot locally with animal bounding boxes
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

        # 1. Render animal intrusion bounding box overlay on snapshot
        annotated_snapshot = frame.copy()
        if animal_result.boxes:
            dummy_det = {"has_disease": False, "disease_name": "None", "plant_name": "Wildlife Intrusion"}
            annotated_snapshot = self.detector.draw_hud(
                frame=annotated_snapshot,
                fps=30.0,
                detection_data=dummy_det,
                local_yolo_boxes=[],
                animal_boxes=animal_result.boxes,
                animal_result=animal_result,
                hud_expanded=True
            )

        # 2. Save Automatic Snapshot locally
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', top_animal)
        snapshot_path = self.camera.save_snapshot(annotated_snapshot, prefix=f"animal_{clean_name}")

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
                self.rtdb.push_snapshot(animal_payload)
                logger.info(f"Animal intrusion alert for '{top_animal}' sent to Firebase: {photo_url}")
            except Exception as e:
                logger.error(f"Failed to sync animal intrusion event to Firebase: {e}")

        self.upload_executor.submit(upload_animal_sync)

    def _handle_manual_snapshot(self, annotated_frame: np.ndarray):
        """
        Triggered when farmer manually captures a photo ('s' key / app command):
        1. Saves snapshot locally
        2. Uploads to Cloud Hosting / Firebase Storage
        3. Pushes metadata with photo_url to Firebase /snapshots (capped at 200 items FIFO)
        4. Updates live_status with latest photo URL
        """
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        dt_str = time.strftime("%Y-%m-%d %H:%M:%S")
        snapshot_path = self.camera.save_snapshot(annotated_frame, prefix="manual_snapshot")
        logger.info(f"Manual snapshot saved locally: {snapshot_path}")

        def upload_manual_sync():
            try:
                photo_url = self.storage.upload_image(
                    local_image_path=snapshot_path,
                    metadata={"type": "manual_snapshot", "timestamp": now_iso}
                )

                payload = {
                    "timestamp": now_iso,
                    "datetime": dt_str,
                    "type": "manual_snapshot",
                    "disease_name": self.latest_result.top_class if self.latest_result.has_disease else "Healthy Foliage",
                    "confidence": round(self.latest_result.confidence, 4),
                    "severity": self.latest_result.severity,
                    "sensors": self.latest_sensor_data,
                    "photo_url": photo_url or ""
                }

                self.rtdb.push_snapshot(payload)
                self.rtdb.update_live_status({
                    "last_updated": dt_str,
                    "latest_photo_url": photo_url or "",
                    "sensors": self.latest_sensor_data
                })
                logger.info(f"Manual snapshot successfully sent to Firebase /snapshots: {photo_url}")
            except Exception as e:
                logger.error(f"Failed to upload manual snapshot to Firebase: {e}")

        self.upload_executor.submit(upload_manual_sync)

    def start(self):
        """Starts the main system pipeline with high-speed 30 FPS streaming and decoupled AI inference."""
        self.running = True

        # 1. Start SQLite-to-Firebase offline sync worker
        self.sync_worker.start()

        # 2. Start Sensor polling background thread
        sensor_thread = threading.Thread(target=self._sensor_loop, daemon=True)
        sensor_thread.start()

        # 3. Start Outbound SRT Live Video Streaming Engine (Pi -> Cloud VPS / Media Server)
        self.streamer.start()

        # 4. Open High-Speed USB Camera
        camera_ok = self.camera.open()
        if not camera_ok:
            logger.warning("Camera not detected at startup. System will run sensor monitoring and retry camera connection.")

        # 5. Start Decoupled AI Vision Worker Thread (Runs at 100% full model precision in background)
        ai_thread = threading.Thread(target=self._ai_worker_loop, daemon=True)
        ai_thread.start()

        # 6. Start Dual-Mode PTZ Camera Servo Engine (Auto Sweep & Mobile App Manual Direction)
        self.servo.start()

        # 7. Initialize Local Desktop Camera Live Monitor Window
        if self.gui_available:
            try:
                cv2.namedWindow("AgroEye - Plant Disease & Animal Intrusion Monitor", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("AgroEye - Plant Disease & Animal Intrusion Monitor", 1280, 720)
                splash_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(splash_frame, "AgroEye Vision AI - Initializing Hardware...", (320, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.imshow("AgroEye - Plant Disease & Animal Intrusion Monitor", splash_frame)
                cv2.waitKey(1)
                logger.info("🖥️ Local Camera Live Monitor Window initialized on display.")
            except Exception as e:
                err_str = str(e)
                if "not implemented" in err_str.lower() or "headless" in err_str.lower():
                    logger.warning("⚠️ OpenCV headless package detected in venv (GUI disabled). To enable the camera screen on the Pi, run:")
                    logger.warning("   source venv/bin/activate && pip uninstall -y opencv-python-headless && pip install opencv-python")
                else:
                    logger.warning(f"Local GUI window init skipped ({e}). Running in headless streaming mode.")
                self.gui_available = False

        logger.info("\n=======================================================")
        logger.info(" System Active - Dual Vision AI (Plant + Animal Detection)")
        lora_status = "433 MHz / SF7 / BW125 (Active)" if (hasattr(self.lora, 'lora') and self.lora.lora.is_ready) else "Standby / Not Connected"
        logger.info(f" LoRa Transmitter    : {lora_status}")
        logger.info(" Live Video Pipeline : WebRTC (1280x720 @ 30 FPS / H.264 / VP8 via Firebase Signaling)")
        logger.info(" Camera PTZ Servo    : ESP32 Hardware Driven (GPIO 18) / USB Serial")
        logger.info(" Controls: [Q] Quit | [C] Cam | [S] Save | [H] HUD | [A] Animal AI | [+/-] Sens")
        logger.info("=======================================================\n")

        fps_history = []
        prev_time = time.time()
        last_cam_retry = time.time()
        last_frame_id = -1

        try:
            while self.running:
                # Camera reconnection handler
                if self.camera.cap is None or not self.camera.cap.isOpened():
                    now = time.time()
                    if now - last_cam_retry >= 2.5:
                        last_cam_retry = now
                        self.camera.open(verbose=False)

                    if self.gui_available:
                        try:
                            standby_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                            cv2.putText(standby_frame, "AgroEye - Searching for Camera Device...", (330, 320),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2, cv2.LINE_AA)
                            cv2.putText(standby_frame, "Connecting to /dev/video0 or external USB camera", (340, 370),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)

                            if self.latest_sensor_data:
                                t = self.latest_sensor_data.get('temperature')
                                h = self.latest_sensor_data.get('humidity')
                                m = self.latest_sensor_data.get('soil_moisture')
                                s_str = f"Sensors Active -> Temp: {t}C | Hum: {h}% | Soil: {m}%"
                                cv2.putText(standby_frame, s_str, (40, 680),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 1, cv2.LINE_AA)

                            cv2.imshow("AgroEye - Plant Disease & Animal Intrusion Monitor", standby_frame)
                            key = cv2.waitKey(100) & 0xFF
                            if key in [ord('q'), ord('Q'), 27]:
                                break
                        except Exception:
                            pass
                    else:
                        time.sleep(0.5)
                    continue

                # Synchronized frame grab (blocks until new hardware frame arrives, locked at ~30 FPS)
                ret, frame, frame_id = self.camera.get_frame(wait_for_new=True, last_id=last_frame_id, timeout=0.04)
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue
                last_frame_id = frame_id

                # FPS Calculation
                curr_time = time.time()
                dt = curr_time - prev_time
                prev_time = curr_time
                fps = 1.0 / dt if dt > 0 else 30.0
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)

                # Fetch latest detection overlays atomically (< 0.01ms)
                with self.ai_lock:
                    plant_boxes = list(self.latest_plant_boxes)
                    det_data = self.latest_det_data
                    animal_res = self.latest_animal_result

                # Prepare sensor overlay string
                sensor_text = None
                if self.latest_sensor_data:
                    t = self.latest_sensor_data.get('temperature')
                    h = self.latest_sensor_data.get('humidity')
                    m = self.latest_sensor_data.get('soil_moisture')
                    mq = self.latest_sensor_data.get('mq135_raw')
                    srv = self.latest_sensor_data.get('servo_angle')

                    t_str = f"{t}°C" if t is not None else "NULL"
                    h_str = f"{h}%" if h is not None else "NULL"
                    m_str = f"{m}%" if m is not None else "NULL"
                    mq_str = str(mq) if mq is not None else "NULL"
                    srv_str = f"{srv}°" if srv is not None else "NULL"
                    sensor_text = f"Sensors: T:{t_str}  H:{h_str}  Soil:{m_str}  Air:{mq_str}  Pan:{srv_str}"

                # Generate Annotated Frame with AI Bounding Boxes & Sensor HUD (Runs at 25-30 FPS)
                annotated = self.detector.draw_hud(
                    frame=frame,
                    fps=avg_fps,
                    detection_data=det_data,
                    local_yolo_boxes=plant_boxes,
                    animal_boxes=animal_res.boxes if self.enable_animal_ai else [],
                    animal_result=animal_res if self.enable_animal_ai else None,
                    animal_thresh=self.animal_detector.conf_thresh if self.enable_animal_ai else 0.35,
                    hud_expanded=self.hud_expanded,
                    sensor_overlay_text=sensor_text
                )

                # Push live frame to WebRTC Streamer with zero lag (< 0.1ms)
                self.streamer.update_frame(annotated)

                # GUI Display Handling (if desktop/monitor attached)
                if self.gui_available:
                    try:
                        cv2.imshow("AgroEye - Plant Disease & Animal Intrusion Monitor", annotated)
                        key = cv2.waitKey(1) & 0xFF
                        if key in [ord('q'), ord('Q'), 27]:
                            logger.info("Quit command received.")
                            break
                        elif key in [ord('h'), ord('H')]:
                            self.hud_expanded = not self.hud_expanded
                        elif key in [ord('s'), ord('S')]:
                            self._handle_manual_snapshot(annotated)
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
                    except (cv2.error, Exception) as e:
                        logger.warning(f"GUI display not available ({e}). Running seamlessly in headless WebRTC streaming mode.")
                        self.gui_available = False
                        time.sleep(0.005)
                else:
                    time.sleep(0.005)

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

        try:
            self.upload_executor.shutdown(wait=False)
        except Exception:
            pass

        if self.hooter_active:
            try:
                self.sensor_mgr.set_hooter(False)
            except Exception:
                pass

        self.streamer.stop()
        self.servo.stop()
        self.camera.release()
        self.sensor_mgr.stop()
        self.detector.stop()
        self.sync_worker.stop()
        self.lora.lora.close()

        if self.gui_available:
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
