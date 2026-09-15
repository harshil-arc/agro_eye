#!/usr/bin/env python3
"""
=============================================================================
AGRO EYE: Real-Time Plant Disease Detection System (YOLOv8)
=============================================================================
Production-ready smart farming disease detection application for Raspberry Pi 5
with USB Webcam, Ultralytics YOLOv8 (models/disease_model.pt), temporal
confirmation, snapshot storage, and SQLite database logging.

Pipeline:
  USB Webcam -> OpenCV Video Capture -> YOLOv8 disease_model.pt
  -> Bounding Box + Class Name + Confidence -> Temporal Confirmation
  -> Confirmed Disease -> Save Snapshot -> Save Detection in SQLite
  -> (Future: LoRa + Firebase Cloud Sync)
=============================================================================
"""

import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import sys
import time
import signal
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import (
    MODEL_PATH,
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FPS,
    CAMERA_ID,
    CAMERA_RETRY_INTERVAL,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    IMAGE_SIZE,
    CONFIRMATION_FRAMES,
    CONFIRMATION_WINDOW_SECONDS,
    SNAPSHOT_DIRECTORY,
    SNAPSHOT_COOLDOWN_SECONDS,
    DATABASE_PATH,
    ENABLE_GUI_DISPLAY,
    WINDOW_NAME
)
from utils.logger import logger
from database import init_database, insert_disease_detection

# Global shutdown flag for signal handling
SHUTDOWN_REQUESTED = False


def signal_handler(signum, frame):
    """Handles SIGINT and SIGTERM for graceful application shutdown."""
    global SHUTDOWN_REQUESTED
    logger.info("Shutdown signal received. Stopping Agro Eye gracefully...")
    SHUTDOWN_REQUESTED = True


# Register OS signals
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ===========================================================================
# 1. MODEL LOADING & VERIFICATION
# ===========================================================================
def load_model(model_path: str = MODEL_PATH):
    """
    Verifies that the trained YOLOv8 model exists at models/disease_model.pt,
    loads it using Ultralytics YOLO, and dynamically reads all class names.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        error_msg = (
            "\n"
            "============================================================\n"
            "ERROR: disease_model.pt not found.\n"
            f"Place the trained model at: {model_path}\n"
            "============================================================\n"
        )
        sys.stderr.write(error_msg)
        logger.error(f"Trained model not found at '{model_path}'. Please copy disease_model.pt into models/ directory.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
        logger.info(f"Loading YOLOv8 model from: {model_path}")
        model = YOLO(model_path)
        
        # Read class names dynamically from the trained model
        class_names = model.names
        if isinstance(class_names, list):
            class_dict = {i: name for i, name in enumerate(class_names)}
        elif isinstance(class_names, dict):
            class_dict = class_names
        else:
            class_dict = {0: "Plant Disease"}

        # Print structured startup banner
        print("\n========================================")
        print(" AGRO EYE DISEASE DETECTION")
        print("========================================")
        print(f"Model: {model_path}")
        print("Classes:")
        for cls_id, cls_name in class_dict.items():
            print(f"  {cls_id}: {cls_name}")
        print(f"Camera: USB Camera (Index: {CAMERA_INDEX})")
        print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print("Status: READY")
        print("========================================\n")

        logger.info(f"YOLOv8 model loaded successfully with {len(class_dict)} classes.")
        return model, class_dict

    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        sys.exit(1)


# ===========================================================================
# 2. CAMERA INITIALIZATION & RECONNECTION
# ===========================================================================
def initialize_camera(
    camera_index: int = CAMERA_INDEX,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    fps: int = FPS
) -> Optional[cv2.VideoCapture]:
    """
    Opens the USB webcam using OpenCV, configures resolution and FPS,
    and tests frame acquisition. Handles camera absence gracefully.
    """
    logger.info(f"Initializing USB camera at index {camera_index} ({width}x{height} @ {fps}fps)...")
    
    # Try preferred backend (V4L2 on Linux/Pi, DirectShow/MSMF on Windows, Default fallback)
    cap = None
    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    elif sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        logger.error(f"ERROR: USB camera not detected at index {camera_index}.")
        return None

    # Configure hardware parameters
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Test read frame
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        logger.warning(f"Camera opened at index {camera_index} but failed to capture test frame.")
        cap.release()
        return None

    logger.info(f"USB Camera initialized successfully on index {camera_index}.")
    return cap


# ===========================================================================
# 3. DATABASE INITIALIZATION
# ===========================================================================
def initialize_database(db_path: str = DATABASE_PATH) -> None:
    """
    Initializes SQLite database and creates disease_detections table if not existing.
    Never deletes or overwrites existing records.
    """
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    init_database(db_path)
    logger.info(f"Agro Eye SQLite database ready at: {db_path}")


# ===========================================================================
# 4. YOLO INFERENCE
# ===========================================================================
def run_detection(
    model: Any,
    frame: np.ndarray,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
    imgsz: int = IMAGE_SIZE
) -> List[Dict[str, Any]]:
    """
    Runs YOLOv8 disease inference on a single video frame.
    Returns a structured list of detection dictionaries.
    """
    if frame is None or model is None:
        return []

    try:
        # Run inference using Ultralytics YOLO
        results = model(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False
        )

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    conf = float(box.conf[0].cpu().item())
                    cls_id = int(box.cls[0].cpu().item())
                    cls_name = model.names.get(cls_id, f"Disease_{cls_id}")
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

                    detections.append({
                        "class_id": cls_id,
                        "class_name": str(cls_name),
                        "confidence": round(conf, 4),
                        "box": xyxy  # [x1, y1, x2, y2]
                    })

        return detections

    except Exception as e:
        logger.error(f"Error during YOLO inference: {e}")
        return []


# ===========================================================================
# 5. TEMPORAL CONFIRMATION TRACKER
# ===========================================================================
class TemporalConfirmationTracker:
    """
    Sliding-window temporal confirmation tracker to eliminate false positives.
    Requires a disease to be observed across at least CONFIRMATION_FRAMES
    within CONFIRMATION_WINDOW_SECONDS before marking it as CONFIRMED.
    """
    def __init__(
        self,
        required_frames: int = CONFIRMATION_FRAMES,
        window_seconds: float = CONFIRMATION_WINDOW_SECONDS
    ):
        self.required_frames = required_frames
        self.window_seconds = window_seconds
        # Stores history tuples: (timestamp, disease_name, confidence)
        self.history: List[Tuple[float, str, float]] = []

    def update(self, detections: List[Dict[str, Any]]) -> Tuple[bool, str, float, str, int]:
        """
        Updates the tracker with detections from the latest frame.
        Returns:
          - is_confirmed: bool
          - dominant_disease: str
          - dominant_conf: float
          - status_str: str ('CONFIRMED', 'DETECTING (x/y)', 'Monitoring')
          - count: int (number of frames observed in current window)
        """
        now = time.time()

        # 1. Prune entries outside the time window
        cutoff = now - self.window_seconds
        self.history = [entry for entry in self.history if entry[0] >= cutoff]

        # 2. Append current detections into history
        if detections:
            # Pick detection with highest confidence in current frame
            best_detection = max(detections, key=lambda d: d["confidence"])
            self.history.append((now, best_detection["class_name"], best_detection["confidence"]))

        # 3. If no history in window -> Monitoring
        if not self.history:
            return False, "No disease detected", 0.0, "Monitoring", 0

        # 4. Count occurrences of each disease in the window
        counts: Dict[str, int] = {}
        confs: Dict[str, List[float]] = {}
        for _, name, conf in self.history:
            counts[name] = counts.get(name, 0) + 1
            confs.setdefault(name, []).append(conf)

        # 5. Identify dominant disease
        dominant_name = max(counts, key=counts.get)
        count = counts[dominant_name]
        avg_conf = sum(confs[dominant_name]) / len(confs[dominant_name])

        # 6. Check confirmation threshold
        if count >= self.required_frames:
            status_str = "CONFIRMED"
            return True, dominant_name, avg_conf, status_str, count
        else:
            status_str = f"DETECTING ({count}/{self.required_frames})"
            return False, dominant_name, avg_conf, status_str, count


def confirm_detection(
    tracker: TemporalConfirmationTracker,
    detections: List[Dict[str, Any]]
) -> Tuple[bool, str, float, str, int]:
    """Helper function to evaluate temporal confirmation on current frame detections."""
    return tracker.update(detections)


# ===========================================================================
# 6. SNAPSHOT STORAGE SYSTEM
# ===========================================================================
def save_snapshot(
    annotated_frame: np.ndarray,
    disease_name: str,
    confidence: float,
    snapshot_dir: str = SNAPSHOT_DIRECTORY
) -> Optional[str]:
    """
    Saves an annotated snapshot image of a confirmed disease detection.
    Filename: YYYY-MM-DD_HH-MM-SS_<sanitized_disease_name>_<conf_int>.jpg
    Example: 2026-09-15_17-30-42_tomato_late_blight_91.jpg
    """
    try:
        Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
        
        # Sanitize disease name for filesystem safety
        clean_name = re.sub(r'[^a-zA-Z0-9]+', '_', disease_name.strip().lower()).strip('_')
        now_dt = datetime.now()
        timestamp_prefix = now_dt.strftime("%Y-%m-%d_%H-%M-%S")
        conf_percent = int(round(confidence * 100))
        
        filename = f"{timestamp_prefix}_{clean_name}_{conf_percent}.jpg"
        filepath = os.path.join(snapshot_dir, filename)

        # Write JPEG image
        success = cv2.imwrite(filepath, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if success:
            logger.info(f"Snapshot saved: {filepath}")
            return filepath
        else:
            logger.error(f"Failed to write snapshot to: {filepath}")
            return None

    except Exception as e:
        logger.error(f"Error saving snapshot: {e}")
        return None


# ===========================================================================
# 7. SQLITE DATABASE INSERTION
# ===========================================================================
def save_detection_to_database(
    db_path: str,
    disease_name: str,
    confidence: float,
    snapshot_path: str,
    camera_id: str = CAMERA_ID,
    status: str = "confirmed"
) -> Optional[int]:
    """
    Inserts a confirmed disease detection record into SQLite database table disease_detections.
    Uses parameterized SQL to guarantee data integrity.
    """
    try:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record_id = insert_disease_detection(
            timestamp=timestamp_str,
            disease_name=disease_name,
            confidence=confidence,
            snapshot_path=snapshot_path,
            camera_id=camera_id,
            status=status,
            synced_to_firebase=0,
            db_path=db_path
        )
        logger.info(
            f"Database record #{record_id} saved -> {disease_name} "
            f"(Conf: {confidence * 100:.1f}%, Path: {snapshot_path})"
        )
        return record_id

    except Exception as e:
        logger.error(f"Database insertion error: {e}")
        return None


# ===========================================================================
# 8. VISUAL OVERLAY & DRAWING
# ===========================================================================
def draw_detection(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    dominant_disease: str,
    dominant_conf: float,
    status_str: str,
    fps: float
) -> np.ndarray:
    """
    Draws YOLO bounding boxes, class labels with confidence, and top HUD banner
    onto the video frame for live display and snapshot recording.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # 1. Draw Bounding Boxes for all detections
    for det in detections:
        box = det["box"]
        x1, y1, x2, y2 = box
        cls_name = det["class_name"]
        conf = det["confidence"]
        label = f"{cls_name} {int(round(conf * 100))}%"

        # Bounding box color (Green for high confidence, Orange/Yellow otherwise)
        box_color = (0, 230, 0) if conf >= 0.75 else (0, 165, 255)

        # Draw Rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # Label background badge
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        badge_y1 = max(0, y1 - th - 8)
        badge_y2 = y1
        badge_x2 = min(w, x1 + tw + 10)

        cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), box_color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 5, badge_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    # 2. Draw Top Semi-Transparent HUD Bar
    hud_height = 68
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Header Title
    cv2.putText(
        annotated,
        "AGRO EYE Plant Disease Detection",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Status Pill / Color
    if status_str == "CONFIRMED":
        status_color = (0, 255, 0)  # Bright Green
    elif "DETECTING" in status_str:
        status_color = (0, 200, 255)  # Yellow
    else:
        status_color = (200, 200, 200)  # Light Gray

    # Subtitle with live metrics
    if dominant_disease and dominant_disease != "No disease detected":
        info_text = (
            f"Disease: {dominant_disease}  |  "
            f"Confidence: {dominant_conf * 100:.1f}%  |  "
            f"Status: {status_str}  |  "
            f"FPS: {fps:.1f}"
        )
    else:
        info_text = f"Disease: No disease detected  |  Status: Monitoring  |  FPS: {fps:.1f}"

    cv2.putText(
        annotated,
        info_text,
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        status_color,
        1,
        cv2.LINE_AA
    )

    # Bottom subtle timestamp watermark
    timestamp_tag = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        annotated,
        f"AgroEye Node | {timestamp_tag}",
        (12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    return annotated


# ===========================================================================
# 9. CLEANUP & SHUTDOWN
# ===========================================================================
def cleanup(cap: Optional[cv2.VideoCapture], window_name: str = WINDOW_NAME) -> None:
    """Releases webcam hardware resources and destroys OpenCV display windows."""
    logger.info("Releasing camera and UI resources...")
    if cap is not None and cap.isOpened():
        cap.release()
    if ENABLE_GUI_DISPLAY:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    logger.info("Agro Eye Plant Disease Detection stopped safely.")


# ===========================================================================
# 10. MAIN APPLICATION LOOP
# ===========================================================================
def main():
    """
    Main application entry point.
    Initializes YOLO model, SQLite database, webcam stream, temporal confirmation,
    and runs real-time continuous plant disease detection.
    """
    global SHUTDOWN_REQUESTED

    logger.info("Starting Agro Eye Plant Disease Detection System...")

    # 1. Initialize SQLite Database
    initialize_database(DATABASE_PATH)

    # 2. Load Trained YOLOv8 Model
    model, class_dict = load_model(MODEL_PATH)

    # 3. Initialize USB Webcam
    cap = initialize_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS)
    if cap is None:
        logger.warning(
            f"Could not open camera index {CAMERA_INDEX}. "
            "Please verify USB camera connection or configure CAMERA_INDEX in config/config.py."
        )

    # 4. Initialize Temporal Confirmation Tracker
    tracker = TemporalConfirmationTracker(
        required_frames=CONFIRMATION_FRAMES,
        window_seconds=CONFIRMATION_WINDOW_SECONDS
    )

    # Cooldown dictionary: {disease_name: last_snapshot_timestamp}
    snapshot_cooldowns: Dict[str, float] = {}

    # Performance & FPS calculation
    fps_counter = 0
    fps_timer = time.time()
    current_fps = 0.0

    last_reconnect_attempt = 0.0
    gui_failed = False

    logger.info("Continuous live detection loop started. Press 'q' or Ctrl+C to stop.")

    try:
        while not SHUTDOWN_REQUESTED:
            loop_start_time = time.time()

            # Handle camera reconnection if disconnected
            if cap is None or not cap.isOpened():
                if loop_start_time - last_reconnect_attempt >= CAMERA_RETRY_INTERVAL:
                    last_reconnect_attempt = loop_start_time
                    logger.info("Attempting to reconnect USB camera...")
                    cap = initialize_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS)
                time.sleep(0.1)
                continue

            # Read frame from USB camera
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Camera frame read failed. Checking device connection...")
                cap.release()
                cap = None
                continue

            # Run YOLOv8 Disease Inference
            detections = run_detection(
                model=model,
                frame=frame,
                conf_threshold=CONFIDENCE_THRESHOLD,
                iou_threshold=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE
            )

            # Update Temporal Confirmation Logic
            is_confirmed, dominant_disease, dominant_conf, status_str, count = confirm_detection(tracker, detections)

            # Calculate live FPS
            fps_counter += 1
            if loop_start_time - fps_timer >= 1.0:
                current_fps = fps_counter / (loop_start_time - fps_timer)
                fps_counter = 0
                fps_timer = loop_start_time

            # Generate visual annotated frame
            annotated_frame = draw_detection(
                frame=frame,
                detections=detections,
                dominant_disease=dominant_disease,
                dominant_conf=dominant_conf,
                status_str=status_str,
                fps=current_fps
            )

            # Trigger Snapshot & Database Storage upon Confirmation
            if is_confirmed and dominant_disease and dominant_disease != "No disease detected":
                now_sec = time.time()
                last_snap = snapshot_cooldowns.get(dominant_disease, 0.0)

                if (now_sec - last_snap) >= SNAPSHOT_COOLDOWN_SECONDS:
                    snapshot_cooldowns[dominant_disease] = now_sec
                    logger.warning(
                        f"🚨 CONFIRMED DISEASE DETECTED: {dominant_disease} "
                        f"(Confidence: {dominant_conf * 100:.1f}%)"
                    )

                    # 1. Save annotated snapshot image
                    snapshot_path = save_snapshot(
                        annotated_frame=annotated_frame,
                        disease_name=dominant_disease,
                        confidence=dominant_conf,
                        snapshot_dir=SNAPSHOT_DIRECTORY
                    )

                    # 2. Insert record into SQLite database
                    if snapshot_path:
                        save_detection_to_database(
                            db_path=DATABASE_PATH,
                            disease_name=dominant_disease,
                            confidence=dominant_conf,
                            snapshot_path=snapshot_path,
                            camera_id=CAMERA_ID,
                            status="confirmed"
                        )

            # Display live camera window (if GUI display is enabled)
            if ENABLE_GUI_DISPLAY and not gui_failed:
                try:
                    cv2.imshow(WINDOW_NAME, annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):  # 'q' or ESC key
                        logger.info("User requested exit from OpenCV window.")
                        break
                except Exception as e:
                    logger.warning(f"OpenCV GUI display unavailable (headless mode active): {e}")
                    gui_failed = True

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")
    except Exception as e:
        logger.error(f"Unexpected error in main detection loop: {e}", exc_info=True)
    finally:
        cleanup(cap, WINDOW_NAME)


if __name__ == "__main__":
    main()

