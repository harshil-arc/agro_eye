import os
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import platform
import time
import cv2
import numpy as np
from typing import Optional, Tuple
from config import CAMERA_INDICES, CAMERA_WIDTH, CAMERA_HEIGHT, ROI_SIZE, SNAPSHOT_DIR
from utils.logger import logger

class USBCamera:
    def __init__(self, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT, roi_size: int = ROI_SIZE):
        self.width = width
        self.height = height
        self.roi_size = roi_size
        self.cap: Optional[cv2.VideoCapture] = None
        self.active_index: Optional[int] = None
        self.consecutive_errors: int = 0
        self._logged_no_cam: bool = False

    def _get_backends(self):
        """Returns ordered list of optimal video capture backends for the OS."""
        system = platform.system()
        if system == "Windows":
            return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif system == "Linux":
            return [cv2.CAP_V4L2, cv2.CAP_ANY]
        else:
            return [cv2.CAP_ANY]

    def open(self, verbose: bool = True) -> bool:
        """Searches and opens an available USB camera across configured indices and backends."""
        self.release()
        if verbose and not self._logged_no_cam:
            logger.info("Searching for USB camera (prioritizing Index 1)...")
        backends = self._get_backends()

        # Prioritize Camera 1 (External USB Webcam) over Camera 0 (Laptop)
        search_indices = [1, 0, 2]

        for camera_index in search_indices:
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(camera_index, backend)
                    if cap.isOpened():
                        # Set resolution
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

                        # Test reading a test frame
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            self.cap = cap
                            self.active_index = camera_index
                            self.consecutive_errors = 0
                            self._logged_no_cam = False
                            cam_type = "USB External Webcam" if camera_index == 1 else f"Camera Device {camera_index}"
                            logger.info(f"Camera opened at index {camera_index} ({cam_type}, Backend: {backend})")
                            return True

                        cap.release()
                except Exception as e:
                    logger.debug(f"Camera open error (Index {camera_index}, Backend {backend}): {e}")
                    continue

        if not self._logged_no_cam:
            logger.warning("No camera detected. System will continue monitoring sensors and retry camera connection.")
            self._logged_no_cam = True
        return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a single frame from the camera with auto-reconnect on dropped connections."""
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.consecutive_errors += 1
            if self.consecutive_errors >= 5:
                logger.warning("Multiple camera frame grab failures. Releasing camera to allow reconnect...")
                self.release()
            return False, None

        self.consecutive_errors = 0
        return ret, frame

    def get_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extracts center square Region of Interest (ROI) for disease analysis.
        Returns (cropped_frame, (x1, y1, x2, y2)).
        """
        h, w, _ = frame.shape
        box_size = min(self.roi_size, w, h)
        x1 = max(0, (w - box_size) // 2)
        y1 = max(0, (h - box_size) // 2)
        x2 = min(w, x1 + box_size)
        y2 = min(h, y1 + box_size)
        crop = frame[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2)

    def save_snapshot(self, frame: np.ndarray, prefix: str = "snapshot") -> str:
        """Saves a frame to disk with a timestamped filename and returns the file path."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jpg"
        file_path = str(SNAPSHOT_DIR / filename)
        cv2.imwrite(file_path, frame)
        logger.info(f"Snapshot saved locally: {file_path}")
        return file_path

    def switch_camera(self) -> bool:
        """Switches to the next available camera index."""
        current = self.active_index if self.active_index is not None else 0
        next_index = 0 if current == 1 else 1
        logger.info(f"Switching camera to Index {next_index}...")
        self.release()
        backends = self._get_backends()
        for backend in backends:
            try:
                cap = cv2.VideoCapture(next_index, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        self.cap = cap
                        self.active_index = next_index
                        cam_type = "USB External Webcam" if next_index == 1 else f"Camera Device {next_index}"
                        logger.info(f"Successfully switched to index {next_index} ({cam_type})")
                        return True
                    cap.release()
            except Exception:
                pass

        # Fallback to reopen
        return self.open(verbose=True)

    def release(self):
        """Releases the camera hardware handle."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.active_index = None
