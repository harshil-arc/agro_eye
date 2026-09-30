"""
AgroEye Camera Pan Servo Controller (ESP32 Serial Bridge)
Connects to ESP32 over USB Serial to control camera rotation.
Zero Raspberry Pi GPIO power consumption or PWM overhead.
"""

import os
import time
import threading
from typing import Optional, Dict, Any

from config import (
    ENABLE_SERVO,
    SERVO_AUTO_MIN_ANGLE, SERVO_AUTO_MAX_ANGLE,
    SERVO_AUTO_STEP_DEG, SERVO_AUTO_INTERVAL, SERVO_AUTO_PAUSE_SEC
)
from utils.logger import logger
from firebase.realtime_db import RealtimeDatabaseManager


class CameraControlState:
    """Data structure representing live Pan camera control state."""
    def __init__(
        self,
        mode: str = "auto",
        pan_angle: int = 90,
        x_coord: int = 0,
        command: str = "mode_change",
        step_size: int = 5,
        last_updated: str = "",
        timestamp: int = 0,
        source: str = "esp32_bridge"
    ):
        self.mode = mode
        self.pan_angle = pan_angle
        self.x_coord = x_coord
        self.command = command
        self.step_size = step_size
        self.last_updated = last_updated or time.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.timestamp = timestamp or int(time.time() * 1000)
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "pan_angle": int(self.pan_angle),
            "tilt_angle": 90,
            "x_coord": int(self.pan_angle - 90),
            "y_coord": 0,
            "command": self.command,
            "step_size": self.step_size,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timestamp": int(time.time() * 1000),
            "source": self.source
        }


class ServoController:
    """
    ESP32-Driven Camera Pan Servo Controller.
    - All servo PWM is generated natively on the ESP32 (GPIO 18)
    - Raspberry Pi communicates with ESP32 over USB Serial (115200 baud)
    - Zero Pi GPIO pin usage, avoiding brownouts and voltage drops.
    """
    def __init__(
        self,
        sensor_manager=None,
        min_angle: int = SERVO_AUTO_MIN_ANGLE,
        max_angle: int = SERVO_AUTO_MAX_ANGLE,
        step_deg: int = SERVO_AUTO_STEP_DEG,
        step_interval: float = SERVO_AUTO_INTERVAL,
        pause_sec: float = SERVO_AUTO_PAUSE_SEC
    ):
        self.enabled = ENABLE_SERVO
        self.sensor_mgr = sensor_manager
        self.min_angle = max(0, min(180, min_angle))
        self.max_angle = max(0, min(180, max_angle))
        self.step_deg = step_deg
        self.step_interval = step_interval
        self.pause_sec = pause_sec

        self.current_pan: float = 90.0
        self.target_pan: float = 90.0
        self.mode: str = "auto"
        self.sweep_direction: int = 1

        self.running: bool = False
        self.lock = threading.Lock()
        self.rtdb = RealtimeDatabaseManager()

        # Threads
        self.watchdog_thread: Optional[threading.Thread] = None

        # Track remote updates
        self.last_remote_timestamp: int = 0
        self.last_cloud_sync_time: float = 0.0

    def attach_sensor_manager(self, sensor_manager):
        """Attaches sensor manager instance for ESP32 serial pass-through."""
        self.sensor_mgr = sensor_manager

    def set_mode(self, mode: str):
        """Switches between 'auto' and 'manual' modes."""
        norm_mode = mode.lower().strip()
        if norm_mode not in ("auto", "manual"):
            return

        with self.lock:
            if self.mode != norm_mode:
                self.mode = norm_mode
                logger.info(f"ESP32 Servo Mode changed to: {self.mode.upper()}")
                if self.sensor_mgr:
                    self.sensor_mgr.esp32_receiver.send_command(f"MODE:{norm_mode.upper()}")
                self._push_state_to_cloud(command="mode_change")

    def set_position(self, pan: float, command: str = "set_coords", **kwargs):
        """Directly positions the pan servo via ESP32 in Manual mode."""
        clamped_pan = max(0.0, min(180.0, round(float(pan), 1)))

        with self.lock:
            self.mode = "manual"
            self.target_pan = clamped_pan
            self.current_pan = clamped_pan

        # Send Mode and Angle commands to ESP32 over serial
        if self.sensor_mgr:
            self.sensor_mgr.esp32_receiver.send_command("MODE:MANUAL")
            self.sensor_mgr.send_servo_angle(int(clamped_pan))

        logger.info(f"Manual Camera Servo Position via ESP32: Pan={clamped_pan} deg")
        self._push_state_to_cloud(command=command)

    def _push_state_to_cloud(self, command: str = "update"):
        """Publishes current servo telemetry to Firebase /camera_control."""
        state = CameraControlState(
            mode=self.mode,
            pan_angle=int(round(self.current_pan)),
            command=command,
            source="esp32_bridge"
        )
        self.rtdb.update_camera_control(state.to_dict())

    def _firebase_watchdog_loop(self):
        """
        Polls Firebase Realtime Database (/camera_control) for live user commands
        dispatched from the Farmer's Mobile/Web App and relays to ESP32 immediately.
        """
        logger.info("Firebase Camera PTZ Control Watchdog (ESP32 Bridge) started.")
        while self.running:
            try:
                remote_data = self.rtdb.get_camera_control()
                if remote_data and isinstance(remote_data, dict):
                    remote_ts = int(remote_data.get("timestamp") or 0)
                    remote_source = str(remote_data.get("source") or "")

                    # Only process commands originating from user app that are newer
                    if remote_ts > self.last_remote_timestamp and remote_source not in ("raspberry_pi", "esp32_bridge"):
                        self.last_remote_timestamp = remote_ts
                        remote_mode = str(remote_data.get("mode", "auto")).lower()
                        remote_pan = float(remote_data.get("pan_angle", 90))
                        remote_cmd = str(remote_data.get("command", "remote_update"))

                        with self.lock:
                            prev_mode = self.mode
                            self.mode = remote_mode

                            if remote_mode == "manual":
                                if remote_cmd == "mode_change" and prev_mode != "manual":
                                    # User merely flipped the switch to Manual Mode -> Freeze at current position immediately
                                    logger.info("Farmer App switched to MANUAL MODE -> Freezing camera servo at current position.")
                                    if self.sensor_mgr:
                                        self.sensor_mgr.esp32_receiver.send_command("MODE:MANUAL")
                                else:
                                    # User actively dragged angle slider / pressed direction button
                                    self.target_pan = max(0.0, min(180.0, remote_pan))
                                    self.current_pan = self.target_pan
                                    logger.info(f"Manual Pan Command from Farmer App -> ESP32 Pan: {self.current_pan}° (Cmd: {remote_cmd})")
                                    if self.sensor_mgr:
                                        self.sensor_mgr.esp32_receiver.send_command("MODE:MANUAL")
                                        self.sensor_mgr.send_servo_angle(int(self.current_pan))
                            else:
                                if prev_mode != "auto":
                                    logger.info("Farmer App switched to AUTO SWEEP MODE -> Resuming camera auto-sweep.")
                                    if self.sensor_mgr:
                                        self.sensor_mgr.esp32_receiver.send_command("MODE:AUTO")

            except Exception as e:
                logger.debug(f"Firebase camera control poll error: {e}")

            time.sleep(0.3)  # 300ms poll interval

    def start(self):
        """Starts the Firebase control watchdog thread."""
        if not self.enabled:
            logger.info("Camera Servo Controller is disabled in config.")
            return

        self.running = True

        # Start Firebase Remote Command Watchdog Thread
        self.watchdog_thread = threading.Thread(target=self._firebase_watchdog_loop, daemon=True)
        self.watchdog_thread.start()
        logger.info("ESP32 Camera Servo Bridge active.")

    def stop(self):
        """Safely stops threads."""
        self.running = False
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            self.watchdog_thread.join(timeout=0.5)
        logger.info("Camera Servo Controller stopped.")
