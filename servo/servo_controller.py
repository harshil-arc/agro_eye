"""
AgroEye Dual-Mode PTZ Camera Servo Controller
Supports:
1. Auto Mode: Continuous smooth horizontal scanning/oscillation for 360/180-degree farm inspection.
2. Manual Mode: User-directed pan/tilt position chosen via the Farmer's Mobile/Web App.
3. Live Bi-Directional Firebase RTDB Synchronization: Real-time command intake and live angle telemetry.
4. Universal Compatibility: Native Raspberry Pi GPIO PWM (Hardware/Software) with graceful simulation fallback on PC.
"""

import os
import time
import threading
from typing import Optional, Dict, Any

from config import (
    SERVO_PAN_PIN, SERVO_TILT_PIN, ENABLE_SERVO,
    SERVO_AUTO_MIN_ANGLE, SERVO_AUTO_MAX_ANGLE,
    SERVO_AUTO_STEP_DEG, SERVO_AUTO_INTERVAL, SERVO_AUTO_PAUSE_SEC
)
from utils.logger import logger
from firebase.realtime_db import RealtimeDatabaseManager

# Hardware GPIO detection
RPI_GPIO_AVAILABLE = False
try:
    import RPi.GPIO as GPIO
    RPI_GPIO_AVAILABLE = True
except Exception:
    RPI_GPIO_AVAILABLE = False


class CameraControlState:
    """Data structure representing live PTZ camera control state."""
    def __init__(
        self,
        mode: str = "auto",
        pan_angle: int = 90,
        tilt_angle: int = 90,
        x_coord: int = 0,
        y_coord: int = 0,
        command: str = "mode_change",
        step_size: int = 5,
        last_updated: str = "",
        timestamp: int = 0,
        source: str = "raspberry_pi"
    ):
        self.mode = mode
        self.pan_angle = pan_angle
        self.tilt_angle = tilt_angle
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.command = command
        self.step_size = step_size
        self.last_updated = last_updated or time.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.timestamp = timestamp or int(time.time() * 1000)
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "pan_angle": int(self.pan_angle),
            "tilt_angle": int(self.tilt_angle),
            "x_coord": int(self.pan_angle - 90),
            "y_coord": int(self.tilt_angle - 90),
            "command": self.command,
            "step_size": self.step_size,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timestamp": int(time.time() * 1000),
            "source": self.source
        }


class ServoController:
    """
    High-Precision Dual-Mode Camera Servo Driver.
    - Pan Servo Pin: GPIO 18 (Physical Pin 12, Hardware PWM0 on Raspberry Pi)
    - Tilt Servo Pin: GPIO 13 (Physical Pin 33, Hardware PWM1 on Raspberry Pi)
    """
    def __init__(
        self,
        pan_pin: int = SERVO_PAN_PIN,
        tilt_pin: int = SERVO_TILT_PIN,
        min_angle: int = SERVO_AUTO_MIN_ANGLE,
        max_angle: int = SERVO_AUTO_MAX_ANGLE,
        step_deg: int = SERVO_AUTO_STEP_DEG,
        step_interval: float = SERVO_AUTO_INTERVAL,
        pause_sec: float = SERVO_AUTO_PAUSE_SEC
    ):
        self.enabled = ENABLE_SERVO
        self.pan_pin = pan_pin
        self.tilt_pin = tilt_pin
        self.min_angle = max(0, min(180, min_angle))
        self.max_angle = max(0, min(180, max_angle))
        self.step_deg = step_deg
        self.step_interval = step_interval
        self.pause_sec = pause_sec

        self.current_pan: float = 90.0
        self.current_tilt: float = 90.0
        self.target_pan: float = 90.0
        self.target_tilt: float = 90.0
        self.mode: str = "auto"
        self.sweep_direction: int = 1  # +1 for sweeping right, -1 for sweeping left

        self.running: bool = False
        self.lock = threading.Lock()
        self.rtdb = RealtimeDatabaseManager()

        # Threads
        self.servo_thread: Optional[threading.Thread] = None
        self.watchdog_thread: Optional[threading.Thread] = None

        # Hardware PWM handles
        self.pan_pwm = None
        self.tilt_pwm = None
        self.hardware_active = False

        # Track remote updates
        self.last_remote_timestamp: int = 0
        self.last_cloud_sync_time: float = 0.0

        if self.enabled:
            self._init_hardware()

    def _init_hardware(self):
        """Initializes Raspberry Pi GPIO PWM for the servos if on Pi hardware."""
        if not RPI_GPIO_AVAILABLE:
            logger.info(f"Servo Controller initialized in SIMULATION mode (Pan: GPIO {self.pan_pin} | Tilt: GPIO {self.tilt_pin}).")
            return

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.pan_pin, GPIO.OUT)
            GPIO.setup(self.tilt_pin, GPIO.OUT)

            # 50 Hz PWM standard for hobby RC servos (SG90 / MG995 / MG996R)
            self.pan_pwm = GPIO.PWM(self.pan_pin, 50)
            self.tilt_pwm = GPIO.PWM(self.tilt_pin, 50)

            self.pan_pwm.start(0)
            self.tilt_pwm.start(0)
            self.hardware_active = True
            logger.info(f"Servo Hardware PWM active on Raspberry Pi (Pan: GPIO {self.pan_pin} [Pin 12] | Tilt: GPIO {self.tilt_pin} [Pin 33]).")
            
            # Move to neutral center 90°
            self._write_hardware_angle(90, 90)
        except Exception as e:
            logger.warning(f"Failed to initialize Raspberry Pi GPIO PWM for servo: {e}. Running in simulation fallback.")
            self.hardware_active = False

    def _angle_to_duty(self, angle: float) -> float:
        """Converts angle (0 - 180 degrees) to 50Hz PWM duty cycle (2.5% - 12.5%)."""
        clamped = max(0.0, min(180.0, float(angle)))
        return 2.5 + (clamped / 180.0) * 10.0

    def _write_hardware_angle(self, pan_angle: float, tilt_angle: float):
        """Applies PWM duty cycle to the physical servo pins."""
        if not self.hardware_active:
            return

        try:
            if self.pan_pwm:
                pan_duty = self._angle_to_duty(pan_angle)
                self.pan_pwm.ChangeDutyCycle(pan_duty)

            if self.tilt_pwm:
                tilt_duty = self._angle_to_duty(tilt_angle)
                self.tilt_pwm.ChangeDutyCycle(tilt_duty)

            # Brief pause to let physical motor position itself, then reduce jitter
            time.sleep(0.04)
        except Exception as e:
            logger.debug(f"Hardware servo PWM write error: {e}")

    def set_mode(self, mode: str):
        """Switches between 'auto' and 'manual' modes."""
        norm_mode = mode.lower().strip()
        if norm_mode not in ("auto", "manual"):
            return

        with self.lock:
            if self.mode != norm_mode:
                self.mode = norm_mode
                logger.info(f"Camera PTZ Servo Mode changed to: {self.mode.upper()}")
                self._push_state_to_cloud(command="mode_change")

    def set_position(self, pan: float, tilt: float = 90.0, command: str = "set_coords"):
        """Directly positions the camera in Manual mode."""
        clamped_pan = max(0.0, min(180.0, round(float(pan), 1)))
        clamped_tilt = max(0.0, min(180.0, round(float(tilt), 1)))

        with self.lock:
            self.mode = "manual"
            self.target_pan = clamped_pan
            self.target_tilt = clamped_tilt
            self.current_pan = clamped_pan
            self.current_tilt = clamped_tilt

        self._write_hardware_angle(clamped_pan, clamped_tilt)
        logger.info(f"Manual Camera Servo Position: Pan={clamped_pan} deg | Tilt={clamped_tilt} deg")
        self._push_state_to_cloud(command=command)

    def _push_state_to_cloud(self, command: str = "update"):
        """Publishes current servo telemetry to Firebase /camera_control."""
        state = CameraControlState(
            mode=self.mode,
            pan_angle=int(round(self.current_pan)),
            tilt_angle=int(round(self.current_tilt)),
            command=command,
            source="raspberry_pi"
        )
        self.rtdb.update_camera_control(state.to_dict())

    def _auto_sweep_step(self):
        """Performs one step in the auto-sweep oscillation cycle."""
        with self.lock:
            current = self.current_pan
            direction = self.sweep_direction

            next_pan = current + (direction * self.step_deg)
            is_at_limit = False

            if next_pan >= self.max_angle:
                next_pan = self.max_angle
                self.sweep_direction = -1
                is_at_limit = True
            elif next_pan <= self.min_angle:
                next_pan = self.min_angle
                self.sweep_direction = 1
                is_at_limit = True

            self.current_pan = next_pan
            self.target_pan = next_pan
            pan_val = self.current_pan
            tilt_val = self.current_tilt

        # Move hardware servo
        self._write_hardware_angle(pan_val, tilt_val)

        # Sync with cloud periodically (every 1.2 seconds during sweep)
        now = time.time()
        if now - self.last_cloud_sync_time >= 1.2:
            self.last_cloud_sync_time = now
            self._push_state_to_cloud(command="auto_sweep")

        # Pause at extremities for deep foliage vision scanning
        if is_at_limit:
            time.sleep(self.pause_sec)
        else:
            time.sleep(self.step_interval)

    def _servo_worker_loop(self):
        """Continuous thread executing auto rotation or holding manual position."""
        logger.info("Camera Servo Motor Driver worker started.")
        while self.running:
            try:
                if self.mode == "auto":
                    self._auto_sweep_step()
                else:
                    # In manual mode, sleep lightly to yield CPU
                    time.sleep(0.08)
            except Exception as e:
                logger.error(f"Error in Servo worker loop: {e}")
                time.sleep(0.1)

    def _firebase_watchdog_loop(self):
        """
        Polls Firebase Realtime Database (/camera_control) for live user commands
        dispatched from the Farmer's Mobile/Web App.
        """
        logger.info("Firebase Camera PTZ Control Watchdog started.")
        while self.running:
            try:
                remote_data = self.rtdb.get_camera_control()
                if remote_data and isinstance(remote_data, dict):
                    remote_ts = int(remote_data.get("timestamp") or 0)
                    remote_source = str(remote_data.get("source") or "")
                    
                    # Only process commands originating from user app that are newer
                    if remote_ts > self.last_remote_timestamp and remote_source != "raspberry_pi":
                        self.last_remote_timestamp = remote_ts
                        remote_mode = str(remote_data.get("mode", "auto")).lower()
                        remote_pan = float(remote_data.get("pan_angle", 90))
                        remote_tilt = float(remote_data.get("tilt_angle", 90))
                        remote_cmd = str(remote_data.get("command", "remote_update"))

                        with self.lock:
                            prev_mode = self.mode
                            self.mode = remote_mode
                            
                            if remote_mode == "manual":
                                self.target_pan = max(0.0, min(180.0, remote_pan))
                                self.target_tilt = max(0.0, min(180.0, remote_tilt))
                                self.current_pan = self.target_pan
                                self.current_tilt = self.target_tilt
                                logger.info(f"Remote Manual PTZ Command from Farmer App -> Pan: {self.current_pan} deg | Tilt: {self.current_tilt} deg (Cmd: {remote_cmd})")
                                self._write_hardware_angle(self.current_pan, self.current_tilt)
                            else:
                                if prev_mode != "auto":
                                    logger.info("Remote Auto Sweep Mode activated from Farmer App.")

            except Exception as e:
                logger.debug(f"Firebase camera control poll error: {e}")

            time.sleep(0.3)  # High responsiveness (300ms poll interval)

    def start(self):
        """Starts the servo rotation and Firebase control watchdog threads."""
        if not self.enabled:
            logger.info("Servo Controller is disabled in configuration.")
            return

        self.running = True
        
        # 1. Start Auto Sweep / Position Worker Thread
        self.servo_thread = threading.Thread(target=self._servo_worker_loop, daemon=True)
        self.servo_thread.start()

        # 2. Start Firebase Remote Command Watchdog Thread
        self.watchdog_thread = threading.Thread(target=self._firebase_watchdog_loop, daemon=True)
        self.watchdog_thread.start()

        logger.info(f"PTZ Camera Servo Engine started (Mode: {self.mode.upper()} | Pan Range: {self.min_angle} deg - {self.max_angle} deg)")

    def stop(self):
        """Safely stops threads and releases GPIO PWM handles."""
        self.running = False
        if self.servo_thread and self.servo_thread.is_alive():
            self.servo_thread.join(timeout=0.5)
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            self.watchdog_thread.join(timeout=0.5)

        if self.hardware_active:
            try:
                if self.pan_pwm:
                    self.pan_pwm.stop()
                if self.tilt_pwm:
                    self.tilt_pwm.stop()
                if RPI_GPIO_AVAILABLE:
                    GPIO.cleanup([self.pan_pin, self.tilt_pin])
            except Exception:
                pass
            self.hardware_active = False

        logger.info("Camera Servo Controller stopped safely.")
