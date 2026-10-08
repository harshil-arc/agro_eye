import os
import glob
import json
import re
import math
import threading
import time
from typing import Dict, Any, Optional, List
from config import ESP32_SERIAL_PORT, ESP32_BAUDRATE
from utils.logger import logger

class ESP32SensorReceiver:
    """
    Reads genuine sensor telemetry stream over USB Serial from ESP32 / Arduino.
    Features:
    - Continuous auto-discovery of USB COM ports on Windows & Linux
    - Multi-format resilient parsing: JSON (all key aliases), Key-Value text, and CSV
    - Smart persistent caching: Retains last known valid non-null sensor readings so
      temporary 1-wire timing glitches or servo status updates never wipe out live telemetry.
    - Proactive logging for connection state, raw lines, and exact hardware errors
    """
    def __init__(
        self,
        port: str = ESP32_SERIAL_PORT,
        baudrate: int = ESP32_BAUDRATE
    ):
        self.preferred_port = port
        self.baudrate = baudrate
        self.ser = None
        self.connected_port: Optional[str] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Live state & persistent cache
        self.cached_readings: Dict[str, Any] = {
            "temperature": None,
            "humidity": None,
            "soil_moisture": None,
            "soil_raw": None,
            "mq135_raw": None,
            "mq135_voltage": None,
            "servo_angle": 90,
            "servo_mode": "auto",
            "hooter": "OFF"
        }
        self.cached_times: Dict[str, float] = {}
        self.latest_data: Dict[str, Any] = {}
        self.last_received_time: Optional[float] = None
        self.last_error: Optional[str] = None
        self.last_log_time: float = 0.0
        self.cache_ttl: float = 60.0  # Retain last valid reading for up to 60s

    def _find_available_ports(self) -> List[str]:
        """Scans system for active USB serial ports with cross-platform fallback."""
        ports_list = []
        try:
            import serial.tools.list_ports
            detected = serial.tools.list_ports.comports()
            for p in detected:
                if p.device not in ports_list:
                    ports_list.append(p.device)
        except Exception:
            pass

        # Linux/Pi glob scan for USB serial devices
        if os.name != "nt":
            for pattern in [
                "/dev/ttyUSB*",
                "/dev/ttyACM*",
                "/dev/serial/by-id/*",
                "/dev/serial/by-path/*",
                "/dev/ttyAMA*"
            ]:
                for dev in glob.glob(pattern):
                    if dev not in ports_list:
                        ports_list.append(dev)

        # Prioritize preferred port if available
        if self.preferred_port:
            if self.preferred_port in ports_list:
                ports_list.remove(self.preferred_port)
                ports_list.insert(0, self.preferred_port)
            elif not ports_list:
                ports_list.append(self.preferred_port)

        return ports_list

    def _try_open_port(self, port_name: str) -> bool:
        """Attempts to open a specific serial port with baudrate auto-detection (115200 first for ESP32)."""
        try:
            import serial
            candidate_bauds = [115200, 9600, 57600]
            if self.baudrate not in candidate_bauds:
                candidate_bauds.insert(0, self.baudrate)

            for baud in candidate_bauds:
                try:
                    self.ser = serial.Serial(
                        port_name,
                        baud,
                        timeout=1.0,
                        write_timeout=1.0,
                        dsrdtr=False,
                        rtscts=False
                    )
                    # Explicitly release DTR and RTS so ESP32 auto-reset circuit does not freeze CPU in reset/bootloader
                    try:
                        self.ser.dtr = False
                        self.ser.rts = False
                    except Exception:
                        pass

                    self.connected_port = port_name
                    self.baudrate = baud
                    self.last_error = None
                    logger.info(f"Connected to Arduino/ESP32 on {port_name} at {baud} baud.")
                    return True
                except Exception:
                    continue

            self.last_error = f"Could not open {port_name} at bauds {candidate_bauds}"
            return False
        except Exception as e:
            err_str = f"{type(e).__name__}: {e}"
            if "PermissionError" in err_str or "Access is denied" in err_str:
                self.last_error = f"{port_name} Access Denied (Port is busy or open in Arduino Serial Monitor)"
            else:
                self.last_error = err_str
            return False

    def start(self):
        """Starts background reader thread."""
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        logger.info("Arduino/ESP32 Serial receiver thread started.")

    def stop(self):
        """Stops background reader thread."""
        self.running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        self.connected_port = None
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Arduino/ESP32 Serial receiver stopped.")

    def _clean_num(self, val: Any) -> Optional[float]:
        """Cleans and validates numeric readings, rejecting None, NaN, inf, or error flags."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            if math.isnan(val) or math.isinf(val):
                return None
            return float(val)
        val_str = str(val).strip().replace("°C", "").replace("C", "").replace("%", "").replace("V", "").replace("v", "")
        if val_str.lower() in ("null", "none", "nan", "inf", "-999", "-1", ""):
            return None
        try:
            f = float(val_str)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except (ValueError, TypeError):
            return None

    def _parse_line(self, raw_line: str) -> Optional[Dict[str, Any]]:
        """Parses a serial line using JSON, Key-Value extraction, or CSV formats."""
        # Silently skip divider and banner lines
        if re.match(r'^[-\s=_*#]{3,}$', raw_line):
            return None
        if "Multi-Sensor Monitor" in raw_line:
            return None
        if "Failed to read" in raw_line:
            logger.warning(f"Arduino Sensor Warning: {raw_line}")
            return None

        parsed: Dict[str, Any] = {}

        # 1. Try JSON parsing
        try:
            raw_data = json.loads(raw_line)
            if isinstance(raw_data, dict):
                # Check status message
                if "status" in raw_data and "source" not in raw_data and not any(k in raw_data for k in ("temp", "temperature", "hum", "humidity", "soil", "mq")):
                    logger.info(f"Arduino/ESP32 status message: {raw_data['status']}")
                    if "servo_angle" in raw_data:
                        angle = self._clean_num(raw_data["servo_angle"])
                        if angle is not None:
                            return {"servo_angle": int(angle)}
                    return None

                for k, v in raw_data.items():
                    k_clean = k.lower().replace(" ", "_").replace("-", "_")
                    c_val = self._clean_num(v)

                    if k_clean in ("temperature", "temp", "t", "temp_c", "dht_temp"):
                        if c_val is not None:
                            parsed["temperature"] = round(c_val, 1)
                    elif k_clean in ("humidity", "hum", "h", "rh", "dht_hum"):
                        if c_val is not None:
                            parsed["humidity"] = round(c_val, 1)
                    elif k_clean in ("soil_moisture", "soil", "moisture", "sm", "soilmoisture", "soil_percent", "soil_pct"):
                        if c_val is not None:
                            parsed["soil_moisture"] = round(c_val, 1)
                    elif k_clean in ("soil_raw", "soilraw", "raw_soil"):
                        if c_val is not None:
                            parsed["soil_raw"] = int(c_val)
                    elif k_clean in ("mq135_raw", "mq135", "air", "air_quality", "gas", "mq", "airquality", "gas_raw"):
                        if c_val is not None:
                            parsed["mq135_raw"] = int(c_val)
                    elif k_clean in ("mq135_voltage", "mq_voltage", "voltage", "mq135_v", "air_voltage"):
                        if c_val is not None:
                            parsed["mq135_voltage"] = round(c_val, 2)
                    elif k_clean in ("servo_angle", "servo_pan", "servo_degree", "servo_deg", "pan_angle", "angle", "servo"):
                        if c_val is not None:
                            parsed["servo_angle"] = int(c_val)
                    elif k_clean in ("servo_mode", "mode"):
                        if v is not None:
                            parsed["servo_mode"] = str(v).lower()
                    elif k_clean in ("hooter", "hooter_active", "relay", "relay_state", "alarm"):
                        if v is not None:
                            parsed["hooter"] = "ON" if str(v).upper() in ("ON", "1", "TRUE", "LOW") else "OFF"

                if parsed:
                    return parsed
        except json.JSONDecodeError:
            pass

        # 2. Try Regex Key-Value parsing
        # Temperature
        temp_m = re.search(r'(?:temp(?:erature)?|t(?:emp_c)?|dht_temp)\s*[:=]?\s*([-+]?[0-9]*\.?[0-9]+)', raw_line, re.I)
        if not temp_m:
            temp_m = re.search(r'([-+]?[0-9]*\.?[0-9]+)\s*(?:°c|\*c|c\b)', raw_line, re.I)
        if temp_m:
            tv = self._clean_num(temp_m.group(1))
            if tv is not None:
                parsed["temperature"] = round(tv, 1)

        # Humidity
        hum_m = re.search(r'(?:hum(?:idity)?|h(?:um)?|rh|dht_hum)\s*[:=]?\s*([-+]?[0-9]*\.?[0-9]+)', raw_line, re.I)
        if not hum_m:
            hum_m = re.search(r'([-+]?[0-9]*\.?[0-9]+)\s*%\s*(?:humidity|hum)', raw_line, re.I)
        if hum_m:
            hv = self._clean_num(hum_m.group(1))
            if hv is not None:
                parsed["humidity"] = round(hv, 1)

        # Soil Raw
        soil_raw_m = re.search(r'(?:soil\s*(?:moisture)?\s*raw|raw\s*soil)\s*[:=]?\s*([0-9]+)', raw_line, re.I)
        if soil_raw_m:
            srv = self._clean_num(soil_raw_m.group(1))
            if srv is not None:
                parsed["soil_raw"] = int(srv)

        # Soil Moisture (%)
        soil_m = re.search(r'(?:soil\s*(?:moisture)?|moisture)\s*(?:\(%\))?\s*[:=]?\s*([-+]?[0-9]*\.?[0-9]+)', raw_line, re.I)
        if soil_m:
            smv = self._clean_num(soil_m.group(1))
            if smv is not None:
                parsed["soil_moisture"] = round(smv, 1)

        # MQ Gas / Air Quality
        mq_m = re.search(r'(?:mq(?:-?135)?(?:\s*gas)?(?:\s*sensor)?|gas(?:\s*sensor)?|air(?:\s*quality)?)\s*(?:raw)?\s*[:=]?\s*([0-9]+)', raw_line, re.I)
        if mq_m:
            mqv = self._clean_num(mq_m.group(1))
            if mqv is not None:
                parsed["mq135_raw"] = int(mqv)

        # Voltage
        mq_v_m = re.search(r'(?:voltage|mq(?:135)?_?v)\s*[:=]?\s*([-+]?[0-9]*\.?[0-9]+)', raw_line, re.I)
        if mq_v_m:
            vv = self._clean_num(mq_v_m.group(1))
            if vv is not None:
                parsed["mq135_voltage"] = round(vv, 2)

        # Servo Angle / Degree
        servo_m = re.search(r'(?:servo\s*(?:angle|deg(?:ree)?|pan)?|angle|pan)\s*[:=]?\s*([0-9]+)', raw_line, re.I)
        if servo_m:
            sav = self._clean_num(servo_m.group(1))
            if sav is not None:
                parsed["servo_angle"] = int(sav)

        # Servo Mode
        mode_m = re.search(r'(?:servo\s*mode|mode)\s*[:=]?\s*(auto|manual)', raw_line, re.I)
        if mode_m:
            parsed["servo_mode"] = mode_m.group(1).lower()

        # Hooter / Relay State
        hooter_m = re.search(r'(?:hooter|relay)\s*(?:state|active)?\s*[:=]?\s*\"?(on|off|true|false|1|0)\"?', raw_line, re.I)
        if hooter_m:
            parsed["hooter"] = "ON" if hooter_m.group(1).upper() in ("ON", "1", "TRUE", "LOW") else "OFF"

        if parsed:
            return parsed

        # 3. Try CSV parsing (e.g. 29.9,40.5,53.5,2422,1874,1.51)
        csv_parts = [p.strip() for p in raw_line.split(",") if p.strip()]
        if len(csv_parts) >= 3:
            try:
                nums = [float(p) for p in csv_parts if self._clean_num(p) is not None]
                if len(nums) >= 3:
                    parsed["temperature"] = round(nums[0], 1)
                    parsed["humidity"] = round(nums[1], 1)
                    parsed["soil_moisture"] = round(nums[2], 1)
                    if len(nums) >= 4:
                        parsed["soil_raw"] = int(nums[3])
                    if len(nums) >= 5:
                        parsed["mq135_raw"] = int(nums[4])
                    if len(nums) >= 6:
                        parsed["mq135_voltage"] = round(nums[5], 2)
                    return parsed
            except Exception:
                pass

        # Log unparsed text lines if printable and not empty
        cleaned_str = ''.join(c for c in raw_line if c.isprintable()).strip()
        if cleaned_str:
            logger.info(f"Arduino/ESP32 Serial Output: {cleaned_str}")
        return None

    def _read_loop(self):
        while self.running:
            # 1. Ensure serial port is connected
            if self.ser is None or not self.ser.is_open:
                available_ports = self._find_available_ports()
                if not available_ports:
                    now = time.time()
                    if now - self.last_log_time >= 5.0:
                        self.last_log_time = now
                        logger.warning("No Arduino/ESP32 USB serial device found. Please connect Arduino via USB cable.")
                    time.sleep(2.0)
                    continue

                connected = False
                for p in available_ports:
                    if self._try_open_port(p):
                        connected = True
                        break

                if not connected:
                    now = time.time()
                    if now - self.last_log_time >= 15.0:
                        self.last_log_time = now
                        if self.last_error:
                            logger.warning(f"Could not connect to Arduino/ESP32 ({self.last_error}). Retrying...")
                    time.sleep(2.0)
                    continue

            # 2. Read from connected port
            try:
                raw_bytes = self.ser.readline()
                if not raw_bytes:
                    continue

                line = raw_bytes.decode("utf-8", errors="ignore").replace("\x00", "").strip()
                if not line:
                    continue

                parsed_data = self._parse_line(line)
                if parsed_data:
                    now = time.time()
                    with self.lock:
                        # Merge newly received non-null readings into cache
                        for k, v in parsed_data.items():
                            if v is not None:
                                self.cached_readings[k] = v
                                self.cached_times[k] = now

                        # Auto-calculate missing derived fields if raw/voltage exist
                        if self.cached_readings.get("mq135_raw") is not None and self.cached_readings.get("mq135_voltage") is None:
                            self.cached_readings["mq135_voltage"] = round((self.cached_readings["mq135_raw"] / 4095.0) * 3.3, 2)

                        # Assemble live snapshot respecting cache TTL
                        snapshot = {"source": "esp32_serial"}
                        for key in ("temperature", "humidity", "soil_moisture", "soil_raw", "mq135_raw", "mq135_voltage", "servo_angle", "servo_mode", "hooter"):
                            val = self.cached_readings.get(key)
                            t_stamp = self.cached_times.get(key, 0.0)
                            # Control states (servo, hooter) persist indefinitely; telemetry values respect TTL
                            if key in ("servo_angle", "servo_mode", "hooter") or (now - t_stamp) <= self.cache_ttl:
                                snapshot[key] = val
                            else:
                                snapshot[key] = None

                        self.latest_data = snapshot
                        self.last_received_time = now

            except Exception as e:
                if not self.running:
                    break
                self.last_error = f"{type(e).__name__}: {e}"
                port_str = str(self.connected_port) if self.connected_port else "USB"
                logger.error(f"Error reading from Arduino/ESP32 on {port_str}: {self.last_error}")
                if self.ser:
                    try:
                        self.ser.close()
                    except Exception:
                        pass
                self.ser = None
                self.connected_port = None
                time.sleep(1.0)

    def send_command(self, cmd: str) -> bool:
        """Sends a text command to ESP32 over USB Serial."""
        if not self.ser or not self.ser.is_open:
            logger.warning(f"Cannot send command '{cmd}' to ESP32: USB Serial port is not connected.")
            return False
        try:
            cmd_bytes = (cmd.strip() + "\n").encode("utf-8")
            self.ser.write(cmd_bytes)
            self.ser.flush()
            logger.info(f"-> Transmitted to ESP32 ({self.connected_port}): {cmd.strip()}")
            return True
        except Exception as e:
            logger.error(f"Failed to send command '{cmd}' to ESP32: {e}")
            return False

    def send_servo_angle(self, angle: int) -> bool:
        """Sends a SERVO:<angle> command to ESP32 to position the camera servo."""
        clamped = max(0, min(180, int(angle)))
        return self.send_command(f"SERVO:{clamped}")

    def set_hooter(self, state: bool) -> bool:
        """
        Controls the Hooter Relay connected to ESP32 GPIO 26 via USB Serial.
        state=True  -> sends 'HOOTER:ON' (ESP32 drives GPIO 26 HIGH on Elephant Detection)
        state=False -> sends 'HOOTER:OFF' (ESP32 drives GPIO 26 LOW when cleared)
        """
        cmd = "HOOTER:ON" if state else "HOOTER:OFF"
        logger.info(f"ESP32 Relay Command: {cmd} (GPIO 26 -> {'HIGH (ACTIVE)' if state else 'LOW (INACTIVE)'})")
        return self.send_command(cmd)

    def is_connected(self) -> bool:
        """Returns True if USB serial port is currently open."""
        return bool(self.ser and self.ser.is_open)

    def get_latest_readings(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent genuine reading from the ESP32 stream if available."""
        with self.lock:
            if not self.latest_data:
                if self.is_connected():
                    return {
                        "source": "esp32_serial",
                        "temperature": None,
                        "humidity": None,
                        "soil_moisture": None,
                        "soil_raw": None,
                        "mq135_raw": None,
                        "mq135_voltage": None,
                        "servo_angle": self.cached_readings.get("servo_angle", 90),
                        "servo_mode": self.cached_readings.get("servo_mode", "auto"),
                        "hooter": self.cached_readings.get("hooter", "OFF"),
                        "is_connected": True
                    }
                return None
            res = dict(self.latest_data)
            res["is_connected"] = self.is_connected() or bool(self.latest_data)
            return res
