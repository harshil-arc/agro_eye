import json
import re
import threading
import time
import platform
from typing import Dict, Any, Optional, List
from config import ESP32_SERIAL_PORT, ESP32_BAUDRATE
from utils.logger import logger

class ESP32SensorReceiver:
    """
    Reads genuine sensor telemetry stream over USB Serial from ESP32.
    Features:
    - Continuous auto-discovery of USB COM ports on Windows & Linux
    - Flexible parsing: JSON (all key variants), Key-Value text (Temp: 25 Hum: 60), CSV
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
        self.latest_data: Dict[str, Any] = {}
        self.last_received_time: Optional[float] = None
        self.last_error: Optional[str] = None
        self.last_log_time: float = 0.0

    def _find_available_ports(self) -> List[str]:
        """Scans system for active USB serial ports."""
        ports_list = []
        try:
            import serial.tools.list_ports
            detected = serial.tools.list_ports.comports()
            for p in detected:
                ports_list.append(p.device)
        except Exception:
            pass

        # Prioritize preferred port if available
        if self.preferred_port and self.preferred_port in ports_list:
            ports_list.remove(self.preferred_port)
            ports_list.insert(0, self.preferred_port)
        elif self.preferred_port and not ports_list:
            ports_list.append(self.preferred_port)

        return ports_list

    def _try_open_port(self, port_name: str) -> bool:
        """Attempts to open a specific serial port with baudrate auto-detection (9600 first for Arduino)."""
        try:
            import serial
            candidate_bauds = [9600, 115200, 57600]
            if self.baudrate not in candidate_bauds:
                candidate_bauds.insert(0, self.baudrate)

            for baud in candidate_bauds:
                try:
                    self.ser = serial.Serial(port_name, baud, timeout=1.5)
                    self.connected_port = port_name
                    self.baudrate = baud
                    self.last_error = None
                    self.consecutive_empty = 0
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
        self.consecutive_empty = 0
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

    def _parse_line(self, raw_line: str) -> Optional[Dict[str, Any]]:
        """Parses a serial line using JSON or Regex Key-Value extraction."""
        # Silently skip divider and banner lines
        if re.match(r'^[-\s=_*#]{3,}$', raw_line):
            return None
        if "Multi-Sensor Monitor" in raw_line:
            return None
        if "Failed to read" in raw_line:
            logger.warning(f"Arduino Sensor Warning: {raw_line}")
            return None

        # 1. Try JSON parsing
        try:
            raw_data = json.loads(raw_line)
            if isinstance(raw_data, dict):
                if "status" in raw_data and not any(k in raw_data for k in ("temp", "temperature", "hum", "humidity")):
                    logger.info(f"Arduino/ESP32 status message: {raw_data['status']}")
                    return None

                parsed = {}
                for k, v in raw_data.items():
                    k_clean = k.lower().replace(" ", "_").replace("-", "_")
                    try:
                        val = float(v) if v is not None else None
                    except (ValueError, TypeError):
                        val = v

                    if k_clean in ("temperature", "temp", "t", "temp_c"):
                        parsed["temperature"] = val
                    elif k_clean in ("humidity", "hum", "h", "rh"):
                        parsed["humidity"] = val
                    elif k_clean in ("soil_moisture", "soil", "moisture", "sm", "soilmoisture"):
                        parsed["soil_moisture"] = val
                    elif k_clean in ("soil_raw", "soilraw", "raw_soil"):
                        parsed["soil_raw"] = val
                    elif k_clean in ("mq135_raw", "mq135", "air", "air_quality", "gas", "mq", "airquality"):
                        parsed["mq135_raw"] = val
                    elif k_clean in ("mq135_voltage", "mq_voltage", "voltage", "mq135_v", "air_voltage"):
                        parsed["mq135_voltage"] = val

                if any(parsed.get(k) is not None for k in ("temperature", "humidity", "soil_moisture", "soil_raw", "mq135_raw", "mq135_voltage")):
                    return parsed
        except json.JSONDecodeError:
            pass

        # 2. Try Regex Key-Value parsing
        parsed = {}

        # Temperature
        temp_m = re.search(r'(?:temp(?:erature)?|t)\s*(?:sensor)?\s*[:=]?\s*([0-9.]+)', raw_line, re.I)
        if not temp_m:
            temp_m = re.search(r'([0-9.]+)\s*(?:°c|\*c|c\b)', raw_line, re.I)
        if temp_m:
            parsed["temperature"] = float(temp_m.group(1))

        # Humidity
        hum_m = re.search(r'(?:hum(?:idity)?|h)\s*(?:sensor)?\s*[:=]?\s*([0-9.]+)', raw_line, re.I)
        if not hum_m:
            hum_m = re.search(r'([0-9.]+)\s*%\s*(?:humidity|hum)', raw_line, re.I)
        if hum_m:
            parsed["humidity"] = float(hum_m.group(1))

        # Soil Raw
        soil_raw_m = re.search(r'(?:soil\s*(?:moisture)?\s*raw|raw\s*soil)\s*[:=]?\s*([0-9.]+)', raw_line, re.I)
        if soil_raw_m:
            parsed["soil_raw"] = float(soil_raw_m.group(1))

        # Soil Moisture (%)
        soil_m = re.search(r'(?:soil\s*moisture|moisture)\s*(?:\(%\))?\s*[:=]?\s*([0-9.]+)', raw_line, re.I)
        if not soil_m and not soil_raw_m:
            soil_m = re.search(r'soil\s*[:=]\s*([0-9.]+)', raw_line, re.I)
        if soil_m:
            parsed["soil_moisture"] = float(soil_m.group(1))

        # MQ Gas / Air Quality
        mq_m = re.search(r'(?:mq(?:-?135)?(?:\s*gas)?(?:\s*sensor)?|gas(?:\s*sensor)?|air(?:\s*quality)?)\s*(?:raw)?\s*[:=]?\s*([0-9.]+)', raw_line, re.I)
        if mq_m:
            parsed["mq135_raw"] = float(mq_m.group(1))

        # Voltage
        mq_v_m = re.search(r'voltage\s*[:=]?\s*([0-9.]+)', raw_line, re.I)
        if mq_v_m:
            parsed["mq135_voltage"] = float(mq_v_m.group(1))

        if any(parsed.get(k) is not None for k in ("temperature", "humidity", "soil_moisture", "soil_raw", "mq135_raw", "mq135_voltage")):
            return parsed

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
                    if now - self.last_log_time >= 5.0:
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
                    with self.lock:
                        for k, v in parsed_data.items():
                            if v is not None:
                                self.latest_data[k] = v
                        self.latest_data["source"] = "esp32_serial"
                        self.last_received_time = time.time()

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


    def get_latest_readings(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent genuine reading from the ESP32 stream if available."""
        with self.lock:
            if not self.latest_data:
                return None
            return dict(self.latest_data)
