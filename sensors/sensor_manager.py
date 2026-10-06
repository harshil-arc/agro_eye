import time
from typing import Dict, Any, List, Optional
from config import (
    TEMP_HIGH_THRESHOLD, TEMP_LOW_THRESHOLD, HUMIDITY_HIGH_THRESHOLD,
    SOIL_DRY_THRESHOLD, SOIL_WET_THRESHOLD, MQ135_ALERT_THRESHOLD
)
from sensors.esp32_sensor import ESP32SensorReceiver
from utils.logger import logger

class SensorManager:
    def __init__(self):
        self.esp32_receiver = ESP32SensorReceiver()
        self.esp32_receiver.start()

    def stop(self):
        """Stops background sensor threads."""
        self.esp32_receiver.stop()

    def send_servo_angle(self, angle: int) -> bool:
        """Sends servo position command to ESP32."""
        return self.esp32_receiver.send_servo_angle(angle)

    def set_hooter(self, state: bool) -> bool:
        """Sends Hooter relay state (ON/OFF) to ESP32 over USB Serial."""
        return self.esp32_receiver.set_hooter(state)

    def read_all(self) -> Dict[str, Any]:
        """
        Polls ESP32 sensor stream and returns readings.
        If ESP32 or sensors are not connected, returns a valid telemetry dictionary
        with NULL (None) values so the system NEVER stops or hangs.
        """
        esp32_data = self.esp32_receiver.get_latest_readings()
        payload: Dict[str, Any] = {
            "timestamp": int(time.time()),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "esp32_sensor",
            "temperature": None,
            "humidity": None,
            "soil_moisture": None,
            "soil_raw": None,
            "mq135_raw": None,
            "mq135_voltage": None,
            "servo_angle": None,
            "servo_mode": "auto",
            "hooter": "OFF",
            "is_connected": False
        }

        if esp32_data:
            payload["is_connected"] = True
            payload.update(esp32_data)

        return payload

    def check_alerts(self, readings: Dict[str, Any]) -> List[str]:
        """Evaluates sensor thresholds on real readings and returns alert messages."""
        if not readings:
            return []

        alerts: List[str] = []

        # Temperature (Ignore None or <= 0.0 boot / disconnected placeholder)
        temp = readings.get("temperature")
        if temp is not None and isinstance(temp, (int, float)) and temp > 0.0:
            if temp >= TEMP_HIGH_THRESHOLD:
                alerts.append(f"HIGH_TEMP:{temp}C")
            elif temp <= TEMP_LOW_THRESHOLD:
                alerts.append(f"LOW_TEMP:{temp}C")

        # Humidity (Ignore None or <= 0.0 placeholder)
        hum = readings.get("humidity")
        if hum is not None and isinstance(hum, (int, float)) and hum > 0.0 and hum >= HUMIDITY_HIGH_THRESHOLD:
            alerts.append(f"HIGH_HUMIDITY:{hum}%")

        # Soil Moisture
        moisture = readings.get("soil_moisture")
        if moisture is not None and isinstance(moisture, (int, float)):
            if moisture <= SOIL_DRY_THRESHOLD:
                alerts.append(f"SOIL_DRY:{moisture}%")
            elif moisture >= SOIL_WET_THRESHOLD:
                alerts.append(f"SOIL_WATERLOGGED:{moisture}%")

        # Air Quality / Gas (MQ135)
        mq135_raw = readings.get("mq135_raw")
        if mq135_raw is not None and isinstance(mq135_raw, (int, float)) and mq135_raw >= MQ135_ALERT_THRESHOLD:
            alerts.append(f"POOR_AIR_QUALITY_MQ135:{mq135_raw}")

        return alerts
