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

    def read_all(self) -> Optional[Dict[str, Any]]:
        """
        Polls ESP32 sensor stream and returns genuine readings.
        Returns None if no data has been received yet to avoid false readings.
        """
        esp32_data = self.esp32_receiver.get_latest_readings()
        if not esp32_data:
            return None

        payload: Dict[str, Any] = {
            "timestamp": int(time.time()),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "esp32_sensor"
        }
        payload.update(esp32_data)
        return payload

    def check_alerts(self, readings: Dict[str, Any]) -> List[str]:
        """Evaluates sensor thresholds on real readings and returns alert messages."""
        if not readings:
            return []

        alerts: List[str] = []

        # Temperature
        temp = readings.get("temperature")
        if temp is not None:
            if temp >= TEMP_HIGH_THRESHOLD:
                alerts.append(f"HIGH_TEMP:{temp}C")
            elif temp <= TEMP_LOW_THRESHOLD:
                alerts.append(f"LOW_TEMP:{temp}C")

        # Humidity
        hum = readings.get("humidity")
        if hum is not None and hum >= HUMIDITY_HIGH_THRESHOLD:
            alerts.append(f"HIGH_HUMIDITY:{hum}%")

        # Soil Moisture
        moisture = readings.get("soil_moisture")
        if moisture is not None:
            if moisture <= SOIL_DRY_THRESHOLD:
                alerts.append(f"SOIL_DRY:{moisture}%")
            elif moisture >= SOIL_WET_THRESHOLD:
                alerts.append(f"SOIL_WATERLOGGED:{moisture}%")

        # Air Quality / Gas (MQ135)
        mq135_raw = readings.get("mq135_raw")
        if mq135_raw is not None and mq135_raw >= MQ135_ALERT_THRESHOLD:
            alerts.append(f"POOR_AIR_QUALITY_MQ135:{mq135_raw}")

        return alerts
