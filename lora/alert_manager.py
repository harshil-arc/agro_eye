import json
import time
from typing import Dict, Any, Optional
from config import LORA_ALERT_COOLDOWN
from lora.lora_interface import LoRaInterface
from database import DatabaseRepository, get_latest_sensor_readings
from utils.logger import logger

class LoRaAlertManager:
    """
    Manages packet serialization and transmission over LoRa SX127x (loralibPi5).
    Transmits:
    1. Recent sensor readings from Arduino (including latest detected disease status).
    2. Real-time Emergency Disease Detection Alerts with confidence metrics.
    """
    def __init__(self, cooldown: float = LORA_ALERT_COOLDOWN):
        self.lora = LoRaInterface()
        self.repo = DatabaseRepository()
        self.cooldown = cooldown
        self.last_alert_time: float = 0.0
        self.latest_disease_name: str = "None"
        self.latest_disease_conf: float = 0.0

    def update_latest_disease(self, disease_name: str, confidence: float):
        """Updates cached disease info so telemetry packets carry current disease diagnosis."""
        self.latest_disease_name = disease_name
        self.latest_disease_conf = confidence

    def send_latest_sensor_data(self, direct_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Broadcasts sensor telemetry packet over LoRa.
        Includes live sensor metrics + recent disease status.
        """
        row = direct_data
        if not row:
            recent = get_latest_sensor_readings(limit=1)
            if recent:
                row = recent[0]

        if not row:
            logger.debug("LoRa: No sensor entries to transmit.")
            return False

        payload = {
            "type": "SENSOR",
            "temp": row.get("temperature"),
            "hum": row.get("humidity"),
            "soil": row.get("soil_moisture"),
            "soil_raw": row.get("soil_raw"),
            "mq": row.get("mq135_raw"),
            "mq_v": row.get("mq135_voltage"),
            "disease": self.latest_disease_name if self.latest_disease_name != "None" else "None",
            "conf": round(self.latest_disease_conf, 2) if self.latest_disease_conf > 0 else 0.0,
            "time": time.strftime("%H:%M:%S")
        }

        packet_str = json.dumps(payload, separators=(',', ':'))
        if not self.lora.is_ready:
            return False
        success = self.lora.transmit(packet_str)
        if not success:
            logger.error(f"LoRa Failed to send sensor packet: {self.lora.last_error or 'TransmissionError'}")
        return success

    def trigger_disease_alert(self, disease_name: str, confidence: float) -> bool:
        """
        Dispatches an immediate high-priority disease alert packet over LoRa.
        """
        self.update_latest_disease(disease_name, confidence)

        now = time.time()
        if (now - self.last_alert_time) < self.cooldown:
            logger.debug(f"LoRa disease alert throttled by cooldown ({int(now - self.last_alert_time)}s < {self.cooldown}s)")

        payload = {
            "type": "ALERT",
            "disease": disease_name,
            "conf": round(confidence, 2),
            "time": time.strftime("%H:%M:%S")
        }

        packet_str = json.dumps(payload, separators=(',', ':'))
        success = self.lora.transmit(packet_str)

        # Log alert to local SQLite database
        dt_str = time.strftime("%Y-%m-%d %H:%M:%S")
        self.repo.insert_lora_alert(
            timestamp=int(now),
            datetime_str=dt_str,
            alert_type="DISEASE_ALERT",
            message=packet_str,
            sent_status=1 if success else 0
        )

        self.last_alert_time = now
        return success

    def trigger_sensor_alert(self, alert_code: str) -> bool:
        """Dispatches an emergency sensor threshold alert over LoRa."""
        payload = {
            "type": "SENSOR_ALERT",
            "alert": alert_code,
            "time": time.strftime("%H:%M:%S")
        }
        packet_str = json.dumps(payload, separators=(',', ':'))
        return self.lora.transmit(packet_str)
