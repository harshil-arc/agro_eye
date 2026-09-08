import random
from typing import Dict, Any, Optional
from sensors.base_sensor import BaseSensor
from utils.logger import logger

class SoilMoistureSensor(BaseSensor):
    def __init__(self, channel: int = 0, mock: bool = False):
        super().__init__(name="Soil_Moisture_Sensor", mock=mock)
        self.channel = channel
        self.adc_chan = None

        if not self.mock:
            try:
                import board
                import busio
                import adafruit_ads1x15.ads1115 as ADS
                from adafruit_ads1x15.analog_in import AnalogIn

                i2c = busio.I2C(board.SCL, board.SDA)
                ads = ADS.ADS1115(i2c)
                channels = [ADS.P0, ADS.P1, ADS.P2, ADS.P3]
                self.adc_chan = AnalogIn(ads, channels[channel])
                logger.info(f"ADS1115 Soil Moisture Sensor initialized on Channel {channel}")
            except Exception as e:
                logger.warning(f"Failed to initialize ADC for Soil Moisture ({e}). Falling back to simulation mode.")
                self.mock = True

    def read(self) -> Dict[str, Optional[float]]:
        """
        Reads soil moisture percentage (0% = dry, 100% = submerged).
        Maps analog voltage/ADC raw counts to calibrated percentage.
        """
        if self.mock or self.adc_chan is None:
            # Generate realistic synthetic data
            moisture = round(55.0 + random.uniform(-15.0, 10.0), 2)
            return {"soil_moisture": moisture}

        try:
            # Calibrate raw voltage: Air ~ 2.8V (0%), Water ~ 1.2V (100%)
            voltage = self.adc_chan.voltage
            # Linear calibration mapping
            air_voltage = 2.8
            water_voltage = 1.2
            moisture = ((air_voltage - voltage) / (air_voltage - water_voltage)) * 100.0
            moisture = max(0.0, min(100.0, round(moisture, 2)))
            return {"soil_moisture": moisture}
        except Exception as e:
            logger.error(f"Error reading Soil Moisture sensor: {e}")
            return {"soil_moisture": None}
