import random
from typing import Dict, Any, Optional
from sensors.base_sensor import BaseSensor
from utils.logger import logger

class TempHumiditySensor(BaseSensor):
    def __init__(self, pin: int = 4, sensor_type: str = "DHT22", mock: bool = False):
        super().__init__(name="DHT_Sensor", mock=mock)
        self.pin = pin
        self.sensor_type = sensor_type
        self.device = None

        if not self.mock:
            try:
                import board
                import adafruit_dht
                pin_obj = getattr(board, f"D{pin}", None)
                if pin_obj is None:
                    raise ValueError(f"Invalid GPIO pin D{pin}")
                
                if sensor_type.upper() == "DHT11":
                    self.device = adafruit_dht.DHT11(pin_obj)
                else:
                    self.device = adafruit_dht.DHT22(pin_obj)
                logger.info(f"{sensor_type} initialized on GPIO pin {pin}")
            except Exception as e:
                logger.warning(f"Failed to initialize physical {sensor_type} ({e}). Falling back to simulation mode.")
                self.mock = True

    def read(self) -> Dict[str, Optional[float]]:
        """Reads temperature (°C) and relative humidity (%)."""
        if self.mock or self.device is None:
            # Generate realistic synthetic data
            temp = round(24.0 + random.uniform(-3.0, 5.0), 2)
            hum = round(60.0 + random.uniform(-10.0, 15.0), 2)
            return {"temperature": temp, "humidity": hum}

        try:
            temperature = round(float(self.device.temperature), 2)
            humidity = round(float(self.device.humidity), 2)
            return {"temperature": temperature, "humidity": humidity}
        except RuntimeError as error:
            # DHT sensors often report transient checksum/read errors
            logger.debug(f"DHT read transient error: {error.args[0]}")
            return {"temperature": None, "humidity": None}
        except Exception as error:
            logger.error(f"Error reading DHT sensor: {error}")
            return {"temperature": None, "humidity": None}
