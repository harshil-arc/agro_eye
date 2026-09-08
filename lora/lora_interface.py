import os
import sys
import time
import json
from pathlib import Path
from typing import Optional
from config import (
    LORA_FREQUENCY, LORA_SPREADING_FACTOR, LORA_BANDWIDTH,
    LORA_CODING_RATE, LORA_SYNC_WORD, LORA_TX_POWER
)
from utils.logger import logger

# Automatically discover and add RaspberryPi-LoRaLib to sys.path if cloned in home or parent directories
for candidate_dir in [
    os.path.expanduser("~/RaspberryPi-LoRaLib"),
    os.path.expanduser("~/RaspberryPi-LoRaLib/loralibPi5"),
    "/home/gullu/RaspberryPi-LoRaLib",
    "/home/pi/RaspberryPi-LoRaLib",
    str(Path(__file__).resolve().parent.parent / "RaspberryPi-LoRaLib"),
    str(Path(__file__).resolve().parent.parent.parent / "RaspberryPi-LoRaLib")
]:
    if os.path.exists(candidate_dir) and candidate_dir not in sys.path:
        sys.path.insert(0, candidate_dir)


class LoRaInterface:
    """
    Raspberry Pi 5 LoRa SX127x transceiver driver using loralibPi5.
    Configured for 433MHz, SF7, BW125kHz, CR 4/5, SyncWord 0x12, 17 dBm PA_BOOST.
    """
    def __init__(self):
        self.driver = None
        self.is_ready = False
        self.last_error = None
        self._initialize_driver()

    def _initialize_driver(self):
        """Initializes loralibPi5 on Raspberry Pi 5 SPI interface."""
        try:
            import loralibPi5 as loralib
            self.driver = loralib

            logger.info("Initializing Raspberry Pi 5 LoRa (loralibPi5)...")
            self.driver.initialize()
            self.driver.configPower(LORA_TX_POWER)
            self.driver.setFrequency(LORA_FREQUENCY)
            self.driver.setBandwidth(LORA_BANDWIDTH)
            self.driver.setCodingRate(LORA_CODING_RATE)
            self.driver.setHeader(False)        # False = explicit header (matches ESP32 default)
            self.driver.setSF(LORA_SPREADING_FACTOR)
            self.driver.setContMode(False)
            self.driver.setCRC(True)
            self.driver.optimizeLowRate(LORA_SPREADING_FACTOR, LORA_BANDWIDTH)
            self.driver.setSyncWord(LORA_SYNC_WORD)
            self.driver.configPower(LORA_TX_POWER)  # Enables PA_BOOST + sets TX power

            self.is_ready = True
            logger.info(f"LoRa SX127x initialized successfully (Freq: {LORA_FREQUENCY}Hz, SF: {LORA_SPREADING_FACTOR}, BW: {LORA_BANDWIDTH}kHz, SyncWord: 0x{LORA_SYNC_WORD:02X}, Power: {LORA_TX_POWER}dBm)")
        except ImportError as e:
            self.last_error = f"ImportError: {e}"
            logger.error(f"LoRa Hardware Error: '{self.last_error}'. Make sure 'loralibPi5' is installed on Raspberry Pi 5.")
            self.is_ready = False
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            logger.error(f"LoRa Initialization Failure: '{self.last_error}'")
            self.is_ready = False

    def transmit(self, message: str) -> bool:
        """
        Transmits a raw string/JSON packet over LoRa.
        """
        if not self.is_ready or self.driver is None:
            logger.warning(f"LoRa TX Skipped (Transceiver not ready: {self.last_error or 'Hardware uninitialized'})")
            return False

        try:
            self.driver.transmit(str(message))
            logger.info(f"LoRa TX [Transmitted Successfully ✅] Payload: {message}")
            return True
        except Exception as e:
            err_name = f"{type(e).__name__}: {e}"
            self.last_error = err_name
            logger.error(f"LoRa TX [Transmission Failed ❌] Error: '{err_name}' | Payload: {message}")
            return False


    def close(self):
        """Releases LoRa resources."""
        self.is_ready = False
