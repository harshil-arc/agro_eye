import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from config import (
    ENABLE_LORA,
    LORA_FREQUENCY, LORA_SPREADING_FACTOR, LORA_BANDWIDTH,
    LORA_CODING_RATE, LORA_SYNC_WORD, LORA_TX_POWER
)
from utils.logger import logger

# Automatically discover and add RaspberryPi-LoRaLib to sys.path if cloned in home or parent directories
CANDIDATE_DIRS = [
    os.path.expanduser("~/RaspberryPi-LoRaLib"),
    os.path.expanduser("~/RaspberryPi-LoRaLib/loralibPi5"),
    "/home/gullu/RaspberryPi-LoRaLib",
    "/home/pi/RaspberryPi-LoRaLib",
    str(Path(__file__).resolve().parent.parent / "RaspberryPi-LoRaLib"),
    str(Path(__file__).resolve().parent.parent.parent / "RaspberryPi-LoRaLib")
]

for candidate_dir in CANDIDATE_DIRS:
    if os.path.exists(candidate_dir) and candidate_dir not in sys.path:
        sys.path.insert(0, candidate_dir)


def _probe_lora_hardware() -> Tuple[bool, str]:
    """
    Safely tests LoRa transceiver responsiveness in an isolated subprocess.
    This prevents any C-level exit(1) in loralib.initialize() from abruptly
    killing the main application when wiring is loose or SPI is disabled.
    """
    paths_init = f"import sys\nfor p in {repr(CANDIDATE_DIRS)}:\n    if p not in sys.path:\n        sys.path.insert(0, p)\n"
    probe_code = (
        f"{paths_init}"
        "try:\n"
        "    import loralibPi5 as loralib\n"
        "    loralib.initialize()\n"
        "    sys.exit(0)\n"
        "except Exception as e:\n"
        "    print(f'Exception: {e}')\n"
        "    sys.exit(2)\n"
    )
    try:
        res = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=3.0
        )
        output = (res.stdout + " " + res.stderr).strip()
        if res.returncode == 0:
            return True, "OK"
        else:
            return False, output or f"Transceiver probe exited with code {res.returncode}"
    except subprocess.TimeoutExpired:
        return False, "Hardware probe timed out (SPI bus unresponsive)"
    except Exception as e:
        return False, str(e)


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
        """Initializes loralibPi5 on Raspberry Pi 5 SPI interface with safety checks."""
        if not ENABLE_LORA:
            logger.info("LoRa module is disabled in configuration (ENABLE_LORA=False).")
            self.is_ready = False
            return

        # Check if running on Linux and verify SPI bus availability
        if os.name != "nt":
            spi_nodes = ["/dev/spidev0.0", "/dev/spidev0.1", "/dev/spidev1.0", "/dev/spidev1.1"]
            if not any(os.path.exists(p) for p in spi_nodes):
                self.last_error = "SPI interface is disabled (/dev/spidev* not found)"
                logger.warning(
                    f"LoRa Warning: {self.last_error}. "
                    "To enable SPI on Raspberry Pi: run 'sudo raspi-config' -> Interface Options -> SPI -> Yes. "
                    "System will continue operating normally in offline/cloud mode without LoRa."
                )
                self.is_ready = False
                return

        # Run isolated hardware probe first to protect main process from C exit(1)
        logger.info("Probing Raspberry Pi 5 LoRa SX127x transceiver (loralibPi5)...")
        is_ok, probe_msg = _probe_lora_hardware()
        if not is_ok:
            self.last_error = probe_msg
            logger.warning(
                f"LoRa Hardware Probe: '{probe_msg}'. "
                "The SX127x transceiver did not respond on SPI. "
                "Please check 3.3V/GND power, SPI wiring (MOSI, MISO, SCK, CS, RST), or SPI enabled in raspi-config. "
                "System will continue operating normally without LoRa."
            )
            self.is_ready = False
            return

        # Hardware confirmed responding, now safely initialize in main process
        try:
            import loralibPi5 as loralib
            self.driver = loralib

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
            logger.warning(f"LoRa Initialization Warning: '{self.last_error}'. Continuing without LoRa.")
            self.is_ready = False

    def transmit(self, message: str) -> bool:
        """
        Transmits a raw string/JSON packet over LoRa.
        """
        if not self.is_ready or self.driver is None:
            logger.debug(f"LoRa TX Skipped (Transceiver not ready: {self.last_error or 'Hardware uninitialized'})")
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
