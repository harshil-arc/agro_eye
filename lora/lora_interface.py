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
        # Catch non-zero exit codes OR diagnostic messages emitted by C library
        bad_keywords = ("unrecognized", "transceiver", "failed", "error", "exception", "cannot open", "not found")
        if res.returncode != 0 or any(bad in output.lower() for bad in bad_keywords):
            return False, output or f"Transceiver probe exited with code {res.returncode}"
        return True, "OK"
    except subprocess.TimeoutExpired:
        return False, "Hardware probe timed out (SPI bus unresponsive)"
    except Exception as e:
        return False, str(e)


import struct
import ctypes
try:
    import fcntl
except ImportError:
    fcntl = None


class RawIoctlSPIDev:
    """Zero-dependency Linux /dev/spidev SPI driver using native ioctl."""
    def __init__(self, bus: int = 0, device: int = 0):
        self.bus = bus
        self.device = device
        self.fd = -1

    def open(self, speed_hz: int = 5000000) -> bool:
        if os.name == "nt" or not fcntl:
            return False
        dev_path = f"/dev/spidev{self.bus}.{self.device}"
        if not os.path.exists(dev_path):
            return False
        try:
            self.fd = os.open(dev_path, os.O_RDWR)
            fcntl.ioctl(self.fd, 0x40016b01, struct.pack("B", 0))          # Mode 0
            fcntl.ioctl(self.fd, 0x40046b04, struct.pack("<I", speed_hz)) # Speed Hz
            fcntl.ioctl(self.fd, 0x40016b03, struct.pack("B", 8))          # 8 bits
            return True
        except Exception:
            if self.fd >= 0:
                try:
                    os.close(self.fd)
                except Exception:
                    pass
                self.fd = -1
            return False

    def xfer2(self, data):
        if self.fd < 0:
            return [0] * len(data)
        length = len(data)
        tx_buf = (ctypes.c_uint8 * length)(*data)
        rx_buf = (ctypes.c_uint8 * length)()

        transfer = struct.pack(
            "=QQIIHBBBBH",
            ctypes.addressof(tx_buf),
            ctypes.addressof(rx_buf),
            length,
            5000000,
            0,
            8,
            0,
            0,
            0,
            0
        )
        try:
            fcntl.ioctl(self.fd, 0x40206b00, transfer)
            return list(rx_buf)
        except Exception:
            return [0] * length

    def close(self):
        if self.fd >= 0:
            try:
                os.close(self.fd)
            except Exception:
                pass
            self.fd = -1


# -------------------------------------------------------------
# DIRECT PYTHON SPIDEV SX127X DRIVER
# -------------------------------------------------------------
class DirectSPILoRaDriver:
    """
    Direct SPI hardware driver for Semtech SX1276/77/78/79 on Linux / Raspberry Pi.
    Supports python-spidev module and native Linux ioctl fallback.
    No external C library compilation required.
    """
    REG_FIFO = 0x00
    REG_OP_MODE = 0x01
    REG_FRF_MSB = 0x06
    REG_FRF_MID = 0x07
    REG_FRF_LSB = 0x08
    REG_PA_CONFIG = 0x09
    REG_PA_RAMP = 0x0A
    REG_OCP = 0x0B
    REG_LNA = 0x0C
    REG_FIFO_ADDR_PTR = 0x0D
    REG_FIFO_TX_BASE_ADDR = 0x0E
    REG_FIFO_RX_BASE_ADDR = 0x0F
    REG_IRQ_FLAGS = 0x12
    REG_MODEM_CONFIG_1 = 0x1D
    REG_MODEM_CONFIG_2 = 0x1E
    REG_PREAMBLE_MSB = 0x20
    REG_PREAMBLE_LSB = 0x21
    REG_PAYLOAD_LENGTH = 0x22
    REG_MODEM_CONFIG_3 = 0x26
    REG_SYNC_WORD = 0x39
    REG_VERSION = 0x42
    REG_PA_DAC = 0x4D

    MODE_LONG_RANGE_MODE = 0x80
    MODE_SLEEP = 0x00
    MODE_STDBY = 0x01
    MODE_TX = 0x03

    def __init__(self, bus: int = 0, device: int = 0):
        self.bus = bus
        self.device = device
        self.spi = None
        self.version = 0
        self.freq_hz = 433000000
        self.sf = 7
        self.bw_khz = 125
        self.cr = 1
        self.op_mode_base = self.MODE_LONG_RANGE_MODE | 0x08  # Default Low Frequency Mode (433MHz)

    def open(self, freq_hz: int = 433000000, sf: int = 7, bw_khz: int = 125, cr: int = 1, sync_word: int = 0x12, power_dbm: int = 17) -> bool:
        self.freq_hz = freq_hz
        self.sf = sf
        self.bw_khz = bw_khz
        self.cr = cr
        # Bit 3 (0x08) of REG_OP_MODE is LowFrequencyModeOn for 410-525 MHz band
        is_low_freq = (freq_hz < 525000000)
        self.op_mode_base = self.MODE_LONG_RANGE_MODE | (0x08 if is_low_freq else 0x00)

        try:
            # 1. Try python-spidev if installed
            try:
                import spidev
                self.spi = spidev.SpiDev()
                self.spi.open(self.bus, self.device)
                self.spi.max_speed_hz = 5000000  # 5 MHz
                self.spi.mode = 0
                self.version = self._read_reg(self.REG_VERSION)
            except Exception:
                self.spi = None

            # 2. Native Linux raw ioctl fallback (zero dependencies)
            if self.version != 0x12:
                try:
                    raw_spi = RawIoctlSPIDev(self.bus, self.device)
                    if raw_spi.open(5000000):
                        self.spi = raw_spi
                        self.version = self._read_reg(self.REG_VERSION)
                except Exception:
                    pass

            # 3. Try device 1 (CE1) if CE0 failed
            if self.version != 0x12 and self.device == 0:
                try:
                    if self.spi:
                        self.spi.close()
                    raw_spi = RawIoctlSPIDev(self.bus, 1)
                    if raw_spi.open(5000000):
                        self.spi = raw_spi
                        self.device = 1
                        self.version = self._read_reg(self.REG_VERSION)
                except Exception:
                    pass

            if self.version != 0x12:
                logger.warning(f"SX127x SPI read version=0x{self.version:02X} (Expected 0x12). Check wiring.")
                return False

            # 4. Put into Sleep mode to switch to LoRa mode
            self._write_reg(self.REG_OP_MODE, self.MODE_LONG_RANGE_MODE | self.MODE_SLEEP)
            time.sleep(0.02)
            self._write_reg(self.REG_OP_MODE, self.op_mode_base | self.MODE_SLEEP)
            time.sleep(0.02)

            # 5. Put into Standby mode
            self._write_reg(self.REG_OP_MODE, self.op_mode_base | self.MODE_STDBY)
            time.sleep(0.02)

            # 6. Unmask IRQs
            self._write_reg(0x11, 0x00)  # REG_IRQ_FLAGS_MASK

            # 7. Set Frequency
            frf = int((freq_hz << 19) / 32000000)
            self._write_reg(self.REG_FRF_MSB, (frf >> 16) & 0xFF)
            self._write_reg(self.REG_FRF_MID, (frf >> 8) & 0xFF)
            self._write_reg(self.REG_FRF_LSB, frf & 0xFF)

            # 8. Set Power (PA_BOOST)
            if power_dbm > 17:
                self._write_reg(self.REG_PA_DAC, 0x87)
                self._write_reg(self.REG_PA_CONFIG, 0x80 | 0x70 | min(15, max(0, power_dbm - 5)))
            else:
                self._write_reg(self.REG_PA_DAC, 0x84)
                self._write_reg(self.REG_PA_CONFIG, 0x80 | 0x70 | min(15, max(0, power_dbm - 2)))
            self._write_reg(self.REG_OCP, 0x2B)  # 100mA current limit
            self._write_reg(self.REG_LNA, 0x23)  # Maximum LNA gain + boost

            # 9. Set Bandwidth (125kHz=0x70), Coding Rate (4/5=0x02), Explicit Header (0x00)
            bw_val = 0x70 if bw_khz <= 125 else (0x80 if bw_khz <= 250 else 0x90)
            cr_val = 0x02  # CR 4/5
            if cr == 2 or cr == 6:
                cr_val = 0x04  # CR 4/6
            elif cr == 3 or cr == 7:
                cr_val = 0x06  # CR 4/7
            elif cr == 4 or cr == 8:
                cr_val = 0x08  # CR 4/8
            self._write_reg(self.REG_MODEM_CONFIG_1, bw_val | cr_val | 0x00)

            # 10. Set Spreading Factor & CRC enabled (0x04)
            sf_val = (sf & 0x0F) << 4
            self._write_reg(self.REG_MODEM_CONFIG_2, sf_val | 0x04)

            # 11. Set AGC Auto On (0x04) & Low Data Rate Optimize (0x08 when symbol duration > 16ms)
            sym_duration = (1 << sf) / (bw_khz * 1000.0)
            ldro = 0x08 if sym_duration > 0.016 else 0x00
            self._write_reg(self.REG_MODEM_CONFIG_3, ldro | 0x04)

            # 12. Set Preamble (8 symbols)
            self._write_reg(self.REG_PREAMBLE_MSB, 0x00)
            self._write_reg(self.REG_PREAMBLE_LSB, 0x08)

            # 13. Set Sync Word
            self._write_reg(self.REG_SYNC_WORD, sync_word & 0xFF)

            # 14. Set FIFO base
            self._write_reg(self.REG_FIFO_TX_BASE_ADDR, 0x00)
            self._write_reg(self.REG_FIFO_RX_BASE_ADDR, 0x00)
            self._write_reg(self.REG_FIFO_ADDR_PTR, 0x00)

            return True
        except Exception as e:
            logger.debug(f"Direct SPI Driver Open Error: {e}")
            return False

    def _read_reg(self, reg: int) -> int:
        resp = self.spi.xfer2([reg & 0x7F, 0x00])
        return resp[1] if len(resp) > 1 else 0

    def _write_reg(self, reg: int, val: int):
        self.spi.xfer2([reg | 0x80, val & 0xFF])

    def transmit(self, payload_str: str) -> bool:
        if not self.spi:
            return False
        try:
            # Standby mode
            self._write_reg(self.REG_OP_MODE, self.op_mode_base | self.MODE_STDBY)
            time.sleep(0.005)

            # Clear IRQ flags
            self._write_reg(self.REG_IRQ_FLAGS, 0xFF)

            # Reset FIFO ptr to TX base
            self._write_reg(self.REG_FIFO_TX_BASE_ADDR, 0x00)
            self._write_reg(self.REG_FIFO_ADDR_PTR, 0x00)

            # Write payload bytes
            raw_bytes = payload_str.encode('utf-8')
            self._write_reg(self.REG_PAYLOAD_LENGTH, len(raw_bytes))
            self.spi.xfer2([self.REG_FIFO | 0x80] + list(raw_bytes))

            # Put in TX mode
            self._write_reg(self.REG_OP_MODE, self.op_mode_base | self.MODE_TX)

            # Dynamic timeout calculated from estimated Time-on-Air (ToA)
            sym_duration = (1 << self.sf) / (self.bw_khz * 1000.0)
            est_toa = (8 + 4.25 + len(raw_bytes) * 2.5) * sym_duration
            timeout = max(3.5, est_toa * 3.0 + 1.5)

            # Wait for TxDone IRQ flag (bit 3) or auto-transition to Standby
            start_t = time.time()
            tx_done = False
            while (time.time() - start_t) < timeout:
                irq = self._read_reg(self.REG_IRQ_FLAGS)
                if irq & 0x08:  # TxDone
                    tx_done = True
                    break
                # Check if SX127x automatically completed TX and returned to STDBY (mode 0x01)
                cur_mode = self._read_reg(self.REG_OP_MODE) & 0x07
                if cur_mode == self.MODE_STDBY:
                    tx_done = True
                    break
                time.sleep(0.01)

            # Clear IRQ & return to standby
            self._write_reg(self.REG_IRQ_FLAGS, 0xFF)
            self._write_reg(self.REG_OP_MODE, self.op_mode_base | self.MODE_STDBY)
            return tx_done
        except Exception as e:
            logger.error(f"Direct SPI TX Error: {e}")
            return False

    def close(self):
        if self.spi:
            try:
                self.spi.close()
            except Exception:
                pass
            self.spi = None


class LoRaInterface:
    """
    Raspberry Pi 5/4/3 LoRa SX127x transceiver driver.
    Supports:
    1. Direct Linux Python SPI (spidev) with register-level control.
    2. loralibPi5 / RaspberryPi-LoRaLib.
    Configured for 433MHz, SF7, BW125kHz, CR 4/5, SyncWord 0x12, 17 dBm PA_BOOST.
    """
    def __init__(self):
        self.driver = None
        self.driver_type = None
        self.is_ready = False
        self.last_error = None
        self._initialize_driver()

    def _initialize_driver(self):
        """Initializes LoRa driver on Raspberry Pi with safety checks and multi-backend fallback."""
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

        # 1. First Attempt: Direct Python SPI Driver (No C-lib dependencies)
        try:
            direct_drv = DirectSPILoRaDriver(bus=0, device=0)
            if direct_drv.open(
                freq_hz=LORA_FREQUENCY,
                sf=LORA_SPREADING_FACTOR,
                bw_khz=LORA_BANDWIDTH,
                cr=LORA_CODING_RATE,
                sync_word=LORA_SYNC_WORD,
                power_dbm=LORA_TX_POWER
            ):
                self.driver = direct_drv
                self.driver_type = "DirectSPI"
                self.is_ready = True
                logger.info(f"LoRa SX127x initialized via Direct SPI (Silicon Version: 0x{direct_drv.version:02X}, Freq: {LORA_FREQUENCY}Hz, SF: {LORA_SPREADING_FACTOR}, SyncWord: 0x{LORA_SYNC_WORD:02X})")
                return
        except Exception as e:
            logger.debug(f"Direct SPI initialization attempt: {e}")

        # 2. Second Attempt: loralibPi5 wrapper
        logger.info("Probing Raspberry Pi 5 LoRa SX127x transceiver (loralibPi5)...")
        is_ok, probe_msg = _probe_lora_hardware()
        if is_ok:
            try:
                import loralibPi5 as loralib
                self.driver = loralib
                self.driver.initialize()
                self.driver.configPower(LORA_TX_POWER)
                self.driver.setFrequency(LORA_FREQUENCY)
                self.driver.setBandwidth(LORA_BANDWIDTH)
                self.driver.setCodingRate(LORA_CODING_RATE)
                self.driver.setHeader(False)
                self.driver.setSF(LORA_SPREADING_FACTOR)
                self.driver.setContMode(False)
                self.driver.setCRC(True)
                self.driver.optimizeLowRate(LORA_SPREADING_FACTOR, LORA_BANDWIDTH)
                self.driver.setSyncWord(LORA_SYNC_WORD)
                self.driver.configPower(LORA_TX_POWER)

                self.driver_type = "loralibPi5"
                self.is_ready = True
                logger.info(f"LoRa SX127x initialized via loralibPi5 (Freq: {LORA_FREQUENCY}Hz, SF: {LORA_SPREADING_FACTOR}, SyncWord: 0x{LORA_SYNC_WORD:02X})")
                return
            except Exception as e:
                self.last_error = f"loralibPi5 Init Error: {e}"
        else:
            self.last_error = probe_msg

        logger.warning(
            f"LoRa Hardware Notice: '{self.last_error}'. "
            "Please check 3.3V/GND power, SPI wiring (MOSI, MISO, SCK, CS, RST), or SPI enabled in raspi-config. "
            "System will continue operating normally without LoRa."
        )
        self.is_ready = False

    def transmit(self, message: str) -> bool:
        """Transmits a raw string/JSON packet over LoRa."""
        if not self.is_ready or self.driver is None:
            logger.debug(f"LoRa TX Skipped (Transceiver not ready: {self.last_error or 'Hardware uninitialized'})")
            return False

        try:
            if self.driver_type == "DirectSPI":
                success = self.driver.transmit(str(message))
            else:
                self.driver.transmit(str(message))
                success = True

            if success:
                logger.info(f"LoRa TX [Transmitted Successfully ✅] Payload: {message}")
            else:
                logger.error(f"LoRa TX [Transmission Latch Failed ❌] Payload: {message}")
            return success
        except Exception as e:
            err_name = f"{type(e).__name__}: {e}"
            self.last_error = err_name
            logger.error(f"LoRa TX [Transmission Failed ❌] Error: '{err_name}' | Payload: {message}")
            return False

    def close(self):
        """Releases LoRa resources."""
        self.is_ready = False
        if self.driver_type == "DirectSPI" and self.driver:
            self.driver.close()

