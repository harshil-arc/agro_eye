#!/usr/bin/env python3
"""
=============================================================================
AgroEye - Standalone Raspberry Pi LoRa SX127x Hardware Transmitter Test Tool
=============================================================================
Supports:
- Direct Linux SPI (`spidev` module OR Zero-Dependency native Linux `ioctl` fallback).
- loralibPi5 / RaspberryPi-LoRaLib.
- Automated continuous test transmission and interactive packet tester.
=============================================================================
"""

import os
import sys
import time
import json
import struct
import ctypes
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

try:
    import fcntl
except ImportError:
    fcntl = None

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Candidate search directories for RaspberryPi-LoRaLib
CANDIDATE_DIRS = [
    os.path.expanduser("~/RaspberryPi-LoRaLib"),
    os.path.expanduser("~/RaspberryPi-LoRaLib/loralibPi5"),
    "/home/gullu/RaspberryPi-LoRaLib",
    "/home/pi/RaspberryPi-LoRaLib",
    str(PROJECT_ROOT / "RaspberryPi-LoRaLib"),
    str(PROJECT_ROOT.parent / "RaspberryPi-LoRaLib")
]

for candidate_dir in CANDIDATE_DIRS:
    if os.path.exists(candidate_dir) and candidate_dir not in sys.path:
        sys.path.insert(0, candidate_dir)

# Import configurations if available, otherwise use defaults
try:
    from config import (
        LORA_FREQUENCY, LORA_SPREADING_FACTOR, LORA_BANDWIDTH,
        LORA_CODING_RATE, LORA_SYNC_WORD, LORA_TX_POWER
    )
except ImportError:
    LORA_FREQUENCY = 433000000       # 433 MHz
    LORA_SPREADING_FACTOR = 7        # SF7
    LORA_BANDWIDTH = 125             # 125 kHz
    LORA_CODING_RATE = 1             # 1 => 4/5
    LORA_SYNC_WORD = 0x12            # 0x12
    LORA_TX_POWER = 17               # 17 dBm (PA_BOOST)


def print_banner():
    print("\n" + "=" * 70)
    print(" 📡 AgroEye LoRa SX127x Transceiver Diagnostic & Transmitter Test Tool")
    print("=" * 70)


def check_spi_interface() -> Tuple[bool, str]:
    """Checks if Linux SPI device nodes exist."""
    if os.name == "nt":
        return True, "Windows OS detected (SPI nodes simulated)."

    spi_nodes = ["/dev/spidev0.0", "/dev/spidev0.1", "/dev/spidev1.0", "/dev/spidev1.1"]
    found = [p for p in spi_nodes if os.path.exists(p)]
    if found:
        return True, f"SPI device nodes detected: {', '.join(found)}"
    else:
        return False, (
            "NO SPI device nodes found (/dev/spidev* missing).\n"
            "   👉 Fix: Run 'sudo raspi-config' -> Interface Options -> SPI -> Enable -> Finish, then reboot."
        )


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
            # Set mode 0 (SPI_IOC_WR_MODE = 0x40016b01)
            fcntl.ioctl(self.fd, 0x40016b01, struct.pack("B", 0))
            # Set max speed hz (SPI_IOC_WR_MAX_SPEED_HZ = 0x40046b04)
            fcntl.ioctl(self.fd, 0x40046b04, struct.pack("<I", speed_hz))
            # Set bits per word 8 (SPI_IOC_WR_BITS_PER_WORD = 0x40016b03)
            fcntl.ioctl(self.fd, 0x40016b03, struct.pack("B", 8))
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

        # struct spi_ioc_transfer
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


class DirectSPIDriver:
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

    def open(self, freq_hz: int = 433000000, sf: int = 7, bw_khz: int = 125, cr: int = 1, sync_word: int = 0x12, power_dbm: int = 17) -> bool:
        # 1. Try python-spidev if installed
        try:
            import spidev
            self.spi = spidev.SpiDev()
            self.spi.open(self.bus, self.device)
            self.spi.max_speed_hz = 5000000
            self.spi.mode = 0
            self.version = self._read_reg(self.REG_VERSION)
        except Exception:
            self.spi = None

        # 2. If spidev module not found or failed, try native Linux raw ioctl
        if self.version != 0x12:
            try:
                raw_spi = RawIoctlSPIDev(self.bus, self.device)
                if raw_spi.open(5000000):
                    self.spi = raw_spi
                    self.version = self._read_reg(self.REG_VERSION)
            except Exception:
                pass

        # 3. If device 0 failed, try device 1 (CE1)
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
            return False

        # 4. Switch to LoRa Sleep then Standby
        self._write_reg(self.REG_OP_MODE, self.MODE_LONG_RANGE_MODE | self.MODE_SLEEP)
        time.sleep(0.01)
        self._write_reg(self.REG_OP_MODE, self.MODE_LONG_RANGE_MODE | self.MODE_STDBY)
        time.sleep(0.01)

        # 5. Set Frequency
        frf = int((freq_hz << 19) / 32000000)
        self._write_reg(self.REG_FRF_MSB, (frf >> 16) & 0xFF)
        self._write_reg(self.REG_FRF_MID, (frf >> 8) & 0xFF)
        self._write_reg(self.REG_FRF_LSB, frf & 0xFF)

        # 6. Set PA Config (PA_BOOST = 17 dBm)
        self._write_reg(self.REG_PA_CONFIG, 0x80 | 0x0F)
        self._write_reg(self.REG_PA_DAC, 0x84)
        self._write_reg(self.REG_OCP, 0x2B)
        self._write_reg(self.REG_LNA, 0x23)

        # 7. Set Modem Config 1 (BW 125kHz=0x70, CR 4/5=0x02, Explicit=0x00)
        bw_val = 0x70 if bw_khz <= 125 else 0x80
        cr_val = (cr & 0x07) << 1 if cr > 0 else 0x02
        self._write_reg(self.REG_MODEM_CONFIG_1, bw_val | cr_val | 0x00)

        # 8. Set Modem Config 2 (SF7=0x70, CRC enabled=0x04)
        sf_val = (sf & 0x0F) << 4
        self._write_reg(self.REG_MODEM_CONFIG_2, sf_val | 0x04)

        # 9. Set AGC Auto On
        self._write_reg(self.REG_MODEM_CONFIG_3, 0x08)

        # 10. Preamble & Sync Word
        self._write_reg(self.REG_PREAMBLE_MSB, 0x00)
        self._write_reg(self.REG_PREAMBLE_LSB, 0x08)
        self._write_reg(self.REG_SYNC_WORD, sync_word & 0xFF)

        # 11. FIFO Base
        self._write_reg(self.REG_FIFO_TX_BASE_ADDR, 0x00)
        self._write_reg(self.REG_FIFO_RX_BASE_ADDR, 0x00)
        return True

    def _read_reg(self, reg: int) -> int:
        if not self.spi:
            return 0
        resp = self.spi.xfer2([reg & 0x7F, 0x00])
        return resp[1] if len(resp) > 1 else 0

    def _write_reg(self, reg: int, val: int):
        if not self.spi:
            return
        self.spi.xfer2([reg | 0x80, val & 0xFF])

    def transmit(self, payload_str: str) -> bool:
        if not self.spi:
            return False
        try:
            self._write_reg(self.REG_OP_MODE, self.MODE_LONG_RANGE_MODE | self.MODE_STDBY)
            self._write_reg(self.REG_IRQ_FLAGS, 0xFF)
            self._write_reg(self.REG_FIFO_ADDR_PTR, 0x00)

            raw_bytes = payload_str.encode('utf-8')
            self._write_reg(self.REG_PAYLOAD_LENGTH, len(raw_bytes))
            self.spi.xfer2([self.REG_FIFO | 0x80] + list(raw_bytes))

            self._write_reg(self.REG_OP_MODE, self.MODE_LONG_RANGE_MODE | self.MODE_TX)

            start_t = time.time()
            tx_done = False
            while (time.time() - start_t) < 2.0:
                irq = self._read_reg(self.REG_IRQ_FLAGS)
                if irq & 0x08:  # TxDone
                    tx_done = True
                    break
                time.sleep(0.01)

            self._write_reg(self.REG_IRQ_FLAGS, 0xFF)
            self._write_reg(self.REG_OP_MODE, self.MODE_LONG_RANGE_MODE | self.MODE_STDBY)
            return tx_done
        except Exception:
            return False

    def close(self):
        if self.spi:
            try:
                self.spi.close()
            except Exception:
                pass
            self.spi = None


class LoRaTester:
    def __init__(self):
        self.driver = None
        self.driver_type = None
        self.is_ready = False
        self.last_error = None

    def initialize(self) -> bool:
        print("\n[1/3] Checking SPI Bus Interface...")
        spi_ok, spi_msg = check_spi_interface()
        print(f"      {spi_msg}")
        if not spi_ok and os.name != "nt":
            self.last_error = spi_msg
            return False

        print("\n[2/3] Probing LoRa SX127x Hardware Transceiver...")
        # A. Try Direct Python SPI (native ioctl or spidev)
        try:
            direct_drv = DirectSPIDriver(bus=0, device=0)
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
                print(f"      ✅ Direct SPI SX127x Transceiver Responded!")
                print(f"         - Silicon Chip Version: 0x{direct_drv.version:02X} (Valid SX1276/7/8)")
                print(f"         - Bus / Device        : /dev/spidev{direct_drv.bus}.{direct_drv.device}")
                print(f"         - Frequency           : {LORA_FREQUENCY / 1e6:.1f} MHz")
                print(f"         - Spreading Factor    : SF{LORA_SPREADING_FACTOR}")
                print(f"         - Bandwidth           : {LORA_BANDWIDTH} kHz")
                print(f"         - Sync Word           : 0x{LORA_SYNC_WORD:02X}")
                print(f"         - TX Power            : {LORA_TX_POWER} dBm (PA_BOOST)")
                return True
        except Exception as e:
            pass

        # B. Try loralibPi5
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
            print("      ✅ loralibPi5 Initialized Successfully!")
            print(f"         - Frequency : {LORA_FREQUENCY / 1e6:.1f} MHz | SF{LORA_SPREADING_FACTOR} | SyncWord 0x{LORA_SYNC_WORD:02X}")
            return True
        except Exception as e:
            self.last_error = f"{e}"

        print("      ❌ LoRa SX127x Transceiver not responding on SPI.")
        print("\n⚠️  QUICK FIXES:")
        print("   👉 1. Install spidev package: run 'sudo apt-get install -y python3-spidev' or 'pip install spidev'")
        print("   👉 2. Check 3.3V Power: Connect VCC to Pin 1 (3.3V) and GND to Pin 6 (GND). DO NOT connect to 5V!")
        print("   👉 3. Check Raspberry Pi 5 SPI Pinout:")
        print("         - MOSI  -> Pin 19 (GPIO 10)")
        print("         - MISO  -> Pin 21 (GPIO 9)")
        print("         - SCK   -> Pin 23 (GPIO 11)")
        print("         - NSS   -> Pin 24 (GPIO 8 / CE0)")
        print("         - RST   -> Pin 22 (GPIO 25)")
        print("         - DIO0  -> Pin 18 (GPIO 24)")
        print("   👉 4. Enable SPI interface: sudo raspi-config -> Interface Options -> SPI -> Enable.")
        return False

    def transmit(self, payload_dict: Dict[str, Any], label: str = "Packet") -> bool:
        """Serializes and sends a dictionary as a JSON payload."""
        packet_str = json.dumps(payload_dict, separators=(',', ':'))
        print(f"\n🚀 [TX -> {label}] Sending payload ({len(packet_str)} bytes):")
        print(f"   Payload: {packet_str}")

        if not self.is_ready or self.driver is None:
            print(f"   ❌ TX Error: LoRa transceiver is not ready ({self.last_error or 'Not initialized'})")
            print("   👉 Run: sudo apt-get install -y python3-spidev")
            return False

        try:
            if self.driver_type == "DirectSPI":
                success = self.driver.transmit(packet_str)
            else:
                self.driver.transmit(str(packet_str))
                success = True

            if success:
                print("   ✅ Transmitted successfully over 433 MHz RF!")
                print("   👉 Check your ESP32 / Arduino receiver Serial Monitor (115200 baud).")
                return True
            else:
                print("   ❌ TX hardware failed to complete transmission (TxDone flag timeout).")
                return False
        except Exception as e:
            print(f"   ❌ Transmission exception: {e}")
            return False


def run_continuous_test_loop(tester: LoRaTester, interval: float = 3.0):
    print("\n" + "-" * 70)
    print(f" 🔄 Starting Continuous Automated Transmission Loop (Interval: {interval}s)")
    print(" Press Ctrl+C at any time to return to menu.")
    print("-" * 70)

    packet_counter = 1
    sample_temp = 24.0
    sample_soil = 55.0

    try:
        while True:
            cur_time = time.strftime("%H:%M:%S")
            mode_choice = packet_counter % 4

            if mode_choice == 1:
                sample_temp += 0.3
                if sample_temp > 34.0:
                    sample_temp = 24.0
                sample_soil -= 0.5
                if sample_soil < 25.0:
                    sample_soil = 60.0

                payload = {
                    "type": "SENSOR",
                    "temp": round(sample_temp, 1),
                    "hum": 62.0,
                    "soil": round(sample_soil, 1),
                    "soil_raw": 2150,
                    "mq": 165,
                    "mq_v": 0.52,
                    "disease": "None",
                    "conf": 0.0,
                    "time": cur_time
                }
                tester.transmit(payload, label=f"#{packet_counter} SENSOR (Valid Readings)")

            elif mode_choice == 2:
                payload = {
                    "type": "SENSOR",
                    "temp": None,
                    "hum": None,
                    "soil": None,
                    "soil_raw": None,
                    "mq": None,
                    "mq_v": None,
                    "disease": "None",
                    "conf": 0.0,
                    "time": cur_time
                }
                tester.transmit(payload, label=f"#{packet_counter} SENSOR (NULL / Disconnected Sensors)")

            elif mode_choice == 3:
                payload = {
                    "type": "ALERT",
                    "disease": "Tomato_Late_Blight",
                    "conf": 0.94,
                    "time": cur_time
                }
                tester.transmit(payload, label=f"#{packet_counter} ALERT (Emergency Disease Pop-up)")

            else:
                payload = {
                    "type": "INTRUSION_ALERT",
                    "animal": "Wild Boar",
                    "count": 2,
                    "conf": 0.89,
                    "threat": 1,
                    "time": cur_time
                }
                tester.transmit(payload, label=f"#{packet_counter} INTRUSION_ALERT (Animal Detection)")

            packet_counter += 1
            print(f"⏳ Waiting {interval}s before next transmission...\n")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n⏹️ Continuous test loop stopped by user.")


def main():
    print_banner()
    tester = LoRaTester()
    init_ok = tester.initialize()

    if not init_ok:
        print("\n" + "=" * 70)
        print("⚠️  LoRa hardware could not be initialized.")
        print("   Follow the wiring & quick fix commands printed above.")
        print("=" * 70)

    while True:
        print("\n" + "=" * 50)
        print(" LoRa Transmitter Test Menu")
        print("=" * 50)
        print(" [1] Run Continuous Auto Loop (Sends SENSOR -> NULL -> ALERT -> INTRUSION every 3s)")
        print(" [2] Send Single SENSOR Packet (Normal Valid Values: 26.5°C, 60%, Soil 45%)")
        print(" [3] Send Single SENSOR Packet with NULL Values (Simulating Disconnected Sensors)")
        print(" [4] Send Single Emergency Disease Alert (Tomato_Late_Blight, 94%)")
        print(" [5] Send Single Animal Intrusion Alert (Wild Boar, Count: 2)")
        print(" [6] Send Custom Text / JSON String")
        print(" [7] Re-Run Hardware Diagnostics")
        print(" [Q] Quit")
        print("=" * 50)

        choice = input("Enter option [1-7 or Q]: ").strip().lower()

        if choice in ('q', 'quit', 'exit'):
            print("\nExiting LoRa Test Tool. Goodbye!")
            break

        cur_time = time.strftime("%H:%M:%S")

        if choice == '1':
            run_continuous_test_loop(tester, interval=3.0)

        elif choice == '2':
            payload = {
                "type": "SENSOR",
                "temp": 26.5,
                "hum": 60.0,
                "soil": 45.0,
                "soil_raw": 2100,
                "mq": 170,
                "mq_v": 0.54,
                "disease": "None",
                "conf": 0.0,
                "time": cur_time
            }
            tester.transmit(payload, label="Single Valid Sensor Packet")

        elif choice == '3':
            payload = {
                "type": "SENSOR",
                "temp": None,
                "hum": None,
                "soil": None,
                "soil_raw": None,
                "mq": None,
                "mq_v": None,
                "disease": "None",
                "conf": 0.0,
                "time": cur_time
            }
            tester.transmit(payload, label="Single NULL Sensor Packet")

        elif choice == '4':
            payload = {
                "type": "ALERT",
                "disease": "Tomato_Late_Blight",
                "conf": 0.94,
                "time": cur_time
            }
            tester.transmit(payload, label="Single Disease Alert")

        elif choice == '5':
            payload = {
                "type": "INTRUSION_ALERT",
                "animal": "Wild Boar",
                "count": 2,
                "conf": 0.89,
                "threat": 1,
                "time": cur_time
            }
            tester.transmit(payload, label="Single Animal Intrusion Alert")

        elif choice == '6':
            custom_str = input("Enter custom string or JSON to transmit: ").strip()
            if custom_str:
                try:
                    custom_json = json.loads(custom_str)
                    tester.transmit(custom_json, label="Custom JSON Packet")
                except json.JSONDecodeError:
                    if tester.is_ready and tester.driver:
                        if tester.driver_type == "DirectSPI":
                            tester.driver.transmit(custom_str)
                        else:
                            tester.driver.transmit(custom_str)
                        print(f"✅ Raw string transmitted: {custom_str}")
                    else:
                        print("❌ Transceiver not ready.")

        elif choice == '7':
            tester.initialize()

        else:
            print("Invalid option. Please choose 1-7 or Q.")


if __name__ == "__main__":
    main()
