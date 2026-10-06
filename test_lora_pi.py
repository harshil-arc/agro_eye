#!/usr/bin/env python3
"""
=============================================================================
AgroEye - Standalone Raspberry Pi 5 LoRa SX127x Hardware Transmitter Test Tool
=============================================================================
Use this script to verify that:
1. SPI bus is enabled and active on the Raspberry Pi.
2. LoRa SX127x transceiver hardware is wired correctly and responsive.
3. Transmitted packets (Sensor telemetry with/without NULL values, Disease Alerts,
   and Animal Intrusion Alerts) are received successfully on the ESP32 / Arduino OLED node.
=============================================================================
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

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


def probe_loralib_hardware() -> Tuple[bool, str]:
    """Safely runs a probe in an isolated subprocess to test loralibPi5 without C exit(1)."""
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
            timeout=4.0
        )
        output = (res.stdout + " " + res.stderr).strip()
        bad_keywords = ("unrecognized", "transceiver", "failed", "error", "exception", "cannot open", "not found")
        if res.returncode != 0 or any(bad in output.lower() for bad in bad_keywords):
            return False, output or f"Transceiver probe exited with code {res.returncode}"
        return True, "Hardware transceiver responded successfully."
    except subprocess.TimeoutExpired:
        return False, "Hardware probe timed out (SPI bus unresponsive)."
    except Exception as e:
        return False, str(e)


class LoRaTester:
    def __init__(self):
        self.driver = None
        self.is_ready = False
        self.last_error = None

    def initialize(self) -> bool:
        print("\n[1/3] Checking SPI hardware bus...")
        spi_ok, spi_msg = check_spi_interface()
        print(f"      {spi_msg}")
        if not spi_ok and os.name != "nt":
            self.last_error = spi_msg
            return False

        print("\n[2/3] Probing LoRa SX127x Transceiver...")
        hw_ok, hw_msg = probe_loralib_hardware()
        print(f"      Result: {hw_msg}")
        if not hw_ok:
            self.last_error = hw_msg
            print("\n⚠️  LoRa Transceiver Probe Failed. Please verify:")
            print("   1. Raspberry Pi 5 SPI Wiring:")
            print("      - VCC   -> Pin 1 (3.3V) [DO NOT connect to 5V!]")
            print("      - GND   -> Pin 6 (GND)")
            print("      - MOSI  -> Pin 19 (GPIO 10 / SPI0_MOSI)")
            print("      - MISO  -> Pin 21 (GPIO 9  / SPI0_MISO)")
            print("      - SCK   -> Pin 23 (GPIO 11 / SPI0_SCLK)")
            print("      - NSS   -> Pin 24 (GPIO 8  / SPI0_CE0)")
            print("      - RST   -> Pin 22 (GPIO 25)")
            print("      - DIO0  -> Pin 18 (GPIO 24)")
            print("   2. Install RaspberryPi-LoRaLib for Pi 5:")
            print("      git clone https://github.com/mchlab/RaspberryPi-LoRaLib.git ~/RaspberryPi-LoRaLib")
            print("      cd ~/RaspberryPi-LoRaLib/loralibPi5 && make")
            return False

        print("\n[3/3] Initializing RF Parameters...")
        try:
            import loralibPi5 as loralib
            self.driver = loralib

            self.driver.initialize()
            self.driver.configPower(LORA_TX_POWER)
            self.driver.setFrequency(LORA_FREQUENCY)
            self.driver.setBandwidth(LORA_BANDWIDTH)
            self.driver.setCodingRate(LORA_CODING_RATE)
            self.driver.setHeader(False)        # False = explicit header (matches ESP32 / Arduino default)
            self.driver.setSF(LORA_SPREADING_FACTOR)
            self.driver.setContMode(False)
            self.driver.setCRC(True)
            self.driver.optimizeLowRate(LORA_SPREADING_FACTOR, LORA_BANDWIDTH)
            self.driver.setSyncWord(LORA_SYNC_WORD)
            self.driver.configPower(LORA_TX_POWER)

            self.is_ready = True
            print("      ✅ LoRa SX127x Initialized Successfully!")
            print(f"         - Frequency      : {LORA_FREQUENCY / 1e6:.1f} MHz")
            print(f"         - Spreading Factor: SF{LORA_SPREADING_FACTOR}")
            print(f"         - Bandwidth       : {LORA_BANDWIDTH} kHz")
            print(f"         - Coding Rate     : 4/{LORA_CODING_RATE + 4}")
            print(f"         - Sync Word       : 0x{LORA_SYNC_WORD:02X}")
            print(f"         - TX Power        : {LORA_TX_POWER} dBm (PA_BOOST)")
            return True
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            print(f"      ❌ Failed to initialize LoRa: {self.last_error}")
            return False

    def transmit(self, payload_dict: Dict[str, Any], label: str = "Packet") -> bool:
        """Serializes and sends a dictionary as a JSON payload."""
        packet_str = json.dumps(payload_dict, separators=(',', ':'))
        print(f"\n🚀 [TX -> {label}] Sending payload ({len(packet_str)} bytes):")
        print(f"   Payload: {packet_str}")

        if not self.is_ready or self.driver is None:
            print(f"   ❌ TX Error: LoRa transceiver is not ready ({self.last_error or 'Not initialized'})")
            return False

        try:
            self.driver.transmit(str(packet_str))
            print("   ✅ Transmitted successfully over 433 MHz RF!")
            print("   👉 Check your ESP32 / Arduino OLED receiver screen for this packet.")
            return True
        except Exception as e:
            print(f"   ❌ Transmission failed: {e}")
            return False


def run_continuous_test_loop(tester: LoRaTester, interval: float = 3.0):
    """Continuously transmits alternating packets every few seconds."""
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
                # Normal Sensor Telemetry Packet
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
                # Sensor Telemetry Packet with NULL / Disconnected Sensors
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
                # Emergency Disease Alert Packet
                payload = {
                    "type": "ALERT",
                    "disease": "Tomato_Late_Blight",
                    "conf": 0.94,
                    "time": cur_time
                }
                tester.transmit(payload, label=f"#{packet_counter} ALERT (Emergency Disease Pop-up)")

            else:
                # Animal Intrusion Alert Packet
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
        print("⚠️  LoRa hardware could not be initialized in normal mode.")
        print("   If you are running on Windows, loralibPi5 is only available on Raspberry Pi 5.")
        print("   On Raspberry Pi 5, follow the wiring & library steps printed above.")
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
