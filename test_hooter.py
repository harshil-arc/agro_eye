import os
import sys
import time

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from sensors.esp32_sensor import ESP32SensorReceiver

def main():
    print("=" * 60)
    print(" AgroEye ESP32 Hooter Relay Pin 26 Tester")
    print("=" * 60)
    print("Connecting to ESP32 over USB Serial...")

    receiver = ESP32SensorReceiver()
    receiver.start()

    time.sleep(2.5)

    if not receiver.is_connected():
        print("[FAIL] ESP32 is not connected. Please check USB cable and port permissions.")
        receiver.stop()
        sys.exit(1)

    print(f"[PASS] Connected to ESP32 on port: {receiver.connected_port}")
    print("\n--- Test 1: Sending HOOTER:ON (GPIO 26 -> HIGH) ---")
    receiver.set_hooter(True)
    
    print("Hooter relay should now be CLICKED / ACTIVE (Pin 26 = HIGH).")
    print("Listening for 4 seconds...")
    for i in range(4, 0, -1):
        print(f"  {i} seconds remaining...")
        time.sleep(1.0)

    print("\n--- Test 2: Sending HOOTER:OFF (GPIO 26 -> LOW) ---")
    receiver.set_hooter(False)
    print("Hooter relay should now be RELEASED / INACTIVE (Pin 26 = LOW).")
    time.sleep(2.0)

    readings = receiver.get_latest_readings()
    print("\nLatest ESP32 Status:")
    print(f"  Telemetry: {readings}")

    print("\nStopping test...")
    receiver.stop()
    print("[DONE] Hooter test finished.")

if __name__ == "__main__":
    main()
