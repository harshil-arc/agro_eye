import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"

DATA_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR.mkdir(exist_ok=True)

# Load environment variables from .env file if present
ENV_PATH = BASE_DIR / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ENV_PATH)
except ImportError:
    # Fallback basic manual parser if python-dotenv is not installed yet
    if ENV_PATH.exists():
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

# ============================================================
# SYSTEM & HARDWARE OPERATION MODE
# ============================================================
ENABLE_GUI_DISPLAY = os.environ.get("ENABLE_GUI_DISPLAY", "True").lower() in ("true", "1", "yes")

# ============================================================
# CAMERA & AI INFERENCE CONFIGURATION
# ============================================================
CAMERA_INDICES = [0, 1, 2]
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
ROI_SIZE = 360

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_AI_INTERVAL = float(os.environ.get("GEMINI_INTERVAL", "0.8"))

MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "detecting-diseases/5")
API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "https://detect.roboflow.com")
CONFIDENCE_THRESHOLD = 0.25
INFERENCE_INTERVAL = 0.5  # In seconds (throttle API rate)



# ============================================================
# SENSOR POLLING & THRESHOLDS
# ============================================================
SENSOR_POLL_INTERVAL = 5.0  # seconds between sensor readings

# ESP32 / Arduino Serial Sensor Receiver (JSON / Text stream over Serial)
ESP32_SERIAL_PORT = os.environ.get("ESP32_PORT", "/dev/ttyUSB0")
ESP32_BAUDRATE = int(os.environ.get("ESP32_BAUDRATE", "9600"))

# Thresholds for Alerts
TEMP_HIGH_THRESHOLD = 38.0     # °C
TEMP_LOW_THRESHOLD = 10.0      # °C
HUMIDITY_HIGH_THRESHOLD = 90.0   # %
SOIL_DRY_THRESHOLD = 30.0      # % (Alert if moisture < 30%)
SOIL_WET_THRESHOLD = 85.0      # % (Alert if moisture > 85%)
MQ135_ALERT_THRESHOLD = 600    # Raw ADC count for poor air quality / gas leak

# Direct Hardware Pins (if used directly on Pi instead of ESP32)
DHT_PIN = int(os.environ.get("DHT_PIN", "4"))
DHT_TYPE = os.environ.get("DHT_TYPE", "DHT22")
SOIL_MOISTURE_CHANNEL = int(os.environ.get("SOIL_CHANNEL", "0"))

# ============================================================
# FIREBASE CONFIGURATION
# ============================================================
FIREBASE_CREDENTIALS_PATH = os.environ.get(
    "FIREBASE_CREDENTIALS", 
    str(BASE_DIR / "config" / "service_account.json")
)
FIREBASE_DATABASE_URL = os.environ.get(
    "FIREBASE_DB_URL", 
    "https://plant-detection-system-default-rtdb.firebaseio.com/"
)
FIREBASE_STORAGE_BUCKET = os.environ.get(
    "FIREBASE_STORAGE_BUCKET", 
    "plant-detection-system.appspot.com"
)

# ============================================================
# LORA MODULE CONFIGURATION (Raspberry Pi 5 loralibPi5)
# ============================================================
LORA_FREQUENCY = int(os.environ.get("LORA_FREQ", "433000000"))  # 433 MHz
LORA_SPREADING_FACTOR = int(os.environ.get("LORA_SF", "7"))
LORA_BANDWIDTH = int(os.environ.get("LORA_BW", "125"))          # 125 kHz
LORA_CODING_RATE = int(os.environ.get("LORA_CR", "1"))          # 1 => 4/5
LORA_SYNC_WORD = int(os.environ.get("LORA_SYNC_WORD", "0x12"), 16)
LORA_TX_POWER = int(os.environ.get("LORA_POWER", "17"))         # 17 dBm (PA_BOOST)
LORA_ALERT_COOLDOWN = 2.5  # seconds between repeated packet broadcasts

# ============================================================
# LOCAL DATABASE CONFIGURATION
# ============================================================
SQLITE_DB_PATH = str(DATA_DIR / "plant_system.db")
SYNC_INTERVAL = 10.0  # seconds to retry flushing offline buffer to Firebase
