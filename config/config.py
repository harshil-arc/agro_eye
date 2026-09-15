import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
SNAPSHOT_DIR = BASE_DIR / "snapshots"
DATABASE_DIR = BASE_DIR / "database"
LOGS_DIR = BASE_DIR / "logs"

# Ensure required runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env file if present
ENV_PATH = BASE_DIR / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ENV_PATH)
except ImportError:
    if ENV_PATH.exists():
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

# ============================================================
# 1. YOLOv8 TRAINED MODEL CONFIGURATION
# ============================================================
MODEL_PATH = os.environ.get("MODEL_PATH", str(MODELS_DIR / "disease_model.pt"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.55"))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", "0.45"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "640"))

# ============================================================
# 2. CAMERA CONFIGURATION (USB Webcam)
# ============================================================
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", "480"))
FPS = int(os.environ.get("FPS", "20"))
CAMERA_ID = os.environ.get("CAMERA_ID", "usb_camera_0")
CAMERA_RETRY_INTERVAL = float(os.environ.get("CAMERA_RETRY_INTERVAL", "2.0"))

# ============================================================
# 3. TEMPORAL CONFIRMATION LOGIC
# ============================================================
# Detections of the same disease required within the window to confirm
CONFIRMATION_FRAMES = int(os.environ.get("CONFIRMATION_FRAMES", "5"))
CONFIRMATION_WINDOW_SECONDS = float(os.environ.get("CONFIRMATION_WINDOW_SECONDS", "3.0"))

# ============================================================
# 4. SNAPSHOT & COOLDOWN SETTINGS
# ============================================================
SNAPSHOT_DIRECTORY = os.environ.get("SNAPSHOT_DIRECTORY", str(SNAPSHOT_DIR))
# Minimum seconds between snapshots of the same confirmed disease
SNAPSHOT_COOLDOWN_SECONDS = float(os.environ.get("SNAPSHOT_COOLDOWN_SECONDS", "60.0"))

# ============================================================
# 5. SQLITE DATABASE CONFIGURATION
# ============================================================
DATABASE_PATH = os.environ.get("DATABASE_PATH", str(DATABASE_DIR / "farm_system.db"))

# ============================================================
# 6. LOGGING CONFIGURATION
# ============================================================
LOG_DIRECTORY = os.environ.get("LOG_DIRECTORY", str(LOGS_DIR))
LOG_FILE = os.environ.get("LOG_FILE", str(LOGS_DIR / "agro_eye.log"))

# ============================================================
# 7. GUI & DISPLAY CONFIGURATION
# ============================================================
ENABLE_GUI_DISPLAY = os.environ.get("ENABLE_GUI_DISPLAY", "True").lower() in ("true", "1", "yes")
WINDOW_NAME = "AGRO EYE - Plant Disease Detection"

# ============================================================
# 8. FIREBASE & CLOUD CONFIGURATION (Future Cloud Sync)
# ============================================================
FIREBASE_CREDENTIALS_PATH = os.environ.get(
    "FIREBASE_CREDENTIALS", 
    str(BASE_DIR / "config" / "service_account.json")
)
FIREBASE_DATABASE_URL = os.environ.get(
    "FIREBASE_DB_URL", 
    "https://sample-629de-default-rtdb.firebaseio.com/"
)
FIREBASE_STORAGE_BUCKET = os.environ.get(
    "FIREBASE_STORAGE_BUCKET", 
    "sample-629de.appspot.com"
)

# ============================================================
# 9. LORA MODULE CONFIGURATION (Future LoRa Alert Broadcast)
# ============================================================
LORA_FREQUENCY = int(os.environ.get("LORA_FREQ", "433000000"))  # 433 MHz
LORA_SPREADING_FACTOR = int(os.environ.get("LORA_SF", "7"))
LORA_BANDWIDTH = int(os.environ.get("LORA_BW", "125"))          # 125 kHz
LORA_CODING_RATE = int(os.environ.get("LORA_CR", "1"))          # 1 => 4/5
LORA_SYNC_WORD = int(os.environ.get("LORA_SYNC_WORD", "0x12"), 16)
LORA_TX_POWER = int(os.environ.get("LORA_POWER", "17"))         # 17 dBm (PA_BOOST)
LORA_ALERT_COOLDOWN = 2.5  # seconds between repeated packet broadcasts

# ============================================================
# 10. SENSORS & SERIAL TELEMETRY (Hardware Bridge)
# ============================================================
SENSOR_POLL_INTERVAL = 5.0
ESP32_SERIAL_PORT = os.environ.get("ESP32_PORT", "/dev/ttyUSB0")
ESP32_BAUDRATE = int(os.environ.get("ESP32_BAUDRATE", "9600"))
SQLITE_DB_PATH = DATABASE_PATH
SYNC_INTERVAL = 10.0

# Sensor Alert Thresholds
TEMP_HIGH_THRESHOLD = float(os.environ.get("TEMP_HIGH_THRESHOLD", "38.0"))
TEMP_LOW_THRESHOLD = float(os.environ.get("TEMP_LOW_THRESHOLD", "10.0"))
HUMIDITY_HIGH_THRESHOLD = float(os.environ.get("HUMIDITY_HIGH_THRESHOLD", "90.0"))
SOIL_DRY_THRESHOLD = float(os.environ.get("SOIL_DRY_THRESHOLD", "30.0"))
SOIL_WET_THRESHOLD = float(os.environ.get("SOIL_WET_THRESHOLD", "85.0"))
MQ135_ALERT_THRESHOLD = int(os.environ.get("MQ135_ALERT_THRESHOLD", "600"))



