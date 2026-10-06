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
# CAMERA & OFFLINE AI INFERENCE CONFIGURATION
# ============================================================
# Prioritize external USB camera (index 1) over laptop webcam (index 0)
PREFERRED_CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "1"))
CAMERA_INDICES = [PREFERRED_CAMERA_INDEX, 0, 2] if PREFERRED_CAMERA_INDEX != 0 else [0, 1, 2]

CAMERA_WIDTH = int(os.environ.get("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "720"))
ROI_SIZE = int(os.environ.get("ROI_SIZE", "480"))

# 100% Offline YOLOv8 Plant Disease Model
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", str(BASE_DIR / "models" / "disease_model.pt"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.50"))

# Animal & Wildlife Detection Model
ANIMAL_MODEL_PATH = os.environ.get("ANIMAL_MODEL_PATH", str(BASE_DIR / "models" / "yolo11m.pt"))
ANIMAL_CONF_THRESHOLD = float(os.environ.get("ANIMAL_CONF_THRESHOLD", "0.45"))
ENABLE_ANIMAL_DETECTION = os.environ.get("ENABLE_ANIMAL_DETECTION", "True").lower() in ("true", "1", "yes")

# ============================================================
# SENSOR POLLING & THRESHOLDS
# ============================================================
SENSOR_POLL_INTERVAL = 5.0  # seconds between sensor readings

# ESP32 / Arduino Serial Sensor Receiver (JSON / Text stream over Serial)
ESP32_SERIAL_PORT = os.environ.get("ESP32_PORT", "/dev/ttyUSB0")
ESP32_BAUDRATE = int(os.environ.get("ESP32_BAUDRATE", "115200"))

# Thresholds for Alerts
TEMP_HIGH_THRESHOLD = 38.0     # °C
TEMP_LOW_THRESHOLD = 10.0      # °C
HUMIDITY_HIGH_THRESHOLD = 90.0   # %
SOIL_DRY_THRESHOLD = 30.0      # % (Alert if moisture < 30%)
SOIL_WET_THRESHOLD = 85.0      # % (Alert if moisture > 85%)
MQ135_ALERT_THRESHOLD = 3300    # Raw 12-bit ADC count (~2.7V) for air quality alert

# Direct Hardware Pins (if used directly on Pi instead of ESP32)
DHT_PIN = int(os.environ.get("DHT_PIN", "4"))
DHT_TYPE = os.environ.get("DHT_TYPE", "DHT22")
SOIL_MOISTURE_CHANNEL = int(os.environ.get("SOIL_CHANNEL", "0"))

# ============================================================
# CAMERA SERVO CONFIGURATION (Controlled natively via ESP32)
# ============================================================
# Pan Servo is wired to ESP32 (GPIO 18). Raspberry Pi GPIO is not used.
ENABLE_SERVO = os.environ.get("ENABLE_SERVO", "True").lower() in ("true", "1", "yes")

# Auto-Sweep Rotation Boundaries (degrees)
SERVO_AUTO_MIN_ANGLE = int(os.environ.get("SERVO_AUTO_MIN_ANGLE", "30"))
SERVO_AUTO_MAX_ANGLE = int(os.environ.get("SERVO_AUTO_MAX_ANGLE", "150"))
SERVO_AUTO_STEP_DEG = int(os.environ.get("SERVO_AUTO_STEP_DEG", "3"))
SERVO_AUTO_INTERVAL = float(os.environ.get("SERVO_AUTO_INTERVAL", "0.12"))  # step period in seconds
SERVO_AUTO_PAUSE_SEC = float(os.environ.get("SERVO_AUTO_PAUSE_SEC", "1.5")) # pause at limits for vision scan
SERVO_CONTROL_PATH = "camera_control"

# ============================================================
# FIREBASE CONFIGURATION
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
MAX_FIREBASE_SNAPSHOTS = int(os.environ.get("MAX_FIREBASE_SNAPSHOTS", "200"))

# ============================================================
# LORA MODULE CONFIGURATION (Raspberry Pi 5 loralibPi5)
# ============================================================
ENABLE_LORA = os.environ.get("ENABLE_LORA", "True").lower() in ("true", "1", "yes")
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

# ============================================================
# OUTBOUND SRT LIVE VIDEO STREAMING CONFIGURATION (Pi -> Cloud VPS / Media Server)
# ============================================================
ENABLE_STREAMING = os.environ.get("ENABLE_STREAMING", "True").lower() in ("true", "1", "yes")

# Outgoing SRT Media Server Destination (e.g. srt://vps.example.com:6000 or srt://1.2.3.4:8890)
SRT_SERVER_URL = os.environ.get("SRT_SERVER_URL", "srt://127.0.0.1:6000")
SRT_STREAM_ID = os.environ.get("SRT_STREAM_ID", "publish/live/agroeye")
SRT_LATENCY_MS = int(os.environ.get("SRT_LATENCY_MS", "120"))  # Ultra-low latency buffer (ms)
SRT_RECONNECT_DELAY = float(os.environ.get("SRT_RECONNECT_DELAY", "3.0"))  # Seconds to retry on disconnect

# Stream Resolution, FPS and Encoding Bitrate
STREAM_FPS = int(os.environ.get("STREAM_FPS", "30"))
STREAM_WIDTH = int(os.environ.get("STREAM_WIDTH", "1280"))
STREAM_HEIGHT = int(os.environ.get("STREAM_HEIGHT", "720"))
STREAM_BITRATE = os.environ.get("STREAM_BITRATE", "2000k")

# Cloud Media Server / HLS Playback URL for Mobile App
STREAM_DEVICE_ID = os.environ.get("STREAM_DEVICE_ID", os.environ.get("WEBRTC_DEVICE_ID", "pi_agroeye_01"))
HLS_STREAM_URL = os.environ.get("HLS_STREAM_URL", os.environ.get("LIVE_STREAM_URL", "https://your-cloud-video-server/live/agroeye_01/index.m3u8"))
LIVE_STREAM_NODE_PATH = f"live_stream/{STREAM_DEVICE_ID}"

WEBRTC_DEVICE_ID = STREAM_DEVICE_ID
WEBRTC_SESSION_PATH = f"webrtc_sessions/{WEBRTC_DEVICE_ID}"

STUN_SERVERS = [
    "stun:stun.l.google.com:19302",
    "stun:stun1.l.google.com:19302",
    "stun:stun.cloudflare.com:3478"
]

TURN_SERVERS_CONFIG = [
    {
        "urls": ["turn:openrelay.metered.ca:80"],
        "username": "openrelayproject",
        "credential": "openrelayproject"
    },
    {
        "urls": ["turn:openrelay.metered.ca:443"],
        "username": "openrelayproject",
        "credential": "openrelayproject"
    },
    {
        "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
        "username": "openrelayproject",
        "credential": "openrelayproject"
    }
]



