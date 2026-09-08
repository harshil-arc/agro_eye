/*
 * =================================================================================
 * Universal Arduino / ESP32 LoRa Receiver + 0.96" I2C OLED Display + 15s Button Wake-up
 * =================================================================================
 * Compatible with BOTH:
 *   - ESP32 (NodeMCU / WROOM / DOIT ESP32 DevKit)
 *   - Arduino Uno / Nano (AVR ATmega328P)
 *
 * Features:
 *   - Continuous LoRa packet reception (never drops packets, auto re-arms RX mode).
 *   - Pushbutton Wake-up: Turns OLED ON for exactly 15 seconds.
 *   - Live countdown timer (15s -> 0s) and real-time metric updates.
 *   - True OLED Panel Power-Off on timeout.
 *   - Auto-wakes OLED on Emergency Disease Alert.
 * =================================================================================
 */

#include <SPI.h>
#include <Wire.h>
#include <LoRa.h>
#include <ArduinoJson.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// -------------------------------------------------------------
// PIN DEFINITIONS
// -------------------------------------------------------------
#if defined(ESP32)
  #define LORA_SS       5
  #define LORA_RST      14
  #define LORA_DIO0     2
  #define BUTTON_PIN    4       // Pushbutton between GPIO 4 and GND
  #define I2C_SDA       21
  #define I2C_SCL       22
#else
  // Arduino Uno / Nano (AVR)
  #define LORA_SS       10
  #define LORA_RST      9
  #define LORA_DIO0     2
  #define BUTTON_PIN    3       // Pushbutton between Pin D3 and GND
#endif

#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
#define SCREEN_ADDRESS 0x3C     // Default I2C address

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// -------------------------------------------------------------
// LORA RF PARAMETERS (Matches Raspberry Pi 5 loralibPi5)
// -------------------------------------------------------------
#define LORA_BAND       433E6    // 433 MHz
#define LORA_SF         7        // Spreading Factor 7
#define LORA_BW         125E3    // Bandwidth 125 kHz
#define LORA_CR         5        // Coding Rate 4/5
#define LORA_SYNC_WORD  0x12     // Custom Sync Word

// -------------------------------------------------------------
// DISPLAY & TIMING STATE
// -------------------------------------------------------------
const unsigned long DISPLAY_DURATION_MS = 15000; // 15 seconds timeout
unsigned long displayStartTime = 0;
unsigned long lastDisplayRefresh = 0;
bool isDisplayActive = false;

// Button Debounce & Edge Trigger
int lastButtonReading = HIGH;
int buttonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

// Cached Sensor Telemetry Data
struct SensorData {
  float temp = 0.0;
  float hum = 0.0;
  float soil = 0.0;
  int soil_raw = 0;
  int mq = 0;
  float mq_v = 0.0;
  String time_str = "--:--:--";
  int rssi = 0;
  float snr = 0.0;
  bool has_data = false;
  unsigned long last_packet_time = 0;
} latestSensors;

// Cached Disease Alert Data
struct DiseaseAlert {
  String disease_name = "";
  float confidence = 0.0;
  String time_str = "";
  bool is_active = false;
  unsigned long alert_time = 0;
} latestAlert;

// -------------------------------------------------------------
// FUNCTION DECLARATIONS
// -------------------------------------------------------------
void turnDisplayOn();
void turnDisplayOff();
void updateOLED();
void renderSensorScreen();
void renderAlertScreen();
void processIncomingLoRa();

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println(F("\n================================================"));
  Serial.println(F("  LoRa Receiver + OLED Display Node (Universal)"));
  Serial.println(F("================================================"));

  // Configure Button with Internal Pull-Up
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  // Initialize I2C OLED Display
#if defined(ESP32)
  Wire.begin(I2C_SDA, I2C_SCL);
#else
  Wire.begin();
#endif

  if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("[WARNING] SSD1306 allocation failed. Check I2C wiring."));
  } else {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(15, 20);
    display.println(F("Smart Farm Node"));
    display.setCursor(10, 35);
    display.println(F("LoRa Receiver Ready"));
    display.display();
    delay(1500);
    turnDisplayOff(); // Start with display OFF to conserve power
  }

  // Initialize LoRa SX1278
  LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);
  if (!LoRa.begin(LORA_BAND)) {
    Serial.println(F("[ERROR] LoRa SX1278 initialization failed!"));
    Serial.println(F("Check SPI wiring, 3.3V power, and pin definitions."));
    turnDisplayOn();
    display.clearDisplay();
    display.setCursor(0, 25);
    display.println(F("LoRa Init Failed!"));
    display.display();
    while (1) {
      delay(1000);
    }
  }

  // Set matching RF configurations
  LoRa.setSpreadingFactor(LORA_SF);
  LoRa.setSignalBandwidth(LORA_BW);
  LoRa.setCodingRate4(LORA_CR);
  LoRa.enableCrc();
  LoRa.setSyncWord(LORA_SYNC_WORD);

  // Put radio into explicit continuous receive mode
  LoRa.receive();

  Serial.println(F("[OK] LoRa SX1278 initialized (433MHz, SF7, BW125, CR4/5, SyncWord 0x12)."));
  Serial.println(F("Listening for continuous telemetry..."));
  Serial.println(F("Press Pushbutton to turn ON OLED display for 15 seconds.\n"));
}

void loop() {
  // 1. Process Incoming LoRa Packets (Continuous Reception)
  processIncomingLoRa();

  // 2. Check Pushbutton (Debounced Edge Trigger)
  int reading = digitalRead(BUTTON_PIN);
  if (reading != lastButtonReading) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != buttonState) {
      buttonState = reading;
      if (buttonState == LOW) {
        Serial.println(F("[BUTTON] Pressed! Waking OLED display for 15 seconds."));
        turnDisplayOn();
      }
    }
  }
  lastButtonReading = reading;

  // 3. Manage OLED Refresh & 15-Second Power-Off Timer
  if (isDisplayActive) {
    unsigned long elapsed = millis() - displayStartTime;

    if (elapsed >= DISPLAY_DURATION_MS) {
      turnDisplayOff();
      Serial.println(F("[OLED] 15s timeout reached -> Display turned OFF."));
    } else {
      if (millis() - lastDisplayRefresh >= 250) {
        lastDisplayRefresh = millis();
        updateOLED();
      }
    }
  }
}

// -------------------------------------------------------------
// PROCESS INCOMING LORA PACKETS & RE-ARM RECEIVER
// -------------------------------------------------------------
void processIncomingLoRa() {
  int packetSize = LoRa.parsePacket();
  if (packetSize) {
    String incomingPayload = "";
    while (LoRa.available()) {
      incomingPayload += (char)LoRa.read();
    }

    int rssi = LoRa.packetRssi();
    float snr = LoRa.packetSnr();

    // Re-arm LoRa receiver immediately for the next packet
    LoRa.receive();

    StaticJsonDocument<512> doc;
    DeserializationError error = deserializeJson(doc, incomingPayload);

    if (!error) {
      const char* type = doc["type"] | "UNKNOWN";

      // A. Sensor Telemetry Packet (includes live metrics + recent disease status)
      if (strcmp(type, "SENSOR") == 0) {
        latestSensors.temp = doc["temp"] | 0.0;
        latestSensors.hum = doc["hum"] | 0.0;
        latestSensors.soil = doc["soil"] | 0.0;
        latestSensors.soil_raw = doc["soil_raw"] | 0;
        latestSensors.mq = doc["mq"] | 0;
        latestSensors.mq_v = doc["mq_v"] | 0.0;
        latestSensors.time_str = String((const char*)(doc["time"] | "--:--:--"));
        latestSensors.rssi = rssi;
        latestSensors.snr = snr;
        latestSensors.has_data = true;
        latestSensors.last_packet_time = millis();

        const char* embedded_disease = doc["disease"] | "None";
        float embedded_conf = doc["conf"] | 0.0;
        if (strcmp(embedded_disease, "None") != 0 && strlen(embedded_disease) > 0) {
          latestAlert.disease_name = String(embedded_disease);
          latestAlert.confidence = embedded_conf;
          latestAlert.time_str = latestSensors.time_str;
          latestAlert.is_active = true;
          latestAlert.alert_time = millis();
        }

        Serial.print(F("[LORA] Telemetry: Temp:"));
        Serial.print(latestSensors.temp, 1);
        Serial.print(F("C | Hum:"));
        Serial.print(latestSensors.hum, 1);
        Serial.print(F("% | Soil:"));
        Serial.print(latestSensors.soil, 1);
        Serial.print(F("% | Air:"));
        Serial.print(latestSensors.mq);
        Serial.print(F(" | Disease:"));
        Serial.print(embedded_disease);
        Serial.print(F(" | RSSI:"));
        Serial.print(rssi);
        Serial.println(F(" dBm"));

        if (isDisplayActive) {
          updateOLED();
        }
      }

      // B. Emergency Disease Alert Packet
      else if (strcmp(type, "ALERT") == 0) {
        latestAlert.disease_name = String((const char*)(doc["disease"] | "Unknown"));
        latestAlert.confidence = doc["conf"] | 0.0;
        latestAlert.time_str = String((const char*)(doc["time"] | "--:--:--"));
        latestAlert.is_active = true;
        latestAlert.alert_time = millis();

        Serial.print(F("🚨 [EMERGENCY DISEASE ALERT] "));
        Serial.print(latestAlert.disease_name);
        Serial.print(F(" ("));
        Serial.print(latestAlert.confidence * 100.0, 1);
        Serial.print(F("%) at "));
        Serial.print(latestAlert.time_str);
        Serial.print(F(" | RSSI:"));
        Serial.print(rssi);
        Serial.println(F(" dBm"));

        // Auto-wake OLED display on emergency alert
        turnDisplayOn();
      }
    } else {
      Serial.print(F("[LORA RAW] "));
      Serial.print(incomingPayload);
      Serial.print(F(" (RSSI: "));
      Serial.print(rssi);
      Serial.println(F(" dBm)"));
    }
  }
}

// -------------------------------------------------------------
// TURN DISPLAY ON & RESET 15s TIMER
// -------------------------------------------------------------
void turnDisplayOn() {
  display.ssd1306_command(SSD1306_DISPLAYON);
  displayStartTime = millis();
  isDisplayActive = true;
  lastDisplayRefresh = millis();
  updateOLED();
}

// -------------------------------------------------------------
// TURN DISPLAY COMPLETELY OFF (POWER SAVING)
// -------------------------------------------------------------
void turnDisplayOff() {
  display.clearDisplay();
  display.display();
  display.ssd1306_command(SSD1306_DISPLAYOFF);
  isDisplayActive = false;
}

// -------------------------------------------------------------
// RENDER APPROPRIATE SCREEN
// -------------------------------------------------------------
void updateOLED() {
  if (latestAlert.is_active && (millis() - latestAlert.alert_time < 30000)) {
    renderAlertScreen();
  } else {
    renderSensorScreen();
  }
}

// -------------------------------------------------------------
// RENDER SENSOR TELEMETRY ON OLED (128x64)
// -------------------------------------------------------------
void renderSensorScreen() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  // Top Title Bar
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print(F("PLANT TELEMETRY"));

  // Remaining active countdown
  long remainingMs = DISPLAY_DURATION_MS - (millis() - displayStartTime);
  int remainingSec = (remainingMs > 0) ? (remainingMs / 1000 + 1) : 0;
  display.setCursor(102, 0);
  if (remainingSec < 10) display.print(F(" "));
  display.print(remainingSec);
  display.print(F("s"));

  // Header Divider
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  if (!latestSensors.has_data) {
    display.setCursor(15, 25);
    display.println(F("Waiting for LoRa"));
    display.setCursor(20, 38);
    display.println(F("telemetry..."));
  } else {
    // Row 1: Temperature & Humidity
    display.setCursor(0, 15);
    display.print(F("Temp: "));
    display.print(latestSensors.temp, 1);
    display.print(F(" C"));

    display.setCursor(78, 15);
    display.print(F("H: "));
    display.print((int)latestSensors.hum);
    display.print(F("%"));

    // Row 2: Soil Moisture
    display.setCursor(0, 28);
    display.print(F("Soil Moist: "));
    display.print(latestSensors.soil, 1);
    display.print(F("%"));

    // Row 3: MQ Gas / Air Quality
    display.setCursor(0, 41);
    display.print(F("Air Quality: "));
    display.print(latestSensors.mq);

    // Row 4: Signal & Timestamp Footer
    display.drawLine(0, 52, 127, 52, SSD1306_WHITE);
    display.setCursor(0, 55);
    display.print(F("RSSI:"));
    display.print(latestSensors.rssi);
    display.print(F("dBm"));

    display.setCursor(75, 55);
    display.print(latestSensors.time_str);
  }

  display.display();
}

// -------------------------------------------------------------
// RENDER DISEASE EMERGENCY ALERT SCREEN
// -------------------------------------------------------------
void renderAlertScreen() {
  display.clearDisplay();

  // Inverted header bar
  display.fillRect(0, 0, 128, 12, SSD1306_WHITE);
  display.setTextColor(SSD1306_BLACK, SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(8, 2);
  display.print(F("! DISEASE ALERT !"));

  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 18);
  display.println(latestAlert.disease_name);

  display.setCursor(0, 34);
  display.print(F("Conf: "));
  display.print(latestAlert.confidence * 100.0, 1);
  display.print(F("%"));

  display.setCursor(0, 48);
  display.print(F("Time: "));
  display.print(latestAlert.time_str);

  display.display();
}
