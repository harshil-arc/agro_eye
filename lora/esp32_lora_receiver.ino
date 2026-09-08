/*
 * =================================================================================
 * ESP32 LoRa Receiver + 0.96" I2C OLED Display + 15s Pushbutton Wake-up
 * =================================================================================
 * Features:
 *   - Continuous LoRa reception: Never drops packets (auto re-arms RX mode).
 *   - Pushbutton Wake-up: Pressing the button turns the OLED ON for exactly 15 seconds.
 *   - Live OLED Updates: Live countdown timer (15s -> 0s) and real-time metric updates.
 *   - True Display Power-Off: Completely turns OFF the OLED display after 15 seconds.
 *   - Emergency Alert Override: Disease alerts automatically wake the display.
 *
 * Hardware Wiring:
 *
 *   1. LoRa SX1278 (SPI):
 *      ESP32 Pin     SX1278 Pin
 *      ------------------------
 *      GPIO 5        NSS / CS / SS
 *      GPIO 14       RST / RESET
 *      GPIO 2        DIO0 / IRQ
 *      GPIO 18       SCK
 *      GPIO 19       MISO
 *      GPIO 23       MOSI
 *      3.3V          VCC (⚠️ 3.3V Only! Do NOT connect to 5V)
 *      GND           GND
 *
 *   2. 0.96" I2C OLED (SSD1306 128x64):
 *      ESP32 Pin     OLED Pin
 *      ------------------------
 *      GPIO 21       SDA
 *      GPIO 22       SCL
 *      3.3V or 5V    VCC
 *      GND           GND
 *
 *   3. Push Button (Wake-up):
 *      ESP32 Pin     Button Pin
 *      ------------------------
 *      GPIO 4        Button Leg 1
 *      GND           Button Leg 2 (Uses Internal Pull-Up, No Resistor Needed)
 *
 * Required Arduino IDE Libraries:
 *   - "LoRa" by Sandeep Mistry
 *   - "ArduinoJson" (v6 or v7) by Benoit Blanchon
 *   - "Adafruit SSD1306" by Adafruit
 *   - "Adafruit GFX Library" by Adafruit
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
#define LORA_SS       5
#define LORA_RST      14
#define LORA_DIO0     2

#define BUTTON_PIN    4       // Pushbutton connected between GPIO 4 and GND

#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
#define SCREEN_ADDRESS 0x3C  // Default I2C address (some screens use 0x3D)

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// -------------------------------------------------------------
// LORA RF PARAMETERS (Exact match with Raspberry Pi 5 loralibPi5)
// -------------------------------------------------------------
#define LORA_BAND       433E6    // 433 MHz
#define LORA_SF         7        // Spreading Factor 7
#define LORA_BW         125E3    // Bandwidth 125 kHz
#define LORA_CR         5        // Coding Rate 4/5
#define LORA_SYNC_WORD  0x12     // Custom Sync Word

// -------------------------------------------------------------
// DISPLAY & TIMING STATE
// -------------------------------------------------------------
const unsigned long DISPLAY_DURATION_MS = 15000; // 15 seconds display timeout
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

  Serial.println("\n================================================");
  Serial.println("  ESP32 LoRa Receiver + OLED Display Node");
  Serial.println("================================================");

  // Configure Button with Internal Pull-Up
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  // Initialize I2C OLED Display
  Wire.begin(21, 22); // SDA=GPIO 21, SCL=GPIO 22
  if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("[WARNING] SSD1306 allocation failed. Check I2C wiring."));
  } else {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(15, 20);
    display.println("Smart Farm Node");
    display.setCursor(10, 35);
    display.println("LoRa Receiver Ready");
    display.display();
    delay(1500);
    turnDisplayOff(); // Start with display OFF to conserve power
  }

  // Initialize LoRa SX1278
  LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);
  if (!LoRa.begin(LORA_BAND)) {
    Serial.println("[ERROR] LoRa SX1278 initialization failed!");
    Serial.println("Please check SPI wiring, 3.3V power, and pin definitions.");
    turnDisplayOn();
    display.clearDisplay();
    display.setCursor(0, 25);
    display.println("LoRa Init Failed!");
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

  Serial.println("[OK] LoRa SX1278 initialized successfully (433MHz, SF7, BW125, CR4/5).");
  Serial.println("Listening for continuous telemetry...");
  Serial.println("Press Pushbutton to turn ON the OLED display for 15 seconds.\n");
}

void loop() {
  // -----------------------------------------------------------
  // 1. PROCESS INCOMING LORA PACKETS (Continuous Reception)
  // -----------------------------------------------------------
  processIncomingLoRa();

  // -----------------------------------------------------------
  // 2. CHECK PUSHBUTTON (Debounced Edge Trigger)
  // -----------------------------------------------------------
  int reading = digitalRead(BUTTON_PIN);
  if (reading != lastButtonReading) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != buttonState) {
      buttonState = reading;
      // Button was just pressed (transition from HIGH to LOW)
      if (buttonState == LOW) {
        Serial.println("[BUTTON] Pressed! Waking OLED display for 15 seconds.");
        turnDisplayOn();
      }
    }
  }
  lastButtonReading = reading;

  // -----------------------------------------------------------
  // 3. MANAGE OLED REFRESH & 15-SECOND POWER-OFF TIMER
  // -----------------------------------------------------------
  if (isDisplayActive) {
    unsigned long elapsed = millis() - displayStartTime;

    if (elapsed >= DISPLAY_DURATION_MS) {
      // 15 seconds reached -> Turn OFF OLED completely
      turnDisplayOff();
      Serial.println("[OLED] 15s timeout reached -> Display turned OFF.");
    } else {
      // Refresh display every 250ms to keep countdown timer & metrics live
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

        Serial.printf("[LORA] Telemetry: Temp:%.1fC | Hum:%.1f%% | Soil:%.1f%% | Air:%d | Disease:%s | RSSI:%d dBm\n",
                      latestSensors.temp, latestSensors.hum, latestSensors.soil, latestSensors.mq, embedded_disease, rssi);

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

        Serial.printf("🚨 [EMERGENCY DISEASE ALERT] %s (%.1f%%) at %s | RSSI:%d dBm\n",
                      latestAlert.disease_name.c_str(), latestAlert.confidence * 100.0,
                      latestAlert.time_str.c_str(), rssi);

        // Auto-wake OLED display on emergency alert
        turnDisplayOn();
      }
    } else {
      Serial.printf("[LORA RAW] %s (RSSI: %d dBm)\n", incomingPayload.c_str(), rssi);
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
  display.print("PLANT TELEMETRY");

  // Remaining active countdown (e.g. "15s", "14s" ...)
  long remainingMs = DISPLAY_DURATION_MS - (millis() - displayStartTime);
  int remainingSec = (remainingMs > 0) ? (remainingMs / 1000 + 1) : 0;
  display.setCursor(102, 0);
  display.printf("%2ds", remainingSec);

  // Horizontal Header Divider
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  if (!latestSensors.has_data) {
    display.setCursor(15, 25);
    display.println("Waiting for LoRa");
    display.setCursor(20, 38);
    display.println("telemetry...");
  } else {
    // Row 1: Temperature & Humidity
    display.setCursor(0, 15);
    display.printf("Temp: %.1f C", latestSensors.temp);
    display.setCursor(78, 15);
    display.printf("H: %.0f%%", latestSensors.hum);

    // Row 2: Soil Moisture
    display.setCursor(0, 28);
    display.printf("Soil Moist: %.1f%%", latestSensors.soil);

    // Row 3: MQ Gas / Air Quality
    display.setCursor(0, 41);
    display.printf("Air Quality: %d", latestSensors.mq);

    // Row 4: Signal & Timestamp Footer
    display.drawLine(0, 52, 127, 52, SSD1306_WHITE);
    display.setCursor(0, 55);
    display.printf("RSSI:%ddBm", latestSensors.rssi);
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
  display.print("! DISEASE ALERT !");

  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 18);
  display.println(latestAlert.disease_name);

  display.setCursor(0, 34);
  display.printf("Conf: %.1f%%", latestAlert.confidence * 100.0);

  display.setCursor(0, 48);
  display.printf("Time: %s", latestAlert.time_str.c_str());

  display.display();
}


