/*
 * ==============================================================================
 * AgroEye ESP32 Multi-Sensor + Servo Controller (USB Serial to Raspberry Pi)
 * ==============================================================================
 * Hardware: ESP32 Dev Module (WROOM / NodeMCU-32S) + DHT11/22 + Soil + MQ135 + Pan Servo
 *
 * Pin Connections:
 *   ESP32 Pin      Hardware / Sensor Pin
 *   -------------------------------------------------
 *   GPIO 4         DHT11 / DHT22 Data Pin (Requires Pullup to 3.3V)
 *   GPIO 34 (ADC1) Soil Moisture Sensor Analog Out (AOUT)
 *   GPIO 35 (ADC1) MQ-135 Gas / Air Quality Analog Out (AOUT)
 *   GPIO 18        Camera Pan Servo Signal Pin (PWM)
 *   GPIO 26        Hooter Relay Signal Pin (HIGH = Active/ON, LOW = Inactive/OFF)
 *   VIN (5V)       Servo VCC & Relay VCC (Red Wire) - MUST BE 5V (Not 3.3V!)
 *   3.3V           DHT / Soil Sensor VCC
 *   GND            Common Ground (Black/Brown Wire)
 * ==============================================================================
 */

#include <DHT.h>
#include <ESP32Servo.h>

// ---------- Feature Enable Configuration ----------
#define ENABLE_DHT        true    // Set to true for DHT11 / DHT22
#define ENABLE_SOIL       true    // Set to true for Soil Moisture Sensor
#define ENABLE_MQ135      true    // Set to true for MQ-135 Air Quality Sensor
#define ENABLE_SERVO      true    // Set to true for Camera Pan Servo
#define ENABLE_HOOTER     true    // Set to true for Hooter Relay (Active-HIGH on GPIO 26)

// ---------- Pin Definitions ----------
#define DHTPIN            4       // GPIO 4 for DHT Data
#define DHTTYPE           DHT11   // Set to DHT11 (Blue) or DHT22 (White)

#define SOIL_PIN          34      // GPIO 34 (ADC1_CH6) for Soil Moisture
#define MQ135_PIN         35      // GPIO 35 (ADC1_CH7) for MQ-135 Air Quality
#define SERVO_PIN         18      // GPIO 18 for Pan Servo PWM Control
#define HOOTER_PIN        26      // GPIO 26 for Relay (HIGH on Elephant Detection)

// ---------- Calibration Parameters (ESP32 12-bit ADC: 0 - 4095) ----------
// Soil Moisture Sensor:
// In dry air: ~3500 - 4095 (0% moisture)
// In water: ~1200 - 1500 (100% moisture)
#define SOIL_DRY_RAW           3500
#define SOIL_WET_RAW           1200

#if ENABLE_DHT
DHT dht(DHTPIN, DHTTYPE);
float lastGoodTemp = 28.0;
float lastGoodHum = 50.0;
bool hasReadDHTOnce = false;
#endif

#if ENABLE_SERVO
Servo panServo;
bool isManualMode = false;   // In Manual mode, auto-sweep halts immediately
int currentServoAngle = 90;
int servoDirection = 1;      // 1 = sweeping up, -1 = sweeping down
unsigned long lastServoStep = 0;
const int servoMinAngle = 30;
const int servoMaxAngle = 150;
const int servoStepDeg = 2;
const unsigned long servoStepInterval = 50; // ms per step for smooth sweep
#endif

#if ENABLE_HOOTER
bool isHooterActive = false;
unsigned long lastHooterCmdTime = 0;
const unsigned long hooterSafetyTimeout = 15000; // 15s failsafe auto-off if Pi disconnects
#endif

unsigned long lastReadTime = 0;
const unsigned long readInterval = 2000; // 2 seconds between telemetry packets

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50); // Fast non-blocking timeout for commands
  delay(500);

  // Configure ADC resolution to 12 bits (0-4095) and full 3.3V attenuation
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

#if ENABLE_HOOTER
  // Active-HIGH Relay: Initialize LOW first so the relay starts in the OFF/inactive state
  pinMode(HOOTER_PIN, OUTPUT);
  digitalWrite(HOOTER_PIN, LOW);
#endif

#if ENABLE_DHT
  pinMode(DHTPIN, INPUT_PULLUP);
  dht.begin();
#endif

#if ENABLE_SERVO
  // Allocate hardware PWM timers for ESP32Servo
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  panServo.setPeriodHertz(50);             // Standard 50Hz servo
  panServo.attach(SERVO_PIN, 500, 2500);   // Standard 500us to 2500us SG90/MG995 pulses
  panServo.write(currentServoAngle);
#endif

  Serial.println("{\"status\":\"ESP32_READY\",\"baud\":115200,\"servo_angle\":90,\"hooter\":\"OFF\",\"relay_pin\":26}");
  Serial.flush();
}

void loop() {
  // 1. Process incoming commands from Raspberry Pi
  while (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() == 0) continue;

    if (cmd.equalsIgnoreCase("MODE:MANUAL") || cmd.equalsIgnoreCase("FREEZE")) {
#if ENABLE_SERVO
      isManualMode = true;
      panServo.write(currentServoAngle); // Instantly freeze at exact current angle
#endif
    } 
    else if (cmd.equalsIgnoreCase("MODE:AUTO") || cmd.equalsIgnoreCase("RESUME")) {
#if ENABLE_SERVO
      isManualMode = false;
      lastServoStep = millis();
#endif
    } 
    else if (cmd.startsWith("SERVO:") || cmd.startsWith("ANGLE:")) {
      int splitIndex = cmd.indexOf(':');
      int targetAngle = cmd.substring(splitIndex + 1).toInt();
      targetAngle = constrain(targetAngle, 0, 180);
#if ENABLE_SERVO
      isManualMode = true;
      currentServoAngle = targetAngle;
      panServo.write(currentServoAngle);
#endif
    }
    else if (cmd.equalsIgnoreCase("HOOTER:ON") || cmd.equalsIgnoreCase("RELAY:ON") || cmd.equalsIgnoreCase("HOOTER_ON") || cmd.equalsIgnoreCase("HOOTER:1")) {
#if ENABLE_HOOTER
      isHooterActive = true;
      lastHooterCmdTime = millis();
      digitalWrite(HOOTER_PIN, HIGH); // Drive pin 26 HIGH on Elephant Detection
      Serial.println("{\"event\":\"HOOTER_TRIGGER\",\"hooter\":\"ON\",\"relay_pin\":26,\"state\":\"HIGH\"}");
      Serial.flush();
#endif
    }
    else if (cmd.equalsIgnoreCase("HOOTER:OFF") || cmd.equalsIgnoreCase("RELAY:OFF") || cmd.equalsIgnoreCase("HOOTER_OFF") || cmd.equalsIgnoreCase("HOOTER:0")) {
#if ENABLE_HOOTER
      isHooterActive = false;
      digitalWrite(HOOTER_PIN, LOW); // Drive pin 26 LOW when Elephant is cleared
      Serial.println("{\"event\":\"HOOTER_CLEARED\",\"hooter\":\"OFF\",\"relay_pin\":26,\"state\":\"LOW\"}");
      Serial.flush();
#endif
    }
  }

#if ENABLE_HOOTER
  // Safety Failsafe: Automatically release relay to LOW if no command received within timeout
  if (isHooterActive && (millis() - lastHooterCmdTime >= hooterSafetyTimeout)) {
    isHooterActive = false;
    digitalWrite(HOOTER_PIN, LOW);
  }
#endif

#if ENABLE_SERVO
  // 2. Smooth auto-sweep oscillation ONLY when in Auto mode
  if (!isManualMode && (millis() - lastServoStep >= servoStepInterval)) {
    lastServoStep = millis();
    currentServoAngle += (servoDirection * servoStepDeg);
    if (currentServoAngle >= servoMaxAngle) {
      currentServoAngle = servoMaxAngle;
      servoDirection = -1;
    } else if (currentServoAngle <= servoMinAngle) {
      currentServoAngle = servoMinAngle;
      servoDirection = 1;
    }
    panServo.write(currentServoAngle);
  }
#endif

  // 3. Periodic sensor telemetry output (every 2 seconds)
  if (millis() - lastReadTime >= readInterval) {
    lastReadTime = millis();
    readAndTransmitTelemetry();
  }
}

void readAndTransmitTelemetry() {
  float temperature = 0.0;
  float humidity = 0.0;
  bool dhtValid = false;

  // 1. Read DHT Sensor
#if ENABLE_DHT
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  if (!isnan(h) && !isnan(t) && h > 0.0 && t > -40.0 && t < 85.0) {
    temperature = t;
    humidity = h;
    lastGoodTemp = t;
    lastGoodHum = h;
    hasReadDHTOnce = true;
    dhtValid = true;
  } else if (hasReadDHTOnce) {
    // Preserve last known valid reading across brief 1-wire timing hiccups
    temperature = lastGoodTemp;
    humidity = lastGoodHum;
    dhtValid = true;
  }
#endif

  // 2. Read Soil Moisture (Analog GPIO 34)
  int soilRaw = 0;
  float soilPercent = 0.0;
#if ENABLE_SOIL
  soilRaw = analogRead(SOIL_PIN);
  if (soilRaw > SOIL_WET_RAW) {
    soilPercent = ((float)(SOIL_DRY_RAW - soilRaw) / (float)(SOIL_DRY_RAW - SOIL_WET_RAW)) * 100.0;
  } else {
    soilPercent = 100.0;
  }
  soilPercent = constrain(soilPercent, 0.0, 100.0);
#endif

  // 3. Read MQ-135 Air Quality (Analog GPIO 35)
  int mqRaw = 0;
  float mqVoltage = 0.0;
#if ENABLE_MQ135
  mqRaw = analogRead(MQ135_PIN);
  mqVoltage = (mqRaw / 4095.0) * 3.3; // ESP32 ADC reference is 3.3V
#endif

  // 4. Output Clean JSON Line to USB Serial
  Serial.print("{\"source\":\"esp32_sensor\"");

  if (dhtValid) {
    Serial.print(",\"temperature\":");
    Serial.print(temperature, 1);
    Serial.print(",\"humidity\":");
    Serial.print(humidity, 1);
  } else {
    Serial.print(",\"temperature\":null,\"humidity\":null");
  }

#if ENABLE_SOIL
  Serial.print(",\"soil_moisture\":");
  Serial.print(soilPercent, 1);
  Serial.print(",\"soil_raw\":");
  Serial.print(soilRaw);
#else
  Serial.print(",\"soil_moisture\":null,\"soil_raw\":null");
#endif

#if ENABLE_MQ135
  Serial.print(",\"mq135_raw\":");
  Serial.print(mqRaw);
  Serial.print(",\"mq135_voltage\":");
  Serial.print(mqVoltage, 2);
#else
  Serial.print(",\"mq135_raw\":null,\"mq135_voltage\":null");
#endif

#if ENABLE_SERVO
  Serial.print(",\"servo_angle\":");
  Serial.print(currentServoAngle);
  Serial.print(",\"servo_mode\":\"");
  Serial.print(isManualMode ? "manual" : "auto");
  Serial.print("\"");
#else
  Serial.print(",\"servo_angle\":null,\"servo_mode\":null");
#endif

#if ENABLE_HOOTER
  Serial.print(",\"hooter\":\"");
  Serial.print(isHooterActive ? "ON" : "OFF");
  Serial.print("\",\"relay_pin\":26");
#else
  Serial.print(",\"hooter\":null,\"relay_pin\":null");
#endif

  Serial.println("}");
  Serial.flush();
}
