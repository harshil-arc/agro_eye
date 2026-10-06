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
 *   VIN (5V)       Servo VCC (Red Wire) - MUST BE 5V (Not 3.3V!)
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

// ---------- Pin Definitions ----------
#define DHTPIN            4       // GPIO 4 for DHT Data
#define DHTTYPE           DHT11   // Set to DHT11 (Blue) or DHT22 (White)

#define SOIL_PIN          34      // GPIO 34 (ADC1_CH6) for Soil Moisture
#define MQ135_PIN         35      // GPIO 35 (ADC1_CH7) for MQ-135 Air Quality
#define SERVO_PIN         18      // GPIO 18 for Pan Servo PWM Control

// ---------- Calibration & Connection Thresholds (ESP32 12-bit ADC: 0 - 4095) ----------
// Soil Moisture Sensor:
// Disconnected / open circuit: raw < 300 (near 0)
// In air (dry): ~3200 - 4095 (0% moisture)
// In water (wet): ~1200 - 1600 (100% moisture)
#define SOIL_DISCONNECTED_RAW  300
#define SOIL_DRY_RAW           3600
#define SOIL_WET_RAW           1400

// MQ-135 Air Quality Sensor:
// Disconnected / unpowered: raw < 100 (near 0.0V)
// In clean ambient air: ~200 - 1500 (> 0.1V)
#define MQ135_DISCONNECTED_RAW 100

#if ENABLE_DHT
DHT dht(DHTPIN, DHTTYPE);
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

unsigned long lastReadTime = 0;
const unsigned long readInterval = 2000; // 2 seconds between telemetry packets

void setup() {
  Serial.begin(115200);
  delay(500);

  // Configure ADC resolution to 12 bits (0-4095)
  analogReadResolution(12);

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

  Serial.println("{\"status\":\"ESP32_READY\",\"baud\":115200,\"servo_angle\":90}");
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
  }

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

  // 1. Read DHT with retry loop
#if ENABLE_DHT
  for (int attempt = 0; attempt < 3; attempt++) {
    humidity = dht.readHumidity();
    temperature = dht.readTemperature();
    if (!isnan(humidity) && !isnan(temperature) && humidity > 0.0 && temperature > -40.0 && temperature < 85.0) {
      dhtValid = true;
      break;
    }
    delay(50);
  }
#endif

  // 2. Read Soil Moisture (Analog GPIO 34)
  int soilRaw = 0;
  float soilPercent = 0.0;
  bool soilValid = false;
#if ENABLE_SOIL
  soilRaw = analogRead(SOIL_PIN);
  // Sensor is considered physically connected when raw reading is >= SOIL_DISCONNECTED_RAW
  if (soilRaw >= SOIL_DISCONNECTED_RAW) {
    soilValid = true;
    soilPercent = ((float)(SOIL_DRY_RAW - soilRaw) / (float)(SOIL_DRY_RAW - SOIL_WET_RAW)) * 100.0;
    if (soilPercent < 0.0) soilPercent = 0.0;
    if (soilPercent > 100.0) soilPercent = 100.0;
  }
#endif

  // 3. Read MQ-135 Air Quality (Analog GPIO 35)
  int mqRaw = 0;
  float mqVoltage = 0.0;
  bool mqValid = false;
#if ENABLE_MQ135
  mqRaw = analogRead(MQ135_PIN);
  mqVoltage = (mqRaw / 4095.0) * 3.3; // ESP32 ADC reference is 3.3V
  // MQ-135 is considered physically connected when raw reading is >= MQ135_DISCONNECTED_RAW
  if (mqRaw >= MQ135_DISCONNECTED_RAW) {
    mqValid = true;
  }
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
  if (soilValid) {
    Serial.print(",\"soil_moisture\":");
    Serial.print(soilPercent, 1);
    Serial.print(",\"soil_raw\":");
    Serial.print(soilRaw);
  } else {
    Serial.print(",\"soil_moisture\":null,\"soil_raw\":null");
  }
#else
  Serial.print(",\"soil_moisture\":null,\"soil_raw\":null");
#endif

#if ENABLE_MQ135
  if (mqValid) {
    Serial.print(",\"mq135_raw\":");
    Serial.print(mqRaw);
    Serial.print(",\"mq135_voltage\":");
    Serial.print(mqVoltage, 2);
  } else {
    Serial.print(",\"mq135_raw\":null,\"mq135_voltage\":null");
  }
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

  Serial.println("}");
  Serial.flush();
}
