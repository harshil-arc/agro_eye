/*
 * ==============================================================================
 * AgroEye ESP32 Multi-Sensor + Servo Controller (USB Serial to Raspberry Pi)
 * ==============================================================================
 * Hardware: ESP32 Dev Module (WROOM / NodeMCU-32S) + DHT11/22 + Soil + MQ135 + Pan Servo
 *
 * Pin Connections:
 *   ESP32 Pin      Hardware / Sensor Pin
 *   -------------------------------------------------
 *   GPIO 4         DHT11 / DHT22 Data Pin (Pullup to 3.3V)
 *   GPIO 34 (ADC1) Soil Moisture Sensor Analog Out (AOUT)
 *   GPIO 35 (ADC1) MQ-135 Gas / Air Quality Analog Out (AOUT)
 *   GPIO 18        Camera Pan Servo Signal Pin (PWM)
 *   VIN (5V)       Servo VCC (Red Wire) & 5V Sensors
 *   3.3V           DHT / Low-Voltage Sensor VCC
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
#define DHTTYPE           DHT11   // Change to DHT22 if using white DHT22 sensor

#define SOIL_PIN          34      // GPIO 34 (ADC1_CH6) for Soil Moisture
#define MQ135_PIN         35      // GPIO 35 (ADC1_CH7) for MQ-135 Air Quality
#define SERVO_PIN         18      // GPIO 18 for Pan Servo PWM Control

// ---------- Calibration (ESP32 12-bit ADC: 0 - 4095) ----------
// In air (completely dry): ~3200 - 4095
// In water (completely wet): ~1200 - 1600
#define SOIL_DRY_RAW      3600
#define SOIL_WET_RAW      1400

#if ENABLE_DHT
DHT dht(DHTPIN, DHTTYPE);
#endif

#if ENABLE_SERVO
Servo panServo;
int currentServoAngle = 90;
int servoDirection = 1;      // 1 = sweeping up, -1 = sweeping down
unsigned long lastServoStep = 0;
const int servoMinAngle = 30;
const int servoMaxAngle = 150;
const int servoStepDeg = 2;
const unsigned long servoStepInterval = 60; // ms per step for smooth sweep
#endif

unsigned long lastReadTime = 0;
const unsigned long readInterval = 2000; // 2 seconds between telemetry packets

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Configure ADC resolution to 12 bits (0-4095)
  analogReadResolution(12);

#if ENABLE_DHT
  dht.begin();
#endif

#if ENABLE_SERVO
  // Allow allocation of all timers for ESP32Servo
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  panServo.setPeriodHertz(50); // Standard 50hz servo
  panServo.attach(SERVO_PIN, 500, 2400); // Attach GPIO 18 with 500us - 2400us pulses
  panServo.write(currentServoAngle);
#endif
}

void loop() {
  // Check for incoming serial commands from Raspberry Pi (e.g. angle commands)
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.startsWith("SERVO:")) {
      int targetAngle = cmd.substring(6).toInt();
      targetAngle = constrain(targetAngle, 0, 180);
#if ENABLE_SERVO
      currentServoAngle = targetAngle;
      panServo.write(currentServoAngle);
#endif
    }
  }

#if ENABLE_SERVO
  // Smooth auto-sweep oscillation for camera coverage
  if (millis() - lastServoStep >= servoStepInterval) {
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

  // Periodic sensor telemetry output
  if (millis() - lastReadTime >= readInterval) {
    lastReadTime = millis();
    readAndTransmitTelemetry();
  }
}

void readAndTransmitTelemetry() {
  float temperature = 0.0;
  float humidity = 0.0;
  bool dhtValid = false;

  // 1. Read DHT
#if ENABLE_DHT
  humidity = dht.readHumidity();
  temperature = dht.readTemperature(); // Celsius
  if (!isnan(humidity) && !isnan(temperature)) {
    dhtValid = true;
  }
#endif

  // 2. Read Soil Moisture (Analog GPIO 34)
  int soilRaw = 0;
  float soilPercent = 0.0;
#if ENABLE_SOIL
  soilRaw = analogRead(SOIL_PIN);
  // Map and constrain 0-100%
  soilPercent = ((float)(SOIL_DRY_RAW - soilRaw) / (float)(SOIL_DRY_RAW - SOIL_WET_RAW)) * 100.0;
  if (soilPercent < 0.0) soilPercent = 0.0;
  if (soilPercent > 100.0) soilPercent = 100.0;
#endif

  // 3. Read MQ-135 Air Quality (Analog GPIO 35)
  int mqRaw = 0;
  float mqVoltage = 0.0;
#if ENABLE_MQ135
  mqRaw = analogRead(MQ135_PIN);
  mqVoltage = (mqRaw / 4095.0) * 3.3; // ESP32 ADC reference is 3.3V
#endif

  // 4. Output Compact JSON Line to USB Serial (Read by Raspberry Pi)
  Serial.print("{\"source\":\"esp32_sensor\"");

  if (dhtValid) {
    Serial.print(",\"temperature\":");
    Serial.print(temperature, 1);
    Serial.print(",\"humidity\":");
    Serial.print(humidity, 1);
  }

#if ENABLE_SOIL
  Serial.print(",\"soil_moisture\":");
  Serial.print(soilPercent, 1);
  Serial.print(",\"soil_raw\":");
  Serial.print(soilRaw);
#endif

#if ENABLE_MQ135
  Serial.print(",\"mq135_raw\":");
  Serial.print(mqRaw);
  Serial.print(",\"mq135_voltage\":");
  Serial.print(mqVoltage, 2);
#endif

  Serial.println("}");
}
