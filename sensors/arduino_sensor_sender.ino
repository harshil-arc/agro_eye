/*
 * ==============================================================================
 * Arduino Multi-Sensor Sender (USB Serial to Raspberry Pi)
 * =============================================================================
 * Hardware: Arduino Uno / Nano + JHTS 11 + MQ Gas Sensor + Soil Moisture Sensor
 *
 * Pin Connections:
 *   Arduino Pin    Sensor Pin
 *   -----------------------------
 *   Digital 4      DHT11 Data Pin
 *   Analog A0      MQ Gas Sensor Analog Out (AOUT)
 *   Analog A1      Soil Moisture Sensor Analog Out
 *   5V             VCC for Sensors
 *   GND            GND
 * =============================================================================
 */

#include <DHT.h>

// ---------- Sensor Enable Configuration ----------
// Set to true only for sensors physically connected to your Arduino:
#define ENABLE_DHT11       true    // Temperature & Humidity on Digital Pin 4
#define ENABLE_MQ_GAS      false   // MQ Gas Sensor on Analog Pin A0 (Set true when connected)
#define ENABLE_SOIL        false   // Soil Moisture Sensor on Analog Pin A1 (Set true when connected)

// ---------- Pin Definitions (Arduino Uno/Nano) ----------
#define DHTPIN        4        // DHT11 data pin (digital)
#define DHTTYPE       DHT11    // DHT sensor type

#define MQ_PIN        A0       // MQ gas sensor analog pin
#define SOIL_PIN      A1       // Soil moisture sensor analog pin

// ---------- Calibration values (10-bit ADC: 0-1023) ----------
#define SOIL_DRY_VALUE   1023
#define SOIL_WET_VALUE   300

#if ENABLE_DHT11
DHT dht(DHTPIN, DHTTYPE);
#endif

unsigned long lastReadTime = 0;
const unsigned long readInterval = 2000; // read every 2 seconds

void setup() {
  Serial.begin(9600);
  delay(1000);

  Serial.println("Arduino Multi-Sensor Monitor");
  Serial.println("----------------------------");

#if ENABLE_DHT11
  dht.begin();
#endif
}

void loop() {
  if (millis() - lastReadTime >= readInterval) {
    lastReadTime = millis();
    readAllSensors();
  }
}

void readAllSensors() {
  // ---------- 1. DHT11 (Temperature & Humidity) ----------
#if ENABLE_DHT11
  float humidity = dht.readHumidity();
  float tempC = dht.readTemperature();       // Celsius
  float tempF = dht.readTemperature(true);   // Fahrenheit

  if (isnan(humidity) || isnan(tempC)) {
    Serial.println("Failed to read from DHT11 sensor!");
  } else {
    Serial.print("Temperature: ");
    Serial.print(tempC);
    Serial.print(" *C (");
    Serial.print(tempF);
    Serial.print(" *F)  Humidity: ");
    Serial.print(humidity);
    Serial.println(" %");
  }
#endif

  // ---------- 2. MQ Gas Sensor (Analog A0) ----------
#if ENABLE_MQ_GAS
  int mqRaw = analogRead(MQ_PIN);
  float mqVoltage = mqRaw * (5.0 / 1023.0);   // Arduino ADC ref is 5V
  Serial.print("MQ Gas Sensor Raw: ");
  Serial.print(mqRaw);
  Serial.print("  Voltage: ");
  Serial.print(mqVoltage, 2);
  Serial.println(" V");
#endif

  // ---------- 3. Soil Moisture Sensor (Analog A1) ----------
#if ENABLE_SOIL
  int soilRaw = analogRead(SOIL_PIN);
  int soilPercent = map(soilRaw, SOIL_DRY_VALUE, SOIL_WET_VALUE, 0, 100);
  soilPercent = constrain(soilPercent, 0, 100);

  Serial.print("Soil Moisture Raw: ");
  Serial.print(soilRaw);
  Serial.print("  Moisture: ");
  Serial.print(soilPercent);
  Serial.println(" %");
#endif

  Serial.println("----------------------------");
}
