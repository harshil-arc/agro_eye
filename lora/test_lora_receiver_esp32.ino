/*
 * ==============================================================================
 * AgroEye - Ultra-Simple Standalone LoRa Receiver Test Sketch (ESP32 / Arduino)
 * ==============================================================================
 * Upload this sketch to your ESP32 / Arduino using the Arduino IDE.
 * Open the Serial Monitor at 115200 Baud.
 * It will print EVERY packet received over LoRa (raw text + RSSI + SNR).
 * ==============================================================================
 */

#include <SPI.h>
#include <LoRa.h>

// ---------- PIN DEFINITIONS ----------
#if defined(ESP32)
  // Standard ESP32 DevKit Wiring:
  #define LORA_SS       5     // GPIO 5 (NSS / CS)
  #define LORA_RST      14    // GPIO 14 (RST)
  #define LORA_DIO0     2     // GPIO 2 (DIO0)
  #define LED_PIN       2     // Onboard Blue LED (Blinks on packet RX)
#else
  // Standard Arduino Uno / Nano:
  #define LORA_SS       10    // Pin D10 (NSS)
  #define LORA_RST      9     // Pin D9 (RST)
  #define LORA_DIO0     2     // Pin D2 (DIO0)
  #define LED_PIN       13    // Onboard LED
#endif

// ---------- RF PARAMETERS (Must match Raspberry Pi) ----------
#define LORA_BAND       433E6    // 433 MHz
#define LORA_SF         7        // SF7
#define LORA_BW         125E3    // 125 kHz
#define LORA_CR         5        // Coding Rate 4/5
#define LORA_SYNC_WORD  0x12     // Custom Sync Word (0x12)

unsigned long packetCount = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.println(F("\n======================================================="));
  Serial.println(F("  AgroEye - Standalone LoRa Receiver Test (433 MHz)   "));
  Serial.println(F("======================================================="));

  // Initialize SPI & LoRa
  LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);

  if (!LoRa.begin(LORA_BAND)) {
    Serial.println(F("❌ ERROR: LoRa SX1278 initialization failed!"));
    Serial.println(F("Please verify SPI wiring (SCK, MISO, MOSI, SS=5, RST=14, DIO0=2) and 3.3V power."));
    while (1) {
      digitalWrite(LED_PIN, HIGH);
      delay(200);
      digitalWrite(LED_PIN, LOW);
      delay(200);
    }
  }

  // Configure matching RF settings
  LoRa.setSpreadingFactor(LORA_SF);
  LoRa.setSignalBandwidth(LORA_BW);
  LoRa.setCodingRate4(LORA_CR);
  LoRa.enableCrc();
  LoRa.setSyncWord(LORA_SYNC_WORD);

  Serial.println(F("✅ LoRa SX1278 Initialized Successfully!"));
  Serial.println(F("   - Frequency : 433.0 MHz"));
  Serial.println(F("   - SF        : 7"));
  Serial.println(F("   - BW        : 125 kHz"));
  Serial.println(F("   - Sync Word : 0x12"));
  Serial.println(F("Listening for incoming packets from Raspberry Pi...\n"));
}

void loop() {
  // Check for received packet
  int packetSize = LoRa.parsePacket();
  if (packetSize) {
    packetCount++;
    digitalWrite(LED_PIN, HIGH);

    String receivedText = "";
    while (LoRa.available()) {
      receivedText += (char)LoRa.read();
    }

    int rssi = LoRa.packetRssi();
    float snr = LoRa.packetSnr();

    Serial.print(F("📥 [RX #"));
    Serial.print(packetCount);
    Serial.print(F("] Size: "));
    Serial.print(packetSize);
    Serial.print(F("B | RSSI: "));
    Serial.print(rssi);
    Serial.print(F(" dBm | SNR: "));
    Serial.print(snr, 1);
    Serial.println(F(" dB"));

    Serial.print(F("   Payload: "));
    Serial.println(receivedText);
    Serial.println(F("-------------------------------------------------------"));

    delay(80);
    digitalWrite(LED_PIN, LOW);
  }
}
