/*
 * ESP32 EV Telemetry Transmitter
 * 
 * Hardware:
 *  - ESP32 DevKit V1
 *  - SX1278 / Ra-02 LoRa Module (SPI)
 *  - INA219 High-Side DC Voltage & Current Sensor (I2C)
 *  - DS18B20 1-Wire Digital Temperature Sensor
 *  - Hall Effect Sensor (Pulse Interrupt) or GPS (Serial)
 * 
 * Packet format: "speed,voltage,current,temperature"
 * Cadence: ~1 packet per second
 */

#include <SPI.h>
#include <LoRa.h>
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ================= Pin Configuration =================
// SX1278 LoRa SPI Pins
#define LORA_SS    5
#define LORA_RST   14
#define LORA_DIO0  2
#define LORA_FREQ  433E6  // Set to 433E6, 868E6, or 915E6 matching your module

// INA219 I2C Pins (Default ESP32 Wire)
#define I2C_SDA    21
#define I2C_SCL    22

// DS18B20 1-Wire Bus Pin
#define ONE_WIRE_BUS 4

// Speed Sensor (Hall-Effect or Optical pulse)
#define SPEED_PIN  13

// Onboard LED for transmission heartbeat
#define LED_PIN    2

// ================= Speed Measurement Configuration =================
// Wheel circumference in meters (e.g., 10-inch scooter wheel ~ 0.8m, bicycle wheel ~ 2.1m)
const float WHEEL_CIRCUMFERENCE_METERS = 1.05f; 
const int PULSES_PER_REVOLUTION = 1; // Number of magnets on wheel

volatile unsigned long pulseCount = 0;
unsigned long lastPulseTime = 0;
portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;

void IRAM_ATTR onSpeedPulse() {
    portENTER_CRITICAL_ISR(&mux);
    pulseCount++;
    portEXIT_CRITICAL_ISR(&mux);
}

// ================= Sensor Objects =================
Adafruit_INA219 ina219;
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature tempSensors(&oneWire);

bool ina219Available = false;
bool loraAvailable = false;

// Multi-cell battery pack voltage scaling:
// INA219 measures up to 26V directly. If using a voltage divider for 36V/48V/72V packs,
// configure DIVIDER_RATIO = (R1 + R2) / R2. If measuring directly <= 26V, ratio = 1.0.
const float VOLTAGE_DIVIDER_RATIO = 1.0f;

// Transmission timing
unsigned long lastTxTime = 0;
const unsigned long TX_INTERVAL_MS = 1000; // 1 Hz cadence

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n[ESP32 Telemetry] Initializing...");

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    // 1. Hall Effect Sensor Setup
    pinMode(SPEED_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(SPEED_PIN), onSpeedPulse, FALLING);
    Serial.println(" -> Speed pulse interrupt configured on GPIO 13");

    // 2. INA219 Setup
    Wire.begin(I2C_SDA, I2C_SCL);
    if (ina219.begin()) {
        ina219Available = true;
        ina219.setCalibration_32V_2A();
        Serial.println(" -> INA219 voltage/current sensor initialized");
    } else {
        Serial.println(" [!] Warning: INA219 not detected. Will report simulated/zero power data.");
    }

    // 3. DS18B20 Setup
    tempSensors.begin();
    tempSensors.setWaitForConversion(false);
    tempSensors.requestTemperatures();
    Serial.println(" -> DS18B20 temperature sensor initialized on GPIO 4");

    // 4. SX1278 LoRa Setup
    LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);
    if (LoRa.begin(LORA_FREQ)) {
        loraAvailable = true;
        LoRa.setSpreadingFactor(7);
        LoRa.setSignalBandwidth(125E3);
        LoRa.setCodingRate4(5);
        LoRa.setSyncWord(0x12);
        LoRa.enableCrc();
        Serial.println(" -> SX1278 LoRa initialized at 433 MHz (SF7, 125kHz, CR4/5)");
    } else {
        Serial.println(" [!] Error: SX1278 LoRa module init failed! Check SPI wiring.");
    }

    Serial.println("[ESP32 Telemetry] Initialization complete. Streaming packets...\n");
}

float calculateSpeedKmH(unsigned long elapsedMs) {
    if (elapsedMs == 0) return 0.0f;

    unsigned long pulses = 0;
    portENTER_CRITICAL(&mux);
    pulses = pulseCount;
    pulseCount = 0; // Reset for next window
    portEXIT_CRITICAL(&mux);

    float revolutions = (float)pulses / (float)PULSES_PER_REVOLUTION;
    float distanceMeters = revolutions * WHEEL_CIRCUMFERENCE_METERS;
    float speedMps = distanceMeters / ((float)elapsedMs / 1000.0f);
    float speedKmh = speedMps * 3.6f;

    return speedKmh;
}

void loop() {
    unsigned long now = millis();
    if (now - lastTxTime >= TX_INTERVAL_MS) {
        unsigned long deltaMs = now - lastTxTime;
        lastTxTime = now;

        // 1. Measure Speed
        float speedKmh = calculateSpeedKmH(deltaMs);

        // 2. Measure Voltage & Current
        float busVoltageV = 0.0f;
        float currentA = 0.0f;

        if (ina219Available) {
            float busVoltageRaw = ina219.getBusVoltage_V();
            float shuntVoltage_mV = ina219.getShuntVoltage_mV();
            busVoltageV = (busVoltageRaw + (shuntVoltage_mV / 1000.0f)) * VOLTAGE_DIVIDER_RATIO;
            // Telemetry convention: negative current = battery discharge, positive = charging/regen
            currentA = -(ina219.getCurrent_mA() / 1000.0f);
        } else {
            // Fallback: nominal cell pack voltage
            busVoltageV = 48.0f;
            currentA = -5.0f;
        }

        // 3. Measure Temperature
        float tempC = tempSensors.getTempCByIndex(0);
        if (tempC < -50.0f || tempC > 125.0f) {
            tempC = 25.0f;
        }
        tempSensors.requestTemperatures();

        // 4. Format comma-separated packet: "speed,voltage,current,temperature"
        char packetBuf[64];
        snprintf(packetBuf, sizeof(packetBuf), "%.2f,%.2f,%.2f,%.2f",
                 speedKmh, busVoltageV, currentA, tempC);

        // 5. Transmit over LoRa
        if (loraAvailable) {
            digitalWrite(LED_PIN, HIGH);
            LoRa.beginPacket();
            LoRa.print(packetBuf);
            LoRa.endPacket();
            digitalWrite(LED_PIN, LOW);
        }

        Serial.printf("[TX] Packet: %s  (delta: %lums)\n", packetBuf, deltaMs);
    }
}
