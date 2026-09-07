# ESP32 EV Telemetry Transmitter Firmware

This firmware reads battery pack voltage, current, and temperature, measures vehicle speed, formats a comma-delimited telemetry packet, and transmits it wirelessly over SX1278 LoRa point-to-point every 1 second.

## Hardware Wiring

### 1. SX1278 / Ra-02 LoRa Module (SPI)
| SX1278 Pin | ESP32 GPIO Pin | Description |
|---|---|---|
| **VCC** | 3.3V | 3.3V power supply (do not connect to 5V) |
| **GND** | GND | Ground |
| **SCK** | GPIO 18 | SPI Clock |
| **MISO** | GPIO 19 | SPI Master-In Slave-Out |
| **MOSI** | GPIO 23 | SPI Master-Out Slave-In |
| **NSS / CS** | GPIO 5 | Chip Select |
| **RST** | GPIO 14 | Reset |
| **DIO0** | GPIO 2 | Interrupt / Rx-Tx Done |

### 2. INA219 Voltage & Current Sensor (I2C)
| INA219 Pin | ESP32 GPIO Pin | Description |
|---|---|---|
| **VCC** | 3.3V / 5V | Power supply |
| **GND** | GND | Ground |
| **SDA** | GPIO 21 | I2C Data |
| **SCL** | GPIO 22 | I2C Clock |
| **VIN+ / VIN-** | In series with load / battery | High-side shunt resistor input |

> **Note on Voltage Range**: The stock INA219 chip measures up to 26V DC. For packs higher than 26V (e.g. 36V, 48V, 72V), either:
> 1. Use an external voltage divider connected to an ESP32 ADC pin and update `VOLTAGE_DIVIDER_RATIO`.
> 2. Use an INA226 (rated up to 36V) or a dedicated high-voltage isolated Hall current/voltage sensor (e.g., LEM or ACS758 + divider).

### 3. DS18B20 Temperature Sensor (1-Wire)
| DS18B20 Pin | ESP32 GPIO Pin | Description |
|---|---|---|
| **VDD** | 3.3V | Power |
| **GND** | GND | Ground |
| **DQ (Data)** | GPIO 4 | 1-Wire Data line with **4.7kΩ pullup resistor to 3.3V** |

### 4. Vehicle Speed Sensor
- **Hall-Effect Pulse Sensor (Default)**:
  - Signal pin connected to **GPIO 13** (internal pull-up enabled, interrupt on FALLING edge).
  - Configured for `WHEEL_CIRCUMFERENCE_METERS` (default 1.05m).
- **GPS Alternative (NEO-6M / NEO-8M)**:
  - If using GPS instead of Hall effect: connect GPS TX to ESP32 RX2 (GPIO 16) and GPS RX to ESP32 TX2 (GPIO 17). Use `TinyGPSPlus` library to extract `gps.speed.kmph()`.

---

## Packet Specification

Packets are transmitted as standard ASCII strings:
```
<speed_kmh>,<voltage_v>,<current_a>,<temperature_c>
```
Example:
```
24.50,48.20,-12.40,31.50
```
- **speed**: vehicle speed in km/h (float)
- **voltage**: pack voltage in Volts (float)
- **current**: battery current in Amperes, where negative indicates discharge and positive indicates regen/charge (float)
- **temperature**: pack temperature in °C (float)

---

## Required Arduino Libraries
Install via the Arduino IDE Library Manager:
- `LoRa` by Sandeep Mistry (v0.8.0 or newer)
- `Adafruit INA219` by Adafruit
- `OneWire` by Paul Stoffregen
- `DallasTemperature` by Miles Burton
