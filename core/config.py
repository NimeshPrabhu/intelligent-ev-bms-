"""
Centralized Configuration for EV Battery Management System & Telemetry Dashboard.
Loads settings from environment variables or defaults suitable for Raspberry Pi 3B+.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ================= Operational Mode =================
# Set to False when physical ESP2/LoRa/Serial is connected
USE_SIMULATED_HARDWARE = os.environ.get("USE_SIMULATED_HARDWARE", "True").lower() in ("1", "true", "yes")

# ================= Battery Pack Specifications =================
# Default: 3S 11.1V (3 x 3.7V) 2500mAh (2.5Ah) Li-ion Pack
NOMINAL_PACK_VOLTAGE = float(os.environ.get("NOMINAL_PACK_VOLTAGE", 11.1))
SERIES_CELLS = int(os.environ.get("SERIES_CELLS", 3))
PARALLEL_CELLS = int(os.environ.get("PARALLEL_CELLS", 1))
BATTERY_CAPACITY_AH = float(os.environ.get("BATTERY_CAPACITY_AH", 2.5))
NOMINAL_PACK_KWH = (NOMINAL_PACK_VOLTAGE * BATTERY_CAPACITY_AH) / 1000.0
NOMINAL_EFFICIENCY_WH_KM = float(os.environ.get("NOMINAL_EFFICIENCY_WH_KM", 22.0))

# ================= Safety & Fault Thresholds =================
# Voltage thresholds (Volts)
VOLTAGE_OVERVOLT_CRITICAL = float(os.environ.get("VOLTAGE_OVERVOLT_CRITICAL", 12.65))
VOLTAGE_OVERVOLT_WARNING  = float(os.environ.get("VOLTAGE_OVERVOLT_WARNING", 12.40))
VOLTAGE_UNDERVOLT_WARNING = float(os.environ.get("VOLTAGE_UNDERVOLT_WARNING", 9.90))
VOLTAGE_UNDERVOLT_CRITICAL = float(os.environ.get("VOLTAGE_UNDERVOLT_CRITICAL", 9.00))

# Current thresholds (Amperes, Positive = Discharge, Negative = Charge/Regen)
CURRENT_DISCHARGE_WARN = float(os.environ.get("CURRENT_DISCHARGE_WARN", 5.0))
CURRENT_DISCHARGE_CRIT = float(os.environ.get("CURRENT_DISCHARGE_CRIT", 7.5))
CURRENT_CHARGE_WARN    = float(os.environ.get("CURRENT_CHARGE_WARN", -2.5))
CURRENT_CHARGE_CRIT    = float(os.environ.get("CURRENT_CHARGE_CRIT", -3.5))

# Temperature thresholds (Celsius)
TEMP_OVERTEMP_CRITICAL = float(os.environ.get("TEMP_OVERTEMP_CRITICAL", 55.0))
TEMP_OVERTEMP_WARNING  = float(os.environ.get("TEMP_OVERTEMP_WARNING", 45.0))
TEMP_UNDERTEMP_WARNING = float(os.environ.get("TEMP_UNDERTEMP_WARNING", 0.0))

# Debounce & Hysteresis Timing (seconds)
FAULT_TRIP_DEBOUNCE_S = float(os.environ.get("FAULT_TRIP_DEBOUNCE_S", 0.3))
FAULT_RECOVERY_DEBOUNCE_S = float(os.environ.get("FAULT_RECOVERY_DEBOUNCE_S", 2.0))

# Stale Telemetry Timeout (seconds)
STALE_DATA_TIMEOUT_S = float(os.environ.get("STALE_DATA_TIMEOUT_S", 3.0))

# ================= Hardware & Serial Configuration =================
SERIAL_PORT = os.environ.get("SERIAL_PORT", None)
SERIAL_BAUDRATE = int(os.environ.get("SERIAL_BAUDRATE", 115200))
LORA_FREQ = float(os.environ.get("LORA_FREQ", 433.0))

# ================= Security & Control =================
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "bms-secret-key-2026")

# ================= File Paths =================
DB_PATH = str(BASE_DIR / "readings.db")
MODEL_PATH = str(BASE_DIR / "soc_soh_model.tflite")
SCALER_PATH = str(BASE_DIR / "scaler.pkl")
STATIC_DIR = str(BASE_DIR / "static")
