"""
Battery Fault Detection Engine
Configured for 3S 11.1V (3 x 3.7V) 2500mAh Li-ion battery pack.

Detects:
- Undervoltage (Warning & Critical)
- Overvoltage (Warning & Critical)
- Overcurrent (Discharge & Charge/Regen)
- Overtemperature & Under-temperature
"""

from typing import Dict, List, Any, Optional

# ================= 3S Battery Safety Thresholds =================
# Nominal: 11.1V (3.7V/cell), Peak: 12.6V (4.2V/cell), Cutoff: 9.0V (3.0V/cell)
VOLTAGE_OVERVOLT_CRITICAL = 12.65  # > 4.22V / cell
VOLTAGE_OVERVOLT_WARNING  = 12.40  # > 4.13V / cell
VOLTAGE_UNDERVOLT_WARNING = 9.90   # < 3.30V / cell
VOLTAGE_UNDERVOLT_CRITICAL = 9.00  # < 3.00V / cell

# Current thresholds (2500mAh / 2.5Ah capacity)
# Note: In telemetry, discharge current is negative, charge is positive
CURRENT_DISCHARGE_WARN    = -5.0   # > 2C discharge (5.0A)
CURRENT_DISCHARGE_CRIT    = -7.5   # > 3C discharge (7.5A)
CURRENT_CHARGE_WARN       = 2.5    # > 1C charge (2.5A)
CURRENT_CHARGE_CRIT       = 3.5    # > 1.4C charge (3.5A)

# Temperature thresholds (Celsius)
TEMP_OVERTEMP_CRITICAL    = 55.0
TEMP_OVERTEMP_WARNING     = 45.0
TEMP_UNDERTEMP_WARNING    = 0.0

def evaluate_battery_health(voltage: float, current: float, temperature: float) -> Dict[str, Any]:
    """
    Evaluates battery pack electrical and thermal parameters against safety thresholds.
    
    Returns a dictionary:
    {
        "status": "NORMAL" | "WARNING" | "CRITICAL",
        "alerts": [
            {
                "type": str,        # "UNDERVOLTAGE", "OVERVOLTAGE", "OVERCURRENT", "OVERTEMP"
                "severity": str,    # "WARNING" | "CRITICAL"
                "message": str,
                "value": float,
                "threshold": float
            }, ...
        ]
    }
    """
    alerts: List[Dict[str, Any]] = []

    # 1. Voltage Checks
    if voltage >= VOLTAGE_OVERVOLT_CRITICAL:
        alerts.append({
            "type": "OVERVOLTAGE",
            "severity": "CRITICAL",
            "message": f"Critical Overvoltage: {voltage:.2f}V exceeds limit ({VOLTAGE_OVERVOLT_CRITICAL:.2f}V)",
            "value": round(voltage, 2),
            "threshold": VOLTAGE_OVERVOLT_CRITICAL
        })
    elif voltage >= VOLTAGE_OVERVOLT_WARNING:
        alerts.append({
            "type": "OVERVOLTAGE",
            "severity": "WARNING",
            "message": f"High Voltage Warning: {voltage:.2f}V approaching peak ({VOLTAGE_OVERVOLT_WARNING:.2f}V)",
            "value": round(voltage, 2),
            "threshold": VOLTAGE_OVERVOLT_WARNING
        })
    elif voltage <= VOLTAGE_UNDERVOLT_CRITICAL:
        alerts.append({
            "type": "UNDERVOLTAGE",
            "severity": "CRITICAL",
            "message": f"Critical Undervoltage: {voltage:.2f}V below cutoff ({VOLTAGE_UNDERVOLT_CRITICAL:.2f}V)",
            "value": round(voltage, 2),
            "threshold": VOLTAGE_UNDERVOLT_CRITICAL
        })
    elif voltage <= VOLTAGE_UNDERVOLT_WARNING:
        alerts.append({
            "type": "UNDERVOLTAGE",
            "severity": "WARNING",
            "message": f"Low Battery Warning: {voltage:.2f}V ({voltage/3.0:.2f}V/cell)",
            "value": round(voltage, 2),
            "threshold": VOLTAGE_UNDERVOLT_WARNING
        })

    # 2. Current Checks (Negative = Discharge, Positive = Charge)
    if current <= CURRENT_DISCHARGE_CRIT:
        alerts.append({
            "type": "OVERCURRENT",
            "severity": "CRITICAL",
            "message": f"Critical Overcurrent: Discharge {abs(current):.2f}A exceeds limit ({abs(CURRENT_DISCHARGE_CRIT):.1f}A)",
            "value": round(current, 2),
            "threshold": CURRENT_DISCHARGE_CRIT
        })
    elif current <= CURRENT_DISCHARGE_WARN:
        alerts.append({
            "type": "OVERCURRENT",
            "severity": "WARNING",
            "message": f"High Current Warning: Discharge {abs(current):.2f}A exceeding 2C ({abs(CURRENT_DISCHARGE_WARN):.1f}A)",
            "value": round(current, 2),
            "threshold": CURRENT_DISCHARGE_WARN
        })
    elif current >= CURRENT_CHARGE_CRIT:
        alerts.append({
            "type": "OVERCURRENT",
            "severity": "CRITICAL",
            "message": f"Critical Charge Current: {current:.2f}A exceeds charge limit ({CURRENT_CHARGE_CRIT:.1f}A)",
            "value": round(current, 2),
            "threshold": CURRENT_CHARGE_CRIT
        })
    elif current >= CURRENT_CHARGE_WARN:
        alerts.append({
            "type": "OVERCURRENT",
            "severity": "WARNING",
            "message": f"High Charge Current Warning: {current:.2f}A ({CURRENT_CHARGE_WARN:.1f}A)",
            "value": round(current, 2),
            "threshold": CURRENT_CHARGE_WARN
        })

    # 3. Temperature Checks
    if temperature >= TEMP_OVERTEMP_CRITICAL:
        alerts.append({
            "type": "OVERTEMP",
            "severity": "CRITICAL",
            "message": f"Thermal Fault: Temperature {temperature:.1f}°C exceeds safe cutoff ({TEMP_OVERTEMP_CRITICAL:.1f}°C)",
            "value": round(temperature, 1),
            "threshold": TEMP_OVERTEMP_CRITICAL
        })
    elif temperature >= TEMP_OVERTEMP_WARNING:
        alerts.append({
            "type": "OVERTEMP",
            "severity": "WARNING",
            "message": f"High Temperature Warning: Pack at {temperature:.1f}°C ({TEMP_OVERTEMP_WARNING:.1f}°C)",
            "value": round(temperature, 1),
            "threshold": TEMP_OVERTEMP_WARNING
        })
    elif temperature <= TEMP_UNDERTEMP_WARNING:
        alerts.append({
            "type": "UNDERTEMP",
            "severity": "WARNING",
            "message": f"Low Temperature Warning: Pack at {temperature:.1f}°C (Sub-zero)",
            "value": round(temperature, 1),
            "threshold": TEMP_UNDERTEMP_WARNING
        })

    # Determine overall status
    if any(a["severity"] == "CRITICAL" for a in alerts):
        status = "CRITICAL"
    elif any(a["severity"] == "WARNING" for a in alerts):
        status = "WARNING"
    else:
        status = "NORMAL"

    return {
        "status": status,
        "alerts": alerts
    }
