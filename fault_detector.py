"""
Battery Safety & Fault Detection Engine with Hysteresis and Anti-Chatter Filtering.
Prevents relay oscillation, distinguishes warning from critical alerts, and audits events.
"""

import time
from typing import Dict, List, Any, Optional
import database
from core.config import (
    VOLTAGE_OVERVOLT_CRITICAL,
    VOLTAGE_OVERVOLT_WARNING,
    VOLTAGE_UNDERVOLT_WARNING,
    VOLTAGE_UNDERVOLT_CRITICAL,
    CURRENT_DISCHARGE_WARN,
    CURRENT_DISCHARGE_CRIT,
    CURRENT_CHARGE_WARN,
    CURRENT_CHARGE_CRIT,
    TEMP_OVERTEMP_CRITICAL,
    TEMP_OVERTEMP_WARNING,
    TEMP_UNDERTEMP_WARNING,
    FAULT_TRIP_DEBOUNCE_S,
    FAULT_RECOVERY_DEBOUNCE_S
)

class FaultDetector:
    def __init__(self):
        # Tracks pending fault candidates: { fault_key: first_detected_timestamp }
        self._pending_faults: Dict[str, float] = {}
        # Tracks active confirmed faults: { fault_key: { severity, alert_dict, tripped_at } }
        self._confirmed_faults: Dict[str, Dict[str, Any]] = {}
        # Tracks fault recovery candidates: { fault_key: first_cleared_timestamp }
        self._clearing_faults: Dict[str, float] = {}
        self.last_status = "NORMAL"

    def evaluate(self, voltage: float, current: float, temperature: float, now: Optional[float] = None) -> Dict[str, Any]:
        """
        Evaluates parameters using hysteresis filtering.
        Returns:
        {
            "status": "NORMAL" | "WARNING" | "CRITICAL",
            "relay_command": "KEEP" | "TRIP" | "RESET",
            "alerts": [ ... ]
        }
        """
        if now is None:
            now = time.time()

        raw_candidates: Dict[str, Dict[str, Any]] = {}

        # 1. Voltage Checks
        if voltage >= VOLTAGE_OVERVOLT_CRITICAL:
            raw_candidates["OVERVOLT"] = {
                "type": "OVERVOLTAGE",
                "severity": "CRITICAL",
                "message": f"Critical Overvoltage: {voltage:.2f}V exceeds limit ({VOLTAGE_OVERVOLT_CRITICAL:.2f}V)",
                "value": round(voltage, 2),
                "threshold": VOLTAGE_OVERVOLT_CRITICAL
            }
        elif voltage >= VOLTAGE_OVERVOLT_WARNING:
            raw_candidates["OVERVOLT"] = {
                "type": "OVERVOLTAGE",
                "severity": "WARNING",
                "message": f"High Voltage Warning: {voltage:.2f}V approaching peak ({VOLTAGE_OVERVOLT_WARNING:.2f}V)",
                "value": round(voltage, 2),
                "threshold": VOLTAGE_OVERVOLT_WARNING
            }
        elif voltage <= VOLTAGE_UNDERVOLT_CRITICAL:
            raw_candidates["UNDERVOLT"] = {
                "type": "UNDERVOLTAGE",
                "severity": "CRITICAL",
                "message": f"Critical Undervoltage: {voltage:.2f}V below cutoff ({VOLTAGE_UNDERVOLT_CRITICAL:.2f}V)",
                "value": round(voltage, 2),
                "threshold": VOLTAGE_UNDERVOLT_CRITICAL
            }
        elif voltage <= VOLTAGE_UNDERVOLT_WARNING:
            raw_candidates["UNDERVOLT"] = {
                "type": "UNDERVOLTAGE",
                "severity": "WARNING",
                "message": f"Low Battery Warning: {voltage:.2f}V (Approaching cutoff)",
                "value": round(voltage, 2),
                "threshold": VOLTAGE_UNDERVOLT_WARNING
            }

        # 2. Current Checks (Positive = Discharge, Negative = Charge/Regen)
        if current >= CURRENT_DISCHARGE_CRIT:
            raw_candidates["OVERCURRENT"] = {
                "type": "OVERCURRENT",
                "severity": "CRITICAL",
                "message": f"Critical Overcurrent: Discharge {current:.2f}A exceeds cutoff ({CURRENT_DISCHARGE_CRIT:.1f}A)",
                "value": round(current, 2),
                "threshold": CURRENT_DISCHARGE_CRIT
            }
        elif current >= CURRENT_DISCHARGE_WARN:
            raw_candidates["OVERCURRENT"] = {
                "type": "OVERCURRENT",
                "severity": "WARNING",
                "message": f"High Discharge Warning: {current:.2f}A approaching limit ({CURRENT_DISCHARGE_WARN:.1f}A)",
                "value": round(current, 2),
                "threshold": CURRENT_DISCHARGE_WARN
            }
        elif current <= CURRENT_CHARGE_CRIT:
            raw_candidates["OVERCHARGE"] = {
                "type": "OVERCURRENT",
                "severity": "CRITICAL",
                "message": f"Critical Charge Current: {abs(current):.2f}A exceeds charge cutoff ({abs(CURRENT_CHARGE_CRIT):.1f}A)",
                "value": round(current, 2),
                "threshold": CURRENT_CHARGE_CRIT
            }
        elif current <= CURRENT_CHARGE_WARN:
            raw_candidates["OVERCHARGE"] = {
                "type": "OVERCURRENT",
                "severity": "WARNING",
                "message": f"High Charge Warning: {abs(current):.2f}A exceeding normal rate",
                "value": round(current, 2),
                "threshold": CURRENT_CHARGE_WARN
            }

        # 3. Temperature Checks
        if temperature >= TEMP_OVERTEMP_CRITICAL:
            raw_candidates["OVERTEMP"] = {
                "type": "OVERTEMP",
                "severity": "CRITICAL",
                "message": f"Thermal Cutoff: Temperature {temperature:.1f}°C exceeds safe limit ({TEMP_OVERTEMP_CRITICAL:.1f}°C)",
                "value": round(temperature, 1),
                "threshold": TEMP_OVERTEMP_CRITICAL
            }
        elif temperature >= TEMP_OVERTEMP_WARNING:
            raw_candidates["OVERTEMP"] = {
                "type": "OVERTEMP",
                "severity": "WARNING",
                "message": f"High Temperature Warning: Pack at {temperature:.1f}°C ({TEMP_OVERTEMP_WARNING:.1f}°C)",
                "value": round(temperature, 1),
                "threshold": TEMP_OVERTEMP_WARNING
            }
        elif temperature <= TEMP_UNDERTEMP_WARNING:
            raw_candidates["UNDERTEMP"] = {
                "type": "UNDERTEMP",
                "severity": "WARNING",
                "message": f"Sub-Zero Temperature Warning: Pack at {temperature:.1f}°C",
                "value": round(temperature, 1),
                "threshold": TEMP_UNDERTEMP_WARNING
            }

        # ================= Hysteresis & Debounce Processing =================
        # A. Process new or persistent candidates
        for key, candidate in raw_candidates.items():
            # If not yet pending, mark detection start
            if key not in self._pending_faults:
                self._pending_faults[key] = now
            
            # Check if condition has persisted longer than trip debounce threshold
            elapsed = now - self._pending_faults[key]
            # Critical faults trip faster, warnings debounce slightly longer
            debounce_time = FAULT_TRIP_DEBOUNCE_S if candidate["severity"] == "CRITICAL" else (FAULT_TRIP_DEBOUNCE_S * 2)

            if elapsed >= debounce_time:
                # If not already confirmed, record event to database
                if key not in self._confirmed_faults:
                    try:
                        database.insert_fault_event({
                            "timestamp": now,
                            "type": candidate["type"],
                            "severity": candidate["severity"],
                            "message": candidate["message"],
                            "value": candidate["value"],
                            "threshold": candidate["threshold"],
                            "resolved": 0
                        })
                    except Exception:
                        pass
                
                # Update confirmed state
                self._confirmed_faults[key] = candidate
                # Clear recovery candidate if it returned
                if key in self._clearing_faults:
                    del self._clearing_faults[key]

        # B. Process clearing faults (hysteresis recovery)
        keys_to_remove = []
        for key in list(self._confirmed_faults.keys()):
            if key not in raw_candidates:
                # Condition has cleared, start recovery timer
                if key not in self._clearing_faults:
                    self._clearing_faults[key] = now
                elif now - self._clearing_faults[key] >= FAULT_RECOVERY_DEBOUNCE_S:
                    # Fault has remained cleared for full recovery window
                    keys_to_remove.append(key)
                    if key in self._pending_faults:
                        del self._pending_faults[key]
                    if key in self._clearing_faults:
                        del self._clearing_faults[key]

        for k in keys_to_remove:
            del self._confirmed_faults[k]

        # Clean up any abandoned pending faults
        for k in list(self._pending_faults.keys()):
            if k not in raw_candidates and k not in self._confirmed_faults:
                del self._pending_faults[k]

        # C. Determine overall status
        alerts = list(self._confirmed_faults.values())
        if any(a["severity"] == "CRITICAL" for a in alerts):
            status = "CRITICAL"
        elif any(a["severity"] == "WARNING" for a in alerts):
            status = "WARNING"
        else:
            status = "NORMAL"

        self.last_status = status
        return {
            "status": status,
            "relay_recommended": (status != "CRITICAL"),
            "alerts": alerts
        }

# Global singleton instance for app-wide continuity
_detector_instance = FaultDetector()

def evaluate_battery_health(voltage: float, current: float, temperature: float, now: Optional[float] = None) -> Dict[str, Any]:
    """Compatibility entry point for evaluating health with hysteresis."""
    return _detector_instance.evaluate(voltage, current, temperature, now=now)

