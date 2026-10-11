"""
Telemetry Processing Service.
Handles packet validation (versioned JSON and legacy CSV), stale-data detection,
Coulomb counting State of Charge (SOC), and usable energy range estimation.
"""

import json
import time
import math
import logging
from typing import Dict, Any, Optional, Tuple

from core.config import (
    NOMINAL_PACK_VOLTAGE,
    SERIES_CELLS,
    BATTERY_CAPACITY_AH,
    NOMINAL_PACK_KWH,
    NOMINAL_EFFICIENCY_WH_KM,
    STALE_DATA_TIMEOUT_S
)

logger = logging.getLogger("TelemetryService")

class TelemetryService:
    def __init__(self):
        self.last_packet_time = 0.0
        self.packet_count = 0
        self.last_seq = 0
        self.dropped_packets = 0
        self.last_soc = 90.0
        self.last_soh = 98.0
        self.manual_relay_override: Optional[int] = None # None = automatic, 0 = force open, 1 = force closed

    def parse_packet(self, raw_input: Any) -> Optional[Dict[str, Any]]:
        """
        Parses incoming telemetry.
        Supports both versioned JSON packets and legacy comma-delimited CSV.
        """
        if not raw_input:
            return None

        now = time.time()
        parsed: Dict[str, Any] = {}

        # 1. Try Parsing as JSON
        if isinstance(raw_input, dict):
            data_dict = raw_input
        elif isinstance(raw_input, str) and raw_input.strip().startswith("{"):
            try:
                data_dict = json.loads(raw_input.strip())
            except Exception:
                data_dict = None
        else:
            data_dict = None

        if data_dict:
            try:
                seq = int(data_dict.get("seq", self.last_seq + 1))
                if self.last_seq > 0 and seq > self.last_seq + 1:
                    self.dropped_packets += (seq - self.last_seq - 1)
                self.last_seq = seq

                v = float(data_dict.get("v", data_dict.get("voltage", 0.0)))
                i = float(data_dict.get("i", data_dict.get("current", 0.0)))
                p = float(data_dict.get("p", data_dict.get("power", abs(v * i))))
                t = float(data_dict.get("temp", data_dict.get("temperature", 25.0)))
                spd = float(data_dict.get("speed", 0.0))
                relay = int(data_dict.get("relay", 1))
                faults = int(data_dict.get("faults", 0))
                source = str(data_dict.get("source", "hw"))

                parsed = {
                    "seq": seq,
                    "speed": max(0.0, round(spd, 1)),
                    "voltage": round(v, 2),
                    "current": round(i, 2),
                    "power": round(p, 2),
                    "temperature": round(t, 1),
                    "relay_state": relay,
                    "fault_mask": faults,
                    "source": source
                }
            except (ValueError, TypeError):
                return None

        # 2. Try Parsing as Legacy CSV: "speed,voltage,current,temperature"
        elif isinstance(raw_input, str):
            tokens = [tok.strip() for tok in raw_input.strip().split(",")]
            if len(tokens) == 4:
                try:
                    spd = float(tokens[0])
                    v = float(tokens[1])
                    i = float(tokens[2])
                    t = float(tokens[3])
                    self.last_seq += 1

                    parsed = {
                        "seq": self.last_seq,
                        "speed": max(0.0, round(spd, 1)),
                        "voltage": round(v, 2),
                        "current": round(i, 2),
                        "power": round(abs(v * i), 2),
                        "temperature": round(t, 1),
                        "relay_state": 1,
                        "fault_mask": 0,
                        "source": "hw"
                    }
                except ValueError:
                    return None
            else:
                return None
        else:
            return None

        # 3. Sanity check physical bounds
        if not (0.0 <= parsed["voltage"] <= 150.0):
            return None
        if not (-200.0 <= parsed["current"] <= 200.0):
            return None
        if not (-40.0 <= parsed["temperature"] <= 125.0):
            return None
        if not (-5.0 <= parsed["speed"] <= 250.0):
            return None

        # 4. Apply manual relay override if active
        if self.manual_relay_override is not None:
            parsed["relay_state"] = self.manual_relay_override

        self.last_packet_time = now
        self.packet_count += 1
        return parsed

    def is_stale(self, now: Optional[float] = None) -> bool:
        """Returns True if no telemetry packet has arrived within timeout window."""
        if self.last_packet_time == 0:
            return True
        t = now or time.time()
        return (t - self.last_packet_time) > STALE_DATA_TIMEOUT_S

    def estimate_soc(self, pack_voltage: float, current: float, dt_s: float) -> float:
        """
        Computes State of Charge (SOC) via Coulomb Counting + OCV drift compensation.
        Current Sign Convention: Positive = Discharge, Negative = Charge/Regen.
        """
        cell_v = pack_voltage / float(SERIES_CELLS) if SERIES_CELLS > 0 else pack_voltage

        # OCV-based baseline SOC lookup for Lithium-ion
        if cell_v >= 4.20:
            ocv_soc = 100.0
        elif cell_v >= 4.05:
            ocv_soc = 90.0 + (cell_v - 4.05) / 0.15 * 10.0
        elif cell_v >= 3.90:
            ocv_soc = 75.0 + (cell_v - 3.90) / 0.15 * 15.0
        elif cell_v >= 3.75:
            ocv_soc = 50.0 + (cell_v - 3.75) / 0.15 * 25.0
        elif cell_v >= 3.60:
            ocv_soc = 25.0 + (cell_v - 3.60) / 0.15 * 25.0
        elif cell_v >= 3.30:
            ocv_soc = 5.0 + (cell_v - 3.30) / 0.30 * 20.0
        else:
            ocv_soc = max(0.0, (cell_v - 3.0) / 0.30 * 5.0)

        # Coulomb Counting update:
        # delta_Ah = - (current * dt) / 3600  (since positive current is discharge)
        dt_s = max(0.05, min(5.0, dt_s))
        delta_ah = -(current * dt_s) / 3600.0
        coulomb_soc = self.last_soc + (delta_ah / BATTERY_CAPACITY_AH) * 100.0

        # When active current is flowing (|I| > 0.5A), internal resistance creates IR voltage distortion.
        # Use pure Coulomb integration under load; only blend OCV when near-idle (|I| <= 0.5A).
        if abs(current) > 0.5:
            combined_soc = coulomb_soc
        else:
            combined_soc = 0.96 * coulomb_soc + 0.04 * ocv_soc


        combined_soc = max(0.0, min(100.0, combined_soc))
        self.last_soc = combined_soc
        return round(combined_soc, 2)


    def estimate_range_km(self, soc: float, soh: float, current: float, speed: float) -> float:
        """Estimates remaining range in kilometers based on battery state and load."""
        if soc <= 0.0 or soh <= 0.0:
            return 0.0

        usable_kwh = NOMINAL_PACK_KWH * (soh / 100.0) * (soc / 100.0)

        if speed > 2.0 and current > 0.5:
            # Positive current is discharge
            power_kw = (NOMINAL_PACK_VOLTAGE * current) / 1000.0
            eff_wh_km = (power_kw * 1000.0) / speed
            eff_wh_km = max(8.0, min(70.0, eff_wh_km))
            blended_eff = 0.7 * NOMINAL_EFFICIENCY_WH_KM + 0.3 * eff_wh_km
        else:
            blended_eff = NOMINAL_EFFICIENCY_WH_KM

        remaining_km = (usable_kwh * 1000.0) / blended_eff
        return max(0.0, round(remaining_km, 1))

telemetry_service = TelemetryService()
