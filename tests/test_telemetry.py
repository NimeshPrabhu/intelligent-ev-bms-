"""
Comprehensive Automated Test Suite for Intelligent EV Battery Management System.
Tests:
- Telemetry packet parsing & validation (JSON & CSV)
- Current sign conventions & Coulomb counting SOC estimation
- Fault detection engine & debounce hysteresis
- Database persistence, schema migrations, and pagination
- FastAPI endpoints & E-Stop relay control
"""

import os
import time
import unittest
import tempfile

from fastapi.testclient import TestClient
from main import app
import database
from services.telemetry_service import telemetry_service
from fault_detector import FaultDetector

class TestBMSCore(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.detector = FaultDetector()

    # ================= Telemetry Parsing Tests =================
    def test_parse_json_packet(self):
        json_packet = {
            "ver": 1,
            "seq": 105,
            "v": 11.25,
            "i": 2.40,
            "p": 27.0,
            "temp": 30.5,
            "speed": 22.0,
            "relay": 1,
            "source": "hw"
        }
        parsed = telemetry_service.parse_packet(json_packet)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["seq"], 105)
        self.assertEqual(parsed["voltage"], 11.25)
        self.assertEqual(parsed["current"], 2.40)
        self.assertEqual(parsed["power"], 27.0)
        self.assertEqual(parsed["relay_state"], 1)

    def test_parse_legacy_csv_packet(self):
        raw_csv = "24.5, 11.40, 1.80, 28.5"
        parsed = telemetry_service.parse_packet(raw_csv)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["speed"], 24.5)
        self.assertEqual(parsed["voltage"], 11.40)
        self.assertEqual(parsed["current"], 1.80)
        self.assertEqual(parsed["temperature"], 28.5)

    def test_parse_malformed_packets(self):
        self.assertIsNone(telemetry_service.parse_packet(""))
        self.assertIsNone(telemetry_service.parse_packet("corrupted,packet"))
        self.assertIsNone(telemetry_service.parse_packet("25.0, -10.0, 1.0, 30.0"))  # Negative voltage out of bounds
        self.assertIsNone(telemetry_service.parse_packet("25.0, 12.0, 1.0, 150.0"))   # Temp > 125C out of bounds

    # ================= Current Sign & Coulomb Counting Tests =================
    def test_coulomb_counting_discharge_reduces_soc(self):
        initial_soc = 90.0
        telemetry_service.last_soc = initial_soc
        # Positive current = discharge (should decrease SOC)
        soc_after = telemetry_service.estimate_soc(pack_voltage=11.1, current=5.0, dt_s=1.0)
        self.assertLess(soc_after, initial_soc)

    def test_coulomb_counting_charge_increases_soc(self):
        initial_soc = 50.0
        telemetry_service.last_soc = initial_soc
        # Negative current = charging (should increase SOC)
        soc_after = telemetry_service.estimate_soc(pack_voltage=11.1, current=-3.0, dt_s=1.0)
        self.assertGreater(soc_after, initial_soc)

    # ================= Fault Detection & Hysteresis Tests =================
    def test_nominal_evaluation(self):
        res = self.detector.evaluate(voltage=11.1, current=2.0, temperature=28.0, now=100.0)
        self.assertEqual(res["status"], "NORMAL")
        self.assertTrue(res["relay_recommended"])
        self.assertEqual(len(res["alerts"]), 0)

    def test_critical_overvoltage_with_debounce(self):
        t0 = 100.0
        # Instant 0: candidate marked pending, not yet confirmed trip
        res1 = self.detector.evaluate(voltage=13.0, current=2.0, temperature=28.0, now=t0)
        self.assertEqual(res1["status"], "NORMAL")

        # After trip debounce window (e.g. +0.4s): confirmed CRITICAL
        res2 = self.detector.evaluate(voltage=13.0, current=2.0, temperature=28.0, now=t0 + 0.4)
        self.assertEqual(res2["status"], "CRITICAL")
        self.assertFalse(res2["relay_recommended"])
        self.assertTrue(any(a["type"] == "OVERVOLTAGE" for a in res2["alerts"]))

    def test_fault_recovery_hysteresis(self):
        t0 = 200.0
        # Trigger overtemp
        self.detector.evaluate(voltage=11.1, current=2.0, temperature=60.0, now=t0)
        res_trip = self.detector.evaluate(voltage=11.1, current=2.0, temperature=60.0, now=t0 + 0.5)
        self.assertEqual(res_trip["status"], "CRITICAL")

        # Temperature returns to normal immediately (e.g. 30C), but within recovery window it stays latched
        res_trans = self.detector.evaluate(voltage=11.1, current=2.0, temperature=30.0, now=t0 + 1.0)
        self.assertEqual(res_trans["status"], "CRITICAL")

        # After recovery debounce window (>2.0s normal), fault finally clears
        res_cleared = self.detector.evaluate(voltage=11.1, current=2.0, temperature=30.0, now=t0 + 3.0)
        self.assertEqual(res_cleared["status"], "NORMAL")

    # ================= Database Tests =================
    def test_database_insert_and_paginate(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        try:
            database.init_db(temp_db)
            database.insert_reading({
                "timestamp": 1000.0, "seq": 1, "speed": 20.0, "voltage": 11.2,
                "current": 2.5, "power": 28.0, "temperature": 30.0, "soc": 85.0,
                "soh": 95.0, "range_km": 15.0, "relay_state": 1, "source": "hw"
            }, db_path=temp_db)

            database.insert_reading({
                "timestamp": 1005.0, "seq": 2, "speed": 25.0, "voltage": 11.1,
                "current": 3.0, "power": 33.3, "temperature": 31.0, "soc": 84.5,
                "soh": 95.0, "range_km": 14.8, "relay_state": 1, "source": "hw"
            }, db_path=temp_db)

            latest = database.get_latest_reading(temp_db)
            self.assertIsNotNone(latest)
            self.assertEqual(latest["seq"], 2)
            self.assertEqual(latest["power"], 33.3)

            history = database.get_history(limit=10, db_path=temp_db)
            self.assertEqual(len(history), 2)
        finally:
            try:
                if os.path.exists(temp_db):
                    os.remove(temp_db)
            except Exception:
                pass


    # ================= REST API & E-Stop Tests =================
    def test_api_latest(self):
        resp = self.client.get("/api/latest")
        self.assertEqual(resp.status_code, 200)

    def test_api_diagnostics(self):
        resp = self.client.get("/api/diagnostics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertIn("uptime_s", data)

    def test_api_relay_control_unauthorized(self):
        # Missing or invalid key should return 401
        resp = self.client.post("/api/control/relay?action=trip")
        self.assertEqual(resp.status_code, 401)

    def test_api_relay_control_estop_success(self):
        resp = self.client.post(
            "/api/control/relay?action=trip",
            headers={"X-Api-Key": "local-dev"}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["relay_state"], 0)

        # Reset relay
        resp_reset = self.client.post(
            "/api/control/relay?action=reset",
            headers={"X-Api-Key": "local-dev"}
        )
        self.assertEqual(resp_reset.status_code, 200)
        self.assertEqual(resp_reset.json()["relay_state"], 1)

if __name__ == "__main__":
    unittest.main()
