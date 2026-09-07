"""
Test Suite for EV Telemetry System
Unit tests: parse_packet, estimate_range_km, SocSohModel feature pipeline
Integration tests: SQLite persistence, FastAPI endpoints, WebSocket data flow
"""

import os
import time
import tempfile
import pytest
from fastapi.testclient import TestClient

from main import app, parse_packet, estimate_range_km, SocSohModel
import database

# ================= Unit Tests: parse_packet =================

def test_parse_packet_valid():
    raw = "34.5,48.20,-12.30,29.8"
    parsed = parse_packet(raw)
    assert parsed is not None
    assert parsed["speed"] == 34.5
    assert parsed["voltage"] == 48.20
    assert parsed["current"] == -12.30
    assert parsed["temperature"] == 29.8

def test_parse_packet_with_whitespace():
    raw = "  22.1 ,  51.4  , -4.50 ,  31.2  "
    parsed = parse_packet(raw)
    assert parsed is not None
    assert parsed["speed"] == 22.1
    assert parsed["voltage"] == 51.40
    assert parsed["current"] == -4.50
    assert parsed["temperature"] == 31.2

def test_parse_packet_malformed():
    # Missing fields
    assert parse_packet("34.5,48.2") is None
    # Extra fields
    assert parse_packet("34.5,48.2,-10.0,25.0,99.9") is None
    # Non-numeric
    assert parse_packet("fast,48.2,-10.0,25.0") is None
    # Empty & None
    assert parse_packet("") is None
    assert parse_packet(None) is None

def test_parse_packet_out_of_bounds():
    # Impossible speed (> 250 km/h)
    assert parse_packet("400.0,48.0,-10.0,25.0") is None
    # Negative voltage (< 0V)
    assert parse_packet("25.0,-10.0,-10.0,25.0") is None
    # Extreme temperature (> 120C)
    assert parse_packet("25.0,48.0,-10.0,150.0") is None


# ================= Unit Tests: estimate_range_km =================

def test_estimate_range_zero_battery():
    assert estimate_range_km(soc=0.0, soh=95.0, current=-10.0, speed=30.0) == 0.0
    assert estimate_range_km(soc=80.0, soh=0.0, current=-10.0, speed=30.0) == 0.0

def test_estimate_range_normal_cruise():
    # 3S 27.75 Wh pack (0.02775 kWh), 80% SOC, 95% SOH -> ~21 Wh usable
    # At ~22 Wh/km nominal -> ~1.0 km
    range_km = estimate_range_km(soc=80.0, soh=95.0, current=-2.0, speed=20.0)
    assert 0.5 <= range_km <= 3.0

def test_estimate_range_stationary():
    # Speed 0 (idle) falls back to nominal consumption
    range_km = estimate_range_km(soc=100.0, soh=100.0, current=-0.2, speed=0.0)
    assert range_km > 0.0
    # 27.75 Wh / 22 Wh/km = ~1.3 km
    assert 0.8 <= range_km <= 2.5


# ================= Unit Tests: Fault Detection Engine =================

from fault_detector import evaluate_battery_health

def test_fault_detector_nominal():
    health = evaluate_battery_health(voltage=11.2, current=-2.0, temperature=28.0)
    assert health["status"] == "NORMAL"
    assert len(health["alerts"]) == 0

def test_fault_detector_undervoltage():
    # Warning < 9.9V
    h_warn = evaluate_battery_health(voltage=9.5, current=-1.5, temperature=27.0)
    assert h_warn["status"] == "WARNING"
    assert any(a["type"] == "UNDERVOLTAGE" for a in h_warn["alerts"])

    # Critical < 9.0V
    h_crit = evaluate_battery_health(voltage=8.7, current=-1.5, temperature=27.0)
    assert h_crit["status"] == "CRITICAL"
    assert any(a["type"] == "UNDERVOLTAGE" and a["severity"] == "CRITICAL" for a in h_crit["alerts"])

def test_fault_detector_overvoltage():
    # Warning > 12.4V
    h_warn = evaluate_battery_health(voltage=12.5, current=1.0, temperature=27.0)
    assert h_warn["status"] == "WARNING"
    assert any(a["type"] == "OVERVOLTAGE" for a in h_warn["alerts"])

    # Critical > 12.65V
    h_crit = evaluate_battery_health(voltage=12.8, current=1.0, temperature=27.0)
    assert h_crit["status"] == "CRITICAL"
    assert any(a["type"] == "OVERVOLTAGE" and a["severity"] == "CRITICAL" for a in h_crit["alerts"])

def test_fault_detector_overcurrent():
    # Discharge Warning > 5.0A (<-5.0A)
    h_warn = evaluate_battery_health(voltage=11.0, current=-6.0, temperature=30.0)
    assert h_warn["status"] == "WARNING"
    assert any(a["type"] == "OVERCURRENT" for a in h_warn["alerts"])

    # Discharge Critical > 7.5A (<-7.5A)
    h_crit = evaluate_battery_health(voltage=11.0, current=-8.5, temperature=30.0)
    assert h_crit["status"] == "CRITICAL"
    assert any(a["type"] == "OVERCURRENT" and a["severity"] == "CRITICAL" for a in h_crit["alerts"])

def test_fault_detector_overtemperature():
    # Thermal Warning > 45C
    h_warn = evaluate_battery_health(voltage=11.1, current=-1.5, temperature=48.0)
    assert h_warn["status"] == "WARNING"
    assert any(a["type"] == "OVERTEMP" for a in h_warn["alerts"])

    # Thermal Critical > 55C
    h_crit = evaluate_battery_health(voltage=11.1, current=-1.5, temperature=58.0)
    assert h_crit["status"] == "CRITICAL"
    assert any(a["type"] == "OVERTEMP" and a["severity"] == "CRITICAL" for a in h_crit["alerts"])


# ================= Unit Tests: SocSohModel Feature Order =================

def test_soc_soh_model_feature_order():
    model = SocSohModel(model_path="soc_soh_model.tflite", scaler_path="scaler.pkl")
    # Feed test values: 11.1V pack (3 cells = 3.7V cell), -2A current, 28C temp
    feats = model.build_features(pack_voltage=11.1, current=-2.0, temperature=28.0)
    assert len(feats) == 3
    assert not any(map(float_is_nan_or_inf, feats))

    # Test update returns reasonable SOC and SOH
    soc, soh = model.update(pack_voltage=11.1, current=-2.0, temperature=28.0)
    assert 0.0 <= soc <= 100.0
    assert 50.0 <= soh <= 105.0

def float_is_nan_or_inf(val):
    import math
    return math.isnan(val) or math.isinf(val)


# ================= Integration Tests: SQLite Persistence & API =================

def test_database_operations(tmp_path):
    temp_db = str(tmp_path / "test_readings.db")

    database.init_db(temp_db)
    # Verify empty initial state
    assert database.get_latest_reading(temp_db) is None
    assert database.get_history(limit=10, db_path=temp_db) == []

    # Insert record
    rec = {
        "timestamp": time.time(),
        "speed": 28.5,
        "voltage": 49.1,
        "current": -6.2,
        "temperature": 27.4,
        "soc": 85.0,
        "soh": 97.2,
        "range_km": 44.3
    }
    row_id = database.insert_reading(rec, db_path=temp_db)
    assert row_id is not None and row_id > 0

    latest = database.get_latest_reading(temp_db)
    assert latest is not None
    assert latest["speed"] == 28.5
    assert latest["voltage"] == 49.1
    assert latest["soc"] == 85.0

    history = database.get_history(limit=5, db_path=temp_db)
    assert len(history) == 1
    assert history[0]["id"] == row_id


def test_api_and_websocket_integration():
    with TestClient(app) as client:
        # 1. Test Dashboard HTML GET /
        resp = client.get("/")
        assert resp.status_code == 200
        assert "EV Telemetry" in resp.text
        assert "Vehicle Speed" in resp.text or "Speedometer" in resp.text
        assert "SOC vs SOH" in resp.text

        # 2. Test GET /api/latest
        resp_latest = client.get("/api/latest")
        assert resp_latest.status_code == 200
        if "voltage" in resp_latest.json():
            assert "health_status" in resp_latest.json()
            assert "alerts" in resp_latest.json()

        # 3. Test GET /api/history
        resp_hist = client.get("/api/history?limit=10")
        assert resp_hist.status_code == 200
        assert isinstance(resp_hist.json(), list)

        # 4. Test WebSocket Endpoint /ws
        with client.websocket_connect("/ws") as websocket:
            # Client should receive initial or streamed packet
            msg = websocket.receive_text()
            assert msg is not None
            import json
            data = json.loads(msg)
            assert "speed" in data
            assert "voltage" in data
            assert "soc" in data
            assert "soh" in data
            assert "health_status" in data
            assert "alerts" in data

