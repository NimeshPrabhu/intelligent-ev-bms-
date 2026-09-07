"""
FastAPI EV Telemetry Server with LoRa Hardware Link, TFLite Inference & WebSocket Broadcast.

Endpoints:
- GET /               : Serves web dashboard
- GET /api/latest     : Most recent reading
- GET /api/history    : Last N readings (default 40)
- WS  /ws             : Live telemetry WebSocket stream

Background Task:
- Receives LoRa packets (or generates simulated data if USE_SIMULATED_HARDWARE = True)
- Runs TFLite LSTM model to estimate SOH and computes live SOC
- Estimates remaining range
- Stores reading in SQLite database
- Broadcasts updates to all connected WebSocket clients
"""

import os
import sys
import time
import json
import math
import pickle
import random
import logging
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import database
from lora_driver import SX1278Driver, LoRaHardwareError
from serial_receiver import SerialReceiver
from fault_detector import evaluate_battery_health

# ================= Configuration =================
USE_SIMULATED_HARDWARE = os.environ.get("USE_SIMULATED_HARDWARE", "True").lower() in ("1", "true", "yes")

# Battery Pack Specifications: 3S 11.1V (3 x 3.7V) 2500mAh (2.5Ah)
NOMINAL_PACK_VOLTAGE = 11.1   # Volts (3S Li-ion nominal)
SERIES_CELLS = 3             # 3 cells in series
PARALLEL_CELLS = 1           # 1P string
NOMINAL_PACK_KWH = 0.02775   # 11.1V * 2.5Ah = 27.75 Wh = 0.02775 kWh
NOMINAL_EFFICIENCY_WH_KM = 22.0  # Wh/km average for light 3S platform

# LoRa Configuration
LORA_FREQ = 433.0             # MHz

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EV-Telemetry")

# ================= TFLite State Estimation Model =================
class SocSohModel:
    def __init__(self, model_path: str = "soc_soh_model.tflite", scaler_path: str = "scaler.pkl", seq_len: int = 100):
        self.seq_len = seq_len
        self.scaler = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.history_buffer = []  # sliding window of [voltage, current, temperature]
        self.last_soh = 98.5
        self.last_soc = 88.0
        self.last_timestamp = time.time()
        self.accumulated_ah = 0.0

        # 1. Load Scaler
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded feature scaler from {scaler_path}")
            except Exception as e:
                logger.warning(f"Failed to load scaler: {e}. Using fallback normalization.")
        else:
            logger.warning(f"Scaler file '{scaler_path}' not found. Defaulting to standard bounds.")

        # 2. Load TFLite Interpreter
        # Try importing tflite_runtime first, then tensorflow
        tflite = None
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                logger.warning("Neither tflite_runtime nor tensorflow.lite available. Running heuristic inference.")

        if tflite and os.path.exists(model_path):
            try:
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                logger.info(f"TFLite model loaded successfully from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to allocate TFLite tensors: {e}. Will use adaptive fallback estimation.")

    def build_features(self, pack_voltage: float, current: float, temperature: float) -> np.ndarray:
        """
        Normalizes input features: [Voltage, Current, Temperature].
        Converts multi-cell pack voltage and pack current to single-cell equivalents matching the trained model's domain.
        """
        cell_voltage = pack_voltage / float(SERIES_CELLS) if SERIES_CELLS > 0 else pack_voltage
        cell_current = current / float(PARALLEL_CELLS) if PARALLEL_CELLS > 0 else current
        raw = np.array([[cell_voltage, cell_current, temperature]], dtype=np.float32)

        if self.scaler is not None:
            try:
                scaled = self.scaler.transform(raw)
                return scaled[0]
            except Exception:
                pass

        # Fallback scaling based on inspected scaler bounds:
        # Voltage: [0.19V, 4.54V] -> ~3.0V - 4.2V nominal
        v_norm = (cell_voltage - 3.0) / (4.2 - 3.0)
        # Current: [-4.04A, 0.03A]
        i_norm = (current - (-4.0)) / (0.0 - (-4.0))
        # Temp: [2.9C, 69.9C]
        t_norm = (temperature - 20.0) / (60.0 - 20.0)
        return np.array([v_norm, i_norm, t_norm], dtype=np.float32)

    def estimate_soc(self, pack_voltage: float, current: float, dt_s: float) -> float:
        """
        Computes State of Charge (SOC) via a hybrid Coulomb Counting + OCV (Open-Circuit Voltage) algorithm.
        """
        cell_v = pack_voltage / float(SERIES_CELLS) if SERIES_CELLS > 0 else pack_voltage

        # OCV-based baseline SOC lookup for Lithium-ion
        # 4.20V -> 100%, 4.05V -> 90%, 3.90V -> 75%, 3.75V -> 50%, 3.60V -> 25%, 3.30V -> 5%, 3.00V -> 0%
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

        # Coulomb Counting update: Current is negative during discharge
        nominal_ah = (NOMINAL_PACK_KWH * 1000.0) / NOMINAL_PACK_VOLTAGE
        delta_ah = (current * dt_s) / 3600.0
        coulomb_soc = self.last_soc + (delta_ah / nominal_ah) * 100.0

        # Complementary filter: 95% Coulomb integration + 5% OCV drift correction
        combined_soc = 0.95 * coulomb_soc + 0.05 * ocv_soc
        combined_soc = max(0.0, min(100.0, combined_soc))
        self.last_soc = combined_soc
        return round(combined_soc, 1)

    def predict_soh(self, pack_voltage: float, current: float, temperature: float) -> float:
        """
        Runs TFLite LSTM inference across the sliding window buffer.
        """
        feats = self.build_features(pack_voltage, current, temperature)
        self.history_buffer.append(feats)
        if len(self.history_buffer) > self.seq_len:
            self.history_buffer.pop(0)

        # If buffer is still warming up, pad with repeats of earliest available sample
        if len(self.history_buffer) < self.seq_len:
            pad_count = self.seq_len - len(self.history_buffer)
            first_elem = self.history_buffer[0]
            seq_array = np.vstack([np.tile(first_elem, (pad_count, 1)), np.array(self.history_buffer)])
        else:
            seq_array = np.array(self.history_buffer)

        # Shape: (1, 100, 3)
        input_tensor = np.expand_dims(seq_array, axis=0).astype(np.float32)

        if self.interpreter is not None:
            try:
                self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
                self.interpreter.invoke()
                out = self.interpreter.get_tensor(self.output_details[0]['index'])
                # Model output is SOH scalar [0.0 - 1.05]
                soh_val = float(out[0][0]) * 100.0
                # Filter smoothing
                self.last_soh = round(0.98 * self.last_soh + 0.02 * soh_val, 1)
                self.last_soh = max(60.0, min(100.0, self.last_soh))
                return self.last_soh
            except Exception as e:
                logger.debug(f"TFLite inference error: {e}. Retaining last SOH estimate.")

        # Heuristic fallback SOH
        return self.last_soh

    def update(self, pack_voltage: float, current: float, temperature: float) -> tuple[float, float]:
        """Processes a new reading and returns (soc, soh)."""
        now = time.time()
        dt_s = max(0.1, min(5.0, now - self.last_timestamp))
        self.last_timestamp = now

        soc = self.estimate_soc(pack_voltage, current, dt_s)
        soh = self.predict_soh(pack_voltage, current, temperature)
        return soc, soh


# ================= Core Telemetry Logic =================
def parse_packet(packet_str: str) -> Optional[Dict[str, float]]:
    """
    Parses a comma-separated packet string: 'speed,voltage,current,temperature'.
    Returns parsed dictionary or None if invalid.
    """
    if not packet_str or not isinstance(packet_str, str):
        return None

    tokens = [t.strip() for t in packet_str.split(",")]
    if len(tokens) != 4:
        return None

    try:
        speed = float(tokens[0])
        voltage = float(tokens[1])
        current = float(tokens[2])
        temperature = float(tokens[3])

        # Validate reasonable physical bounds
        if not (-5.0 <= speed <= 250.0):
            return None
        if not (0.0 <= voltage <= 150.0):
            return None
        if not (-300.0 <= current <= 300.0):
            return None
        if not (-40.0 <= temperature <= 120.0):
            return None

        return {
            "speed": max(0.0, round(speed, 1)),
            "voltage": round(voltage, 2),
            "current": round(current, 2),
            "temperature": round(temperature, 1)
        }
    except ValueError:
        return None

def estimate_range_km(soc: float, soh: float, current: float, speed: float,
                      nominal_kwh: float = NOMINAL_PACK_KWH,
                      nominal_eff_wh_km: float = NOMINAL_EFFICIENCY_WH_KM) -> float:
    """
    Estimates remaining vehicle range in kilometers based on remaining usable energy.
    """
    if soc <= 0.0 or soh <= 0.0:
        return 0.0

    # Usable energy in kWh taking into account cell degradation (SOH) and state of charge (SOC)
    usable_kwh = nominal_kwh * (soh / 100.0) * (soc / 100.0)

    # Dynamic consumption calculation
    if speed > 2.0 and current < -0.5:
        # Instantaneous power: P = V * I (kW)
        power_kw = abs(NOMINAL_PACK_VOLTAGE * current) / 1000.0
        eff_wh_km = (power_kw * 1000.0) / speed
        # Clamp to realistic bounds for 3S light platform
        eff_wh_km = max(8.0, min(60.0, eff_wh_km))
        # Blend with nominal efficiency for smoothing
        blended_eff = 0.7 * nominal_eff_wh_km + 0.3 * eff_wh_km
    else:
        blended_eff = nominal_eff_wh_km

    remaining_km = (usable_kwh * 1000.0) / blended_eff
    return max(0.0, round(remaining_km, 1))


# ================= Hardware Link & Background Worker =================
class HardwareWorker:
    def __init__(self, model: SocSohModel, simulated: bool = USE_SIMULATED_HARDWARE):
        self.model = model
        self.simulated = simulated
        self.is_running = False
        self.active_websockets: List[WebSocket] = []
        self.sim_speed = 32.0
        self.sim_voltage = 50.2
        self.sim_current = -8.5
        self.sim_temp = 29.5

    async def broadcast(self, data: Dict[str, Any]):
        """Sends data to all active WebSocket clients."""
        if not self.active_websockets:
            return
        payload = json.dumps(data)
        dead = []
        for ws in self.active_websockets:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.active_websockets:
                self.active_websockets.remove(ws)

    def generate_simulated_packet(self) -> str:
        """Generates realistic 3S 11.1V 2500mAh telemetry when hardware is not connected."""
        # Speed variation (0 - 45 km/h)
        speed_delta = random.uniform(-2.0, 2.0)
        self.sim_speed = max(0.0, min(45.0, self.sim_speed + speed_delta))

        # Auto-recharge simulation if SOC drops below 15%
        if self.model.last_soc <= 15.0:
            self.model.last_soc = 95.0

        # Current is proportional to speed for 3S 2500mAh pack (0.2A to ~3.8A discharge)
        if self.sim_speed > 1.0:
            target_current = -(0.5 + (self.sim_speed / 12.0) + random.uniform(-0.2, 0.2))
        else:
            target_current = random.uniform(-0.25, -0.05) # Idle draw
        self.sim_current = round(0.8 * self.sim_current + 0.2 * target_current, 2)

        # 3S Pack Voltage (12.6V peak to 9.6V cutoff)
        base_v = 11.1 + ((self.model.last_soc - 50.0) * 0.022)
        voltage_sag = abs(self.sim_current) * 0.08
        self.sim_voltage = round(base_v - voltage_sag + random.uniform(-0.02, 0.02), 2)

        # Temperature increases slowly under load
        self.sim_temp = round(self.sim_temp + (abs(self.sim_current) * 0.002) - 0.003 + random.uniform(-0.02, 0.02), 1)
        self.sim_temp = max(24.0, min(42.0, self.sim_temp))

        return f"{self.sim_speed:.1f},{self.sim_voltage:.2f},{self.sim_current:.2f},{self.sim_temp:.1f}"

    async def run(self):
        """Continuous receiver loop using PySerial for ESP32 SuperMini with auto-reconnect."""
        self.is_running = True
        logger.info(f"Hardware worker starting (Simulated: {self.simulated})")

        serial_receiver = None
        reconnect_delay = 1.0

        while self.is_running:
            raw_packet = None

            if self.simulated:
                await asyncio.sleep(1.0)
                raw_packet = self.generate_simulated_packet()
            else:
                # Physical ESP32 SuperMini over USB PySerial
                if serial_receiver is None or not serial_receiver.is_connected:
                    try:
                        logger.info("Probing serial connection to ESP32 SuperMini...")
                        serial_receiver = SerialReceiver()
                        if serial_receiver.connect():
                            logger.info(f"Connected to ESP32 SuperMini on {serial_receiver.port}")
                            reconnect_delay = 1.0
                        else:
                            await asyncio.sleep(reconnect_delay)
                            reconnect_delay = min(reconnect_delay * 1.5, 10.0)
                            continue
                    except Exception as e:
                        logger.warning(f"Serial connection error: {e}. Retrying in {reconnect_delay:.1f}s...")
                        await asyncio.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 1.5, 10.0)
                        continue

                # Read line from serial
                raw_packet = serial_receiver.read_packet()
                if raw_packet is None:
                    await asyncio.sleep(0.02)
                    continue

            # Process packet
            parsed = parse_packet(raw_packet)
            if not parsed:
                logger.warning(f"Malformed packet received: '{raw_packet}'")
                continue

            # State Estimation & Range
            soc, soh = self.model.update(
                parsed["voltage"],
                parsed["current"],
                parsed["temperature"]
            )
            range_km = estimate_range_km(
                soc=soc,
                soh=soh,
                current=parsed["current"],
                speed=parsed["speed"]
            )

            # Evaluate threshold-based safety faults
            health = evaluate_battery_health(
                voltage=parsed["voltage"],
                current=parsed["current"],
                temperature=parsed["temperature"]
            )

            record = {
                "timestamp": time.time(),
                "speed": parsed["speed"],
                "voltage": parsed["voltage"],
                "current": parsed["current"],
                "temperature": parsed["temperature"],
                "soc": soc,
                "soh": soh,
                "range_km": range_km,
                "health_status": health["status"],   # "NORMAL" | "WARNING" | "CRITICAL"
                "alerts": health["alerts"]          # List of active threshold fault alerts
            }

            # Persist core fields to SQLite
            try:
                database.insert_reading(record)
            except Exception as e:
                logger.error(f"Database insertion failed: {e}")

            # Broadcast over WebSocket
            await self.broadcast(record)

        if serial_receiver:
            serial_receiver.close()


# ================= FastAPI Application =================
model_instance = SocSohModel()
worker = HardwareWorker(model=model_instance, simulated=USE_SIMULATED_HARDWARE)

@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    bg_task = asyncio.create_task(worker.run())
    yield
    worker.is_running = False
    bg_task.cancel()
    try:
        await bg_task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="EV Telemetry Service", lifespan=lifespan)

# Mount static folder
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    dashboard_path = STATIC_DIR / "dashboard.html"
    if dashboard_path.exists():
        return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Dashboard file static/dashboard.html not found.</h2>", status_code=404)

@app.get("/api/latest")
async def api_latest():
    reading = database.get_latest_reading()
    if reading:
        health = evaluate_battery_health(reading["voltage"], reading["current"], reading["temperature"])
        reading["health_status"] = health["status"]
        reading["alerts"] = health["alerts"]
        return JSONResponse(content=reading)
    return JSONResponse(content={"message": "No telemetry readings available yet"}, status_code=200)

@app.get("/api/history")
async def api_history(limit: int = Query(default=40, ge=1, le=500)):
    history = database.get_history(limit=limit)
    for item in history:
        health = evaluate_battery_health(item["voltage"], item["current"], item["temperature"])
        item["health_status"] = health["status"]
        item["alerts"] = health["alerts"]
    return JSONResponse(content=history)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    worker.active_websockets.append(websocket)
    try:
        # Send latest reading immediately on connection
        latest = database.get_latest_reading()
        if latest:
            health = evaluate_battery_health(latest["voltage"], latest["current"], latest["temperature"])
            latest["health_status"] = health["status"]
            latest["alerts"] = health["alerts"]
            await websocket.send_text(json.dumps(latest))
        while True:
            # Keepalive / handle any client pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in worker.active_websockets:
            worker.active_websockets.remove(websocket)
    except Exception:
        if websocket in worker.active_websockets:
            worker.active_websockets.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    print("Starting EV Telemetry Service on http://0.0.0.0:8000 ...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
