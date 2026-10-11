"""
FastAPI EV Battery Management System & Telemetry Server.
Scalable, modular, and optimized for Raspberry Pi 3B+.

Endpoints:
- GET  /                  : Serves real-time IoT dashboard
- GET  /api/latest        : Most recent telemetry reading with health & relay state
- GET  /api/history       : Paginated historical readings (supports time_range filter)
- GET  /api/faults        : Recent safety audit logs and alarms
- GET  /api/diagnostics   : Server, database, and telemetry pipeline diagnostics
- POST /api/control/relay : Manual relay control and Emergency Stop (E-Stop)
- WS   /ws                : High-performance live WebSocket broadcast
"""

import os
import sys
import time
import json
import random
import logging
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Header, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config import (
    USE_SIMULATED_HARDWARE,
    STATIC_DIR,
    ADMIN_API_KEY
)
import database
from fault_detector import evaluate_battery_health
from services.telemetry_service import telemetry_service
from services.ml_service import ml_service
from serial_receiver import SerialReceiver

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EV-BMS")

# ================= Background Ingestion Worker =================
class BMSWorker:
    def __init__(self):
        self.simulated = USE_SIMULATED_HARDWARE
        self.is_running = False
        self.active_websockets: List[WebSocket] = []
        self.start_time = time.time()
        self.sim_speed = 28.0
        self.sim_voltage = 11.4
        self.sim_current = 2.4   # Positive is discharge
        self.sim_temp = 31.0
        self.sim_relay = 1

    async def broadcast(self, data: Dict[str, Any]):
        """Broadcasts telemetry payload concurrently to all connected WebSocket clients."""
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

    def generate_simulated_packet(self) -> Dict[str, Any]:
        """Generates realistic telemetry matching 3S 11.1V 2500mAh battery dynamics."""
        # Speed dynamics
        delta_spd = random.uniform(-1.5, 1.5)
        self.sim_speed = max(0.0, min(50.0, self.sim_speed + delta_spd))

        # Current dynamics: higher speed -> higher discharge current
        if self.sim_speed > 1.0:
            target_i = 0.5 + (self.sim_speed / 14.0) * 3.5 + random.uniform(-0.15, 0.15)
        else:
            target_i = random.uniform(0.1, 0.3)  # Idle draw
        self.sim_current = round(0.85 * self.sim_current + 0.15 * target_i, 2)

        # Voltage sag under load: base ~11.2V
        base_v = 11.1 + ((telemetry_service.last_soc - 50.0) * 0.022)
        sag = self.sim_current * 0.09
        self.sim_voltage = round(base_v - sag + random.uniform(-0.02, 0.02), 2)

        # Temperature increases slowly under load
        self.sim_temp = round(self.sim_temp + (self.sim_current * 0.003) - 0.004 + random.uniform(-0.02, 0.02), 1)
        self.sim_temp = max(24.0, min(45.0, self.sim_temp))

        # Power = V * I (Watts)
        power_w = round(self.sim_voltage * self.sim_current, 2)

        return {
            "ver": 1,
            "seq": telemetry_service.last_seq + 1,
            "uptime_ms": int((time.time() - self.start_time) * 1000),
            "v": self.sim_voltage,
            "i": self.sim_current,
            "p": power_w,
            "temp": self.sim_temp,
            "speed": round(self.sim_speed, 1),
            "relay": self.sim_relay,
            "faults": 0,
            "source": "sim"
        }

    async def run(self):
        """Continuous hardware/simulation reception loop."""
        self.is_running = True
        logger.info(f"BMS Ingestion Worker running (Simulated: {self.simulated})")

        serial_receiver = None
        reconnect_delay = 1.0
        last_calc_time = time.time()

        while self.is_running:
            raw_data = None

            if self.simulated:
                await asyncio.sleep(1.0)
                raw_data = self.generate_simulated_packet()
            else:
                # Real hardware reading over USB Serial / LoRa
                if serial_receiver is None or not serial_receiver.is_connected:
                    try:
                        serial_receiver = SerialReceiver()
                        if serial_receiver.connect():
                            logger.info(f"Connected to ESP2 Gateway on {serial_receiver.port}")
                            reconnect_delay = 1.0
                        else:
                            await asyncio.sleep(reconnect_delay)
                            reconnect_delay = min(reconnect_delay * 1.5, 8.0)
                            continue
                    except Exception as e:
                        logger.warning(f"Connection failed: {e}. Retrying in {reconnect_delay}s")
                        await asyncio.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 1.5, 8.0)
                        continue

                raw_line = serial_receiver.read_packet()
                if not raw_line:
                    await asyncio.sleep(0.02)
                    continue
                raw_data = raw_line

            # Parse & Validate Packet
            parsed = telemetry_service.parse_packet(raw_data)
            if not parsed:
                continue

            now = time.time()
            dt_s = max(0.1, min(5.0, now - last_calc_time))
            last_calc_time = now

            # State of Charge & State of Health Estimation
            soc = telemetry_service.estimate_soc(parsed["voltage"], parsed["current"], dt_s)
            soh = ml_service.predict_soh(parsed["voltage"], parsed["current"], parsed["temperature"])
            range_km = telemetry_service.estimate_range_km(soc, soh, parsed["current"], parsed["speed"])

            # Safety & Fault Evaluation with Hysteresis
            health = evaluate_battery_health(parsed["voltage"], parsed["current"], parsed["temperature"], now=now)

            # Auto-trip relay if critical fault detected
            effective_relay = parsed["relay_state"]
            if not health["relay_recommended"]:
                effective_relay = 0
            if telemetry_service.manual_relay_override is not None:
                effective_relay = telemetry_service.manual_relay_override

            record = {
                "timestamp": now,
                "seq": parsed.get("seq", 0),
                "speed": parsed["speed"],
                "voltage": parsed["voltage"],
                "current": parsed["current"],
                "power": parsed["power"],
                "temperature": parsed["temperature"],
                "soc": soc,
                "soh": soh,
                "range_km": range_km,
                "relay_state": effective_relay,
                "fault_mask": parsed.get("fault_mask", 0),
                "source": parsed.get("source", "sim" if self.simulated else "hw"),
                "is_stale": False,
                "health_status": health["status"],
                "alerts": health["alerts"]
            }

            # Persist to SQLite
            try:
                database.insert_reading(record)
            except Exception as e:
                logger.error(f"Failed to persist reading: {e}")

            # Broadcast over live WebSocket
            await self.broadcast(record)

        if serial_receiver:
            serial_receiver.close()

bms_worker = BMSWorker()

# ================= FastAPI Application Lifecycle =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    bg_task = asyncio.create_task(bms_worker.run())
    yield
    bms_worker.is_running = False
    bg_task.cancel()
    try:
        await bg_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="Intelligent Battery Management System",
    description="Real-time EV Telemetry, TFLite LSTM State Estimation & Hardware Safety Cutoff",
    version="2.0.0",
    lifespan=lifespan
)

# Static Files
static_path = Path(STATIC_DIR)
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# ================= REST Endpoints =================

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    dashboard_file = static_path / "dashboard.html"
    if dashboard_file.exists():
        return HTMLResponse(content=dashboard_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Dashboard file static/dashboard.html not found.</h2>", status_code=404)

@app.get("/api/latest")
async def api_latest():
    """Returns the most recent reading, safety health, and stale indicator."""
    reading = database.get_latest_reading()
    if reading:
        now = time.time()
        health = evaluate_battery_health(reading["voltage"], reading["current"], reading["temperature"], now=now)
        reading["health_status"] = health["status"]
        reading["alerts"] = health["alerts"]
        reading["is_stale"] = telemetry_service.is_stale(now)
        if telemetry_service.manual_relay_override is not None:
            reading["relay_state"] = telemetry_service.manual_relay_override
        return JSONResponse(content=reading)
    return JSONResponse(content={"message": "No telemetry readings available yet"}, status_code=200)

@app.get("/api/history")
async def api_history(
    limit: int = Query(default=40, ge=1, le=500),
    page: int = Query(default=1, ge=1),
    time_range: Optional[int] = Query(default=None, ge=10, le=86400, description="Window in seconds (e.g. 60, 300, 900)")
):
    """Fetches paginated telemetry history with optional time window filtering."""
    history = database.get_history(limit=limit, page=page, time_range_s=time_range)
    return JSONResponse(content=history)

@app.get("/api/faults")
async def api_faults(limit: int = Query(default=20, ge=1, le=100)):
    """Retrieves recent safety alarm audit events."""
    events = database.get_recent_fault_events(limit=limit)
    return JSONResponse(content=events)

@app.get("/api/diagnostics")
async def api_diagnostics():
    """Returns operational diagnostics for Raspberry Pi monitoring."""
    stats = database.get_telemetry_stats()
    return JSONResponse(content={
        "status": "ONLINE",
        "simulated_mode": bms_worker.simulated,
        "active_ws_clients": len(bms_worker.active_websockets),
        "packets_processed": telemetry_service.packet_count,
        "dropped_packets": telemetry_service.dropped_packets,
        "is_stale": telemetry_service.is_stale(),
        "manual_relay_override": telemetry_service.manual_relay_override,
        "database": stats,
        "uptime_s": round(time.time() - bms_worker.start_time, 1)
    })

@app.post("/api/control/relay")
async def api_control_relay(
    action: str = Query(..., pattern="^(trip|reset|auto)$", description="Action: 'trip' (cut power), 'reset' (restore), or 'auto'"),

    x_api_key: Optional[str] = Header(None)
):
    """
    Emergency Stop (E-Stop) and Relay Safety Control Endpoint.
    Requires header: X-Api-Key matching ADMIN_API_KEY (or allows local dev reset).
    """
    if x_api_key != ADMIN_API_KEY and x_api_key != "local-dev":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Api-Key for relay safety control."
        )

    if action == "trip":
        telemetry_service.manual_relay_override = 0
        bms_worker.sim_relay = 0
        msg = "Emergency Stop engaged: Relay tripped (Motor power cut)."
        logger.warning(msg)
    elif action == "reset":
        telemetry_service.manual_relay_override = 1
        bms_worker.sim_relay = 1
        msg = "Relay reset command issued: Relay re-energized."
        logger.info(msg)
    else:
        telemetry_service.manual_relay_override = None
        msg = "Relay returned to autonomous automatic control."
        logger.info(msg)

    return JSONResponse(content={
        "success": True,
        "action": action,
        "relay_state": telemetry_service.manual_relay_override if telemetry_service.manual_relay_override is not None else 1,
        "message": msg
    })

# ================= WebSocket Stream =================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    bms_worker.active_websockets.append(websocket)
    try:
        # Push initial state immediately
        latest = database.get_latest_reading()
        if latest:
            now = time.time()
            health = evaluate_battery_health(latest["voltage"], latest["current"], latest["temperature"], now=now)
            latest["health_status"] = health["status"]
            latest["alerts"] = health["alerts"]
            latest["is_stale"] = telemetry_service.is_stale(now)
            if telemetry_service.manual_relay_override is not None:
                latest["relay_state"] = telemetry_service.manual_relay_override
            await websocket.send_text(json.dumps(latest))

        while True:
            # Keepalive ping/pong
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in bms_worker.active_websockets:
            bms_worker.active_websockets.remove(websocket)
    except Exception:
        if websocket in bms_worker.active_websockets:
            bms_worker.active_websockets.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    print("Starting Intelligent EV BMS on http://0.0.0.0:8000 ...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
