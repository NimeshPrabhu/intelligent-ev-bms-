"""
Serial Receiver for ESP32 SuperMini LoRa Receiver via PySerial.
Reads comma-separated telemetry frames: "speed,voltage,current,temperature"
Includes port auto-detection, reconnection with backoff, and packet filtering.
"""

import os
import time
import logging
from typing import Optional, List

logger = logging.getLogger("SerialReceiver")

try:
    import serial
    import serial.tools.list_ports
    PYSERIAL_AVAILABLE = True
except ImportError:
    PYSERIAL_AVAILABLE = False
    logger.warning("pyserial is not installed. Physical serial link unavailable.")

DEFAULT_BAUD = 115200

def list_available_ports() -> List[str]:
    """Returns a list of detected serial device names."""
    if not PYSERIAL_AVAILABLE:
        return []
    ports = serial.tools.list_ports.comports()
    return [p.device for p in ports]

def find_esp32_port() -> Optional[str]:
    """
    Attempts to locate the ESP32 SuperMini serial port.
    Checks environment variable SERIAL_PORT first, then searches active USB CDC / UART ports.
    """
    env_port = os.environ.get("SERIAL_PORT")
    if env_port:
        return env_port

    if not PYSERIAL_AVAILABLE:
        return None

    ports = serial.tools.list_ports.comports()
    if not ports:
        return None

    # Search for common ESP32 / USB UART bridge keywords
    for p in ports:
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        if any(k in desc or k in hwid for k in ("cp210", "ch340", "ch341", "ftdi", "esp32", "usb to uart", "acm", "usb serial")):
            logger.info(f"Auto-detected ESP32 SuperMini on port: {p.device} ({p.description})")
            return p.device

    # Fallback to the first available serial port
    fallback = ports[0].device
    logger.info(f"Defaulting to first detected serial port: {fallback} ({ports[0].description})")
    return fallback

class SerialReceiver:
    def __init__(self, port: Optional[str] = None, baudrate: int = DEFAULT_BAUD, timeout: float = 0.5):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self.is_connected = False

    def connect(self) -> bool:
        """Establishes connection to the serial port."""
        if not PYSERIAL_AVAILABLE:
            return False

        target_port = self.port or find_esp32_port()
        if not target_port:
            logger.warning("No serial port specified or detected for ESP32 SuperMini.")
            self.is_connected = False
            return False

        try:
            self.ser = serial.Serial(
                port=target_port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout
            )
            self.ser.reset_input_buffer()
            self.is_connected = True
            self.port = target_port
            logger.info(f"Successfully connected to ESP32 SuperMini on {target_port} @ {self.baudrate} baud.")
            return True
        except Exception as e:
            logger.warning(f"Failed to open serial port {target_port}: {e}")
            self.is_connected = False
            if self.ser:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
            return False

    def read_packet(self) -> Optional[str]:
        """
        Reads a single newline-terminated line from the serial port.
        Returns the stripped packet string if valid, or None.
        """
        if not self.is_connected or not self.ser:
            return None

        try:
            line_bytes = self.ser.readline()
            if not line_bytes:
                return None

            line = line_bytes.decode("utf-8", errors="ignore").strip()
            # Verify packet has 4 comma-separated values (ignoring boot banners or debug statements)
            if line and line.count(",") == 3:
                return line
            elif line:
                logger.debug(f"Ignoring non-telemetry serial log: {line}")
            return None
        except Exception as e:
            logger.warning(f"Serial read error on {self.port}: {e}. Disconnecting for reconnect.")
            self.close()
            return None

    def close(self):
        """Closes the serial port cleanly."""
        self.is_connected = False
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
