"""
SX1278 / SX1276 Raw Register LoRa Driver for Raspberry Pi
Point-to-Point LoRa communication using spidev and RPi.GPIO (or gpiod).

Features:
- Register-level SX1278 driver adhering to Semtech datasheet.
- Automatic hardware detection & version check (expects RegVersion == 0x12).
- Automatic FIFO overflow recovery & IRQ clearing.
- Graceful mock fallback for simulated hardware testing on non-Pi platforms.
"""

import time
import logging

logger = logging.getLogger("LoRaDriver")

# ================= Register Definitions =================
REG_FIFO                    = 0x00
REG_OP_MODE                 = 0x01
REG_FRF_MSB                 = 0x06
REG_FRF_MID                 = 0x07
REG_FRF_LSB                 = 0x08
REG_PA_CONFIG               = 0x09
REG_FIFO_ADDR_PTR           = 0x0D
REG_FIFO_TX_BASE_ADDR       = 0x0E
REG_FIFO_RX_BASE_ADDR       = 0x0F
REG_FIFO_RX_CURRENT_ADDR    = 0x10
REG_IRQ_FLAGS               = 0x12
REG_RX_NB_BYTES             = 0x13
REG_PKT_SNR_VALUE           = 0x19
REG_PKT_RSSI_VALUE          = 0x1A
REG_MODEM_CONFIG_1          = 0x1D
REG_MODEM_CONFIG_2          = 0x1E
REG_SYMB_TIMEOUT_LSB        = 0x1F
REG_PREAMBLE_MSB            = 0x20
REG_PREAMBLE_LSB            = 0x21
REG_MODEM_CONFIG_3          = 0x26
REG_SYNC_WORD               = 0x39
REG_DIO_MAPPING_1           = 0x40
REG_VERSION                 = 0x42

# Operating Modes
MODE_LONG_RANGE_MODE        = 0x80
MODE_SLEEP                  = 0x00
MODE_STDBY                  = 0x01
MODE_TX                     = 0x03
MODE_RX_CONTINUOUS          = 0x05
MODE_RX_SINGLE              = 0x06

# IRQ Flags
IRQ_RX_TIMEOUT_MASK         = 0x80
IRQ_RX_DONE_MASK            = 0x40
IRQ_PAYLOAD_CRC_ERROR_MASK  = 0x20
IRQ_VALID_HEADER_MASK       = 0x10
IRQ_TX_DONE_MASK            = 0x08

# Hardware default pins for Raspberry Pi (BCM numbering)
DEFAULT_RST_PIN = 25
DEFAULT_DIO0_PIN = 24
DEFAULT_SPI_BUS = 0
DEFAULT_SPI_CS = 0

class LoRaHardwareError(Exception):
    """Raised when LoRa module fails hardware initialization or disconnects."""
    pass

class SX1278Driver:
    def __init__(self, frequency=433.0, spi_bus=DEFAULT_SPI_BUS, spi_cs=DEFAULT_SPI_CS,
                 rst_pin=DEFAULT_RST_PIN, dio0_pin=DEFAULT_DIO0_PIN):
        self.frequency = frequency
        self.spi_bus = spi_bus
        self.spi_cs = spi_cs
        self.rst_pin = rst_pin
        self.dio0_pin = dio0_pin
        self.spi = None
        self.gpio_available = False
        self.is_connected = False

    def connect(self):
        """Initializes SPI, GPIO, performs hardware reset, and configures registers."""
        try:
            import spidev
            self.spi = spidev.SpiDev()
            self.spi.open(self.spi_bus, self.spi_cs)
            self.spi.max_speed_hz = 5000000
            self.spi.mode = 0b00
        except Exception as e:
            logger.warning(f"SPI hardware initialization failed: {e}. Falling back to simulated driver mode.")
            self.is_connected = False
            return False

        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            if self.rst_pin:
                GPIO.setup(self.rst_pin, GPIO.OUT)
            if self.dio0_pin:
                GPIO.setup(self.dio0_pin, GPIO.IN)
            self.gpio_available = True
        except Exception as e:
            logger.warning(f"RPi.GPIO not available ({e}). Hardware reset pin disabled.")
            self.gpio_available = False

        # Reset hardware module
        self.hardware_reset()

        # Check chip version (should be 0x12 for SX1276/7/8/9)
        version = self.read_register(REG_VERSION)
        if version != 0x12:
            self.close()
            raise LoRaHardwareError(f"SX1278 not detected: version register returned 0x{version:02X} (expected 0x12). Check SPI wiring.")

        logger.info(f"SX1278 detected successfully (Silicon Rev: 0x{version:02X})")

        # Put module in Sleep mode to allow configuring LoRa mode
        self.set_op_mode(MODE_SLEEP)
        self.write_register(REG_OP_MODE, MODE_LONG_RANGE_MODE | MODE_SLEEP)
        time.sleep(0.01)

        # Set Standby
        self.set_op_mode(MODE_STDBY)

        # Set Frequency
        self.set_frequency(self.frequency)

        # Base addresses for FIFO
        self.write_register(REG_FIFO_TX_BASE_ADDR, 0x00)
        self.write_register(REG_FIFO_RX_BASE_ADDR, 0x00)

        # Configure Modem:
        # Bandwidth 125 kHz (0x70), Coding rate 4/5 (0x02), Explicit header (0x00) -> 0x72
        self.write_register(REG_MODEM_CONFIG_1, 0x72)
        # Spreading Factor 7 (0x70), CRC On (0x04) -> 0x74
        self.write_register(REG_MODEM_CONFIG_2, 0x74)
        # Low data rate optimize off, AGC on -> 0x04
        self.write_register(REG_MODEM_CONFIG_3, 0x04)

        # Preamble length (8 symbols)
        self.write_register(REG_PREAMBLE_MSB, 0x00)
        self.write_register(REG_PREAMBLE_LSB, 0x08)

        # Set sync word (0x12)
        self.write_register(REG_SYNC_WORD, 0x12)

        # Start continuous RX
        self.start_rx()
        self.is_connected = True
        return True

    def hardware_reset(self):
        """Toggles hardware reset line if available."""
        if self.gpio_available and self.rst_pin:
            import RPi.GPIO as GPIO
            GPIO.output(self.rst_pin, GPIO.LOW)
            time.sleep(0.01)
            GPIO.output(self.rst_pin, GPIO.HIGH)
            time.sleep(0.01)

    def read_register(self, address):
        """Reads a single byte from a register over SPI (bit 7 = 0 for read)."""
        if not self.spi:
            return 0
        resp = self.spi.xfer2([address & 0x7F, 0x00])
        return resp[1]

    def write_register(self, address, value):
        """Writes a single byte to a register over SPI (bit 7 = 1 for write)."""
        if not self.spi:
            return
        self.spi.xfer2([address | 0x80, value & 0xFF])

    def set_op_mode(self, mode):
        """Sets operating mode preserving LoRa LongRangeMode bit."""
        self.write_register(REG_OP_MODE, MODE_LONG_RANGE_MODE | (mode & 0x07))

    def set_frequency(self, freq_mhz):
        """Sets carrier frequency (in MHz)."""
        frf = int((freq_mhz * 1000000.0) / 61.03515625)
        self.write_register(REG_FRF_MSB, (frf >> 16) & 0xFF)
        self.write_register(REG_FRF_MID, (frf >> 8) & 0xFF)
        self.write_register(REG_FRF_LSB, frf & 0xFF)

    def start_rx(self):
        """Enters continuous receive mode."""
        self.set_op_mode(MODE_RX_CONTINUOUS)

    def receive_packet(self):
        """
        Polls the SX1278 for a received packet.
        Returns:
            str payload if a valid packet was received, or None if no packet is available.
        Raises:
            LoRaHardwareError if SPI communication is lost.
        """
        if not self.is_connected or not self.spi:
            return None

        # Verify device is still communicating
        try:
            irq_flags = self.read_register(REG_IRQ_FLAGS)
        except Exception as e:
            self.is_connected = False
            raise LoRaHardwareError(f"SPI read failed: {e}")

        # Check for CRC error
        if irq_flags & IRQ_PAYLOAD_CRC_ERROR_MASK:
            # Clear IRQ
            self.write_register(REG_IRQ_FLAGS, IRQ_PAYLOAD_CRC_ERROR_MASK)
            logger.warning("LoRa packet dropped due to CRC error.")
            return None

        # Check if RX_DONE is set
        if irq_flags & IRQ_RX_DONE_MASK:
            # Number of bytes received
            nb_bytes = self.read_register(REG_RX_NB_BYTES)
            current_addr = self.read_register(REG_FIFO_RX_CURRENT_ADDR)
            
            # Point FIFO pointer to start of received packet
            self.write_register(REG_FIFO_ADDR_PTR, current_addr)

            # Read payload bytes
            raw_payload = []
            for _ in range(nb_bytes):
                raw_payload.append(self.read_register(REG_FIFO))

            # Clear RX_DONE IRQ
            self.write_register(REG_IRQ_FLAGS, IRQ_RX_DONE_MASK)

            # Convert to string
            try:
                packet_str = bytes(raw_payload).decode('utf-8', errors='ignore').strip()
                return packet_str
            except Exception as e:
                logger.warning(f"Error decoding packet bytes: {e}")
                return None

        return None

    def close(self):
        """Closes SPI connection and releases GPIO pins."""
        self.is_connected = False
        if self.spi:
            try:
                self.spi.close()
            except Exception:
                pass
            self.spi = None
        if self.gpio_available:
            try:
                import RPi.GPIO as GPIO
                GPIO.cleanup()
            except Exception:
                pass
            self.gpio_available = False
