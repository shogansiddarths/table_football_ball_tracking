import subprocess
from threading import Thread, Lock
from time import sleep
import random

class BLESender:
    def __init__(self):
        # Latest values
        self._x = 0
        self._y = 0
        self._possession = 0

        # BLE initialization status flag, 0 = not initialized, 1 = initializing, 2 = initialized
        self._initialized_status = 0
        # Sender loop running flag
        self._running = False

        # Thread to run sender loop
        self._thread = None
        # Lock for thread-safe access to latest values
        self._lock = Lock()

    # Run the given command
    def _run_cmd(self, cmd):
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    # Convert a 16-bit integer to big-endian byte pair
    def _to_big_endian(self, val):
        # Get 8 Most Significant Bits
        high = (val >> 8) & 0xFF
        # Get 8 Least Significant Bits
        low = val & 0xFF

        return high, low

    # Convert a value to a 2-digit hex string
    def _to_hex(self, val):
        return f"{val & 0xFF:02X}"

    # Build payload from x, y, possession
    def _build_payload(self, x, y, possession):
        # Convert x and y to big-endian byte pairs, as it would not fit a single byte
        x_high, x_low = self._to_big_endian(x)
        y_high, y_low = self._to_big_endian(y)

        # Build payload data in hex
        payload = [
            self._to_hex(x_high),
            self._to_hex(x_low),
            self._to_hex(y_high),
            self._to_hex(y_low),
            self._to_hex(possession)
        ]

        return payload

    # Build full hcitool command from payload
    def _build_cmd(self, payload):
        # Use hcitool to send raw BLE advertising packet
        cmd = "sudo hcitool -i hci0 cmd 0x08 0x0008 "

        # Add metadata
        cmd += "1F 02 01 1A 1A FF 00 00 "

        # Add payload
        cmd += " ".join(payload)

        # Calculate remaining padding to add to reach 24 bytes in total
        remaining_padding = 24 - len(payload)

        # Add padding if needed
        if remaining_padding > 0:
            padding = ["00"] * remaining_padding

            cmd += " " + " ".join(padding)

        return cmd

    # Initialize BLE interface
    def _initialize_ble(self):
        # Mark as initializing
        self._initialized_status = 1

        # Turn off BLE interface briefly to reset
        self._run_cmd("sudo hciconfig hci0 down")
        sleep(0.5)

        # Turn it back on
        self._run_cmd("sudo hciconfig hci0 up")
        sleep(0.5)

        # Enable advertising        
        self._run_cmd("sudo hciconfig hci0 leadv 0")

        # Mark as initialized
        self._initialized_status = 2

    # Sending loop
    def _sender_loop(self):
        self._running = True

        # Initialize BLE if not already done
        if self._initialized_status == 0:
            self._initialize_ble()

        # Wait until BLE initialization is complete
        while self._initialized_status != 2:
            sleep(0.25)
        
        while True:
            # Grab latest values under lock
            with self._lock:
                x, y, possession = self._x, self._y, self._possession

            # Build payload and command
            payload = self._build_payload(x, y, possession)
            cmd = self._build_cmd(payload)

            # Run the command
            self._run_cmd(cmd)

            # Sleep for 333ms so we send ~3 times per second
            sleep(0.333)

    # Method to send data through BLE
    def send_data(self, x, y, possession):
        # Update latest values under lock
        with self._lock:
            self._x = x
            self._y = y
            self._possession = possession

        # Start background sender if not already running
        if not self._running:
            self._thread = Thread(target=self._sender_loop, daemon=True)
            self._thread.start()
