#!/usr/bin/env python3
"""Read RFID/NFC tag UIDs with a PN532 over I2C on a Raspberry Pi.

Wiring for I2C (Raspberry Pi 40-pin header):
    PN532 VCC/VIN -> 5V (pin 2 or 4) or 3.3V (pin 1), see your board label
    PN532 GND     -> GND (pin 6)
    PN532 SDA     -> GPIO2 / SDA (pin 3)
    PN532 SCL     -> GPIO3 / SCL (pin 5)

The module must be set to I2C mode. Most boards use address 0x24.
Run `i2cdetect -y 1` and expect to see `24` before starting this script.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import requests

import serial
import subprocess

import board
import busio
from adafruit_pn532.adafruit_pn532 import BusyError
from adafruit_pn532.i2c import PN532_I2C
from rpi_ws281x import PixelStrip, Color


# Load .env if present (ponytail: stdlib parser; use python-dotenv if complex quoting needed)
ENV_FILE = Path(__file__).with_name(".env")
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

BASE_URL = os.getenv("BASE_URL", "https://biw.arthatronic.com/api/v1/scan?instalation_id=3&rfid=")
USERNAME_API = os.getenv("USERNAME_API", "")
API_KEY = os.getenv("API_KEY", "")

SERIAL_PORT = "/dev/ttyACM0"
# SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 500000

IMAGE_FILE = Path(__file__).with_name("photo.txt")
CHUNK = 256

LED_COUNT = 13
LED_PIN = 18
strip = PixelStrip(LED_COUNT, LED_PIN, 800000, 10, False, 100, 0)  # count, pin, freq, DMA, invert, brightness, channel
strip.begin()

pet_name = "ponytail"
can_print = "true"

def led_on(color: Color) -> None:
    """Turn on the LED strip with the given color."""
    for i in range(LED_COUNT):
        strip.setPixelColor(i, color)
    strip.show()

def call_api(UID: str) -> int:
    """Call the API and print the response."""
    global pet_name, can_print

    URL = BASE_URL + UID
    response = requests.get(URL, auth=(USERNAME_API, API_KEY), timeout=5)

    status_response = response.status_code

    if response.ok:
        result = response.json()
        print(result)

        can_print = result["data"]["can_print"]
        image_url = result["data"]["image_url"]
        image_raw_url = result["data"]["raw_image_url"]
        frame_image_url = result["data"]["frame_image_url"]
        print_image_url = result["data"]["print_image_url"]

        def download(url, filename=None):
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            filename = filename
            with open(filename, "wb") as f:
                f.write(r.content)
            return filename

        download(image_url, "photo.png")
        download(image_raw_url, "photo_raw.png")
        download(frame_image_url, "frame_monosnap.png")
        # download(print_image_url, "photo_print.png")

        return status_response

    else:
        led_on(Color(0, 0, 255))
        print(f"Failed: {response.status_code}")

        return status_response

def send_frame(ser: serial.Serial, data: bytes) -> None:
    """Write the image data to the serial port, like `cat image.txt`."""
    for i in range(0, len(data), CHUNK):
        ser.write(data[i:i + CHUNK])
    ser.flush()


def log(message: str) -> None:
    """Print status messages to stderr, keeping stdout clean for UID output."""
    print(message, file=sys.stderr, flush=True)


def format_uid(uid: bytes) -> str:
    """Return a UID as colon-separated uppercase hex, e.g. 04:A2:B3:C4."""
    return "".join(f"{byte:02X}" for byte in uid)


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def connect_pn532(
    i2c: busio.I2C, debug: bool, attempts: int = 5
) -> tuple[PN532_I2C, tuple[int, int, int]]:
    """Create the PN532 object, retrying transient I2C startup errors.

    The PN532 sometimes fails its first ACK after power-up (a known
    clock-stretching issue on Raspberry Pi I2C). Retrying the whole
    handshake usually succeeds within a couple of attempts.
    """
    for attempt in range(1, attempts + 1):
        try:
            pn532 = PN532_I2C(i2c, debug=debug)
            ic, version, revision, _support = pn532.firmware_version
            return pn532, (ic, version, revision)
        except (BusyError, OSError, RuntimeError, ValueError) as exc:
            if attempt == attempts:
                raise
            log(f"PN532 startup attempt {attempt}/{attempts} failed ({exc}); retrying...")
            time.sleep(0.5)
    raise RuntimeError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read RFID/NFC tag UIDs with a PN532 over I2C.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=0.5,
        help="seconds to wait for a tag on each poll",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="print one UID and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show PN532 I2C traffic (verbose, mixes debug output into stdout)",
    )
    return parser.parse_args()


def main() -> int:
    global pet_name, can_print

    args = parse_args()
    i2c = None

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    log(f"Connected to {SERIAL_PORT}")                 # stderr, keeps stdout clean
    time.sleep(2)

    try:
        if not os.path.exists("/dev/i2c-1"):
            log("I2C bus 1 is not available (/dev/i2c-1 is missing).")
            log("Enable I2C with: sudo raspi-config nonint do_i2c 0")
            log("Then reboot and check that /dev/i2c-1 exists.")
            return 1

        try:
            i2c = busio.I2C(board.SCL, board.SDA)
        except (OSError, ValueError, RuntimeError) as exc:
            log(f"Could not open the I2C bus: {exc}")
            log("Enable I2C with: sudo raspi-config nonint do_i2c 0")
            log("Then reboot and check that /dev/i2c-1 exists.")
            log("If your PN532 is connected, `i2cdetect -y 1` should show 24.")
            return 1

        try:
            pn532, (ic, version, revision) = connect_pn532(i2c, args.debug)
        except (BusyError, OSError, RuntimeError, ValueError) as exc:
            log(f"PN532 did not respond at I2C address 0x24: {exc}")
            log("Check the wiring, power, and that the module is set to I2C mode.")
            log("Run `i2cdetect -y 1` and look for address 24.")
            return 1

        log(f"PN532 detected: IC 0x{ic:02X}, firmware {version}.{revision}")
        log("Waiting for an RFID/NFC tag... (Ctrl+C to quit)")

        last_uid: bytes | None = None

        while True:
            try:
                uid = pn532.read_passive_target(timeout=args.timeout)
            except BusyError:
                # The PN532 is still busy with the previous poll; try again.
                continue
            except OSError as exc:
                log(f"I2C error ({exc}); retrying...")
                
                for i in range(2):
                    led_on(Color(255, 0, 0))
                    time.sleep(0.2)
                    led_on(Color(0, 0, 0))
                    time.sleep(0.2)

                time.sleep(1)
                continue
            except RuntimeError as exc:
                log(f"PN532 read error ({exc}); retrying...")

                for i in range(3):
                    led_on(Color(255, 0, 0))
                    time.sleep(0.2)
                    led_on(Color(0, 0, 0))
                    time.sleep(0.2)

                time.sleep(1)
                continue

            if uid is None:
                led_on(Color(100, 0, 255))
                last_uid = None
                time.sleep(0.1)
                continue
            if uid != last_uid:
                log(f"Tag {format_uid(uid)}")
                log(f"Calling API: {BASE_URL}{format_uid(uid)}")
                led_on(Color(255, 255, 0))

                if call_api(format_uid(uid)) == 200:
                    log(f"API call successful for tag {format_uid(uid)}.")
                else:
                    log(f"API call failed for tag {format_uid(uid)}.")
                    led_on(Color(255, 0, 0))
                    time.sleep(1)
                    continue

                log(f"Done reading tag {format_uid(uid)} and downloading images.")

                # Convert the photo into photo.txt
                subprocess.run(
                    ["python3", "image_to_txt.py", "photo.png", "photo.txt",
                    "--fit", "crop", "--width", "200", "--height", "200", "--layout", "grid"],
                    check=True
                )
                image_bytes = IMAGE_FILE.read_bytes()
                log(f"Done converting photo to text format.")

                log(f"Sending image data to serial port {SERIAL_PORT}...")
                send_frame(ser, image_bytes)

                try:
                    subprocess.run(
                        ["venv/bin/python", "make_frame.py", "--name", pet_name,
                        "--photo", "photo_raw.png", "--output", "photo_edited.png",
                        "--auto-exposure", "--target-mean", "200"],
                        check=True
                    )

                    log(f"Convert image")
                except subprocess.CalledProcessError as e:
                    log(f"Convert image failed: {e}")
                    led_on(Color(255, 0, 0))
                    time.sleep(1)
                    continue

                # Print photo using Brother printer
                try:
                    if can_print == "true":
                        subprocess.run(
                            ["venv/bin/python", "ql800_auto.py", "print", "photo_edited.png",
                            "--label", "62red", "--dither"],
                            check=True
                        )

                        led_on(Color(0, 255, 0))
                        log(f"Printing image...")
                    else:
                        subprocess.run(
                            ["venv/bin/python", "ql800_auto.py", "print", "photo_edited.png",
                            "--label", "62red", "--dither"],
                            check=True
                        )
                        
                        log(f"Printing skipped for tag {format_uid(uid)} because can_print is false.")
                        led_on(Color(255, 255, 0))
                        time.sleep(1)
                except subprocess.CalledProcessError as e:
                    log(f"Printing failed")
                    led_on(Color(255, 0, 0))
                    time.sleep(1)

                last_uid = uid
                time.sleep(5)
                if args.once:
                    return 0
    except KeyboardInterrupt:
        log("\nStopped.")
        for i in range(LED_COUNT):
            led_on(Color(0, 0, 0))
        return 0
    finally:
        ser.close()
        if i2c is not None:
            try:
                i2c.deinit()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
