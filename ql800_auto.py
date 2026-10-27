#!/usr/bin/env python3
"""QL-800 USB printing and image calibration for Raspberry Pi and PC.

This script extends the direct Brother QL-800 thermal printer driver with
adaptive auto-exposure and contrast normalization (dynamic gamma + percentile
autocontrast). It solves the problem of photos appearing either over-burnt
(black blobs) or washed out by automatically standardizing the midtone luminance
before Floyd-Steinberg dithering.

Setup on Raspberry Pi OS:
  sudo apt install python3-venv libusb-1.0-0
  python3 -m venv .venv-ql800
  .venv-ql800/bin/pip install -r requirements-ql800.txt
  sudo install -m 0644 70-ql800.rules /etc/udev/rules.d/70-ql800.rules
  sudo usermod -aG lp "$USER"
  sudo udevadm control --reload-rules
  # Reconnect USB and log out/in before printing without sudo.
  # Use .venv-ql800/bin/python for printing commands.

Usage Examples:

  1. Quick visual calibration & comparison on PC or Pi (no printer required):
     python ql800_auto.py preview photo.png --output comparison.png
     python ql800_auto.py preview photo.png --output comparison.png --target-mean 155 --cutoff 1.0

  2. Print with auto-exposure on Raspberry Pi:
     python ql800_auto.py print photo.png --label 62red --dither --auto-exposure

  3. Dry-run simulation with full 3-panel comparison preview:
     python ql800_auto.py print photo.png --label 62red --dither --auto-exposure --dry-run --preview-comparison comp.png

  4. List labels or scan USB bus:
     python ql800_auto.py labels
     python ql800_auto.py discover

Calibration Knobs and Arguments:
  --auto-exposure         Enable adaptive exposure and contrast calibration.
  --target-mean FLOAT     Target average luminance floor (default: 155.0, range: 130-170).
                          Dark photos below this floor are automatically lifted.
                          Bright photos already above this floor remain bright and clean.
  --two-way               Also pull down bright photos (default is lift-only).
  --cutoff FLOAT          Outlier clipping percentage for autocontrast (default: 1.0, range: 0.5-2.5).
                          Ignores glare and deep shadows so facial midtones get stretched.
  --brightness FLOAT      Post-calibration brightness multiplier (default: 1.0).
  --contrast FLOAT        Post-calibration contrast multiplier (default: 1.0).
  --preview-comparison    Saves a 3-panel PNG: [Original | Calibrated | Thermal Dither Simulation].
  -f, --force             Overwrite preview or output files if they already exist.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sys
import time
import warnings

# Suppress noisy brother_ql deprecation logger and Pillow warnings
logging.getLogger("brother_ql").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageOps, ImageStat

# Backward-compatible Floyd-Steinberg dither constant
DITHER_METHOD = getattr(getattr(Image, "Dither", None), "FLOYDSTEINBERG", getattr(Image, "FLOYDSTEINBERG", 3))

MODEL = "QL-800"
VENDOR_ID, PRODUCT_ID = 0x04F9, 0x209B

# Lazy/conditional import for brother_ql so preview mode works on any machine
try:
    from brother_ql.labels import Color, FormFactor, LabelsManager
    from brother_ql.models import ModelsManager
    HAS_BROTHER_QL = True
except ImportError:
    HAS_BROTHER_QL = False


# ==============================================================================
# Image Calibration and Thermal Dither Simulation
# ==============================================================================

def auto_calibrate_image(
    image: Image.Image,
    target_mean: float = 155.0,
    cutoff: float = 1.0,
    gamma_min: float = 0.2,
    gamma_max: float = 3.0,
    lift_only: bool = True,
) -> Image.Image:
    """Adjust image midtones and contrast for optimal 1-bit thermal dithering.

    1. Checks raw mean luminance.
    2. In lift_only mode (default), if the photo is already bright (mean >= target_mean),
       it scales highlights to clean white paper without crushing shadows into black.
    3. For dark/normal photos, applies percentile autocontrast and lifts midtones
       to target_mean so thermal dither patterns remain clear.
    """
    raw_mean = ImageStat.Stat(image.convert("L")).mean[0]

    if lift_only and raw_mean >= target_mean:
        # Photo is already bright: ensure highlights reach pure white paper (255)
        # without stretching midtones/shadows down to black
        gray = image.convert("L")
        max_val = ImageStat.Stat(gray).extrema[0][1]
        if 0 < max_val < 255:
            scale = 255.0 / max_val
            lut = [min(255, int(round(i * scale))) for i in range(256)]
            return image.point(lut * 3 if image.mode == "RGB" else lut)
        return image

    if cutoff > 0:
        stretched = ImageOps.autocontrast(image, cutoff=cutoff)
    else:
        stretched = ImageOps.autocontrast(image, cutoff=0)

    # Measure luminance mean of the stretched image
    gray = stretched.convert("L")
    stat = ImageStat.Stat(gray)
    current_mean = stat.mean[0]

    # Calculate dynamic gamma
    if 5.0 < current_mean < 250.0:
        if current_mean < target_mean:
            # Lift dark/normal photos to target floor
            gamma = math.log(target_mean / 255.0) / math.log(current_mean / 255.0)
            gamma = max(gamma_min, min(1.0, gamma))
        elif not lift_only and abs(current_mean - target_mean) > 1.0:
            # Two-way mode: pull bright photos down
            gamma = math.log(target_mean / 255.0) / math.log(current_mean / 255.0)
            gamma = max(1.0, min(gamma_max, gamma))
        else:
            return stretched

        lut = [int(round(((i / 255.0) ** gamma) * 255.0)) for i in range(256)]
        lut = [max(0, min(255, val)) for val in lut]
        if stretched.mode == "RGB":
            return stretched.point(lut * 3)
        elif stretched.mode == "L":
            return stretched.point(lut)
        else:
            return stretched.point(lut * len(stretched.getbands()))

    return stretched


def simulate_thermal_dither(image: Image.Image, red_mode: bool = False, threshold: float = 70.0) -> Image.Image:
    """Simulate on-screen thermal print appearance (1-bit on white paper)."""
    if red_mode:
        hsv = image.convert("HSV")
        h_ch, s_ch, v_ch = hsv.split()
        thresh_val = min(255, max(0, int((100.0 - threshold) / 100.0 * 255)))

        # Mask red hues (around 0 and 255 in 0-255 scale)
        h_mask = h_ch.point(lambda h: 255 if (h < 28 or h > 212) else 0)
        s_mask = s_ch.point(lambda s: 255 if s > 70 else 0)
        v_mask = v_ch.point(lambda v: 255 if v > 60 else 0)
        red_mask = ImageChops.multiply(ImageChops.multiply(h_mask, s_mask), v_mask)
        red_bin = red_mask.point(lambda x: 255 if x > thresh_val else 0, mode="1")

        # Black layer without red areas
        black_canvas = image.copy()
        black_canvas.paste((255, 255, 255), mask=red_bin)
        black_inverted = ImageOps.invert(black_canvas.convert("L"))
        black_dithered = black_inverted.convert("1", dither=DITHER_METHOD)

        # Composite simulation onto white paper
        sim = Image.new("RGB", image.size, (255, 255, 255))
        sim.paste((20, 20, 20), mask=black_dithered)
        sim.paste((215, 25, 25), mask=red_bin)
        return sim
    else:
        # Standard B/W: inverted so dark pixels trigger hot thermal pins
        gray = image.convert("L")
        inverted = ImageOps.invert(gray)
        dithered_1bit = inverted.convert("1", dither=DITHER_METHOD)

        # 1-bit thermal print simulation: hot pins burn black dots on white paper
        sim = Image.new("RGB", image.size, (255, 255, 255))
        sim.paste((20, 20, 20), mask=dithered_1bit)
        return sim


def calculate_black_density(simulated_image: Image.Image) -> float:
    """Calculate the percentage of non-white dots (thermal burn ratio)."""
    gray = simulated_image.convert("L")
    stat = ImageStat.Stat(gray)
    # 255 is pure white paper, 0 is full black
    return max(0.0, min(100.0, (255.0 - stat.mean[0]) / 255.0 * 100.0))


def build_comparison_image(
    original: Image.Image,
    calibrated: Image.Image,
    simulated: Image.Image,
    info_calibrated: str = "",
) -> Image.Image:
    """Assemble a side-by-side 3-panel comparison image for visual verification."""
    # Ensure all 3 panels have matching dimensions
    w, h = original.size
    calib_resized = calibrated.resize((w, h), Image.Resampling.LANCZOS)
    sim_resized = simulated.resize((w, h), Image.Resampling.NEAREST)

    orig_mean = ImageStat.Stat(original.convert("L")).mean[0]
    calib_mean = ImageStat.Stat(calib_resized.convert("L")).mean[0]
    density = calculate_black_density(sim_resized)

    margin = 16
    gap = 14
    header_h = 38
    total_w = margin * 2 + w * 3 + gap * 2
    total_h = margin * 2 + header_h + h

    canvas = Image.new("RGB", (total_w, total_h), (242, 244, 247))
    draw = ImageDraw.Draw(canvas)

    panels = [
        (f"1. Original (Mean: {orig_mean:.1f})", original),
        (f"2. Calibrated ({info_calibrated or f'Mean: {calib_mean:.1f}'})", calib_resized),
        (f"3. Thermal Dither (Burn: {density:.1f}%)", sim_resized),
    ]

    for idx, (title, img) in enumerate(panels):
        x = margin + idx * (w + gap)
        y = margin + header_h
        draw.text((x + 2, margin + 8), title, fill=(35, 38, 45))
        canvas.paste(img, (x, y))
        draw.rectangle([x - 1, y - 1, x + w, y + h], outline=(185, 190, 198), width=1)

    return canvas


# ==============================================================================
# Printer Driver Core (Brother QL-800)
# ==============================================================================

def supported_labels():
    if not HAS_BROTHER_QL:
        return {}
    return {
        label.identifier: label
        for label in LabelsManager().iter_elements()
        if (not label.restricted_to_models or MODEL in label.restricted_to_models)
        and label.tape_size[0] <= 62
        and label.form_factor in (FormFactor.ENDLESS, FormFactor.DIE_CUT, FormFactor.ROUND_DIE_CUT)
    }


def prepare_image(
    path: Path,
    label,
    *,
    rotate: str = "auto",
    fit: bool = False,
    length_mm: float | None = None,
    brightness: float = 1.0,
    contrast: float = 1.0,
    auto_exposure: bool = False,
    target_mean: float = 155.0,
    cutoff: float = 1.0,
    lift_only: bool = True,
) -> Image.Image:
    """Return RGB artwork at 300 dpi with optional auto-calibration and fitting."""
    if not math.isfinite(brightness) or brightness < 0:
        raise ValueError("--brightness must be a non-negative finite number")
    if not math.isfinite(contrast) or contrast < 0:
        raise ValueError("--contrast must be a non-negative finite number")
    if not math.isfinite(target_mean) or not (0 < target_mean < 255):
        raise ValueError("--target-mean must be a number between 0 and 255")
    if not math.isfinite(cutoff) or not (0 <= cutoff <= 50):
        raise ValueError("--cutoff must be a non-negative percentage <= 50")

    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
    image = Image.new("RGB", rgba.size, "white")
    image.paste(rgba, mask=rgba.getchannel("A"))

    # Step 1: Auto-exposure and contrast normalization
    if auto_exposure:
        image = auto_calibrate_image(image, target_mean=target_mean, cutoff=cutoff, lift_only=lift_only)

    # Step 2: Post-calibration manual adjustments
    if brightness != 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness)
    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)

    width, height = label.dots_printable
    endless = label.form_factor == FormFactor.ENDLESS
    if rotate != "auto":
        image = image.rotate(int(rotate), expand=True)
    elif not endless and image.size == (height, width):
        image = image.rotate(90, expand=True)

    if endless:
        if fit:
            raise ValueError("--fit is for die-cut labels; continuous artwork is scaled to roll width")
        if length_mm is not None:
            if not math.isfinite(length_mm) or length_mm <= 0:
                raise ValueError("--length-mm must be a finite positive printable length")
            height = round(length_mm * 300 / 25.4)
        else:
            height = round(image.height * width / image.width)
        model = next(m for m in ModelsManager().iter_elements() if m.identifier == MODEL)
        low, high = model.min_max_length_dots
        if not low <= height <= high:
            raise ValueError(f"Printable length must be {low}..{high} pixels at 300 dpi")
        target = (width, height)
        if length_mm is not None:
            image = ImageOps.pad(image, target, method=Image.Resampling.LANCZOS, color="white")
        elif image.size != target:
            image = image.resize(target, Image.Resampling.LANCZOS)
    else:
        if length_mm is not None:
            raise ValueError("--length-mm is only valid for continuous rolls")
        if fit:
            image = ImageOps.pad(image, (width, height), method=Image.Resampling.LANCZOS, color="white")
        elif image.size != (width, height):
            raise ValueError(f"Label {label.identifier} requires {width}x{height} pixels; use --fit to scale")
    return image


def make_raster(image: Image.Image, label, *, cut: bool = True, dither: bool = False, threshold: float = 70.0) -> bytes:
    """Convert prepared artwork into Brother QL-800 raster binary commands."""
    from brother_ql.conversion import convert
    from brother_ql.raster import BrotherQLRaster

    if not math.isfinite(threshold) or not 0 <= threshold <= 100:
        raise ValueError("Threshold must be a finite percentage from 0 to 100")
    if image.size[0] != label.dots_printable[0]:
        raise ValueError("Artwork must be prepared at the label's printable width")
    if label.form_factor != FormFactor.ENDLESS and image.size != label.dots_printable:
        raise ValueError("Artwork must match the die-cut printable dimensions")

    red = label.color == Color.BLACK_RED_WHITE
    raster = BrotherQLRaster(MODEL)
    raster.exception_on_warning = True

    # brother_ql 0.9.4 defaults to 200; QL-800 requires 400 bytes invalidate
    def invalidate():
        raster.data += b"\x00" * 400

    raster.add_invalidate = invalidate

    if red and dither:
        from brother_ql.conversion import filtered_hsv
        from brother_ql.devicedependent import label_type_specs

        specs = label_type_specs[label.identifier]
        device_pixel_width = raster.get_pixel_width()
        right_margin = specs["right_margin_dots"]

        new_im = Image.new("RGB", (device_pixel_width, image.height), (255, 255, 255))
        new_im.paste(image, (device_pixel_width - image.width - right_margin, 0))

        thresh_val = min(255, max(0, int((100.0 - threshold) / 100.0 * 255)))

        filter_h = lambda h: 255 if (h < 40 or h > 210) else 0
        filter_s = lambda s: 255 if s > 100 else 0
        filter_v = lambda v: 255 if v > 80 else 0
        red_hsv = filtered_hsv(new_im, filter_h, filter_s, filter_v)
        red_inverted = ImageOps.invert(red_hsv.convert("L"))
        red_im = red_inverted.point(lambda x: 0 if x < thresh_val else 255, mode="1")

        black_canvas = new_im.copy()
        black_canvas.paste((255, 255, 255), mask=red_im)
        black_inverted = ImageOps.invert(black_canvas.convert("L"))
        black_im = black_inverted.convert("1", dither=DITHER_METHOD)

        try:
            raster.add_switch_mode()
        except Exception:
            pass
        raster.add_invalidate()
        raster.add_initialize()
        try:
            raster.add_switch_mode()
        except Exception:
            pass
        raster.add_status_information()
        raster.mtype = 0x0A if label.form_factor == FormFactor.ENDLESS else 0x0B
        raster.mwidth = specs["tape_size"][0]
        raster.mlength = specs["tape_size"][1]
        raster.pquality = 1
        raster.add_media_and_quality(image.height)
        if cut:
            raster.add_autocut(True)
            raster.add_cut_every(1)
        else:
            raster.add_autocut(False)
        raster.two_color_printing = True
        raster.cut_at_end = cut
        raster.add_expanded_mode()
        raster.add_margins(specs["feed_margin"])
        raster.add_raster_data(black_im, red_im)
        raster.add_print()
        data = bytes(raster.data)
    else:
        data = bytes(convert(raster, [image], label.identifier, rotate="0", cut=cut,
                             dither=dither, red=red, threshold=threshold, compress=False))
        if not cut:
            data = data.replace(b"\x1biS", b"\x1biM\x00\x1biS", 1)
    return data


def find_printers():
    import usb.core
    return list(usb.core.find(find_all=True, idVendor=VENDOR_ID, idProduct=PRODUCT_ID))


def device_identifier(device) -> str:
    return f"{device.bus}:{device.address}"


def select_printer(identifier: str | None = None):
    devices = find_printers()
    if identifier:
        devices = [d for d in devices if device_identifier(d) == identifier]
    if len(devices) != 1:
        raise ValueError("Expected exactly one QL-800. Turn Editor Lite OFF, run discover, "
                         "and select its --printer BUS:ADDRESS")
    return devices[0]


def send_raster(data: bytes, device, timeout: int = 30) -> dict:
    """Send print job via USB bulk write and inspect status response."""
    import usb.core
    from brother_ql.backends.pyusb import BrotherQLBackendPyUSB
    from brother_ql.reader import interpret_response

    backend = BrotherQLBackendPyUSB(device)
    result = {"outcome": "unknown", "did_print": False, "ready_for_next_job": False, "printer_state": None}
    try:
        written = backend.write_dev.write(data, 15000)
        if written != len(data):
            raise OSError(f"Partial USB write: {written}/{len(data)} bytes; do not automatically retry")
        pending = bytearray()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                pending.extend(backend.read_dev.read(32, timeout=200))
            except usb.core.USBTimeoutError:
                continue
            while len(pending) >= 32:
                packet = bytes(pending[:32])
                del pending[:32]
                state = interpret_response(packet)
                result["printer_state"] = state
                if state["errors"] or state["status_type"] == "Error occurred":
                    result["outcome"] = "error"
                    return result
                if state["status_type"] == "Printing completed":
                    result["did_print"] = True
                if state["status_type"] == "Phase change" and state["phase_type"] == "Waiting to receive":
                    result["ready_for_next_job"] = True
                if result["did_print"] and result["ready_for_next_job"]:
                    result["outcome"] = "printed"
                    return result
        return result
    finally:
        backend.dispose()


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    # 1. Preview Command (Fast offline testing without needing printer)
    prev = commands.add_parser("preview", help="Quick offline 3-panel comparison preview (PC & Pi)")
    prev.add_argument("image", type=Path, help="Input photo path")
    prev.add_argument("-o", "--output", type=Path, default=Path("preview_comparison.png"),
                      help="Output PNG path for the 3-panel comparison image (default: preview_comparison.png)")
    prev.add_argument("--target-mean", type=float, default=155.0, help="Target average luminance floor (default: 155.0)")
    prev.add_argument("--two-way", action="store_true", help="Also pull down bright photos (default is lift-only)")
    prev.add_argument("--cutoff", type=float, default=1.0, help="Autocontrast outlier clipping percentage (default: 1.0)")
    prev.add_argument("--brightness", type=float, default=1.0, help="Manual post-calibration brightness multiplier")
    prev.add_argument("--contrast", type=float, default=1.0, help="Manual post-calibration contrast multiplier")
    prev.add_argument("--red", action="store_true", help="Simulate 2-color black/red DK-22251 label")
    prev.add_argument("-f", "--force", action="store_true", help="Overwrite output file if it exists")

    # 2. Labels Command
    commands.add_parser("labels", help="List QL-800 label sizes and printable pixels")

    # 3. Discover Command
    commands.add_parser("discover", help="Find USB QL-800 devices; does not print")

    # 4. Print Command
    job = commands.add_parser("print", help="Print one image, or render offline with --dry-run")
    job.add_argument("image", type=Path, help="Input artwork file")
    label_choices = supported_labels() if HAS_BROTHER_QL else None
    job.add_argument("--label", required=True, choices=label_choices, help="Target label size (e.g. 62, 62red, 29x90)")
    job.add_argument("--printer", help="BUS:ADDRESS from discover; required if multiple printers are connected")
    job.add_argument("--rotate", choices=("auto", "0", "90", "180", "270"), default="auto")
    job.add_argument("--fit", action="store_true", help="Scale and center die-cut artwork without cropping")
    job.add_argument("--length-mm", type=float, help="Printable artwork length on continuous roll")
    job.add_argument("--no-cut", action="store_true", help="Disable automatic label cutting")
    job.add_argument("--dither", action="store_true", help="Enable Floyd-Steinberg dithering")
    job.add_argument("--threshold", type=float, default=70.0, help="Binarization threshold when dithering is off (0-100)")
    job.add_argument("--auto-exposure", action="store_true",
                     help="Enable adaptive auto-exposure + contrast normalization for consistent prints")
    job.add_argument("--target-mean", type=float, default=155.0,
                     help="Target average luminance floor for auto-exposure (default: 155.0)")
    job.add_argument("--two-way", action="store_true",
                     help="Also pull down bright photos (default is lift-only)")
    job.add_argument("--cutoff", type=float, default=1.0,
                     help="Outlier clipping percentage for autocontrast (default: 1.0)")
    job.add_argument("--brightness", type=float, default=1.0, help="Manual brightness multiplier (default: 1.0)")
    job.add_argument("--contrast", type=float, default=1.0, help="Manual contrast multiplier (default: 1.0)")
    job.add_argument("--dry-run", action="store_true", help="Do not discover, open, or write to USB printer")
    job.add_argument("-f", "--force", action="store_true", help="Overwrite preview or output files if they exist")
    job.add_argument("--preview", type=Path, help="Save prepared artwork as PNG")
    job.add_argument("--preview-comparison", type=Path,
                     help="Save 3-panel comparison PNG: [Original | Calibrated | Thermal Dither]")
    job.add_argument("--output", type=Path, help="Save raw raster bytes")

    args = parser.parse_args(argv)

    try:
        # Handle Preview Command
        if args.command == "preview":
            if not args.image.exists():
                raise FileNotFoundError(f"Input image not found: {args.image}")
            if args.output.exists() and not args.force:
                raise ValueError(f"Refusing to overwrite {args.output} (use -f/--force to overwrite)")

            with Image.open(args.image) as source:
                rgba = ImageOps.exif_transpose(source).convert("RGBA")
            orig = Image.new("RGB", rgba.size, "white")
            orig.paste(rgba, mask=rgba.getchannel("A"))

            # Keep preview reasonably sized (max 720px width/height)
            orig.thumbnail((720, 720), Image.Resampling.LANCZOS)

            calib = auto_calibrate_image(
                orig,
                target_mean=args.target_mean,
                cutoff=args.cutoff,
                lift_only=not args.two_way,
            )
            if args.brightness != 1.0:
                calib = ImageEnhance.Brightness(calib).enhance(args.brightness)
            if args.contrast != 1.0:
                calib = ImageEnhance.Contrast(calib).enhance(args.contrast)

            sim = simulate_thermal_dither(calib, red_mode=args.red)
            info = f"Target: {args.target_mean:.0f}, Cutoff: {args.cutoff:.1f}%"
            comp = build_comparison_image(orig, calib, sim, info_calibrated=info)

            comp.save(args.output, format="PNG")
            orig_m = ImageStat.Stat(orig.convert("L")).mean[0]
            calib_m = ImageStat.Stat(calib.convert("L")).mean[0]
            burn = calculate_black_density(sim)
            print(f"Comparison preview saved to: {args.output}")
            print(f"Stats: Original Mean = {orig_m:.1f} -> Calibrated Mean = {calib_m:.1f} | Thermal Burn = {burn:.1f}%")
            return 0

        if not HAS_BROTHER_QL:
            raise RuntimeError("The 'brother_ql' package is required for labels/discover/print. "
                               "Install dependencies: pip install -r requirements-ql800.txt")

        # Handle Labels Command
        if args.command == "labels":
            for label in supported_labels().values():
                width, height = label.dots_printable
                print(f"{label.identifier:8} {width}x{height or 'variable':<8} {label.name}")
            return 0

        # Handle Discover Command
        if args.command == "discover":
            devices = find_printers()
            for device in devices:
                print(f"{device_identifier(device)}  Brother QL-800 (04f9:209b)")
            if not devices:
                print("No QL-800 found. Check USB, power, and Editor Lite mode.", file=sys.stderr)
            return 0 if devices else 1

        # Handle Print Command
        paths = [p for p in (args.preview, args.preview_comparison, args.output) if p is not None]
        if len({p.resolve() for p in paths}) != len(paths):
            raise ValueError("Preview, comparison, and raster output must use distinct file paths")
        if not args.force:
            for path in paths:
                if path.exists():
                    raise ValueError(f"Refusing to overwrite {path} (use -f/--force to overwrite)")

        labels = supported_labels()
        if args.label not in labels:
            raise ValueError(f"Unsupported label: {args.label}")
        label = labels[args.label]

        # Prepare image with optional auto-exposure and manual tuning
        image = prepare_image(
            args.image,
            label,
            rotate=args.rotate,
            fit=args.fit,
            length_mm=args.length_mm,
            brightness=args.brightness,
            contrast=args.contrast,
            auto_exposure=args.auto_exposure,
            target_mean=args.target_mean,
            cutoff=args.cutoff,
            lift_only=not args.two_way,
        )

        # Convert to raster
        data = make_raster(image, label, cut=not args.no_cut, dither=args.dither, threshold=args.threshold)

        mode = "wb" if args.force else "xb"
        if args.preview:
            with args.preview.open(mode) as output:
                image.save(output, format="PNG")

        if args.preview_comparison:
            # Generate uncalibrated baseline for comparison
            raw_baseline = prepare_image(
                args.image,
                label,
                rotate=args.rotate,
                fit=args.fit,
                length_mm=args.length_mm,
                brightness=1.0,
                contrast=1.0,
                auto_exposure=False,
            )
            sim = simulate_thermal_dither(image, red_mode=(label.color == Color.BLACK_RED_WHITE))
            info = f"Target: {args.target_mean:.0f}, Cutoff: {args.cutoff:.1f}%" if args.auto_exposure else "Manual"
            comp = build_comparison_image(raw_baseline, image, sim, info_calibrated=info)
            with args.preview_comparison.open(mode) as output:
                comp.save(output, format="PNG")

        if args.output:
            with args.output.open(mode) as output:
                output.write(data)

        print(f"QL-800: {label.identifier}, {image.width}x{image.height} pixels, {len(data)} raster bytes")

        if args.dry_run:
            print("Dry run: no USB device opened and nothing printed.")
            return 0

        result = send_raster(data, select_printer(args.printer))
        print(json.dumps(result, default=str))
        if result["outcome"] != "printed":
            print("Print not confirmed. Check printer/stock before retrying; no automatic retry.", file=sys.stderr)
            return 3
        return 0

    except Exception as error:
        print(f"QL-800 error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    sys.exit(main())
