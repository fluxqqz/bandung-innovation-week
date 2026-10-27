#!/usr/bin/env python3
"""Convert a regular image to the RGB565 text format used by image.txt.

The output is one line of lowercase comma-separated values:

    0x0000, 0x0001, ..., 0xffff,\\n

Each value is one pixel in RGB565, stored row by row.  By default the image
is cropped to fill a 75x77 canvas, matching the existing image.txt.

Use --layout grid to put one image row per line, which makes the file easier
to read.  The default single-line layout is what the current receiver expects.

Examples:
    ./rfid-reader/.venv/bin/python image_to_txt.py logo.png image.txt
    ./rfid-reader/.venv/bin/python image_to_txt.py photo.jpg out.txt --fit contain
    ./rfid-reader/.venv/bin/python image_to_txt.py logo.png out.txt --layout grid
    ./rfid-reader/.venv/bin/python image_to_txt.py logo.png out.txt --preview preview.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageColor, ImageOps

DEFAULT_WIDTH = 200
DEFAULT_HEIGHT = 200
DEFAULT_PIXELS = DEFAULT_WIDTH * DEFAULT_HEIGHT

RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def rgb_color(value: str) -> tuple[int, int, int]:
    try:
        red, green, blue, *_ = ImageColor.getrgb(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a colour: {value!r}") from exc
    return red, green, blue


def load_rgb(path: Path, background: tuple[int, int, int]) -> Image.Image:
    """Open an image, apply EXIF orientation, and flatten transparency."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")

    canvas = Image.new("RGBA", image.size, (*background, 255))
    return Image.alpha_composite(canvas, image).convert("RGB")


def fit_image(
    image: Image.Image,
    size: tuple[int, int],
    fit: str,
    resample: Image.Resampling,
    background: tuple[int, int, int],
) -> Image.Image:
    if image.size == size:
        return image
    if fit == "crop":
        return ImageOps.fit(image, size, method=resample, centering=(0.5, 0.5))
    if fit == "contain":
        return ImageOps.pad(
            image, size, method=resample, color=background, centering=(0.5, 0.5)
        )
    return image.resize(size, resample=resample)


def to_rgb565(image: Image.Image) -> list[int]:
    """Quantize an RGB image to a row-major list of 16-bit RGB565 values."""
    rgb = image.tobytes()
    values = []
    for index in range(0, len(rgb), 3):
        red, green, blue = rgb[index : index + 3]
        values.append(((red >> 3) << 11) | ((green >> 2) << 5) | (blue >> 3))
    return values


def format_values(values: list[int], width: int, layout: str) -> str:
    """Format RGB565 values as one line or as one image row per line."""
    tokens = [f"0x{value:04x}" for value in values]
    if layout == "line":
        return ", ".join(tokens) + ",\n"
    rows = [
        ", ".join(tokens[start : start + width]) + ","
        for start in range(0, len(tokens), width)
    ]
    return "\n".join(rows) + "\n"


def decode_rgb565(value: int) -> tuple[int, int, int]:
    """Expand an RGB565 value to 8-bit RGB using bit replication."""
    red = (value >> 11) & 0x1F
    green = (value >> 5) & 0x3F
    blue = value & 0x1F
    return (
        (red << 3) | (red >> 2),
        (green << 2) | (green >> 4),
        (blue << 3) | (blue >> 2),
    )


def write_preview(
    path: Path, values: list[int], size: tuple[int, int], scale: int
) -> None:
    pixels = bytearray()
    for value in values:
        pixels.extend(decode_rgb565(value))
    image = Image.frombytes("RGB", size, bytes(pixels))
    if scale != 1:
        image = image.resize(
            (image.width * scale, image.height * scale),
            resample=Image.Resampling.NEAREST,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an image to the RGB565 text format used by image.txt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="source image (PNG, JPEG, WebP, ...)")
    parser.add_argument("output", type=Path, help="destination .txt file")
    parser.add_argument(
        "--width",
        type=positive_int,
        default=DEFAULT_WIDTH,
        help="output width in pixels",
    )
    parser.add_argument(
        "--height",
        type=positive_int,
        default=DEFAULT_HEIGHT,
        help="output height in pixels",
    )
    parser.add_argument(
        "--fit",
        choices=("crop", "contain", "stretch"),
        default="crop",
        help="how to fit the source onto the output canvas",
    )
    parser.add_argument(
        "--background",
        type=rgb_color,
        default="#000000",
        metavar="#RRGGBB",
        help="colour used for transparency and letterboxing",
    )
    parser.add_argument(
        "--resample",
        choices=tuple(RESAMPLING),
        default="lanczos",
        help="resampling filter used when scaling",
    )
    parser.add_argument(
        "--layout",
        choices=("line", "grid"),
        default="line",
        help="single line (current receiver format) or one image row per line",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="optional PNG showing the decoded RGB565 output",
    )
    parser.add_argument(
        "--preview-scale",
        type=positive_int,
        default=1,
        help="integer upscale factor for the preview",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    size = (args.width, args.height)
    pixel_count = args.width * args.height

    try:
        image = load_rgb(args.input, args.background)
    except OSError as exc:
        print(f"Could not read {args.input}: {exc}", file=sys.stderr)
        return 1

    image = fit_image(
        image, size, args.fit, RESAMPLING[args.resample], args.background
    )
    values = to_rgb565(image)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(format_values(values, args.width, args.layout).encode("ascii"))

    if args.preview is not None:
        write_preview(args.preview, values, size, args.preview_scale)

    if pixel_count != DEFAULT_PIXELS:
        print(
            f"warning: {args.width}x{args.height} produces {pixel_count} pixels; "
            f"the existing image.txt has {DEFAULT_PIXELS}",
            file=sys.stderr,
        )

    print(
        f"Wrote {pixel_count} pixels ({args.width}x{args.height}, RGB565) "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
