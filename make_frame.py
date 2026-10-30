#!/usr/bin/env python3
"""Insert a photo into frame_monosnap.png: python make_frame.py --name Stephen.

Requires Pillow (already in requirements.txt). Name editing is commented out;
the template's existing name is preserved. --name and --font remain accepted.
The photo placeholder must be dark gray (#7B7B7B), distinct from the artwork.
Use --auto-exposure --target-mean 180 to calibrate only the cropped photo;
print the resulting frame without the printer's --auto-exposure option.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from image_to_txt import fit_image, load_rgb
from ql800_auto import auto_calibrate_image

ROOT = Path(__file__).resolve().parent
# ponytail: coordinates belong to this 1080x1912 template; adjust for a new design.
FRAME_SIZE = (1080, 1912)
PHOTO_BOX = (167, 432, 911, 1473)
NAME_BOX = (100, 1760, 980, 1870)
PHOTO_COLOR = (123, 123, 123)
COLOR_TOLERANCE = 9  # Sum of RGB differences: includes small export artifacts.


def load_font(path: str | None) -> ImageFont.FreeTypeFont:
    if path:
        return ImageFont.truetype(path, 44)
    for candidate in ("comicbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(candidate, 44)
        except OSError:
            pass
    return ImageFont.load_default(size=44)


def make_frame(
    name: str, photo: Path, frame: Path, font_path: str | None = None,
    *, auto_exposure: bool = False, target_mean: float = 155.0,
) -> Image.Image:
    # Name editing is disabled: preserve the name baked into the template.
    # name = name.strip()
    # if not name or any(not character.isprintable() for character in name):
    #     raise ValueError("name must be nonempty, printable, single-line text")
    if not math.isfinite(target_mean) or not 0 < target_mean < 255:
        raise ValueError("--target-mean must be a finite number between 0 and 255")

    canvas = load_rgb(frame, (255, 255, 255))
    if canvas.size != FRAME_SIZE:
        raise ValueError(f"frame must be {FRAME_SIZE[0]}x{FRAME_SIZE[1]} pixels")

    background = canvas.crop(PHOTO_BOX)
    seed = (background.width // 2, 100)
    if sum(abs(a - b) for a, b in zip(background.getpixel(seed), PHOTO_COLOR)) > COLOR_TOLERANCE:
        raise ValueError("frame must have a dark-gray (#7B7B7B) photo placeholder")
    filled = background.copy()
    # ponytail: connected color fill; use transparency if artwork shares this gray.
    ImageDraw.floodfill(filled, seed, (255, 0, 255), thresh=COLOR_TOLERANCE)
    # Fill background split off at the bottom by overlapping artwork (between feet).
    for x in range(1, background.width - 1):
        point = (x, background.height - 2)
        if sum(abs(a - b) for a, b in zip(filled.getpixel(point), PHOTO_COLOR)) <= COLOR_TOLERANCE:
            ImageDraw.floodfill(filled, point, (255, 0, 255), thresh=COLOR_TOLERANCE)
    mask = ImageChops.difference(background, filled).convert("L")
    mask = mask.point(lambda value: 255 if value else 0)
    # Remove the one-pixel fringe blended with the old placeholder color.
    mask = mask.filter(ImageFilter.MaxFilter(3))
    fitted = fit_image(
        load_rgb(photo, (255, 255, 255)),
        background.size,
        "crop",
        Image.Resampling.LANCZOS,
        (255, 255, 255),
    )
    if auto_exposure:
        fitted = auto_calibrate_image(fitted, target_mean=target_mean)
    canvas.paste(fitted, PHOTO_BOX[:2], mask)

    # draw = ImageDraw.Draw(canvas)
    # label = f"-{name}"
    # font = load_font(font_path)
    # left, top, right, bottom = NAME_BOX
    # for size in range(56, 9, -1):
    #     font = font.font_variant(size=size)
    #     x0, y0, x1, y1 = draw.textbbox((0, 0), label, font=font)
    #     if x1 - x0 <= right - left and y1 - y0 <= bottom - top:
    #         break
    # else:
    #     raise ValueError("name is too long to fit in the frame")
    # draw.rectangle((left, top, right - 1, bottom - 1), fill="white")
    # draw.text(
    #     ((left + right - x0 - x1) / 2, (top + bottom - y0 - y1) / 2),
    #     label,
    #     font=font,
    #     fill="black",
    # )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="accepted for compatibility; name editing is disabled")
    parser.add_argument("--photo", type=Path, default=ROOT / "photo.png")
    parser.add_argument("--frame", type=Path, default=ROOT / "frame_monosnap.png")
    parser.add_argument("--output", type=Path, default=ROOT / "frame_custom.png")
    parser.add_argument("--font", help="accepted for compatibility; name editing is disabled")
    parser.add_argument(
        "--auto-exposure", action="store_true",
        help="calibrate the cropped photo before adding the frame",
    )
    parser.add_argument(
        "--target-mean", type=float, default=155.0,
        help="photo exposure target, between 0 and 255 (default: 155)",
    )
    args = parser.parse_args()
    try:
        if args.output.resolve() in (args.frame.resolve(), args.photo.resolve()):
            raise ValueError("output must differ from the frame and photo paths")
        result = make_frame(
            args.name, args.photo, args.frame, args.font,
            auto_exposure=args.auto_exposure, target_mean=args.target_mean,
        )
        result.save(args.output, format="PNG")
    except (OSError, ValueError) as exc:
        print(f"Could not create frame: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
