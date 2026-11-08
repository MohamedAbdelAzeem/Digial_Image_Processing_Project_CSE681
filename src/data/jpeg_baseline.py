"""Pillow JPEG baseline: compress images at several qualities and report bpp, PSNR and decode time."""

import argparse
import io
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from data.paths import KODAK_DIR


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_QUALITIES = (10, 20, 30, 50, 70, 85, 95)


@dataclass
class JpegResult:
    image: str
    quality: int
    num_bytes: int
    bpp: float
    psnr: float
    decode_ms: float


def jpeg_roundtrip(image: Image.Image, quality: int) -> tuple[Image.Image, bytes, float]:
    """Encode an image as JPEG in memory and decode it again.

    Returns the decoded RGB image, the JPEG bitstream and the decode time in milliseconds.
    """
    if not 1 <= quality <= 100:
        raise ValueError(f"JPEG quality must be in [1, 100], got {quality}")

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    encoded = buffer.getvalue()

    start = time.perf_counter()
    with Image.open(io.BytesIO(encoded)) as stream:
        decoded = stream.convert("RGB")
    decode_ms = (time.perf_counter() - start) * 1000.0
    return decoded, encoded, decode_ms


def bits_per_pixel(num_bytes: int, width: int, height: int) -> float:
    return 8.0 * num_bytes / (width * height)


def psnr(original: Image.Image, reconstructed: Image.Image) -> float:
    """PSNR in dB over all RGB channels with a peak value of 255."""
    a = np.asarray(original.convert("RGB"), dtype=np.float64)
    b = np.asarray(reconstructed.convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Image shapes differ: {a.shape} vs {b.shape}")
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return math.inf
    return 10.0 * math.log10(255.0**2 / mse)


def evaluate_image(path: Path, quality: int, save_dir: Path | None = None) -> JpegResult:
    with Image.open(path) as source:
        original = source.convert("RGB")

    decoded, encoded, decode_ms = jpeg_roundtrip(original, quality)
    if decoded.size != original.size:
        raise RuntimeError(f"Decoded size {decoded.size} differs from original {original.size}")

    if save_dir is not None:
        out_dir = save_dir / f"q{quality:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{path.stem}.jpg").write_bytes(encoded)

    width, height = original.size
    return JpegResult(
        image=path.name,
        quality=quality,
        num_bytes=len(encoded),
        bpp=bits_per_pixel(len(encoded), width, height),
        psnr=psnr(original, decoded),
        decode_ms=decode_ms,
    )


def evaluate_folder(
    folder: Path, qualities=DEFAULT_QUALITIES, save_dir: Path | None = None
) -> list[JpegResult]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {folder}")
    paths = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported image files found in {folder}")

    return [evaluate_image(path, quality, save_dir) for quality in qualities for path in paths]


def summarize(results: list[JpegResult]) -> list[dict]:
    """Average bpp, PSNR and decode time per quality (mean over images)."""
    summary = []
    for quality in sorted({r.quality for r in results}):
        rows = [r for r in results if r.quality == quality]
        summary.append(
            {
                "quality": quality,
                "images": len(rows),
                "bpp": float(np.mean([r.bpp for r in rows])),
                "psnr": float(np.mean([r.psnr for r in rows])),
                "decode_ms": float(np.mean([r.decode_ms for r in rows])),
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=KODAK_DIR)
    parser.add_argument("--qualities", type=int, nargs="+", default=list(DEFAULT_QUALITIES))
    parser.add_argument(
        "--save-dir", type=Path, default=None, help="Optionally write the JPEG files for inspection"
    )
    args = parser.parse_args()

    summary = summarize(evaluate_folder(args.image_dir, args.qualities, args.save_dir))

    print(f"JPEG (Pillow) on {args.image_dir}")
    print(f"{'Q':>4} {'images':>6} {'bpp':>8} {'PSNR dB':>8} {'decode ms':>10}")
    for row in summary:
        print(
            f"{row['quality']:>4} {row['images']:>6} {row['bpp']:>8.4f} "
            f"{row['psnr']:>8.2f} {row['decode_ms']:>10.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
