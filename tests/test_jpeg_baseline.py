import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.jpeg_baseline import (  # noqa: E402
    DEFAULT_QUALITIES,
    bits_per_pixel,
    evaluate_folder,
    jpeg_roundtrip,
    psnr,
    summarize,
)
from data.paths import KODAK_DIR  # noqa: E402


def natural_like_image(width=256, height=192, seed=0):
    """Smooth gradients plus mild noise, so JPEG behaves as on photos."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    channels = [xs, ys, 0.5 + 0.5 * np.sin(6 * xs + 4 * ys)]
    pixels = np.stack(channels, axis=-1) * 220 + rng.normal(0, 8, (height, width, 3))
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


@pytest.fixture
def image_dir(tmp_path):
    for i in range(3):
        natural_like_image(seed=i).save(tmp_path / f"img{i}.png")
    return tmp_path


def test_roundtrip_keeps_size_and_mode():
    image = natural_like_image()
    decoded, encoded, decode_ms = jpeg_roundtrip(image, 50)
    assert decoded.size == image.size
    assert decoded.mode == "RGB"
    assert encoded[:2] == b"\xff\xd8"  # JPEG SOI marker
    assert decode_ms >= 0


def test_roundtrip_rejects_bad_quality():
    with pytest.raises(ValueError):
        jpeg_roundtrip(natural_like_image(), 0)


def test_bits_per_pixel():
    assert bits_per_pixel(768 * 512 * 3, 768, 512) == 24.0


def test_psnr_known_values():
    image = natural_like_image()
    assert psnr(image, image) == math.inf
    shifted = Image.fromarray(np.clip(np.asarray(image).astype(np.int16) + 1, 0, 255).astype(np.uint8))
    off_by_one = np.mean(np.asarray(shifted, dtype=np.float64) != np.asarray(image, dtype=np.float64))
    expected = 10 * math.log10(255**2 / off_by_one)
    assert psnr(image, shifted) == pytest.approx(expected)


def test_quality_trades_rate_for_distortion(image_dir):
    summary = summarize(evaluate_folder(image_dir, DEFAULT_QUALITIES))
    bpps = [row["bpp"] for row in summary]
    psnrs = [row["psnr"] for row in summary]
    assert [row["quality"] for row in summary] == sorted(DEFAULT_QUALITIES)
    assert bpps == sorted(bpps)
    assert psnrs == sorted(psnrs)
    assert all(row["images"] == 3 for row in summary)


def test_save_dir_writes_measured_bitstream(image_dir, tmp_path):
    save_dir = tmp_path / "out"
    results = evaluate_folder(image_dir, [30], save_dir)
    for result in results:
        saved = save_dir / "q030" / f"{Path(result.image).stem}.jpg"
        assert saved.stat().st_size == result.num_bytes


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_folder(tmp_path / "nope")


@pytest.mark.skipif(not KODAK_DIR.is_dir(), reason="Kodak not downloaded")
def test_real_kodak_sanity():
    summary = {row["quality"]: row for row in summarize(evaluate_folder(KODAK_DIR))}
    assert all(row["images"] == 24 for row in summary.values())
    assert summary[10]["bpp"] < 0.5
    assert 25 < summary[10]["psnr"] < 30
    assert summary[95]["psnr"] > 38
