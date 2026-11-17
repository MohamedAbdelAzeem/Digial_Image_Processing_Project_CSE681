"""Load a trained ScaleHyperprior and round-trip one RGB image."""

import dataclasses
import math
import struct
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image

from models.hyperprior import build_scale_hyperprior
from settings import ExperimentSettings
from training.checkpoint import CHECKPOINT_FORMAT_VERSION, load_checkpoint

HEADER = struct.Struct(">4s6I")
MAGIC = b"SH01"


def load_codec(checkpoint_path: str | Path):
    """Load a project training checkpoint and initialize its CPU entropy tables."""
    checkpoint = load_checkpoint(checkpoint_path)
    settings = ExperimentSettings(**checkpoint["settings"])
    model = build_scale_hyperprior(settings)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.update(force=True)
    return model, settings


def export_codec_checkpoint(model, settings: ExperimentSettings, path: str | Path) -> None:
    """Export model weights and initialized entropy tables without replacing a file."""
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "settings": dataclasses.asdict(settings),
        "model": model.state_dict(),
    }
    with Path(path).open("xb") as output:
        torch.save(checkpoint, output)


def encode_image(model, image: Image.Image) -> bytes:
    """Encode an RGB image, including dimensions and both streams in the result."""
    image = image.convert("RGB")
    width, height = image.size
    pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = pixels.reshape(height, width, 3).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    padded = functional.pad(tensor, (0, (-width) % 64, 0, (-height) % 64))
    with torch.no_grad():
        compressed = model.compress(padded)
    latent_stream = compressed["strings"][0][0]
    hyperprior_stream = compressed["strings"][1][0]
    shape_height, shape_width = compressed["shape"]
    header = HEADER.pack(
        MAGIC, height, width, shape_height, shape_width, len(latent_stream), len(hyperprior_stream)
    )
    return header + latent_stream + hyperprior_stream


def decode_image(model, bitstream: bytes) -> Image.Image:
    """Decode an image produced by encode_image, restoring its original dimensions."""
    if len(bitstream) < HEADER.size:
        raise ValueError("Invalid or truncated codec bitstream")
    magic, height, width, shape_height, shape_width, latent_size, hyperprior_size = HEADER.unpack_from(bitstream)
    if (magic != MAGIC or min(height, width, shape_height, shape_width, latent_size, hyperprior_size) <= 0
            or len(bitstream) != HEADER.size + latent_size + hyperprior_size):
        raise ValueError("Invalid or truncated codec bitstream")
    start = HEADER.size
    latent_stream = bitstream[start:start + latent_size]
    hyperprior_stream = bitstream[start + latent_size:]
    with torch.no_grad():
        reconstructed = model.decompress(
            [[latent_stream], [hyperprior_stream]], (shape_height, shape_width)
        )["x_hat"].clamp(0, 1)
    if reconstructed.shape[-2] < height or reconstructed.shape[-1] < width:
        raise ValueError("Invalid image dimensions in codec bitstream")
    pixels = (reconstructed[0, :, :height, :width] * 255).round().byte()
    return Image.fromarray(pixels.permute(1, 2, 0).contiguous().numpy(), mode="RGB")


def image_metrics(original: Image.Image, reconstruction: Image.Image, bitstream: bytes) -> dict:
    """Report actual bitstream BPP and 8-bit RGB PSNR for matching images."""
    original = original.convert("RGB")
    reconstruction = reconstruction.convert("RGB")
    if original.size != reconstruction.size:
        raise ValueError("Original and reconstructed image dimensions must match")
    original_pixels = torch.frombuffer(bytearray(original.tobytes()), dtype=torch.uint8).float()
    decoded_pixels = torch.frombuffer(bytearray(reconstruction.tobytes()), dtype=torch.uint8).float()
    mse = (original_pixels - decoded_pixels).square().mean().item()
    width, height = original.size
    return {
        "bytes": len(bitstream),
        "bpp": 8 * len(bitstream) / (width * height),
        "psnr_db": 10 * math.log10(255 ** 2 / mse) if mse else math.inf,
    }