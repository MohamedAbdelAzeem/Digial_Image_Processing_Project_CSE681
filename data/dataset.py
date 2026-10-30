"""PyTorch dataset for randomly cropped DIV2K training images."""

import random
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from data.paths import DIV2K_DIR, DIV2K_VALID_DIR


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class DIV2KDataset(Dataset):
    """Load DIV2K images as normalized RGB crops with random flip augmentation.

    `root` is one folder or a sequence of folders; the default is DIV2K train + valid (800 + 100).
    """

    def __init__(
        self,
        root: str | Path | Sequence[str | Path] = (DIV2K_DIR, DIV2K_VALID_DIR),
        crop_size: int = 256,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
    ) -> None:
        self.roots = [Path(root)] if isinstance(root, (str, Path)) else [Path(r) for r in root]
        self.crop_size = crop_size
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        if crop_size <= 0:
            raise ValueError("crop_size must be a positive integer")
        for folder in self.roots:
            if not folder.is_dir():
                raise FileNotFoundError(f"DIV2K image directory does not exist: {folder}")

        self.image_paths = sorted(
            path
            for folder in self.roots
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise ValueError(f"No supported image files found in {self.roots}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        width, height = image.size
        if width < self.crop_size or height < self.crop_size:
            raise ValueError(
                f"Image {image_path} is {width}x{height}; "
                f"it is smaller than the {self.crop_size}x{self.crop_size} crop"
            )

        left = random.randint(0, width - self.crop_size)
        top = random.randint(0, height - self.crop_size)
        image = image.crop((left, top, left + self.crop_size, top + self.crop_size))

        if self.horizontal_flip and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.vertical_flip and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        return pixels.reshape(self.crop_size, self.crop_size, 3).permute(2, 0, 1).to(
            dtype=torch.float32
        ).div_(255.0).contiguous()