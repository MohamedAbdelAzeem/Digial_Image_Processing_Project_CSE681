import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import DIV2KDataset  # noqa: E402

REAL_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "DIV2K_train_HR"


@pytest.fixture
def synthetic_dir(tmp_path):
    for i in range(4):
        pixels = torch.randint(0, 256, (300, 400, 3), dtype=torch.uint8).numpy()
        Image.fromarray(pixels).save(tmp_path / f"{i:04d}.png")
    return tmp_path


def test_item_contract(synthetic_dir):
    item = DIV2KDataset(synthetic_dir, crop_size=192)[0]
    assert item.shape == (3, 192, 192)
    assert item.dtype == torch.float32
    assert 0.0 <= item.min().item() <= item.max().item() <= 1.0


def test_dataloader_batch(synthetic_dir):
    loader = DataLoader(DIV2KDataset(synthetic_dir), batch_size=4, shuffle=True)
    assert next(iter(loader)).shape == (4, 3, 192, 192)


def test_crops_are_random(synthetic_dir):
    dataset = DIV2KDataset(synthetic_dir)
    assert any(not torch.equal(dataset[0], dataset[0]) for _ in range(10))


def test_small_image_raises(tmp_path):
    Image.new("RGB", (100, 100)).save(tmp_path / "small.png")
    with pytest.raises(ValueError):
        DIV2KDataset(tmp_path, crop_size=192)[0]


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DIV2KDataset(tmp_path / "nope")


@pytest.mark.skipif(not REAL_DIR.is_dir(), reason="DIV2K not downloaded")
def test_real_div2k_batch():
    dataset = DIV2KDataset(REAL_DIR)
    assert len(dataset) == 800
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=True)))
    assert batch.shape == (4, 3, 192, 192)
    assert 0.0 <= batch.min().item() <= batch.max().item() <= 1.0
