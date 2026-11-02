import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import DIV2KDataset  # noqa: E402
from data.paths import DIV2K_DIR as REAL_DIR  # noqa: E402
from data.paths import DIV2K_VALID_DIR as REAL_VALID_DIR  # noqa: E402


@pytest.fixture
def synthetic_dir(tmp_path):
    for i in range(4):
        pixels = torch.randint(0, 256, (300, 400, 3), dtype=torch.uint8).numpy()
        Image.fromarray(pixels).save(tmp_path / f"{i:04d}.png")
    return tmp_path


def test_item_contract(synthetic_dir):
    item = DIV2KDataset(synthetic_dir, crop_size=256)[0]
    assert item.shape == (3, 256, 256)
    assert item.dtype == torch.float32
    assert 0.0 <= item.min().item() <= item.max().item() <= 1.0


def test_dataloader_batch(synthetic_dir):
    loader = DataLoader(DIV2KDataset(synthetic_dir), batch_size=4, shuffle=True)
    assert next(iter(loader)).shape == (4, 3, 256, 256)


def test_crops_are_random(synthetic_dir):
    dataset = DIV2KDataset(synthetic_dir)
    assert any(not torch.equal(dataset[0], dataset[0]) for _ in range(10))


def test_small_image_raises(tmp_path):
    Image.new("RGB", (100, 100)).save(tmp_path / "small.png")
    with pytest.raises(ValueError):
        DIV2KDataset(tmp_path, crop_size=256)[0]


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DIV2KDataset(tmp_path / "nope")


def test_multiple_roots_are_combined(synthetic_dir, tmp_path_factory):
    extra = tmp_path_factory.mktemp("valid")
    Image.new("RGB", (300, 300)).save(extra / "0801.png")
    assert len(DIV2KDataset([synthetic_dir, extra])) == 5
    with pytest.raises(FileNotFoundError):
        DIV2KDataset([synthetic_dir, extra / "nope"])


@pytest.mark.skipif(not REAL_DIR.is_dir(), reason="DIV2K not downloaded")
def test_real_div2k_batch():
    dataset = DIV2KDataset(REAL_DIR)
    assert len(dataset) == 800
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=True)))
    assert batch.shape == (4, 3, 256, 256)
    assert 0.0 <= batch.min().item() <= batch.max().item() <= 1.0


@pytest.mark.skipif(
    not (REAL_DIR.is_dir() and REAL_VALID_DIR.is_dir()), reason="DIV2K train/valid not downloaded"
)
def test_real_div2k_train_plus_valid():
    assert len(DIV2KDataset()) == 900
