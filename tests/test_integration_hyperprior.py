"""A -> B contract: DIV2KDataset batches drive a CompressAI ScaleHyperprior training step."""

import random
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.dataset import DIV2KDataset  # noqa: E402

pytest.importorskip("compressai")
from compressai.losses import RateDistortionLoss  # noqa: E402
from models.hyperprior import build_scale_hyperprior  # noqa: E402
from training.engine import train_one_step  # noqa: E402


@pytest.fixture
def batch(tmp_path):
    for i in range(4):
        pixels = torch.randint(0, 256, (300, 400, 3), dtype=torch.uint8).numpy()
        Image.fromarray(pixels).save(tmp_path / f"{i:04d}.png")
    random.seed(0)
    torch.manual_seed(0)
    loader = DataLoader(DIV2KDataset(tmp_path, crop_size=256), batch_size=2, shuffle=True)
    return next(iter(loader))


@pytest.fixture
def model():
    # Small N/M keeps the test fast; the architecture is the same as the notebook's model.
    return build_scale_hyperprior(num_channels=32, latent_channels=48)


def test_one_training_step(batch, model):
    assert model.N == 32
    assert model.M == 48
    main_params = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
    aux_params = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]
    optimizer = torch.optim.Adam(main_params, lr=1e-4)
    aux_optimizer = torch.optim.Adam(aux_params, lr=1e-3)
    criterion = RateDistortionLoss(lmbda=0.0067)

    weight_before = model.g_a[0].weight.detach().clone()
    result = train_one_step(model, batch, optimizer, aux_optimizer, criterion)

    assert result["x_hat"].shape == batch.shape
    assert all(torch.isfinite(result[key]) for key in ("loss", "bpp_loss", "mse_loss", "aux_loss"))
    assert not torch.equal(model.g_a[0].weight, weight_before)


def test_compress_decompress_roundtrip(batch, model):
    model.eval()
    model.update(force=True)
    with torch.no_grad():
        compressed = model.compress(batch)
        x_hat = model.decompress(compressed["strings"], compressed["shape"])["x_hat"]
    assert x_hat.shape == batch.shape
    assert all(len(s) > 0 for s in compressed["strings"][0])
