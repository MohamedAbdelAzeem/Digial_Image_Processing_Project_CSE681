"""A -> B contract: DIV2KDataset batches drive a CompressAI ScaleHyperprior training step."""

import random
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import DIV2KDataset  # noqa: E402

pytest.importorskip("compressai")
from compressai.losses import RateDistortionLoss  # noqa: E402
from compressai.models import ScaleHyperprior  # noqa: E402


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
    return ScaleHyperprior(N=32, M=48)


def test_one_training_step(batch, model):
    main_params = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
    aux_params = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]
    optimizer = torch.optim.Adam(main_params, lr=1e-4)
    aux_optimizer = torch.optim.Adam(aux_params, lr=1e-3)
    criterion = RateDistortionLoss(lmbda=0.0067)

    model.train()
    output = model(batch)
    assert output["x_hat"].shape == batch.shape

    losses = criterion(output, batch)
    assert all(torch.isfinite(losses[k]) for k in ("loss", "bpp_loss", "mse_loss"))

    weight_before = model.g_a[0].weight.detach().clone()
    optimizer.zero_grad()
    losses["loss"].backward()
    optimizer.step()
    assert not torch.equal(model.g_a[0].weight, weight_before)

    aux_optimizer.zero_grad()
    aux_loss = model.aux_loss()
    aux_loss.backward()
    aux_optimizer.step()
    assert torch.isfinite(aux_loss)


def test_compress_decompress_roundtrip(batch, model):
    model.eval()
    model.update(force=True)
    with torch.no_grad():
        compressed = model.compress(batch)
        x_hat = model.decompress(compressed["strings"], compressed["shape"])["x_hat"]
    assert x_hat.shape == batch.shape
    assert all(len(s) > 0 for s in compressed["strings"][0])
