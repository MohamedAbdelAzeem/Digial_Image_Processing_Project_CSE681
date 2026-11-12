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
from models.hyperprior import build_scale_hyperprior  # noqa: E402
from settings import ExperimentSettings  # noqa: E402
from training.engine import build_training_components, train_one_step  # noqa: E402


@pytest.fixture
def settings():
    return ExperimentSettings(
        num_channels=32,
        latent_channels=48,
        lmbda=0.0067,
        learning_rate=1e-4,
        aux_learning_rate=1e-3,
        crop_size=256,
        seed=0,
        batch_size=2,
        max_steps=2,
        max_epochs=None,
    )


@pytest.fixture
def batch(tmp_path, settings):
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    for i in range(4):
        pixels = torch.randint(0, 256, (300, 400, 3), dtype=torch.uint8).numpy()
        Image.fromarray(pixels).save(tmp_path / f"{i:04d}.png")
    loader = DataLoader(
        DIV2KDataset(tmp_path, crop_size=settings.crop_size),
        batch_size=settings.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(settings.seed),
    )
    return next(iter(loader))


@pytest.fixture
def model(settings):
    # Small N/M keeps the test fast; the architecture is the same as the notebook's model.
    return build_scale_hyperprior(settings=settings)


def test_one_training_step(batch, model, settings):
    assert model.N == 32
    assert model.M == 48
    optimizer, aux_optimizer, criterion = build_training_components(model, settings)
    assert optimizer.param_groups[0]["lr"] == settings.learning_rate
    assert aux_optimizer.param_groups[0]["lr"] == settings.aux_learning_rate
    assert criterion.lmbda == settings.lmbda
    assert batch.shape == (settings.batch_size, 3, settings.crop_size, settings.crop_size)

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


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_channels": 0},
        {"latent_channels": 0},
        {"lmbda": 0},
        {"learning_rate": float("inf")},
        {"aux_learning_rate": -1},
        {"crop_size": 0},
        {"seed": -1},
        {"batch_size": 0},
        {"max_steps": 0},
        {"max_steps": None, "max_epochs": None},
    ],
)
def test_invalid_experiment_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        ExperimentSettings(**overrides)


def test_model_widths_must_match_training_settings(model):
    settings = ExperimentSettings(num_channels=16, latent_channels=24)
    with pytest.raises(ValueError, match="do not match"):
        build_training_components(model, settings)


def test_model_factory_uses_custom_widths():
    settings = ExperimentSettings(num_channels=16, latent_channels=24)
    model = build_scale_hyperprior(settings=settings)
    assert model.N == settings.num_channels
    assert model.M == settings.latent_channels
