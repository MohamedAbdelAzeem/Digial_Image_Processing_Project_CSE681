"""A -> B contract: DIV2KDataset batches drive ScaleHyperprior training and validation."""

import dataclasses
import logging
import math
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
from training.engine import (  # noqa: E402
    build_training_components,
    train,
    train_one_step,
    validate,
)


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
    """Objective: settings reach the model, optimizers and loss, and one step trains the model.

    Steps:
      1. Build the model and training components from `settings` (N=32, M=48).
      2. Check widths, both learning rates, lambda and the batch shape.
      3. Run `train_one_step` on one DIV2KDataset batch.

    Expected: the reconstruction has the batch's shape; loss, bpp_loss, mse_loss and aux_loss are
    finite; the first encoder weight changed.
    """
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
    """Objective: an untrained model can encode a batch to real bitstreams and decode it.

    Steps:
      1. Switch to eval mode and `update(force=True)` to build the entropy-coder tables.
      2. Compress the batch, then decompress the strings with the returned shape.

    compressed image is a dictionary containing the byte strings and the shape of the batch.
    example:
        {
            "strings": [[b"...", b"...", ...]],  # one list of byte strings per image in the batch
            "shape": (batch_size, 3, crop_size, crop_size)
        }
    
    Expected: the decoded batch has the input's shape and every image has a nonempty byte stream.
    This checks the codec path works, not reconstruction quality.
    """
    model.eval()
    model.update(force=True)
    with torch.no_grad():
        compressed = model.compress(batch)
        x_hat = model.decompress(compressed["strings"], compressed["shape"])["x_hat"]
    assert x_hat.shape == batch.shape
    assert all(len(s) > 0 for s in compressed["strings"][0]) # Ensure every image in the batch has a nonempty byte stream


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
    """Objective: invalid experiment settings fail early with a clear error.

    Steps: construct `ExperimentSettings` with one bad value per case (non-positive widths, lambda,
    crop or batch size; non-finite or negative learning rate; negative seed; zero max_steps; no
    step and no epoch limit).

    Expected: every case raises `ValueError`.
    """
    with pytest.raises(ValueError):
        ExperimentSettings(**overrides)


def test_model_widths_must_match_training_settings(model):
    """Objective: training components cannot be built for a model that disagrees with the settings.

    Steps: take the N=32, M=48 model and request components from settings with N=16, M=24.

    Expected: `ValueError` mentioning that the channel counts do not match.
    """
    settings = ExperimentSettings(num_channels=16, latent_channels=24)
    with pytest.raises(ValueError, match="do not match"):
        build_training_components(model, settings)


def test_model_factory_uses_custom_widths():
    """Objective: non-default widths in the settings are what the model is built with.

    Steps: build a model from settings with N=16 and M=24.

    Expected: `model.N == 16` and `model.M == 24`.
    """
    settings = ExperimentSettings(num_channels=16, latent_channels=24)
    model = build_scale_hyperprior(settings=settings)
    assert model.N == settings.num_channels
    assert model.M == settings.latent_channels


def write_noise_images(folder, count):
    folder.mkdir()
    for i in range(count):
        pixels = torch.randint(0, 256, (80, 96, 3), dtype=torch.uint8).numpy()
        Image.fromarray(pixels).save(folder / f"{i:04d}.png")
    return folder


@pytest.fixture
def image_dirs(tmp_path):
    torch.manual_seed(0)
    return write_noise_images(tmp_path / "train", 4), write_noise_images(tmp_path / "valid", 3)


@pytest.fixture
def tiny_settings():
    # Tiny widths keep full runs fast; this model needs crop sizes that are multiples of 64.
    return ExperimentSettings(
        num_channels=8, latent_channels=8, crop_size=64, batch_size=2, max_steps=3, max_epochs=None
    )


@pytest.mark.parametrize(
    ("max_steps", "max_epochs", "validated_steps"),
    [
        (3, None, [2, 3]),
        (None, 2, [2, 4]), 
        (5, 1, [2])],
)
def test_tiny_training_run_stops_at_limit(
    image_dirs, tiny_settings, caplog, max_steps, max_epochs, validated_steps
):
    """Objective: `train` stops at the step or epoch limit and validates after each epoch.

    Steps: run `train` on 4 training and 3 validation images (batch size 2, so 2 steps per epoch)
    with logging every step, for three limits: max_steps=3; max_epochs=2; max_steps=5 with
    max_epochs=1.

    Expected: validation records at steps [2, 3], [2, 4] and [2] respectively (the earlier limit
    wins); all recorded values are finite; the log contains the settings, estimated bpp and
    validation loss.
    """
    caplog.set_level(logging.INFO, logger="training.engine") # Capture INFO logs from the training engine
    settings = dataclasses.replace(tiny_settings, max_steps=max_steps, max_epochs=max_epochs) # Update tiny_settings with the current test's max_steps and max_epochs

    _, history = train(settings, *image_dirs, log_interval=1) 

    assert [record["step"] for record in history] == validated_steps
    assert all(math.isfinite(value) for record in history for value in record.values())
    assert f"Settings: {settings}" in caplog.text
    assert "est_bpp=" in caplog.text
    assert "val_loss=" in caplog.text


def test_fixed_seed_reproduces_training(image_dirs, tiny_settings):
    """Objective: the same seed gives the same initialization, data order, crops and training.

    Steps: run `train` twice with identical settings and data, then compare the histories
    (ignoring elapsed time) and every model tensor.

    Expected: histories are equal and all weights are bitwise identical (CPU run).
    """
    first_model, first_history = train(tiny_settings, *image_dirs)
    second_model, second_history = train(tiny_settings, *image_dirs)

    for record in first_history + second_history:
        del record["elapsed_s"]
    assert first_history == second_history
    second_state = second_model.state_dict()
    assert all(
        torch.equal(value, second_state[name])
        for name, value in first_model.state_dict().items()
    )


def test_validation_is_deterministic_and_read_only(image_dirs, tiny_settings):
    """Objective: validation measures the model without changing it.

    Steps:
      1. Build a fresh model and a validation loader of fixed center crops without flips.
      2. Copy the model state, then call `validate` twice.

    Expected: both calls return identical metrics; the model is back in training mode; every
    weight and buffer equals the copy; no parameter has a gradient.
    """
    model = build_scale_hyperprior(settings=tiny_settings)
    _, _, criterion = build_training_components(model, tiny_settings)
    dataset = DIV2KDataset(
        image_dirs[1],
        crop_size=tiny_settings.crop_size,
        horizontal_flip=False,
        vertical_flip=False,
        random_crop=False,
    )
    loader = DataLoader(dataset, batch_size=tiny_settings.batch_size)
    state_before = {name: value.clone() for name, value in model.state_dict().items()}

    metrics = validate(model, loader, criterion)

    assert validate(model, loader, criterion) == metrics
    assert model.training
    assert all(torch.equal(state_before[name], value) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_train_rejects_invalid_run_arguments(tmp_path):
    """Objective: `train` refuses to run with a leaky data split or a bad logging interval.

    Steps: call `train` (a) with the training folder equal to the validation folder's parent and
    (b) with `log_interval=0`.

    Expected: (a) `ValueError` mentioning overlap; (b) `ValueError` mentioning log_interval. Both
    are raised before any data is read or model is built.
    """
    with pytest.raises(ValueError, match="overlap"):
        train(ExperimentSettings(), tmp_path, tmp_path / "valid")
    with pytest.raises(ValueError, match="log_interval"):
        train(ExperimentSettings(), tmp_path / "train", tmp_path / "valid", log_interval=0)
