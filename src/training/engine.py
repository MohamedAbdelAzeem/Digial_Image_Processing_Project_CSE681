"""Reusable training and validation operations for learned image compression."""

import itertools
import logging
import random
import time
from pathlib import Path

import compressai
import torch
from compressai.losses import RateDistortionLoss
from compressai.models import ScaleHyperprior
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from data.dataset import DIV2KDataset
from data.paths import DIV2K_DIR, DIV2K_VALID_DIR
from models.hyperprior import build_scale_hyperprior
from settings import ExperimentSettings

logger = logging.getLogger(__name__)

TRAIN_METRICS = ("loss", "bpp_loss", "mse_loss", "aux_loss")

# Training component builder: creates optimizers and rate-distortion criterion based on model and settings.
def build_training_components(
    model: ScaleHyperprior,
    settings: ExperimentSettings,
) -> tuple[Optimizer, Optimizer, RateDistortionLoss]:
    """Create the main/auxiliary optimizers and rate-distortion criterion."""
    if model.N != settings.num_channels or model.M != settings.latent_channels:
        raise ValueError("Model channel counts do not match experiment settings")

    main_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.endswith(".quantiles")
    ]
    aux_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith(".quantiles")
    ]
    if not main_parameters or not aux_parameters:
        raise ValueError("Model must provide main parameters and entropy quantiles")

    optimizer = torch.optim.Adam(main_parameters, lr=settings.learning_rate)
    aux_optimizer = torch.optim.Adam(aux_parameters, lr=settings.aux_learning_rate)
    criterion = RateDistortionLoss(lmbda=settings.lmbda)
    return optimizer, aux_optimizer, criterion


def train_one_step(
    model: ScaleHyperprior,
    batch: Tensor,
    optimizer: Optimizer,
    aux_optimizer: Optimizer,
    criterion: RateDistortionLoss,
) -> dict[str, Tensor]:
    """Apply one rate-distortion update and the entropy-bottleneck auxiliary update."""

    batch = batch.to(next(model.parameters()).device, non_blocking=True)
    model.train() # Set model to training mode for the main and auxiliary updates
    optimizer.zero_grad(set_to_none=True) # Reset gradients for main optimizer
    aux_optimizer.zero_grad(set_to_none=True) # Reset gradients for auxiliary optimizer

    output = model(batch) # Forward pass through the model
    losses = criterion(output, batch) # Compute rate-distortion loss

    # Check for non-finite main loss before backpropagation
    if not torch.isfinite(losses["loss"]).all():
        raise FloatingPointError("Non-finite rate-distortion loss")
    
    losses["loss"].backward() # Backpropagate main loss
    optimizer.step() # Update main parameters

    aux_optimizer.zero_grad(set_to_none=True) # Reset gradients for auxiliary optimizer before computing aux loss
    aux_loss = model.aux_loss() # Compute auxiliary loss
    if not torch.isfinite(aux_loss).all():
        raise FloatingPointError("Non-finite auxiliary loss")
    
    aux_loss.backward() # Backpropagate auxiliary loss
    aux_optimizer.step() # Update auxiliary parameters


    result = {name: value.detach() for name, value in losses.items()}
    result["aux_loss"] = aux_loss.detach()
    result["x_hat"] = output["x_hat"].detach()
    return result


def validate(
    model: ScaleHyperprior,
    loader: DataLoader,
    criterion: RateDistortionLoss,
) -> dict[str, float]:
    """Average rate-distortion metrics over `loader` without updating the model."""
    was_training = model.training
    # Eval mode rounds latents instead of adding noise, so repeated validation matches.
    model.eval()
    totals = {"loss": 0.0, "bpp_loss": 0.0, "mse_loss": 0.0}
    image_count = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            losses = criterion(model(batch), batch)
            for name in totals:
                totals[name] += losses[name].item() * len(batch)
            image_count += len(batch)
    model.train(was_training)
    return {name: total / image_count for name, total in totals.items()}


def train(
    settings: ExperimentSettings,
    train_dir: str | Path = DIV2K_DIR,
    valid_dir: str | Path = DIV2K_VALID_DIR,
    *,
    log_interval: int = 10,
) -> tuple[ScaleHyperprior, list[dict[str, float]]]:
    """Train from scratch until a step or epoch limit, validating after each epoch."""
    train_dir = Path(train_dir).resolve()
    valid_dir = Path(valid_dir).resolve()
    if train_dir.is_relative_to(valid_dir) or valid_dir.is_relative_to(train_dir):
        raise ValueError("Training and validation directories must not overlap")
    if log_interval <= 0:
        raise ValueError("log_interval must be a positive integer")
    device = torch.device(settings.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("settings.device is 'cuda' but CUDA is not available")

    # Does nothing if the caller has already configured logging.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # torch drives initialization and quantization noise; random drives crops and flips.
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    # Initialize on the CPU so the starting weights match across devices, then move.
    model = build_scale_hyperprior(settings=settings).to(device)
    optimizer, aux_optimizer, criterion = build_training_components(model, settings)
    # Workers and pinned memory feed the GPU; persistent workers avoid a respawn every epoch.
    loader_options = {
        "batch_size": settings.batch_size,
        "num_workers": settings.num_workers, # Number of DataLoader worker processes; 0 means in-process loading.
        "persistent_workers": settings.num_workers > 0, # Keep worker processes alive across epochs if using multiple workers.
        "pin_memory": device.type == "cuda", # Copy tensors into pinned memory before transferring to CUDA for faster host-to-device transfer.  
    }
    train_loader = DataLoader(
        DIV2KDataset(train_dir, crop_size=settings.crop_size),
        shuffle=True,
        # Its own generator keeps data order unchanged when only model widths change.
        generator=torch.Generator().manual_seed(settings.seed),
        **loader_options,
    )
    valid_loader = DataLoader(
        DIV2KDataset(
            valid_dir,
            crop_size=settings.crop_size,
            horizontal_flip=False,
            vertical_flip=False,
            random_crop=False,
        ),
        # A private generator stops validation passes from advancing the global PyTorch RNG.
        generator=torch.Generator(),
        **loader_options,
    )

    logger.info("Settings: %s", settings)
    logger.info(
        "Data: %d training images in %s; %d validation images in %s",
        len(train_loader.dataset),
        train_dir,
        len(valid_loader.dataset),
        valid_dir,
    )
    logger.info("Software: torch %s, compressai %s", torch.__version__, compressai.__version__)
    if device.type == "cuda":
        logger.info("Device: %s", torch.cuda.get_device_name(device))

    epochs = itertools.count(1) if settings.max_epochs is None else range(1, settings.max_epochs + 1)
    history = []
    step = 0
    start = time.perf_counter()
    for epoch in epochs:
        totals = dict.fromkeys(TRAIN_METRICS, 0.0)
        epoch_steps = 0
        for batch in train_loader:
            result = train_one_step(model, batch, optimizer, aux_optimizer, criterion)
            step += 1
            epoch_steps += 1
            metrics = {name: result[name].item() for name in TRAIN_METRICS}
            for name, value in metrics.items():
                totals[name] += value
            if step % log_interval == 0:
                logger.info(
                    "epoch %(epoch)d step %(step)d: loss=%(loss).4f est_bpp=%(bpp_loss).4f "
                    "mse=%(mse_loss).6f aux=%(aux_loss).2f elapsed=%(elapsed_s).1fs",
                    {
                        "epoch": epoch,
                        "step": step,
                        "elapsed_s": time.perf_counter() - start,
                        **metrics,
                    },
                )
            if step == settings.max_steps:
                break

        record = {"epoch": epoch, "step": step}
        record.update({f"train_{name}": totals[name] / epoch_steps for name in TRAIN_METRICS})
        validation = validate(model, valid_loader, criterion)
        record.update({f"val_{name}": value for name, value in validation.items()})
        record["elapsed_s"] = time.perf_counter() - start
        history.append(record)
        logger.info(
            "epoch %(epoch)d done at step %(step)d: train_loss=%(train_loss).4f "
            "val_loss=%(val_loss).4f val_est_bpp=%(val_bpp_loss).4f val_mse=%(val_mse_loss).6f "
            "elapsed=%(elapsed_s).1fs",
            record,
        )
        if step == settings.max_steps:
            break

    return model, history