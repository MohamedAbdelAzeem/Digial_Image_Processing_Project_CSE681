"""Small reusable training operations for learned image compression."""

import torch
from compressai.losses import RateDistortionLoss
from compressai.models import ScaleHyperprior
from torch import Tensor
from torch.optim import Optimizer

from settings import ExperimentSettings


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
    model.train()
    optimizer.zero_grad(set_to_none=True)
    aux_optimizer.zero_grad(set_to_none=True)

    output = model(batch)
    losses = criterion(output, batch)
    if not torch.isfinite(losses["loss"]).all():
        raise FloatingPointError("Non-finite rate-distortion loss")
    losses["loss"].backward()
    optimizer.step()

    aux_optimizer.zero_grad(set_to_none=True)
    aux_loss = model.aux_loss()
    if not torch.isfinite(aux_loss).all():
        raise FloatingPointError("Non-finite auxiliary loss")
    aux_loss.backward()
    aux_optimizer.step()

    result = {name: value.detach() for name, value in losses.items()}
    result["aux_loss"] = aux_loss.detach()
    result["x_hat"] = output["x_hat"].detach()
    return result