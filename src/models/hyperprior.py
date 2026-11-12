"""Model construction helpers for the learned image-compression project."""

from compressai.models import ScaleHyperprior

from settings import ExperimentSettings


def build_scale_hyperprior(
    num_channels: int | None = None,
    latent_channels: int | None = None,
    *,
    settings: ExperimentSettings | None = None,
) -> ScaleHyperprior:
    """Build a randomly initialized ScaleHyperprior with the requested widths."""
    if settings is not None:
        if num_channels is not None or latent_channels is not None:
            raise ValueError("Pass either settings or explicit channel counts, not both")
        num_channels = settings.num_channels
        latent_channels = settings.latent_channels
    else:
        num_channels = 32 if num_channels is None else num_channels
        latent_channels = 48 if latent_channels is None else latent_channels

    if type(num_channels) is not int or type(latent_channels) is not int:
        raise ValueError("Model channel counts must be integers")
    if num_channels <= 0 or latent_channels <= 0:
        raise ValueError("Model channel counts must be positive")
    return ScaleHyperprior(N=num_channels, M=latent_channels)