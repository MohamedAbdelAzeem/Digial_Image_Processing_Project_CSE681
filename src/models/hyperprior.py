"""Model construction helpers for the learned image-compression project."""

from compressai.models import ScaleHyperprior


def build_scale_hyperprior(
    num_channels: int = 32,
    latent_channels: int = 48,
) -> ScaleHyperprior:
    """Build a randomly initialized ScaleHyperprior with the requested widths."""
    if num_channels <= 0 or latent_channels <= 0:
        raise ValueError("Model channel counts must be positive")
    return ScaleHyperprior(N=num_channels, M=latent_channels)