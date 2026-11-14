"""Model construction helpers for the learned image-compression project."""

from compressai.models import ScaleHyperprior

from settings import ExperimentSettings


def build_scale_hyperprior(settings: ExperimentSettings) -> ScaleHyperprior:
    """Build a randomly initialized ScaleHyperprior with the widths in `settings`."""
    return ScaleHyperprior(N=settings.num_channels, M=settings.latent_channels)