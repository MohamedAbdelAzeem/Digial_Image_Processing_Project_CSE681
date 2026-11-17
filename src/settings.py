"""Validated settings for a bounded ScaleHyperprior experiment."""

from dataclasses import dataclass, fields
import math
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@dataclass(frozen=True)
class ExperimentSettings:
    num_channels: int = 32
    latent_channels: int = 48
    lmbda: float = 0.0067
    learning_rate: float = 1e-4
    aux_learning_rate: float = 1e-3
    crop_size: int = 256
    seed: int = 0
    batch_size: int = 2
    max_steps: int | None = None
    max_epochs: int | None = 1
    num_workers: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive_integers = (
            "num_channels",
            "latent_channels",
            "crop_size",
            "batch_size",
        )
        # Validate that all positive integer fields are indeed positive integers
        for name in positive_integers:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        # Validate that the seed is a non-negative integer
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

        # Validate that all positive float fields are indeed positive floats
        for name in ("lmbda", "learning_rate", "aux_learning_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite positive number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")

        # Validate that max_steps and max_epochs are positive integers or None
        for name in ("max_steps", "max_epochs"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")

        # Validate that at least one of max_steps or max_epochs is set
        if self.max_steps is None and self.max_epochs is None:
            raise ValueError("At least one of max_steps or max_epochs must be set")

        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")

        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")


def load_settings(
    config_path: str | Path,
    overrides: dict | None = None,
    base_path: str | Path = CONFIG_DIR / "base.yaml",
) -> ExperimentSettings:
    """Merge the base config, the given config and `overrides`, in that order."""
    values: dict = {}
    for source in (base_path, config_path):
        loaded = yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{source} must map setting names to values")
        values.update(loaded)
    values.update(overrides or {})

    unknown = sorted(set(values) - {field.name for field in fields(ExperimentSettings)})
    if unknown:
        raise ValueError(f"Unknown setting(s): {', '.join(unknown)}")
    return ExperimentSettings(**values)