"""Save and restore everything needed to resume a ScaleHyperprior training run."""

import dataclasses
import hashlib
import os
import platform
import random
import subprocess
from pathlib import Path

import compressai
import torch
from compressai.models import ScaleHyperprior
from torch.optim import Optimizer

from data.dataset import IMAGE_EXTENSIONS
from settings import ExperimentSettings

CHECKPOINT_FORMAT_VERSION = 1

# Limits and hardware may change on resume; every other setting defines the experiment.
RESUMABLE_SETTINGS = {"max_steps", "max_epochs", "num_workers", "device"}


def data_fingerprint(folder: str | Path) -> dict:
    """Identify an image folder by its file names and sizes, without reading the pixels."""
    folder = Path(folder)
    paths = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(f"{path.relative_to(folder).as_posix()}:{path.stat().st_size}\n".encode())
    return {"image_count": len(paths), "sha256": digest.hexdigest()}


def _git_state() -> dict | None:
    root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.SubprocessError):
        return None


def collect_provenance(train_dir: str | Path, valid_dir: str | Path) -> dict:
    """Describe the code, software and data a run used, so its checkpoints can be identified."""
    return {
        "git": _git_state(),
        "python": platform.python_version(),
        # Plain strings: torch's version type is rejected by the weights-only loader.
        "torch": str(torch.__version__),
        "compressai": str(compressai.__version__),
        "train_data": data_fingerprint(train_dir),
        "valid_data": data_fingerprint(valid_dir),
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: ScaleHyperprior,
    optimizer: Optimizer,
    aux_optimizer: Optimizer,
    settings: ExperimentSettings,
    epoch: int,
    step: int,
    history: list[dict],
    train_generator: torch.Generator,
    provenance: dict | None,
) -> None:
    """Write the training state after a completed epoch."""
    path = Path(path)
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "settings": dataclasses.asdict(settings),
        "epoch": epoch,
        "step": step,
        "history": history,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "aux_optimizer": aux_optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "train_loader": train_generator.get_state(),
        },
        "provenance": provenance,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replacing a finished temporary file means an interrupted save never corrupts the old one.
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path) -> dict:
    """Read a checkpoint onto the CPU; `weights_only` refuses to unpickle arbitrary objects."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format in {path}")
    return checkpoint


def restore_checkpoint(
    checkpoint: dict,
    *,
    model: ScaleHyperprior,
    optimizer: Optimizer,
    aux_optimizer: Optimizer,
    settings: ExperimentSettings,
    train_generator: torch.Generator,
) -> tuple[int, int, list[dict]]:
    """Load saved state into freshly built objects; return the saved epoch, step and history."""
    saved = ExperimentSettings(**checkpoint["settings"])
    changed = [
        field.name
        for field in dataclasses.fields(settings)
        if field.name not in RESUMABLE_SETTINGS
        and getattr(saved, field.name) != getattr(settings, field.name)
    ]
    if changed:
        raise ValueError(f"Cannot resume with different settings: {', '.join(changed)}")

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
    random.setstate(checkpoint["rng"]["python"])
    torch.set_rng_state(checkpoint["rng"]["torch"])
    train_generator.set_state(checkpoint["rng"]["train_loader"])
    return checkpoint["epoch"], checkpoint["step"], checkpoint["history"]
