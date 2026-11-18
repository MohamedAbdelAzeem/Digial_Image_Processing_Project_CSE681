"""Run one configured experiment into its own folder: config, logs, checkpoints and a summary."""

import dataclasses
import json
import logging
import math
import time
from pathlib import Path

import psutil
import torch
import yaml
from torch import Tensor

from data.dataset import DIV2KDataset
from data.paths import DIV2K_DIR, DIV2K_VALID_DIR
from models.hyperprior import build_scale_hyperprior
from settings import ExperimentSettings, load_settings
from training.checkpoint import collect_provenance
from training.engine import train

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
MEGABYTE = 1024 * 1024


def check_codec(model: torch.nn.Module, image: Tensor) -> dict:
    """Encode and decode one image with real bitstreams and report the actual rate."""
    model.eval()
    model.update(force=True)
    x = image.unsqueeze(0).to(next(model.parameters()).device)
    pixels = x.shape[-2] * x.shape[-1]
    with torch.no_grad():
        likelihoods = model(x)["likelihoods"].values()
        compressed = model.compress(x)
        x_hat = model.decompress(compressed["strings"], compressed["shape"])["x_hat"].clamp(0, 1)
    mse = torch.mean((x_hat - x) ** 2).item()
    # Both the latent and the hyperprior streams are needed by the decoder.
    stream_bytes = sum(len(stream) for streams in compressed["strings"] for stream in streams)
    return {
        "image_height": x.shape[-2],
        "image_width": x.shape[-1],
        "stream_bytes": stream_bytes,
        "actual_bpp": 8 * stream_bytes / pixels,
        "estimated_bpp": sum(-torch.log2(l).sum().item() for l in likelihoods) / pixels,
        "psnr_db": 10 * math.log10(1 / mse) if mse > 0 else math.inf,
    }


def _mean_step_seconds(history: list[dict]) -> float:
    """Average the per-epoch step times, weighting each epoch by its number of steps."""
    previous_step, total_seconds = 0, 0.0
    for record in history:
        total_seconds += record["train_step_s"] * (record["step"] - previous_step)
        previous_step = record["step"]
    return total_seconds / previous_step


def run_experiment(
    config_path: str | Path | None,
    runs_root: str | Path,
    *,
    train_dir: str | Path = DIV2K_DIR,
    valid_dir: str | Path = DIV2K_VALID_DIR,
    overrides: dict | None = None,
    resume_run: str | Path | None = None,
    log_interval: int = 10,
) -> dict:
    """Train under `runs_root/<config>-<timestamp>/` and return the summary written there.

    Pass `resume_run` to continue an earlier run folder; `overrides` may then raise the limits.
    """
    if resume_run is None:
        if config_path is None:
            raise ValueError("config_path is required for a new run")
        settings = load_settings(config_path, overrides)
        run_dir = Path(runs_root) / f"{Path(config_path).stem}-{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir.mkdir(parents=True)
    else:
        run_dir = Path(resume_run)
        saved_config = run_dir / "config.yaml"
        settings = load_settings(saved_config, overrides, base_path=saved_config)
    (run_dir / "logs").mkdir(exist_ok=True)
    checkpoint_path = run_dir / "checkpoints" / "latest.pt"
    best_checkpoint_path = run_dir / "checkpoints" / "best.pt"

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(dataclasses.asdict(settings), sort_keys=False), encoding="utf-8"
    )
    provenance = collect_provenance(train_dir, valid_dir)
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    # Configure logging here, at the entry point, before train() could do it without a file.
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    root_logger = logging.getLogger()
    # basicConfig does nothing if logging was already configured, so set the level explicitly.
    previous_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(run_dir / "logs" / "train.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)
    try:
        device = torch.device(settings.device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        model, history = train(
            settings,
            train_dir,
            valid_dir,
            log_interval=log_interval,
            checkpoint_path=checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
            resume_from=checkpoint_path if resume_run is not None else None,
        )
        wall_time = time.perf_counter() - start

        validation_set = DIV2KDataset(
            valid_dir,
            crop_size=settings.crop_size,
            horizontal_flip=False,
            vertical_flip=False,
            random_crop=False,
        )
        codec = check_codec(model, validation_set[0])
        memory = psutil.Process().memory_info()
        cuda = device.type == "cuda"
        summary = {
            "run": run_dir.name,
            "settings": dataclasses.asdict(settings),
            "device": torch.cuda.get_device_name(device) if cuda else "cpu",
            "epochs": history[-1]["epoch"],
            "steps": history[-1]["step"],
            "best_epoch": min(history, key=lambda record: record["val_loss"])["epoch"],
            "best_validation_loss": min(record["val_loss"] for record in history),
            "wall_time_s": wall_time,
            "mean_train_step_s": _mean_step_seconds(history),
            "final_validation": {k: v for k, v in history[-1].items() if k.startswith("val_")},
            "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / MEGABYTE
            if cuda
            else None,
            "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved(device) / MEGABYTE
            if cuda
            else None,
            # Training process only; DataLoader worker processes are not included.
            "peak_host_memory_mb": getattr(memory, "peak_wset", memory.rss) / MEGABYTE,
            "codec_check": codec,
            "history": history,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logging.getLogger(__name__).info(
            "Codec check: %d bytes, %.4f actual bpp, %.4f estimated bpp, %.2f dB PSNR",
            codec["stream_bytes"],
            codec["actual_bpp"],
            codec["estimated_bpp"],
            codec["psnr_db"],
        )
        return summary
    finally:
        root_logger.removeHandler(file_handler)
        root_logger.setLevel(previous_level)
        file_handler.close()
