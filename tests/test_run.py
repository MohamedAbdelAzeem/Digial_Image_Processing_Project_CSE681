"""Stage 5 tooling: YAML configs, run folders, and the training script."""

import dataclasses
import json
import logging
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

pytest.importorskip("compressai")
from settings import CONFIG_DIR, ExperimentSettings, load_settings  # noqa: E402
import training.engine as training_engine  # noqa: E402
from training.run import run_experiment  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402

TINY = {
    "num_channels": 8,
    "latent_channels": 8,
    "crop_size": 64,
    "batch_size": 2,
    "max_steps": 3,
    "max_epochs": None,
    "device": "cpu",
}


@pytest.fixture
def data_root(tmp_path):
    torch.manual_seed(0)
    for name, count in (("DIV2K_train_HR", 4), ("DIV2K_valid_HR", 3)):
        folder = tmp_path / "data" / name
        folder.mkdir(parents=True)
        for i in range(count):
            pixels = torch.randint(0, 256, (80, 96, 3), dtype=torch.uint8).numpy()
            Image.fromarray(pixels).save(folder / f"{i:04d}.png")
    return tmp_path / "data"


@pytest.fixture
def tiny_config(tmp_path):
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(TINY), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "config", sorted(CONFIG_DIR.rglob("*.yaml")), ids=lambda path: path.relative_to(CONFIG_DIR).as_posix()
)
def test_shipped_configs_are_valid(config):
    """Objective: every config in `configs/` loads into valid settings.

    Steps: load each YAML file through `load_settings` (which merges it over `base.yaml`).

    Expected: an `ExperimentSettings` is returned; no unknown keys, mistyped floats or bad limits.
    """
    assert isinstance(load_settings(config), ExperimentSettings)


def test_config_values_override_base_and_overrides_win(tiny_config):
    """Objective: later sources win: base, then the config file, then explicit overrides.

    Steps: load a config that changes widths and limits, with and without `overrides`.

    Expected: the config's values replace the base values; untouched settings keep the base
    value (`lmbda`); overrides replace the config's values.
    """
    base = load_settings(CONFIG_DIR / "base.yaml")
    settings = load_settings(tiny_config)
    assert (settings.num_channels, settings.max_steps) == (8, 3)
    assert settings.lmbda == base.lmbda
    assert load_settings(tiny_config, {"max_steps": 5}).max_steps == 5


def test_n_width_configs_change_only_num_channels():
    base = load_settings(CONFIG_DIR / "base.yaml")
    base_values = dataclasses.asdict(base)

    for name, expected_channels in (("narrow.yaml", 24), ("wide.yaml", 48)):
        settings = load_settings(CONFIG_DIR / "experiments" / name)
        values = dataclasses.asdict(settings)
        assert values["num_channels"] == expected_channels
        assert {key: value for key, value in values.items() if key != "num_channels"} == {
            key: value for key, value in base_values.items() if key != "num_channels"
        }


def test_config_rejects_unknown_and_mistyped_values(tmp_path):
    """Objective: a typo in a config fails loudly instead of being ignored.

    Steps: load configs with a misspelled key, a float written as `1e-4`, and a non-mapping file.

    Expected: `ValueError` in each case (YAML reads `1e-4` as text, which settings reject).
    """
    for text, match in (
        ("learning_rte: 0.001\n", "Unknown setting"),
        ("learning_rate: 1e-4\n", "learning_rate"),
        ("- 1\n- 2\n", "map setting names"),
    ):
        path = tmp_path / "bad.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_settings(path)


def test_run_experiment_writes_a_complete_run_folder(data_root, tiny_config, tmp_path):
    """Objective: one run leaves everything needed to identify, inspect and continue it.

    Steps: run a 3-step experiment on tiny synthetic data into a new runs folder.

    Expected: the run folder holds config.yaml, provenance.json, logs/train.log (with the
    settings), checkpoints/latest.pt and summary.json; the summary has 3 steps, finite
    validation values, positive step time and host memory, no CUDA memory on the CPU, and a codec
    check with a nonempty stream and positive actual and estimated bpp.
    """
    runs = tmp_path / "runs"
    summary = run_experiment(
        tiny_config,
        runs,
        train_dir=data_root / "DIV2K_train_HR",
        valid_dir=data_root / "DIV2K_valid_HR",
    )

    run_dir = runs / summary["run"]
    assert summary["run"].startswith("tiny-")
    for name in (
        "config.yaml",
        "provenance.json",
        "summary.json",
        "checkpoints/latest.pt",
        "checkpoints/best.pt",
    ):
        assert (run_dir / name).is_file()
    assert "Settings: ExperimentSettings" in (run_dir / "logs" / "train.log").read_text()
    assert yaml.safe_load((run_dir / "config.yaml").read_text())["num_channels"] == 8

    assert summary["steps"] == 3
    assert summary["best_validation_loss"] == min(
        record["val_loss"] for record in summary["history"]
    )
    assert summary["best_epoch"] == min(
        summary["history"], key=lambda record: record["val_loss"]
    )["epoch"]
    assert all(math.isfinite(value) for value in summary["final_validation"].values())
    assert summary["mean_train_step_s"] > 0
    assert summary["peak_host_memory_mb"] > 0
    assert summary["peak_cuda_allocated_mb"] is None
    codec = summary["codec_check"]
    assert codec["stream_bytes"] > 0
    assert codec["actual_bpp"] > 0 and codec["estimated_bpp"] > 0
    assert (codec["image_height"], codec["image_width"]) == (64, 64)
    # The run's log file handler must not stay attached to the root logger.
    assert not any(
        isinstance(handler, logging.FileHandler) and str(run_dir) in handler.baseFilename
        for handler in logging.getLogger().handlers
    )


def test_run_experiment_resumes_a_run_folder(data_root, tiny_config, tmp_path):
    """Objective: a run folder can be continued with a larger limit.

    Steps: run to step 2 (one epoch), then resume the same folder with `max_steps=4`.

    Expected: the same folder now reports 4 steps over 2 epochs, and its history holds both
    epochs.
    """
    kwargs = {
        "train_dir": data_root / "DIV2K_train_HR",
        "valid_dir": data_root / "DIV2K_valid_HR",
    }
    first = run_experiment(tiny_config, tmp_path / "runs", overrides={"max_steps": 2}, **kwargs)
    run_dir = tmp_path / "runs" / first["run"]

    second = run_experiment(
        None, tmp_path / "runs", resume_run=run_dir, overrides={"max_steps": 4}, **kwargs
    )

    assert (first["steps"], second["steps"]) == (2, 4)
    assert [record["epoch"] for record in second["history"]] == [1, 2]
    assert second["run"] == first["run"]
    best = load_checkpoint(run_dir / "checkpoints" / "best.pt")
    assert best["best_epoch"] == second["best_epoch"]
    assert best["best_validation_loss"] == second["best_validation_loss"]


def test_resume_preserves_best_checkpoint_when_validation_worsens(
    data_root, tiny_config, tmp_path, monkeypatch
):
    losses = iter((0.5, 0.4, 0.6))
    monkeypatch.setattr(
        training_engine,
        "validate",
        lambda *_args: {"loss": next(losses), "bpp_loss": 0.2, "mse_loss": 0.01},
    )
    kwargs = {
        "train_dir": data_root / "DIV2K_train_HR",
        "valid_dir": data_root / "DIV2K_valid_HR",
    }
    first = run_experiment(
        tiny_config, tmp_path / "runs", overrides={"max_steps": 3}, **kwargs
    )
    run_dir = tmp_path / "runs" / first["run"]
    before_resume = load_checkpoint(run_dir / "checkpoints" / "best.pt")

    resumed = run_experiment(
        None, tmp_path / "runs", resume_run=run_dir, overrides={"max_steps": 5}, **kwargs
    )

    after_resume = load_checkpoint(run_dir / "checkpoints" / "best.pt")
    latest = load_checkpoint(run_dir / "checkpoints" / "latest.pt")
    assert before_resume["best_epoch"] == 2
    assert resumed["history"][-1]["val_loss"] == 0.6
    assert after_resume["best_epoch"] == 2
    assert after_resume["best_validation_loss"] == 0.4
    assert latest["best_epoch"] == after_resume["best_epoch"]
    assert latest["best_validation_loss"] == after_resume["best_validation_loss"]


def test_train_script_runs_from_the_command_line(data_root, tiny_config, tmp_path):
    """Objective: `scripts/train.py` creates a run from a config and overrides.

    Steps: run the script in a subprocess with `--config`, `--data-root`, `--runs-root` and
    `--set max_steps=2`.

    Expected: exit code 0, the run folder is printed, and it contains summary.json with 2 steps.
    """
    runs = tmp_path / "runs"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(tiny_config),
            "--data-root",
            str(data_root),
            "--runs-root",
            str(runs),
            "--set",
            "max_steps=2",
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    assert "Run folder:" in result.stdout
    (run_dir,) = runs.iterdir()
    assert yaml.safe_load((run_dir / "summary.json").read_text())["steps"] == 2


def test_plot_runs_discovers_and_selects_summaries(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    for name in ("first", "second"):
        folder = runs / name
        folder.mkdir()
        summary = {
            "settings": {"lmbda": 0.0067, "num_channels": 32},
            "history": [
                {"epoch": 1, "step": 2, "train_loss": 2.0, "val_loss": 1.8,
                 "train_bpp_loss": 0.5, "val_bpp_loss": 0.4,
                 "train_mse_loss": 0.01, "val_mse_loss": 0.009, "train_aux_loss": 3.0}
            ],
        }
        (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (runs / "unfinished").mkdir()
    output = tmp_path / "comparison.png"

    discovered = subprocess.run(
        [sys.executable, "-m", "scripts.plot_runs", "--runs-root", str(runs), "--output", str(output)],
        cwd=REPOSITORY, capture_output=True, text=True, timeout=60,
    )
    assert discovered.returncode == 0, discovered.stderr
    assert "first" in discovered.stdout and "second" in discovered.stdout
    assert "unfinished" in discovered.stderr
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    selected = subprocess.run(
        [sys.executable, "-m", "scripts.plot_runs", str(runs / "first"),
         "--output", str(tmp_path / "selected.png")],
        cwd=REPOSITORY, capture_output=True, text=True, timeout=60,
    )
    assert selected.returncode == 0, selected.stderr
    assert "first" in selected.stdout and "second" not in selected.stdout

    exists = subprocess.run(
        [sys.executable, "-m", "scripts.plot_runs", str(runs / "first"), "--output", str(output)],
        cwd=REPOSITORY, capture_output=True, text=True, timeout=60,
    )
    assert exists.returncode != 0 and "already exists" in exists.stderr

    missing = subprocess.run(
        [sys.executable, "-m", "scripts.plot_runs", str(runs / "unfinished"),
         "--output", str(tmp_path / "missing.png")],
        cwd=REPOSITORY, capture_output=True, text=True, timeout=60,
    )
    assert missing.returncode != 0 and "summary.json" in missing.stderr


def test_plot_layout_with_long_run_paths(tmp_path):
    from scripts.plot_runs import create_plot
    import matplotlib.pyplot as plt

    history = [{"epoch": 1, "train_loss": 2.0, "val_loss": 1.8,
                "train_bpp_loss": 0.5, "val_bpp_loss": 0.4,
                "train_mse_loss": 0.01, "val_mse_loss": 0.009, "train_aux_loss": 3.0}]
    runs = [(tmp_path / ("long-parent-" * 10) / f"run-{index}", {"history": history})
            for index in range(4)]
    figure = create_plot(runs)
    try:
        figure.canvas.draw()
        axes = figure.axes[:4]
        assert all(axis.get_position().width > 0.3 for axis in axes)
        legend = figure.axes[4].get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        assert labels[:4] == [f"R{index + 1}: run-{index}" for index in range(4)]
        bounds = legend.get_window_extent(figure.canvas.get_renderer())
        assert figure.bbox.contains(bounds.x0, bounds.y0)
        assert figure.bbox.contains(bounds.x1, bounds.y1)
    finally:
        plt.close(figure)


def test_plot_kodak_rd_averages_selected_checkpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import matplotlib.pyplot as plt

    from scripts import plot_kodak_rd

    run_dir = tmp_path / "lambda013"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best.pt").write_bytes(b"test checkpoint")
    images = []
    for index in range(24):
        image_path = tmp_path / f"kodim{index:02d}.png"
        Image.new("RGB", (8, 8), (index, 0, 0)).save(image_path)
        images.append(image_path)

    settings = SimpleNamespace(lmbda=0.013, num_channels=64, latent_channels=96)
    monkeypatch.setattr(plot_kodak_rd, "load_codec", lambda path: (object(), settings))
    monkeypatch.setattr(plot_kodak_rd, "encode_image", lambda model, image: b"bits")
    monkeypatch.setattr(plot_kodak_rd, "decode_image", lambda model, bitstream: object())
    monkeypatch.setattr(
        plot_kodak_rd,
        "image_metrics",
        lambda original, reconstruction, bitstream: {
            "bpp": float(original.getpixel((0, 0))[0]),
            "psnr_db": float(original.getpixel((0, 0))[0] + 20),
        },
    )

    result = plot_kodak_rd.evaluate_checkpoint(run_dir, "best.pt", images)
    assert result["image_count"] == 24
    assert result["mean_bpp"] == 11.5
    assert result["mean_psnr_db"] == 31.5

    figure = plot_kodak_rd.create_plot([result])
    try:
        x_values, y_values = figure.axes[0].lines[0].get_data()
        assert list(x_values) == [11.5]
        assert list(y_values) == [31.5]
    finally:
        plt.close(figure)
