"""Plot per-epoch training and validation metrics from completed run summaries."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

REPOSITORY = Path(__file__).resolve().parents[1]
PANELS = (
    ("Rate-distortion loss", "train_loss", "val_loss"),
    ("Estimated BPP", "train_bpp_loss", "val_bpp_loss"),
    ("MSE", "train_mse_loss", "val_mse_loss"),
    ("Auxiliary loss (training)", "train_aux_loss", None),
)
COMPARISON_SETTINGS = (
    "num_channels", "latent_channels", "lmbda", "seed", "batch_size", "crop_size",
    "max_steps", "max_epochs", "device", "num_workers",
)


def load_run(folder: Path) -> dict:
    """Read and validate the metrics needed for plotting one completed run."""
    summary_path = folder / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read {summary_path}: {error}") from error
    if not isinstance(summary, dict) or not isinstance(summary.get("history"), list):
        raise ValueError(f"{summary_path}: history must be a non-empty list")
    history = summary["history"]
    if not history:
        raise ValueError(f"{summary_path}: history must be a non-empty list")
    fields = {"epoch", "step"}
    for _, train_key, valid_key in PANELS:
        fields.add(train_key)
        if valid_key:
            fields.add(valid_key)
    for record in history:
        if not isinstance(record, dict):
            raise ValueError(f"{summary_path}: every history entry must be an object")
        for field in sorted(fields):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{summary_path}: invalid or missing {field}")
    if not isinstance(summary.get("settings"), dict):
        raise ValueError(f"{summary_path}: settings must be an object")
    return summary


def create_plot(runs: list[tuple[Path, dict]]) -> plt.Figure:
    """Plot metrics with a shared legend that does not shrink the data panels."""
    legend_rows = math.ceil((len(runs) + 2) / 2)
    legend_height = 0.45 * legend_rows + 0.5
    figure = plt.figure(figsize=(13, 8 + legend_height), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=(1, 1, legend_height / 4))
    axes = [figure.add_subplot(grid[row, column]) for row in range(2) for column in range(2)]
    legend_axis = figure.add_subplot(grid[2, :])
    legend_axis.axis("off")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    handles = []

    for index, (folder, summary) in enumerate(runs):
        history = summary["history"]
        epochs = [record["epoch"] for record in history]
        color = colors[index % len(colors)]
        short_name = folder.name
        if len(short_name) > 40:
            short_name = short_name[:37] + "..."
        handles.append(Line2D([0], [0], color=color, marker="o", label=f"R{index + 1}: {short_name}"))
        for axis, (title, train_key, valid_key) in zip(axes, PANELS):
            axis.plot(epochs, [record[train_key] for record in history],
                      color=color, marker="o", linestyle="--")
            if valid_key:
                axis.plot(epochs, [record[valid_key] for record in history],
                          color=color, marker="o")
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(True, alpha=0.25)

    handles.extend((
        Line2D([0], [0], color="black", linestyle="--", label="Training"),
        Line2D([0], [0], color="black", linestyle="-", label="Validation"),
    ))
    legend_axis.legend(handles=handles, loc="upper left", ncol=2, fontsize="small",
                       title="Run colors and line styles", frameon=False)
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", nargs="*", type=Path,
        help="specific run folders; omit to discover all run folders under --runs-root",
    )
    parser.add_argument(
        "--runs-root", type=Path,
        default=Path(os.environ.get("DIP_RUNS_ROOT", REPOSITORY / "runs")),
        help="folder containing runs (default: DIP_RUNS_ROOT or <repository>/runs)",
    )
    parser.add_argument(
        "--output", type=Path, default=REPOSITORY / "results" / "training_metrics.png",
        help="output image path (default: <repository>/results/training_metrics.png)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output image")
    args = parser.parse_args()

    if args.runs:
        folders = args.runs
    else:
        if not args.runs_root.is_dir():
            parser.error(f"runs root does not exist: {args.runs_root}")
        folders = sorted(path for path in args.runs_root.iterdir() if path.is_dir())
    runs = []
    for folder in folders:
        try:
            summary = load_run(folder)
        except ValueError as error:
            if args.runs:
                parser.error(str(error))
            print(f"Skipping {folder}: {error}", file=sys.stderr)
            continue
        runs.append((folder, summary))
    if not runs:
        parser.error("no completed runs with valid summary.json files found")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; choose another path or use --overwrite")

    for index, (folder, summary) in enumerate(runs):
        settings = summary["settings"]
        values = ", ".join(f"{name}={settings.get(name, 'unknown')}" for name in COMPARISON_SETTINGS)
        print(f"R{index + 1} {folder}: {values}, last_step={summary['history'][-1]['step']}")
    print("Compare data splits and provenance separately; step-limited epochs may be partial.")

    figure = create_plot(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    print(f"Saved plot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())