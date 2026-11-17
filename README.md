# Digital Image Processing Project — CSE681

Lightweight Learned Image Compression for Laptop-Class Training.

## Create the project environment

All project dependencies are defined in `environment.yml`.

Install [Miniforge](https://github.com/conda-forge/miniforge) or another Conda
distribution, then open Anaconda Prompt or PowerShell in this repository and run:

```powershell
conda env create -f environment.yml
conda activate dip-project1
```

If you already created this environment using an older version of the YAML file,
update it to match the current dependencies:

```powershell
conda env update -n dip-project1 -f environment.yml --prune
conda activate dip-project1
```

## Download the datasets

From the repository root with `dip-project1` active, run the downloader as a module
from `src` so Python can resolve its `data` package:

```powershell
cd src
python -m data.download_data --data-root "D:/datasets/dip"
```

The default data root is the repository's `data` folder. It must contain separate
`DIV2K_train_HR` and `DIV2K_valid_HR` folders. Kodak and Tecnick are reserved for
final evaluation.

## Train with different configurations

From the repository root, activate the existing environment and start a bounded
GPU smoke run:

```powershell
conda activate dip-project1
python -m scripts.train --config configs/smoke_test.yaml
```

Run configuration are specified using YAML files in the `configs` folder, such as `configs/smoke_test.yaml` for quick tests and `configs/experiments/narrow.yaml` for narrower models.

### Command options

Use exactly one of `--config` for a new run or `--resume` for an existing run.

| Option | Purpose |
| --- | --- |
| `--config PATH` | Start a new run using a YAML configuration. |
| `--resume PATH` | Continue from the latest checkpoint in a run folder. |
| `--set KEY=VALUE` | Override a setting; repeat for multiple overrides. |
| `--data-root PATH` | Use a different folder containing the two DIV2K splits; otherwise use `DIP_DATA_ROOT` or the repository's `data` folder. |
| `--runs-root PATH` | Store new runs elsewhere; otherwise use `DIP_RUNS_ROOT` or the repository's `runs` folder. |
| `--log-interval N` | Log training metrics every N steps; default 10. |

See all options with `python -m scripts.train --help`.

### Configuration precedence

Each new run merges settings in this order; later values replace earlier ones:

```text
configs/base.yaml          shared defaults
-> selected YAML           overrides, e.g. configs/smoke_test.yaml
-> --set                   command-line overrides
```

For example, `base.yaml` defines `num_channels: 32`, `latent_channels: 48`,
`max_epochs: 1` and `device: cpu`. The smoke file changes the limits to
`max_steps: 20` and `max_epochs: null`, and selects `device: cuda`.
This command additionally changes the model width to 24 for a new run:

```powershell
python -m scripts.train --config configs/smoke_test.yaml --set num_channels=24
```

For reusable variants, create a YAML file such as `configs/experiments/narrow.yaml`:

```yaml
num_channels: 24
device: cuda (or cpu    )
max_steps: 20 
max_epochs: null (# no limit)
```

```powershell
python -m scripts.train --config configs/experiments/narrow.yaml --set batch_size=4
```

The variant inherits from `base.yaml`, not `smoke_test.yaml`; include desired
limits and device in the variant. Write scientific notation with a decimal point,
such as `1.0e-4`. When both step and epoch limits are set, the first reached
stops the run. Keep data, seed, lambda and budget fixed when comparing widths.

### Run files

The command prints the run folder, normally `runs/<config>-<timestamp>/`:

```text
<run-folder>/
├── config.yaml             resolved settings
├── provenance.json         code, software and data fingerprints
├── logs/train.log          training and validation log
├── checkpoints/latest.pt   latest saved training state
└── summary.json            metrics, timing, memory and codec check
```

### Compare runs

Plot all completed runs under `runs/` from the repository root:

```powershell
python -m scripts.plot_runs
```

To select specific run folders, pass their paths instead:

```powershell
python -m scripts.plot_runs "runs/<first-run>" "runs/<second-run>" --output "results/comparison.png"
```

For runs stored elsewhere, use `--runs-root "D:/dip-runs"` or set `DIP_RUNS_ROOT`.
The default image is `results/training_metrics.png`; choose another `--output`
or add `--overwrite` to replace it. The plot shows training and validation
rate-distortion loss, estimated BPP and MSE, plus training auxiliary loss.
Estimated BPP is not actual coded BPP. The command prints settings and final
steps for comparison; check data fingerprints and training budgets separately.
Step-limited runs can include a partial epoch. Folders without a valid
`summary.json`, including unfinished runs, are skipped during discovery; an
explicitly selected invalid folder produces an error.

### Rerun from scratch

Pass a previous run's resolved configuration to `--config`. This creates a new
run and initializes a new model; it does not load the old checkpoint:

```powershell
python -m scripts.train --config "runs/<run-folder>/config.yaml"
```

Replace `<run-folder>` with the actual folder name, or use its full path. Use the
same data and software environment for comparison; GPU results may differ.

### Resume an interrupted run

Use the existing run folder and its `checkpoints/latest.pt`:

```powershell
python -m scripts.train --resume "runs/<run-folder>"
```

To extend a completed 20-step smoke run to 40 total steps, use:

```powershell
python -m scripts.train --resume "runs/<run-folder>" --set max_steps=40
```

Use the same data split and environment. Only `max_steps`, `max_epochs`,
`num_workers` and `device` may change on resume; limits count from the start of
the run. Checkpoints are saved after validation at the end of each epoch or agit log
completed step-limited partial epoch. An interruption loses progress since the
last checkpoint; if none exists yet, the run cannot be resumed.

## Render the report template

Following the course guide, install [TeX Live](https://www.tug.org/texlive/)
separately from Conda. Include the LaTeX packages used by the template, such as
`geometry`, `natbib`, `booktabs`, and `hyperref`. Open a new terminal after
installation and confirm that the tools are available:

```powershell
pdflatex --version
bibtex --version
```

Edit `report.tex` and `references.bib`, then run the following commands from the
repository folder, proceeding only if each command succeeds:

```powershell
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

This creates `report.pdf` in the repository. BibTeX generates the bibliography;
the additional LaTeX runs resolve citations and cross-references. Keep one shared
`references.bib` for all three milestones. Conda activation is not required to
render the report.

Alternatively, upload `report.tex` and `references.bib` to
[Overleaf](https://www.overleaf.com/) and click **Recompile**.

## Project code dependencies

C++ build toolchain on Windows; see the
[CompressAI installation instructions](https://interdigitalinc.github.io/CompressAI/installation.html).
