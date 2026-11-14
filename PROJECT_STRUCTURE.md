# Project Structure

Target layout with the current status of each item (checked 2026-11-15). Planned items are described in the Student B plan (`.github/STUDENT_B_DEVELOPMENT_PLAN.md`) or belong to the evaluation work. Run project commands from the repository root with the `dip-project1` environment active. There is no installable package; `src` is added to the import path by the tests.

## Repository layout

Status tags: `[exists]` is in the repository; `[planned: ...]` is not implemented yet, with the stage that introduces it.

```text
Digial_Image_Processing_Project_CSE681/
├── README.md                       # [exists] Setup, data download, common commands
├── AGENTS.md                       # [exists] Instructions for AI coding agents
├── PROJECT_STRUCTURE.md            # [exists] This file
├── environment.yml                 # [exists] Conda environment and dependencies
├── .gitignore                      # [exists] Excludes datasets, runs, results, build artifacts
├── report.tex                      # [exists] Report source
├── references.bib                  # [exists] Bibliography
│
├── .github/
│   └── STUDENT_B_DEVELOPMENT_PLAN.md   # [exists] Staged plan for the training path
│
├── configs/                        # [planned: Stage 5] Reproducible experiment settings
│   ├── base.yaml                   #   Shared defaults
│   ├── smoke_test.yaml             #   Tiny feasibility run
│   └── experiments/                #   One-change-at-a-time model variants
│       ├── small.yaml              #   [planned: Stage 5/6] Compact reference model
│       ├── narrow.yaml             #   [planned: Stage 6] Reduced transform width N only
│       └── small_latent.yaml       #   [planned: Stage 6] Reduced latent channels M only
│
├── src/                            # Reusable project implementation
│   ├── settings.py                 # [exists] Validated ExperimentSettings
│   │
│   ├── data/
│   │   ├── paths.py                # [exists] Resolve data root and dataset folders
│   │   ├── download_data.py        # [exists] Download DIV2K and Kodak
│   │   ├── verify_data.py          # [exists] Check image integrity and layout
│   │   ├── dataset.py              # [exists] Training crops and validation center crops
│   │   ├── jpeg_baseline.py        # [exists] JPEG quality sweep (lives in data/ for now)
│   │   └── README.md               # [exists] Data module usage
│   │
│   ├── models/
│   │   └── hyperprior.py           # [exists] Scratch ScaleHyperprior from settings (no pretrained loading)
│   │
│   ├── training/
│   │   ├── engine.py               # [exists] Training step, validation, train() loop
│   │   └── checkpoint.py           # [exists] Save, restore, resume with provenance
│   │
│   ├── codec/                      # [planned: evaluation] Learned codec inference
│   │   ├── inference.py            #   Encode/decode orchestration and padding
│   │   └── bitstream.py            #   Store all streams and decoder metadata
│   │
│   ├── evaluation/                 # [planned: evaluation] Shared metrics and benchmark protocol
│   │   ├── metrics.py              #   PSNR, MS-SSIM, actual-byte BPP
│   │   ├── benchmark.py            #   Per-image evaluation and CSV output
│   │   ├── timing.py               #   Repeatable timing measurements
│   │   └── analysis.py             #   Rate-distortion and complexity analysis
│   │
│   └── reporting/
│       └── report.py               # [planned: evaluation] Plots and LaTeX tables
│
├── scripts/                        # [planned] Thin command-line entry points
│   ├── train.py                    #   [planned: Stage 5] Start or resume a configured run
│   └── evaluate.py                 #   [planned: evaluation] Evaluate a checkpoint on a test set
│
├── tests/                          # [exists] Fast checks; default tests use synthetic data
│   ├── README.md
│   ├── test_dataset.py
│   ├── test_download_data.py
│   ├── test_integration_hyperprior.py
│   └── test_jpeg_baseline.py
│
├── notebooks/                      # [exists]
│   ├── models/
│   │   └── scale_hyperprior_one_training_step.ipynb
│   └── Evaluation/
│       └── Student_C_JPEG_Baseline_Annotated (1).ipynb
│
├── references/
│   └── elic_points.csv             # [planned: evaluation] Published ELIC rates with source details
│
├── runs/                           # [planned: Stage 5] Generated; ignored by Git
│   └── <experiment_id>/
│       ├── config.yaml             #   Resolved settings
│       ├── provenance.json         #   Code, environment, seed, data fingerprint
│       ├── checkpoints/            #   Latest and best training states
│       └── logs/                   #   Training and validation history, console log
│
└── results/                        # [planned: evaluation] Generated; ignored by Git
    └── <benchmark_id>/
        ├── per_image.csv
        ├── summary.csv
        ├── figures/
        └── provenance.json
```

## Data

Datasets live outside Git under `DIP_DATA_ROOT` (default `src/data/raw`), in the flat layout that `src/data/paths.py` expects:

```text
<DIP_DATA_ROOT>/
├── DIV2K_train_HR/                 # [exists] Training images (800)
├── DIV2K_valid_HR/                 # [exists] Validation and model selection (100)
├── kodak/                          # [exists] Final evaluation only
└── tecnick/                        # [planned] Final evaluation only; not yet in paths.py or download_data.py
```

Never use Kodak or Tecnick for training or model selection. Keep the layout consistent with `src/data/paths.py` and `src/data/download_data.py`.

## Commands

Available now:

```powershell
python -m pytest tests -q
cd src
python -m data.download_data --data-root "D:/datasets/dip"
```

Planned (Stage 5; update to the real arguments when implemented):

```powershell
python -m scripts.train --config configs/smoke_test.yaml
python -m scripts.evaluate --checkpoint runs/experiment_id/checkpoints/best.pt --dataset kodak
```
