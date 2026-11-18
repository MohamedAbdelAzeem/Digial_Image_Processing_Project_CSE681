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
├── configs/                        # [exists] Reproducible experiment settings (YAML; floats need a decimal point)
│   ├── base.yaml                   # [exists] Shared defaults
│   ├── smoke_test.yaml             # [exists] Bounded 20-step feasibility pilot on the GPU
│   └── experiments/                #   One-change-at-a-time model variants
│       ├── small.yaml              #   [planned: Stage 6] Compact reference model
│       ├── narrow.yaml             #   [planned: Stage 6] Reduced transform width N only
│       └── small_latent.yaml       #   [planned: Stage 6] Reduced latent channels M only
│
├── src/                            # Reusable project implementation
│   ├── settings.py                 # [exists] Validated ExperimentSettings and YAML config loading
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
│   │   ├── checkpoint.py           # [exists] Save, restore, resume with provenance
│   │   └── run.py                  # [exists] Run folder, logs, memory and step-time summary, codec check
│   │
│   ├── codec/                      # Learned codec inference
│   │   ├── inference.py            # [exists] Checkpoint loading, image round trip, and trial bitstream
│   │   └── bitstream.py            # [planned: evaluation] Further bitstream tooling
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
├── scripts/                        # Thin command-line entry points
│   ├── train.py                    #   [exists] Start or resume a configured run
│   └── evaluate.py                 #   [planned: evaluation] Evaluate a checkpoint on a test set
│
├── tests/                          # [exists] Fast checks; default tests use synthetic data
│   ├── README.md
│   ├── test_dataset.py
│   ├── test_download_data.py
│   ├── test_integration_hyperprior.py
│   ├── test_jpeg_baseline.py
│   └── test_run.py
│
├── notebooks/                      # [exists]
│   ├── models/
│   │   ├── codec_image_trial.ipynb # [exists] Checkpoint-based image codec trial
│   │   └── scale_hyperprior_one_training_step.ipynb
│   └── Evaluation/
│       └── Student_C_JPEG_Baseline_Annotated (1).ipynb
│
├── references/
│   └── elic_points.csv             # [planned: evaluation] Published ELIC rates with source details
│
├── runs/                           # [exists] Generated; ignored by Git (override with --runs-root or DIP_RUNS_ROOT)
│   └── <config>-<timestamp>/
│       ├── config.yaml             #   [exists] Resolved settings
│       ├── provenance.json         #   [exists] Code, environment, data fingerprint
│       ├── checkpoints/latest.pt   #   [exists] Latest training state
│       ├── checkpoints/best.pt     #   [exists] Lowest validation-loss state
│       ├── logs/train.log          #   [exists] Training and validation log
│       └── summary.json            #   [exists] Step time, peak memory, validation, codec check, history
│
└── results/                        # [planned: evaluation] Generated; ignored by Git
    └── <benchmark_id>/
        ├── per_image.csv
        ├── summary.csv
        ├── figures/
        └── provenance.json
```

## Data

Datasets live outside Git under `DIP_DATA_ROOT` (default `<repository>/data`), in the flat layout that `src/data/paths.py` expects:

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
python -m scripts.train --config configs/smoke_test.yaml --data-root data
python -m scripts.train --resume runs/<run> --set max_epochs=5
cd src
python -m data.download_data --data-root "D:/datasets/dip"
```

Planned (evaluation; update to the real arguments when implemented):

```powershell
python -m scripts.evaluate --checkpoint runs/<run>/checkpoints/latest.pt --dataset kodak
```
