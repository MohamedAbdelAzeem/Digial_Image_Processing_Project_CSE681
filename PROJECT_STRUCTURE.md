# Proposed Project Structure

This is a **target layout for organizing the project**, not an inventory of files that all exist today. It avoids a separate installable `lic` package for now, while keeping source code grouped by responsibility. Run project commands from the repository root with the project environment active.

```text
Digial_Image_Processing_Project_CSE681/
├── README.md                         # Setup, data download, and common commands
├── environment.yml                   # Conda environment and project dependencies
├── .gitignore                        # Excludes local datasets and generated artifacts
│
├── configs/                          # Reproducible training/evaluation settings
│   ├── base.yaml                     # Shared defaults and dataset locations
│   ├── smoke_test.yaml               # Tiny feasibility run
│   └── experiments/                  # One-change-at-a-time model experiments
│       ├── small.yaml                # Compact reference model
│       ├── narrow.yaml               # Reduced transform width
│       └── small_latent.yaml         # Reduced latent channel count
│
├── src/                              # Reusable project implementation
│   ├── data/                         # Dataset paths, acquisition, validation, loaders
│   │   ├── paths.py                  # Resolve data root and dataset locations
│   │   ├── download_data.py          # Download datasets and resume/retry safely
│   │   ├── verify_data.py            # Check image integrity and expected dataset layout
│   │   └── datasets.py               # Training crops and full-resolution evaluation images
│   │
│   ├── models/                       # Learned codec architectures
│   │   └── hyperprior.py             # Construct compact and pretrained scale hyperpriors
│   │
│   ├── training/                     # Training lifecycle
│   │   ├── engine.py                 # Training and validation loops
│   │   └── checkpoint.py             # Save, restore, and resume model/optimizer state
│   │
│   ├── codec/                        # Learned codec inference and file representation
│   │   ├── inference.py              # Encode/decode orchestration and padding
│   │   └── bitstream.py              # Store streams and decoder-required metadata
│   │
│   ├── baselines/                    # Conventional comparison codecs
│   │   └── jpeg.py                   # Pillow JPEG encode/decode quality sweep
│   │
│   ├── evaluation/                   # Shared metrics and benchmark protocol
│   │   ├── metrics.py                # PSNR, MS-SSIM, and actual-byte BPP
│   │   ├── benchmark.py              # Per-image dataset evaluation and CSV output
│   │   ├── timing.py                 # Repeatable timing measurements
│   │   └── analysis.py                # Rate-distortion, matched-rate, and complexity analysis
│   │
│   ├── reporting/                    # Results prepared for inspection and the paper
│   │   └── report.py                 # Plots, summary tables, and LaTeX table output
│   │
│   └── utils/                        # Cross-cutting experiment helpers
│       ├── config.py                 # Load, merge, and validate YAML settings
│       └── reproducibility.py         # Seeds and run/environment provenance
│
├── scripts/                          # Command-line entry points; keep these thin
│   ├── train.py                      # Start or resume a configured training run
│   ├── evaluate.py                   # Evaluate a learned checkpoint on a test set
│   ├── benchmark_jpeg.py             # Run the JPEG baseline
│   ├── compress.py                   # Compress an input image
│   ├── decompress.py                 # Reconstruct an image from a bitstream
│   └── report.py                     # Generate plots and report-ready tables
│
├── tests/                            # Fast automated checks; default tests use synthetic data
│   ├── conftest.py                   # Shared fixtures and synthetic images
│   ├── test_data.py                  # Dataset, crop, and verification behavior
│   ├── test_training.py              # Loss and short training-step behavior
│   ├── test_codec.py                 # Encode/decode and dimension restoration
│   ├── test_metrics.py               # Metric and BPP calculations
│   └── test_jpeg.py                  # JPEG roundtrip and measured file sizes
│
├── notebooks/                        # Guided learning and result exploration
│   ├── data_and_crops.ipynb          # Inspect datasets and training crops
│   ├── hyperprior_experiments.ipynb  # Explain model, training, and inference experiments
│   └── inspect_results.ipynb         # Explore saved CSVs and figures
│
├── references/                       # External comparison data and its provenance
│   ├── elic_points.csv               # Published ELIC rates/metrics with source details
│   └── README.md                     # Dataset, figure/table, and citation notes
│
├── paper/                            # Report source and committed publication figures
│   ├── report.tex                    # Project report
│   ├── references.bib                # Bibliography
│   └── figures/                      # Selected report figures
│
├── data/                             # Local source data; never commit image datasets
│   ├── div2k/
│   │   ├── train/                    # DIV2K training images
│   │   └── val/                      # Held-out DIV2K validation images
│   ├── kodak/                        # Kodak test images; evaluation only
│   └── tecnick/                      # Tecnick test images; evaluation only
│
├── runs/                             # Generated training artifacts; ignored by Git
│   └── experiment_id/
│       ├── config.yaml               # Resolved settings used by this run
│       ├── provenance.json           # Code revision, environment, seed, and data manifest
│       ├── checkpoints/              # Best and resumable training states
│       └── logs/                     # Training/validation history and console log
│
└── results/                          # Generated evaluation artifacts; ignored by Git
    └── benchmark_id/
        ├── provenance.json           # Checkpoint, dataset, protocol, and environment
        ├── per_image.csv             # Individual image measurements
        ├── summary.csv               # Dataset-level aggregate metrics
        ├── figures/                  # Rate-distortion and timing plots
        └── reconstructions/          # Optional decoded images and comparison crops
```

## Command Convention

For the proposed command-line scripts, run from the repository root so imports use the `src` namespace without installing a package:

```powershell
python -m scripts.train --config configs/smoke_test.yaml
python -m scripts.evaluate --checkpoint runs/experiment_id/checkpoints/best.pth --dataset kodak
python -m scripts.compress --checkpoint runs/experiment_id/checkpoints/best.pth --input image.png --output image.bin
python -m scripts.decompress --checkpoint runs/experiment_id/checkpoints/best.pth --input image.bin --output reconstruction.png
```

These are target interfaces; update them to match the actual command-line arguments when the scripts are implemented. Keep DIV2K validation separate from training and use Kodak/Tecnick only for final evaluation. The exact dataset path layout must be kept consistent with `src/data/paths.py` and `src/data/download_data.py`.
