"""Train one configured experiment, for example:

python -m scripts.train --config configs/smoke_test.yaml --data-root data
python -m scripts.train --resume runs/<run> --set max_epochs=5
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from data.paths import DIV2K_DIR, DIV2K_VALID_DIR  # noqa: E402
from training.run import run_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", type=Path, help="YAML file with settings that differ from configs/base.yaml")
    parser.add_argument("--resume", type=Path, help="run folder to continue instead of starting a new run")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(os.environ.get("DIP_RUNS_ROOT", REPOSITORY / "runs")),
        help="folder for run folders (default: $DIP_RUNS_ROOT or <repository>/runs)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="folder holding DIV2K_train_HR and DIV2K_valid_HR (default: $DIP_DATA_ROOT or <repository>/data)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one setting, e.g. --set max_steps=50",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()
    if (args.config is None) == (args.resume is None):
        parser.error("give exactly one of --config or --resume")

    overrides = {}
    for item in args.set:
        key, separator, value = item.partition("=")
        if not separator:
            parser.error(f"--set expects KEY=VALUE, got {item!r}")
        overrides[key] = yaml.safe_load(value)

    train_dir, valid_dir = DIV2K_DIR, DIV2K_VALID_DIR
    if args.data_root is not None:
        train_dir, valid_dir = args.data_root / "DIV2K_train_HR", args.data_root / "DIV2K_valid_HR"

    summary = run_experiment(
        args.config,
        args.runs_root,
        train_dir=train_dir,
        valid_dir=valid_dir,
        overrides=overrides,
        resume_run=args.resume,
        log_interval=args.log_interval,
    )
    print(f"Run folder: {args.runs_root / summary['run'] if args.resume is None else args.resume}")
    print(f"Steps: {summary['steps']}, mean step time: {summary['mean_train_step_s']:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
