"""Single place that says where the raw datasets live on this laptop.

Each student downloads DIV2K and Kodak locally, then either:
  * sets the environment variable DIP_DATA_ROOT to the folder that holds them (recommended,
    no file edits, so no git conflicts), or
  * edits DEFAULT_DATA_ROOT below.
"""

import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "raw"

RAW_DATA_ROOT = Path(os.environ.get("DIP_DATA_ROOT", DEFAULT_DATA_ROOT))
DIV2K_DIR = RAW_DATA_ROOT / "DIV2K_train_HR"
DIV2K_VALID_DIR = RAW_DATA_ROOT / "DIV2K_valid_HR"
KODAK_DIR = RAW_DATA_ROOT / "kodak"
