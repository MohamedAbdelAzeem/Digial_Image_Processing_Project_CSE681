"""Check that the downloaded DIV2K and Kodak image folders are usable."""

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def inspect_folder(name: str, folder: Path) -> bool:
    if not folder.is_dir():
        print(f"FAIL {name}: folder does not exist: {folder}")
        return False

    paths = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        print(f"FAIL {name}: no supported image files found in {folder}")
        return False

    modes = Counter()
    invalid = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                modes[image.mode] += 1
        except (OSError, UnidentifiedImageError) as error:
            invalid.append((path, error))

    print(f"{name}: found {len(paths)} image file(s), decoded {len(paths) - len(invalid)}")
    print(f"  Modes: {dict(modes)}")
    for path, error in invalid:
        print(f"  UNREADABLE {path}: {error}")

    if invalid:
        print(f"FAIL {name}: remove or replace unreadable image files before training")
        return False

    print(f"OK {name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--div2k-dir", type=Path, default=Path("data/raw/DIV2K_train_HR"))
    parser.add_argument("--kodak-dir", type=Path, default=Path("data/raw/kodak"))
    args = parser.parse_args()

    results = [
        inspect_folder("DIV2K train", args.div2k_dir),
        inspect_folder("Kodak", args.kodak_dir),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())