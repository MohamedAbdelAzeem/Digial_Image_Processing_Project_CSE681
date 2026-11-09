"""Download and verify the DIV2K and Kodak datasets used by this project."""

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from data.paths import RAW_DATA_ROOT
from data.verify_data import inspect_folder


DIV2K_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/{name}.zip"
KODAK_URL = "https://r0k.us/graphics/kodak/kodak/kodim{index:02d}.png"
CHUNK_SIZE = 1024 * 1024


def download_file(url: str, destination: Path) -> None:
    partial_path = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DIP-Project-Dataset-Downloader"})
    downloaded = 0

    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial_path.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100 // total
                    print(f"\r  Downloaded {downloaded:,} / {total:,} bytes ({percent}%)", end="")
                elif downloaded % (50 * CHUNK_SIZE) < CHUNK_SIZE:
                    print(f"\r  Downloaded {downloaded:,} bytes", end="")
        print()
        partial_path.replace(destination)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def png_count(folder: Path) -> int:
    return sum(1 for path in folder.rglob("*.png") if path.is_file())


def download_div2k_split(root: Path, name: str, expected_count: int) -> None:
    destination = root / name
    if destination.exists():
        count = png_count(destination) if destination.is_dir() else 0
        if count == expected_count:
            print(f"{name}: already present ({count} images)")
            return
        raise ValueError(f"{destination} already exists but contains {count} PNG images; move it aside and retry")

    archive_path = root / f"{name}.zip"
    if not archive_path.exists():
        print(f"Downloading {name} from the official DIV2K site")
        download_file(DIV2K_URL.format(name=name), archive_path)
    elif not zipfile.is_zipfile(archive_path):
        raise ValueError(f"Existing archive is not a valid ZIP file: {archive_path}")

    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=root) as temporary_dir:
        temporary_root = Path(temporary_dir)
        extracted = temporary_root / "extracted"
        normalized = temporary_root / name
        extracted.mkdir()
        normalized.mkdir()

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extracted)

        images = sorted(path for path in extracted.rglob("*.png") if path.is_file())
        if len(images) != expected_count:
            raise ValueError(f"{name} archive contains {len(images)} PNG images; expected {expected_count}")

        for image_path in images:
            output_path = normalized / image_path.name
            if output_path.exists():
                raise ValueError(f"Duplicate image filename in {name} archive: {image_path.name}")
            shutil.copy2(image_path, output_path)

        normalized.replace(destination)

    archive_path.unlink()
    print(f"Extracted {name} ({expected_count} images)")


def download_kodak(root: Path) -> None:
    destination = root / "kodak"
    destination.mkdir(parents=True, exist_ok=True)

    for index in range(1, 25):
        image_path = destination / f"kodim{index:02d}.png"
        if image_path.exists():
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(f"Existing Kodak image is unreadable: {image_path}") from error
            continue

        print(f"Downloading {image_path.name}")
        download_file(KODAK_URL.format(index=index), image_path)
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            image_path.unlink(missing_ok=True)
            raise ValueError(f"Downloaded Kodak image is unreadable: {image_path}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=RAW_DATA_ROOT)
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()

    try:
        root.mkdir(parents=True, exist_ok=True)
        download_div2k_split(root, "DIV2K_train_HR", 800)
        download_div2k_split(root, "DIV2K_valid_HR", 100)
        download_kodak(root)

        folders = (
            ("DIV2K train", root / "DIV2K_train_HR", 800),
            ("DIV2K valid", root / "DIV2K_valid_HR", 100),
            ("Kodak", root / "kodak", 24),
        )
        valid = True
        for name, folder, expected_count in folders:
            valid = inspect_folder(name, folder) and valid
            count = png_count(folder)
            if count != expected_count:
                print(f"FAIL {name}: found {count} PNG images, expected {expected_count}")
                valid = False
        return 0 if valid else 1
    except (OSError, ValueError, urllib.error.URLError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())