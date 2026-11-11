import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import download_data  # noqa: E402


@pytest.mark.parametrize(
    "working_directory, arguments",
    [
        (".", ["src/data/download_data.py"]),
        ("src", ["-m", "data.download_data"]),
    ],
)
def test_downloader_help(working_directory, arguments):
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-E", *arguments, "--help"],
        cwd=repository / working_directory,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "--data-root" in result.stdout


def test_div2k_split_is_extracted_and_existing_data_is_skipped(tmp_path, monkeypatch):
    def create_test_archive(_url, destination):
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("DIV2K_train_HR/0001.png", b"image one")
            archive.writestr("DIV2K_train_HR/0002.png", b"image two")

    monkeypatch.setattr(download_data, "download_file", create_test_archive)

    download_data.download_div2k_split(tmp_path, "DIV2K_train_HR", 2)

    output = tmp_path / "DIV2K_train_HR"
    assert sorted(path.name for path in output.iterdir()) == ["0001.png", "0002.png"]
    assert not (tmp_path / "DIV2K_train_HR.zip").exists()

    def unexpected_download(*_args):
        pytest.fail("existing complete dataset should not be downloaded again")

    monkeypatch.setattr(download_data, "download_file", unexpected_download)
    download_data.download_div2k_split(tmp_path, "DIV2K_train_HR", 2)