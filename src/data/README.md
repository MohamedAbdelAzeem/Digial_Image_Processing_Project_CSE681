# Student A: Data Setup

## W1: Download and verify

From the repository root, download and verify all three datasets automatically:

```powershell
python -m data.download_data
```

The files are placed in `data/raw/` by default, or in the folder configured by
`DIP_DATA_ROOT`. To choose a folder for one run, pass it directly:

```powershell
python -m data.download_data --data-root "D:\datasets\dip"
```

The downloader fetches the official DIV2K train/validation archives and the 24
Kodak PNG images, extracts them into the layout below, and verifies the images.
It removes each DIV2K ZIP after successful extraction to save disk space.

Each student downloads the datasets on their own laptop into one folder (the
*data root*) with this layout:

```text
<data root>/
├── DIV2K_train_HR/
│   ├── 0001.png
│   └── ...          (800 images)
├── DIV2K_valid_HR/
│   ├── 0801.png
│   └── ...          (100 images)
└── kodak/
    ├── kodim01.png
    └── ...
```

All code reads the location from `data/paths.py` (`RAW_DATA_ROOT`, `DIV2K_DIR`,
`DIV2K_VALID_DIR`, `KODAK_DIR`). The default is `data/raw/`. To use another folder, set the
`DIP_DATA_ROOT` environment variable once (new terminals and VS Code pick it up
after a restart):

```powershell
[Environment]::SetEnvironmentVariable("DIP_DATA_ROOT", "C:\path\to\data_root", "User")
```

Prefer the variable over editing `DEFAULT_DATA_ROOT`, so personal paths never
end up in git.

Training uses both DIV2K splits (train + valid, 800 + 100 images), as the project
spec requires. Kodak is reserved for evaluation and must not be used as
training data. Run all commands from the repository root:

```powershell
python -m data.verify_data
python -m data.verify_data --div2k-dir path\to\DIV2K_train_HR --kodak-dir path\to\kodak
```

The validator checks that each folder exists, contains supported image files,
and that every discovered file can be decoded. It reports image modes but does
not require images to be pre-converted; the W2 dataset converts them to RGB.

## W2: DIV2K dataset loader

Install PyTorch and Pillow in the project environment, then use the loader with
a PyTorch `DataLoader`:

```python
from torch.utils.data import DataLoader

from data.dataset import DIV2KDataset

dataset = DIV2KDataset(crop_size=256)  # DIV2K train + valid from data.paths
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
batch = next(iter(loader))
print(batch.shape)  # torch.Size([4, 3, 256, 256])
print(batch.min().item(), batch.max().item())  # values in [0.0, 1.0]
```

Each item is an RGB `float32` tensor shaped `[3, crop_size, crop_size]` with
values in `[0, 1]`. Random crop coordinates and optional horizontal/vertical
flips are sampled on each access. Images smaller than the crop size raise a
clear error instead of being silently resized.

## W3: Pillow JPEG baseline

`jpeg_baseline.py` encodes each full-size image in memory with Pillow's default
JPEG settings (4:2:0 chroma subsampling), decodes it, and reports the mean
bpp (`8 * bytes / (W * H)`), RGB PSNR (peak 255) and decode time per quality:

```powershell
python -m data.jpeg_baseline
python -m data.jpeg_baseline --qualities 10 50 95 --save-dir results\jpeg_kodak
```

`--save-dir` writes the exact bitstreams that were measured, for visual checks.
Reference output on Kodak (24 images):

| Q | bpp | PSNR (dB) |
|---:|---:|---:|
| 10 | 0.327 | 26.67 |
| 30 | 0.660 | 30.49 |
| 50 | 0.906 | 32.17 |
| 95 | 3.392 | 40.56 |