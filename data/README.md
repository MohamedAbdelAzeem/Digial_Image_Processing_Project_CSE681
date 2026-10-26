# Student A: Data Setup

## W1: Download and verify

Place the extracted DIV2K training images and Kodak images in these folders:

```text
data/raw/
├── DIV2K_train_HR/
│   ├── 0001.png
│   └── ...
└── kodak/
    ├── kodim01.png
    └── ...
```

The DIV2K folder should contain the high-resolution training split, not the
validation images. Kodak is reserved for evaluation and must not be used as
training data. The folder names can differ; pass their actual paths to the
validator:

```powershell
python .\data\verify_data.py
python .\data\verify_data.py --div2k-dir path\to\DIV2K_train_HR --kodak-dir path\to\kodak
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

dataset = DIV2KDataset("data/raw/DIV2K_train_HR", crop_size=192)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
batch = next(iter(loader))
print(batch.shape)  # torch.Size([4, 3, 192, 192])
print(batch.min().item(), batch.max().item())  # values in [0.0, 1.0]
```

Each item is an RGB `float32` tensor shaped `[3, crop_size, crop_size]` with
values in `[0, 1]`. Random crop coordinates and optional horizontal/vertical
flips are sampled on each access. Images smaller than the crop size raise a
clear error instead of being silently resized.