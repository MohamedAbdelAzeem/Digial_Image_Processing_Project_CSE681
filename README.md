# Digital Image Processing Project — CSE681

Lightweight Learned Image Compression for Laptop-Class Training.

## Create the project environment

All project dependencies are defined in `environment.yml`.

Install [Miniforge](https://github.com/conda-forge/miniforge) or another Conda
distribution, then open Anaconda Prompt or PowerShell in this repository and run:

```powershell
conda env create -f environment.yml
conda activate dip-project1
```

If you already created this environment using an older version of the YAML file,
update it to match the current dependencies:

```powershell
conda env update -n dip-project1 -f environment.yml --prune
conda activate dip-project1
```

## Download the datasets

From the repository root with `dip-project1` active, run the downloader as a module
from `src` so Python can resolve its `data` package:

```powershell
cd src
python -m data.download_data --data-root "D:/datasets/dip"
```

## Render the report template

Following the course guide, install [TeX Live](https://www.tug.org/texlive/)
separately from Conda. Include the LaTeX packages used by the template, such as
`geometry`, `natbib`, `booktabs`, and `hyperref`. Open a new terminal after
installation and confirm that the tools are available:

```powershell
pdflatex --version
bibtex --version
```

Edit `report.tex` and `references.bib`, then run the following commands from the
repository folder, proceeding only if each command succeeds:

```powershell
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

This creates `report.pdf` in the repository. BibTeX generates the bibliography;
the additional LaTeX runs resolve citations and cross-references. Keep one shared
`references.bib` for all three milestones. Conda activation is not required to
render the report.

Alternatively, upload `report.tex` and `references.bib` to
[Overleaf](https://www.overleaf.com/) and click **Recompile**.

## Project code dependencies

C++ build toolchain on Windows; see the
[CompressAI installation instructions](https://interdigitalinc.github.io/CompressAI/installation.html).
