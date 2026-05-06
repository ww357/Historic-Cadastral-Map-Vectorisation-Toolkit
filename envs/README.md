# Environment Files

## Two-tier system

| File | Purpose |
|---|---|
| `*.yml` | **Fully pinned** export from the development machine. Use for exact reproduction. |
| `*-base.yml` | **Minimal portable** spec. Use as a starting point on different hardware/OS. |

## Environments

| Environment | Used by | GPU | Python | Key framework |
|---|---|---|---|---|
| `maptools` | steps 01, 02, 05, 06 (text_to_vector), 07 (rasterise) | None | 3.11 | rasterio / geopandas |
| `tf-gpu` | steps 03, 04, 07 (train) | CUDA 12.x | 3.11 | TensorFlow 2.21 |
| `New-MapReader` | step 06 (predict) | CUDA 12.1 | 3.12 | PyTorch 2.2 + detectron2 |
| `MapSAM` | steps 04, 05 (mapsam) | CUDA 11.1 | 3.8 | PyTorch 1.9.1 |

## Development machine

- **GPU**: NVIDIA GeForce RTX 3050 Laptop (4 GB VRAM)
- **Driver**: 581.95 · **CUDA runtime**: 13.0 · **nvcc toolkit**: 12.1.66
- **OS**: Ubuntu (WSL2 on Windows)

## Non-PyPI packages (require manual install)

These packages are **not on PyPI** and will not be installed by the yml files alone:

### New-MapReader
```bash
# 1. MapTextPipeline — install from local repo
pip install -e models/MapTextPipeline/

# 2. detectron2 — build from GitHub (match your CUDA version)
pip install git+https://github.com/facebookresearch/detectron2.git
# OR download a pre-built wheel from:
# https://github.com/facebookresearch/detectron2/releases
```

## Recreating an environment

### Exact reproduction (same hardware, linux-64)
```bash
conda env create -f envs/maptools.yml
conda env create -f envs/tf-gpu.yml
conda env create -f envs/New-MapReader.yml
conda env create -f envs/MapSAM.yml
```

### Different hardware — use base specs
```bash
conda env create -f envs/maptools-base.yml        # no GPU required
conda env create -f envs/tf-gpu-base.yml          # adjust CUDA version in comments if needed
conda env create -f envs/New-MapReader-base.yml   # adjust PyTorch index URL if needed
conda env create -f envs/MapSAM-base.yml          # legacy CUDA 11.1 — see comments
```

## Known compatibility constraints

| Environment | Constraint | Reason |
|---|---|---|
| All | linux-64 pinned exports | Build strings are platform-specific |
| `New-MapReader` | `numpy<2.0` | detectron2 0.6 incompatible with numpy 2.x |
| `tf-gpu` | `numpy<2.0` | TensorFlow 2.x incompatible with numpy 2.x |
| `MapSAM` | Python 3.8, `numpy<1.25` | monai 1.1.0 + scipy 1.7.3 compatibility |
| `MapSAM` | `torch==1.9.1+cu111` | Very old wheel — legacy index required |
