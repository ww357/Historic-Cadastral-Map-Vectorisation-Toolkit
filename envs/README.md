# Environment Files

Three conda environments are used across the pipeline. **Never mix them** — the deep-learning dependencies conflict.

---

## Two-tier file structure

Each environment has two files:

| Tier | Filename pattern | Purpose |
|---|---|---|
| **Pinned** | `<name>.yml` | Exact full export from the development machine. Guarantees bit-for-bit reproduction on identical hardware. Use this first. |
| **Base** | `<name>-base.yml` | Minimal portable spec. Omits build hashes and hardware-specific pins. Use this when the pinned file fails (different OS, CUDA version, or driver). |

---

## Environment summary

| Environment | Steps | Python | Key packages |
|---|---|---|---|
| `maptools` | 01, 02, 05, 06b | 3.11 | rasterio, GDAL, geopandas, shapely, scikit-image, skan, labelme, osam |
| `lines` | 03 boundaries, 04 boundaries, 07 feedback | 3.11 | TensorFlow 2.21, Keras 3.13, numpy 1.x |
| `polygons` | 03 MapSAM, 04 MapSAM, 06a text | 3.12 | PyTorch 2.2.2+cu121, mapreader 1.8.2, monai 1.3.2, detectron2 0.6 |

`maptools` is also used for drawing masks (step 0) and all GeoPackage writes.

---

## Development machine

- OS: WSL2 / Ubuntu on Windows 11
- GPU: NVIDIA RTX 3050 4GB
- Driver: 581.95
- CUDA (system): 12.1
- nvcc: 12.1.66

---

## Creating environments

### maptools
```bash
conda env create -f envs/maptools.yml
# or portable:
conda env create -f envs/maptools-base.yml
```

### lines
```bash
conda env create -f envs/lines.yml
# or portable:
conda env create -f envs/lines-base.yml
```

### polygons
```bash
conda env create -f envs/polygons.yml
# or portable:
conda env create -f envs/polygons-base.yml
```

Then install the two non-PyPI packages (required for both pinned and base):

```bash
conda activate polygons

# 1. MapTextPipeline — local install from the repo
pip install -e models/MapTextPipeline/

# 2. detectron2 — prebuilt wheel (recommended)
pip install detectron2 \
  -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.2/index.html
```

---

## Known constraints

| Constraint | Affected env | Reason |
|---|---|---|
| `numpy<2.0` | lines, polygons | TF 2.21 and detectron2 0.6 both break with numpy 2.x |
| `monai==1.3.2` | polygons | monai >=1.4 requires torch>=2.4, silently upgrades torch and breaks mapreader + torchvision if installed without a version pin |
| `antlr4-python3-runtime==4.9.3` | polygons | hydra-core and omegaconf require exactly the 4.9.x series |
| `tensorflow==2.21.0` | lines | Keras 3.x checkpoint format is not backward compatible with Keras 2.x |
| `torch==2.2.2+cu121` | polygons | mapreader 1.8.2 requires torch<=2.2.2; always install from `https://download.pytorch.org/whl/cu121` |
| Python 3.11 | lines | TF 2.21 wheel is not available for Python 3.12 |

---

## CUDA delivery

| Environment | CUDA delivery method |
|---|---|
| `lines` | pip `nvidia-*-cu12` wheels — installed automatically as tensorflow dependencies, no conda CUDA channel needed |
| `polygons` | conda `nvidia/label/cuda-12.1.0` channel (compiler + headers for building detectron2) + pip `nvidia-*-cu12` wheels (used by torch at runtime) |

The `polygons` environment currently contains **duplicate CUDA 13 pip packages** alongside the correct CUDA 12 ones — a leftover from a failed `pip install monai` without a version pin that temporarily upgraded torch to 2.11.0. These packages are unused and harmless. See the comment at the top of `polygons.yml` for cleanup instructions.

---

## Activating environments

```bash
conda activate maptools     # geospatial steps, annotation, vectorise, text_to_vector
conda activate lines        # U-Net train, predict, feedback
conda activate polygons     # MapSAM train, predict, text spotting inference
```
