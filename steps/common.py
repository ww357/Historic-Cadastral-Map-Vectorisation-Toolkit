"""
Shared helpers used across the pipeline steps.

Kept deliberately dependency-light - standard library + PyYAML only - so it
imports cleanly in every conda environment (maptools / lines / polygons). The one
torch helper imports torch lazily, inside the function, so this module still loads
fine where torch is absent.

Import it by putting the steps/ directory on sys.path, e.g.:

    ROOT = Path(__file__).resolve().parents[2]     # repo root
    sys.path.insert(0, str(ROOT / "steps"))
    from common import load_config, find_raw
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Repo root - steps/ sits one level below it, so parents[1] of this file.
ROOT = Path(__file__).resolve().parents[1]

# Raw map formats in resolution-priority order: georeferenced / wrapped forms
# first, plain images last. GDAL reads world-file + .prj sidecars for jpg/png/tif.
RAW_EXTENSIONS = (".tif", ".tiff", ".vrt", ".jpg", ".jpeg", ".png")


# -- Config ------------------------------------------------------------------

def load_config() -> dict:
    """Load config.yaml from the repo root."""
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


def rel_to_root(p: Path) -> str:
    """Path relative to the repo root for tidy printing; absolute if outside it."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# -- Input file resolution ---------------------------------------------------

def find_raw(raw_root: Path, sheet_id: str) -> Path | None:
    """Return data/raw/<sheet>/<sheet>.<ext> for the first supported extension."""
    for ext in RAW_EXTENSIONS:
        p = raw_root / sheet_id / f"{sheet_id}{ext}"
        if p.exists():
            return p
    return None


def find_mask(mask_dir: Path, sheet_id: str) -> Path | None:
    """Return a map-area mask (data/map_area_masks/<sheet>/<sheet>.png/.tif), or None."""
    for ext in (".png", ".tif", ".tiff"):
        p = mask_dir / f"{sheet_id}{ext}"
        if p.exists():
            return p
    return None


# -- GeoPackage target resolution (the --mended / --gpkg rule) ----------------
#
# One rule everywhere: default is paths.outputs; --mended uses the hand-corrected
# GeoPackage in paths.outputs_mended; --gpkg overrides both. Writers use
# resolve_output_gpkg, feedback readers use resolve_input_gpkg (which warns rather
# than silently ignoring a mended file that exists).

def find_mended(sheet_id: str, cfg: dict) -> Path | None:
    """Hand-corrected GeoPackage for the sheet in paths.outputs_mended, or None.
    Exact name first, then any *.gpkg whose name contains the sheet id
    (e.g. 'Porlock mended.gpkg')."""
    d = ROOT / cfg["paths"].get("outputs_mended", "data/mended outputs")
    if not d.is_dir():
        return None
    exact = d / f"{sheet_id}.gpkg"
    if exact.exists():
        return exact
    hits = sorted(p for p in d.glob("*.gpkg") if sheet_id.lower() in p.stem.lower())
    return hits[0] if hits else None


def _mended_dir(cfg: dict) -> Path:
    return ROOT / cfg["paths"].get("outputs_mended", "data/mended outputs")


def resolve_output_gpkg(sheet_id: str, cfg: dict, gpkg_arg: str | None,
                        mended: bool) -> Path:
    """
    Pick the GeoPackage a vectorise step should WRITE to.

    The target is never inferred from what happens to exist on disk: these steps
    drop and rewrite their own layers, so silently redirecting into a
    hand-corrected file would destroy mending.
    """
    if gpkg_arg:
        p = Path(gpkg_arg)
        return p if p.is_absolute() else ROOT / p
    if not mended:
        return ROOT / cfg["paths"]["outputs"] / f"{sheet_id}.gpkg"
    found = find_mended(sheet_id, cfg)
    if found is None:
        sys.exit(
            f"--mended: no GeoPackage for sheet '{sheet_id}' in {_mended_dir(cfg)}\n"
            f"Looked for '{sheet_id}.gpkg' and any *.gpkg with '{sheet_id}' in the name.\n"
            f"Put the mended file there, or drop --mended to write to "
            f"{cfg['paths']['outputs']}{sheet_id}.gpkg."
        )
    return found


def resolve_input_gpkg(sheet_id: str, cfg: dict, gpkg_arg: str | None,
                       mended: bool) -> Path:
    """
    Pick the GeoPackage a feedback step should READ from.

    If a mended file exists but --mended was not passed, warn loudly rather than
    silently training on the un-mended file - that would discard the corrections
    the feedback loop exists to capture.
    """
    if gpkg_arg:
        p = Path(gpkg_arg)
        return p if p.is_absolute() else ROOT / p
    if mended:
        found = find_mended(sheet_id, cfg)
        if found is None:
            sys.exit(
                f"--mended: no GeoPackage for sheet '{sheet_id}' in {_mended_dir(cfg)}\n"
                f"Looked for '{sheet_id}.gpkg' and any *.gpkg with '{sheet_id}' in the name."
            )
        return found

    default = ROOT / cfg["paths"]["outputs"] / f"{sheet_id}.gpkg"
    available = find_mended(sheet_id, cfg)
    if available is not None:
        print(
            f"\n  ! A mended GeoPackage exists for this sheet:\n"
            f"      {available}\n"
            f"    but --mended was not passed, so corrections in it will be IGNORED\n"
            f"    and training data will come from {default.name} instead.\n"
            f"    Re-run with --mended to use the corrected layers.\n"
        )
    return default


# -- NumPy 2.x / PyTorch compatibility ----------------------------------------

def tensor_from_numpy(arr):
    """torch.from_numpy replacement compatible with NumPy 2.x.

    torch.from_numpy checks the C-level numpy.ndarray type, which changed in
    NumPy 2.0, so it fails with "expected np.ndarray (got numpy.ndarray)" when
    PyTorch was compiled against NumPy 1.x. Routing through memoryview /
    torch.frombuffer bypasses that check regardless of NumPy version.

    torch/numpy are imported lazily so this module still loads where torch is
    absent (only the polygons-env training scripts call this).
    """
    import numpy as np
    import torch

    arr = np.ascontiguousarray(arr)
    return (torch.frombuffer(memoryview(arr), dtype=torch.float32)
                 .reshape(arr.shape)
                 .clone())
