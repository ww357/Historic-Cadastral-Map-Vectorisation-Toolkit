"""
Shared labelme JSON writing, used by every step that pre-fills annotations
(02_annotate/autolines.py, 02_annotate/export_manual.py).

Kept dependency-light (standard library only; labelme itself is imported lazily just to
read its version) so it loads in any conda environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


# `description` tag on shapes written from a manual vector layer (export_manual.py), so a
# re-run can replace exactly those shapes and never touch hand-drawn ones. autolines uses
# its own tag ("autolines") the same way.
MANUAL_LAYER_TAG = "manual-layer"


def labelme_version() -> str:
    try:
        import labelme
        return str(labelme.__version__)
    except Exception:
        return "5.5.0"


def write_labelme_json(json_path: Path, img_path: Path, shapes: list[dict],
                       width: int, height: int, extra_shapes: list[dict]) -> None:
    """
    Write a labelme annotation file: `extra_shapes` (kept as they are) followed by `shapes`.

    imagePath is stored relative to the JSON file: labelme resolves it from there when it
    loads an existing annotation, so a wrong value means the patch won't open.
    """
    rel = os.path.relpath(img_path, json_path.parent).replace("\\", "/")
    data = {
        "version": labelme_version(),
        "flags": {},
        "shapes": extra_shapes + shapes,
        "imagePath": rel,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2))
