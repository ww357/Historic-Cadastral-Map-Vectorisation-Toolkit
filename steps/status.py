"""
Report how far a sheet has progressed through the pipeline.

Answers "where am I on this sheet?" by inspecting what actually exists on disk —
patches, annotations, weights, predictions, GeoPackage layers, feedback tiles —
and suggests the next command to run.

Deliberately depends on nothing beyond the standard library + PyYAML, so it runs
unchanged in all three conda environments (maptools / lines / polygons).
GeoPackage layers are read with sqlite3 rather than geopandas for the same reason.

Usage:
    python steps/status.py                 # one line per sheet found in data/raw/
    python steps/status.py --sheet SHEET   # full breakdown for one sheet
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

TICK, CROSS = "OK", "--"

# Feature folders that are not MapSAM polygon classes; everything else found
# under annotations/ or predictions/ is treated as a MapSAM feature.
_NON_MAPSAM = {"boundaries", "dashed", "text", "parcels", "parcel"}


def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


def _count_pngs(d: Path) -> int:
    return len(list(d.glob("*.png"))) if d.is_dir() else 0


def _gpkg_layers(path: Path) -> list[str]:
    """Layer names in a GeoPackage, via sqlite3 (no geopandas/fiona needed)."""
    if not path.exists():
        return []
    con = sqlite3.connect(str(path))
    try:
        return [r[0] for r in con.execute("SELECT table_name FROM gpkg_contents")]
    except sqlite3.Error:
        return []
    finally:
        con.close()


def _resolve_mended(mended_dir: Path, sheet: str) -> Path | None:
    """Same fuzzy match the vectorise steps use: exact name, else contains sheet."""
    if not mended_dir.is_dir():
        return None
    exact = mended_dir / f"{sheet}.gpkg"
    if exact.exists():
        return exact
    hits = sorted(p for p in mended_dir.glob("*.gpkg") if sheet.lower() in p.stem.lower())
    return hits[0] if hits else None


def gather(sheet: str, cfg: dict) -> dict:
    """Collect every on-disk fact about one sheet. No printing."""
    paths = cfg["paths"]
    boundary_label = cfg.get("annotation", {}).get("boundary_label", "boundary")

    raw       = ROOT / paths["raw"] / sheet / f"{sheet}.tif"
    patch_dir = ROOT / paths["patches"] / "images" / sheet
    meta_csv  = ROOT / paths["patches"] / "metadata" / f"{sheet}_patches.csv"
    ann_root  = ROOT / paths["annotations"]
    pred_root = ROOT / paths["predictions"]
    stitch    = ROOT / paths["stitched"]
    gpkg      = ROOT / paths["outputs"] / f"{sheet}.gpkg"
    mended    = _resolve_mended(ROOT / paths.get("outputs_mended", "data/mended outputs"), sheet)

    # Map-area mask: any of the three spellings patchify/parcels-predict accept.
    mask = next((p for p in (ROOT / paths["masks"] / sheet / f"{sheet}.png",
                             ROOT / paths["masks"] / sheet / f"{sheet}.tif",
                             ROOT / paths["masks"] / f"{sheet}.png") if p.exists()), None)

    # Annotations: label -> patch count (labels are whatever was drawn in labelme).
    annotations: dict[str, int] = {}
    if ann_root.is_dir():
        for d in sorted(ann_root.iterdir()):
            if not d.is_dir():
                continue
            n = _count_pngs(d / sheet / "masks")
            if n:
                annotations[d.name] = n

    # Predictions: folder -> count (text/parcels are single GeoJSON files).
    predictions: dict[str, int] = {}
    if pred_root.is_dir():
        for d in sorted(pred_root.iterdir()):
            if not d.is_dir():
                continue
            if d.name == "text":
                predictions["text"] = 1 if (d / sheet / "text_preds.geojson").exists() else 0
            elif d.name == "parcels":
                predictions["parcels"] = 1 if (d / sheet / "parcel_preds.geojson").exists() else 0
            else:
                predictions[d.name] = _count_pngs(d / sheet)

    # Stitched full-sheet rasters (inputs to vectorise and the parcel watershed).
    stitched = {}
    if stitch.is_dir():
        for d in sorted(stitch.iterdir()):
            if d.is_dir() and (d / f"{sheet}.tif").exists():
                stitched[d.name] = True

    # Weights. Lines are per-sheet; dashed is one pooled model; MapSAM is per feature.
    fin = ROOT / paths["models_finetuned"]
    mapsam_weights = {
        f: sorted(fin.rglob(f"mapsam_{f}*_best.pth"))[-1].name
        for f in annotations
        if f not in _NON_MAPSAM and f != boundary_label
        and sorted(fin.rglob(f"mapsam_{f}*_best.pth"))
    }
    weights = {
        "lines":  (fin / "working" / f"{sheet}_best.weights.h5").exists(),
        "dashed": bool(list((ROOT / paths["models_base"] / "dashed").glob("*.h5"))),
        "mapsam": mapsam_weights,
    }

    # Feedback artefacts produced after mending.
    feedback = {
        "lines":    (ROOT / "data" / "feedback" / boundary_label / sheet / "eligible.csv").exists(),
        "polygons": sorted(
            d.name for d in (ann_root.iterdir() if ann_root.is_dir() else [])
            if d.is_dir() and _count_pngs(d / "feedback" / sheet / "masks")
        ),
    }

    return {
        "sheet": sheet, "raw": raw.exists(), "mask": mask,
        "patches": _count_pngs(patch_dir), "meta": meta_csv.exists(),
        "annotations": annotations, "weights": weights,
        "predictions": predictions, "stitched": stitched,
        "gpkg": gpkg, "layers": _gpkg_layers(gpkg),
        "mended": mended, "mended_layers": _gpkg_layers(mended) if mended else [],
        "feedback": feedback, "boundary_label": boundary_label,
    }


def suggest_next(s: dict) -> str | None:
    """First gap in the canonical order, as a runnable run.py command."""
    sheet = s["sheet"]
    if not s["raw"]:
        return f"# place the GeoTIFF at data/raw/{sheet}/{sheet}.tif first"
    if not s["patches"]:
        flag = " --mask" if s["mask"] else ""
        return f"python run.py patchify --sheet {sheet}{flag}"
    if not s["annotations"]:
        return f"python run.py annotate --sheet {sheet}"

    label = s["boundary_label"]
    # Boundary lines: train -> predict -> vectorise
    if label in s["annotations"]:
        if not s["weights"]["lines"]:
            return f"python run.py train-lines --sheet {sheet}"
        if not s["predictions"].get("boundaries"):
            return f"python run.py predict-lines --sheet {sheet}"
        if "boundaries" not in s["layers"]:
            return f"python run.py vectorise-lines --sheet {sheet}"
    # Dashed lines
    if "dashed" in s["annotations"]:
        if not s["weights"]["dashed"]:
            return "python run.py train-dashed"
        if not s["predictions"].get("dashed"):
            return f"python run.py predict-dashed --sheet {sheet}"
        if "dashed" not in s["layers"]:
            return f"python run.py vectorise-dashed --sheet {sheet}"
    # MapSAM polygon features
    for f in s["annotations"]:
        if f in _NON_MAPSAM or f == label:
            continue
        if f not in s["weights"]["mapsam"]:
            return f"python run.py train-polygons --sheet {sheet} --feature {f}"
        if not s["predictions"].get(f):
            return f"python run.py predict-polygons --sheet {sheet} --feature {f}"
        if f not in s["layers"]:
            return f"python run.py vectorise-polygons --sheet {sheet} --feature {f}"
    # Parcels need the stitched boundary raster to exist first.
    if s["stitched"].get("boundaries") and not s["predictions"].get("parcels"):
        return f"python run.py predict-parcels --sheet {sheet}"
    if s["predictions"].get("parcels") and "parcels" not in s["layers"]:
        return f"python run.py vectorise-parcels --sheet {sheet}"
    return None


def print_report(s: dict) -> None:
    sheet = s["sheet"]
    print(f"\nSheet: {sheet}")
    print("-" * 60)

    print(f"  raw tif      {TICK if s['raw'] else CROSS}")
    print(f"  area mask    {TICK + '  ' + s['mask'].name if s['mask'] else CROSS + '  (optional)'}")

    meta = "" if s["meta"] else "   (metadata CSV missing)"
    print(f"  01 patchify  {TICK if s['patches'] else CROSS}  {s['patches']} patches{meta}")

    if s["annotations"]:
        cls = "  ".join(f"{k}({v})" for k, v in s["annotations"].items())
        print(f"  02 annotate  {TICK}  {cls}")
    else:
        print(f"  02 annotate  {CROSS}")

    w = s["weights"]
    bits = []
    if w["lines"]:
        bits.append(f"lines: working/{sheet}_best.weights.h5")
    if w["dashed"]:
        bits.append("dashed: base/dashed (pooled)")
    bits += [f"{f}: {n}" for f, n in w["mapsam"].items()]
    print(f"  03 finetune  {TICK if bits else CROSS}" + ("  " + "\n" + " " * 15 + ("\n" + " " * 15).join(bits) if bits else ""))

    done = {k: v for k, v in s["predictions"].items() if v}
    if done:
        first = True
        for name, n in s["predictions"].items():
            if not n:
                continue
            shown = "geojson" if name in ("text", "parcels") else f"{n} patches"
            label = f"  04 predict   {TICK}" if first else " " * 15
            print(f"{label}  {name}: {shown}")
            first = False
        missing = [k for k, v in s["predictions"].items() if not v]
        if missing:
            print(f"{' ' * 15}  not run: {', '.join(missing)}")
    else:
        print(f"  04 predict   {CROSS}")

    if s["stitched"]:
        print(f"  (stitched)   {', '.join(sorted(s['stitched']))}")

    if s["layers"]:
        print(f"  05 vectorise {TICK}  {s['gpkg'].name}: {', '.join(sorted(s['layers']))}")
    else:
        print(f"  05 vectorise {CROSS}  no {s['gpkg'].name}")
    if s["mended"]:
        print(f"  (mended)     {s['mended'].name}: {', '.join(sorted(s['mended_layers']))}")

    fb = s["feedback"]
    fb_bits = (["lines"] if fb["lines"] else []) + fb["polygons"]
    print(f"  06 feedback  {TICK + '  ' + ', '.join(fb_bits) if fb_bits else CROSS}")

    nxt = suggest_next(s)
    print("-" * 60)
    print(f"  Next: {nxt}" if nxt else "  Next: nothing outstanding — mend in QGIS, then run feedback.")
    print()


def print_summary(sheets: list[str], cfg: dict) -> None:
    """One line per sheet: counts at each stage."""
    print(f"\n{'sheet':<28} {'patches':>8} {'annot':>6} {'preds':>6}  layers")
    print("-" * 76)
    for sheet in sheets:
        s = gather(sheet, cfg)
        n_pred = sum(1 for v in s["predictions"].values() if v)
        print(f"{sheet:<28} {s['patches']:>8} {len(s['annotations']):>6} {n_pred:>6}  "
              f"{', '.join(sorted(s['layers'])) or '-'}")
    print(f"\nRun 'python run.py status --sheet SHEET' for a full breakdown.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Show pipeline progress for a sheet.")
    ap.add_argument("--sheet", default=None,
                    help="Sheet ID. Omit to list every sheet in data/raw/.")
    args = ap.parse_args()

    cfg = load_config()
    if args.sheet:
        print_report(gather(args.sheet, cfg))
        return

    raw_root = ROOT / cfg["paths"]["raw"]
    sheets = sorted(d.name for d in raw_root.iterdir() if d.is_dir()) if raw_root.is_dir() else []
    if not sheets:
        sys.exit(f"No sheets found in {raw_root}")
    print_summary(sheets, cfg)


if __name__ == "__main__":
    main()
