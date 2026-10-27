"""
Turn chosen patches of your hand-made vector layers into labelme annotations.

The manual layers (imported with `import-manual`) are finished work for THIS map, so they need
no model here. But their patches make good ground truth for fine-tuning on OTHER maps. This
clips each manual layer to the patches you choose and writes the pieces as labelme shapes into
data/annotations/labelme_json/<SHEET_ID>/<patch>.json - exactly where the labelme step keeps
its files. `export-masks` then turns them into masks, Patch_Grid's ann_<feature> columns light
up, and the training scripts read them like any other annotation.

  layer 'boundaries'/'dashed'   -> linestrip shapes, labelled 'boundary' / 'dashed'
  any other line layer          -> linestrip shapes, labelled with the layer name
  polygon layers                -> polygon shapes, labelled with the layer name
                                   (polygon holes are dropped - labelme polygons have none)
  point layers                  -> skipped (nothing to train a mask on)

Shapes carry description "manual-layer". Re-running replaces only those shapes in a patch;
anything you drew by hand in labelme is kept. A patch that already has manual-layer shapes is
skipped unless you pass --overwrite.

Choose patches (one option required):
  --select            the list saved by  `run.py pick-patches --sheet S --name manual`
  --patches ID ...    explicit patch ids (full, or short like r0012_c0031)
  --limit N           N patches spread across the sheet, from those that contain manual features
  --all               every patch that contains manual features

Usage:
    python export_manual.py --sheet SHEET_ID --select [--layers houses bog] [--overwrite]
                            [--empty-patches] [--mended | --gpkg PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from PIL import Image
from shapely.affinity import affine_transform
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "steps"))   # shared helpers
from common import load_config, rel_to_root, resolve_output_gpkg, selection_file   # noqa: E402
from geopackage import manual_layers   # noqa: E402
from annotation_helpers import MANUAL_LAYER_TAG, write_labelme_json   # noqa: E402
from vectorise_common import sheet_georef   # noqa: E402

LINE_LABELS_EXTRA = {"dashed"}                    # line classes besides the boundary label


def to_pixels(geom, georef: dict):
    """World coordinates -> sheet pixel coordinates (identity for an ungeoreferenced map)."""
    if not georef["has_georef"]:
        return geom
    inv = ~georef["transform"]
    return affine_transform(geom, [inv.a, inv.b, inv.d, inv.e, inv.c, inv.f])


def _pts(coords) -> list[list[float]]:
    out: list[list[float]] = []
    for x, y in coords:
        p = [round(float(x), 2), round(float(y), 2)]
        if not out or out[-1] != p:                       # drop repeated points
            out.append(p)
    return out


def shapes_from(geom, label: str, as_lines: bool, stats: Counter) -> list[dict]:
    """labelme shapes for one clipped geometry (patch pixel coordinates)."""
    shapes: list[dict] = []

    def add(points, kind):
        shapes.append({"label": label, "points": points, "group_id": None,
                       "description": MANUAL_LAYER_TAG, "shape_type": kind, "flags": {}})

    for part in shapely.get_parts(geom):
        t = part.geom_type
        if t == "LineString":
            p = _pts(part.coords)
            if len(p) >= 2:
                add(p, "linestrip")
        elif t == "Polygon":
            if as_lines:                                    # a line class drawn as polygons: use the outlines
                for ring in [part.exterior, *part.interiors]:
                    p = _pts(ring.coords)
                    if len(p) >= 2:
                        add(p, "linestrip")
            else:
                p = _pts(part.exterior.coords)[:-1]         # labelme polygons are not closed explicitly
                if len(p) >= 3:
                    add(p, "polygon")
                    stats["holes_dropped"] += len(part.interiors)
        elif t == "Point":
            stats["points_skipped"] += 1
    return shapes


def main() -> None:
    ap = argparse.ArgumentParser(description="Export patches of manual layers as labelme annotations.")
    ap.add_argument("--sheet", required=True, help="Sheet ID")
    ap.add_argument("--select", action="store_true", help="Use the list saved by `pick-patches --name manual`")
    ap.add_argument("--patches", nargs="+", metavar="ID", default=None, help="Explicit patch ids")
    ap.add_argument("--limit", type=int, default=None, help="N patches spread across the sheet")
    ap.add_argument("--all", action="store_true", help="Every patch that contains manual features")
    ap.add_argument("--layers", nargs="+", default=None, help="Only these manual layers (default: all)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace manual-layer shapes already exported for a patch")
    ap.add_argument("--empty-patches", action="store_true",
                    help="Also write an empty annotation for chosen patches with no features "
                         "(a confirmed 'nothing here' example; only the dashed trainer uses these)")
    target = ap.add_mutually_exclusive_group()
    target.add_argument("--mended", action="store_true", help="Read from the hand-corrected GeoPackage")
    target.add_argument("--gpkg", default=None, help="Explicit GeoPackage (overrides --mended and the default)")
    args = ap.parse_args()
    if not (args.select or args.patches or args.limit or args.all):
        ap.error("choose patches with one of: --select, --patches ID ..., --limit N, --all\n"
                 f"  (visual picker:  python run.py pick-patches --sheet {args.sheet} --name manual)")

    cfg, sheet = load_config(), args.sheet
    boundary_label = cfg.get("annotation", {}).get("boundary_label", "boundary")
    gpkg = resolve_output_gpkg(sheet, cfg, args.gpkg, args.mended)
    if not gpkg.exists():
        sys.exit(f"GeoPackage not found: {gpkg}\nImport your layers first:  python run.py import-manual --sheet {sheet}")
    layers = manual_layers(gpkg)
    if args.layers:
        want = {x.lower() for x in args.layers}
        layers = [n for n in layers if n.lower() in want]
    if not layers:
        sys.exit(f"No manual layers in {gpkg.name}" + (f" matching {args.layers}" if args.layers else "") +
                 f"\nImport them first:  python run.py import-manual --sheet {sheet}")

    meta_path = ROOT / cfg["paths"]["patches"] / "metadata" / f"{sheet}_patches.csv"
    patches_dir = ROOT / cfg["paths"]["patches"] / "images" / sheet
    json_dir = ROOT / cfg["paths"]["annotations"] / "labelme_json" / sheet
    if not meta_path.exists() or not patches_dir.exists():
        sys.exit(f"Patches/metadata not found for '{sheet}' - run patchify first.")
    meta = pd.read_csv(meta_path).set_index("patch_id")
    georef = sheet_georef(sheet, cfg)

    # Layers into pixel space once, with a spatial index for the per-patch lookups.
    data = {}
    for name in layers:
        g = gpd.read_file(gpkg, layer=name, columns=[])
        g["geometry"] = [to_pixels(geom, georef) for geom in g.geometry]
        label = boundary_label if name.lower() in ("boundaries", "boundary") else name
        data[name] = (g, g.sindex, label, label == boundary_label or label in LINE_LABELS_EXTRA)

    def inside(c0, r0, w, h):
        """The patch shrunk a hair: a feature that only touches the shared edge belongs to the neighbour."""
        return box(c0 + 0.01, r0 + 0.01, c0 + w - 0.01, r0 + h - 0.01)

    def window(pid: str):
        r = meta.loc[pid]
        with Image.open(patches_dir / f"{pid}.png") as im:
            w, h = im.size
        return int(r.col_off), int(r.row_off), w, h

    have = [pid for pid in meta.index if (patches_dir / f"{pid}.png").exists()]
    if args.select or args.patches:
        wanted: list[str] = []
        if args.select:
            f = selection_file(cfg, sheet, "manual")
            if not f.exists():
                sys.exit(f"--select: no saved list ({rel_to_root(f)}).\n"
                         f"Make one:  python run.py pick-patches --sheet {sheet} --name manual")
            wanted += [ln.strip() for ln in f.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        wanted += list(args.patches or [])
        wanted = [w if w.startswith(sheet) else f"{sheet}_{w}" for w in wanted]
        unknown = [w for w in wanted if w not in set(have)]
        if unknown:
            print(f"Note: {len(unknown)} chosen patch id(s) not found and ignored: {unknown[:3]}")
        chosen = [w for w in wanted if w in set(have)]
    else:
        def has_content(pid):
            c0, r0, w, h = window(pid)
            return any(len(d[1].query(inside(c0, r0, w, h), predicate="intersects")) for d in data.values())
        chosen = [pid for pid in have if has_content(pid)]
        if args.limit and len(chosen) > args.limit:
            chosen = [chosen[i] for i in sorted(set(np.linspace(0, len(chosen) - 1, args.limit).round().astype(int)))]
    if not chosen:
        sys.exit("No patches to export.")

    print(f"Sheet      : {sheet}")
    print(f"Source     : {rel_to_root(gpkg)}   layers: {', '.join(layers)}")
    print(f"Patches    : {len(chosen)}")
    print(f"JSON output: {rel_to_root(json_dir)}\n")

    written = skipped_done = skipped_empty = 0
    per_label: Counter = Counter()
    stats: Counter = Counter()
    for pid in chosen:
        c0, r0, w, h = window(pid)
        win = box(c0, r0, c0 + w, r0 + h)
        shapes: list[dict] = []
        for name, (g, sidx, label, as_lines) in data.items():
            for i in sidx.query(inside(c0, r0, w, h), predicate="intersects"):
                clipped = g.geometry.iloc[i].intersection(win)
                if clipped.is_empty:
                    continue
                shapes += shapes_from(affine_transform(clipped, [1, 0, 0, 1, -c0, -r0]), label, as_lines, stats)

        jp = json_dir / f"{pid}.json"
        keep: list[dict] = []
        if jp.exists():
            try:
                old = json.loads(jp.read_text())
            except (OSError, json.JSONDecodeError):
                print(f"  skipped {pid}: existing JSON could not be read")
                continue
            keep = [s for s in old.get("shapes", []) if s.get("description") != MANUAL_LAYER_TAG]
            if len(keep) != len(old.get("shapes", [])) and not args.overwrite:
                skipped_done += 1                                  # already exported earlier
                continue
        if not shapes:
            if jp.exists() and args.overwrite:
                write_labelme_json(jp, patches_dir / f"{pid}.png", [], w, h, keep)   # drop stale manual shapes
            elif not jp.exists() and args.empty_patches:
                write_labelme_json(jp, patches_dir / f"{pid}.png", [], w, h, [])
                written += 1
            else:
                skipped_empty += 1
            continue
        write_labelme_json(jp, patches_dir / f"{pid}.png", shapes, w, h, keep)
        written += 1
        per_label.update(s["label"] for s in shapes)

    print(f"Wrote {written} annotation file(s)  |  {sum(per_label.values())} shape(s): "
          + ", ".join(f"{k}={v}" for k, v in sorted(per_label.items())))
    if skipped_done:
        print(f"Skipped {skipped_done} patch(es) already exported (use --overwrite to refresh them)")
    if skipped_empty:
        print(f"Skipped {skipped_empty} patch(es) with no features in the chosen layers"
              " (--empty-patches writes them as 'nothing here' examples)")
    if stats["holes_dropped"]:
        print(f"Note: {stats['holes_dropped']} polygon hole(s) dropped (labelme polygons cannot have holes)")
    if stats["points_skipped"]:
        print(f"Note: {stats['points_skipped']} point feature(s) skipped (point layers cannot be turned into masks)")
    print(
        f"\nNext step: turn the annotations into training masks\n"
        f"  conda activate maptools\n"
        f"  python run.py export-masks --sheet {sheet}"
    )


if __name__ == "__main__":
    main()
