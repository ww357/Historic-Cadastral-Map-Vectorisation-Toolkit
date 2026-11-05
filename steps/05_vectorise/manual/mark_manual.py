"""
Tell the toolkit where a layer in the sheet's GeoPackage came from.

For a layer you drew straight into data/outputs/<SHEET_ID>.gpkg in QGIS (instead of predicting
it), or a model layer that had no origin tag yet. It adds / fills the text field `source`:

    manual     you made the layer by hand            (default)
    semi_auto  model output that is mended by hand   (--as semi_auto)

and that one field is what the rest of the toolkit reads: model steps refuse to overwrite a
manual layer, feedback skips it, status lists it, Patch_Grid gets a manual_<layer> column and
`export-manual` can turn its patches into training annotations. Every model step already writes
`source = 'semi_auto'` on the layers it creates; --as semi_auto is for layers written before that
existed.

Training data already in the toolkit is left alone and reported: annotations for the same
feature on this sheet and on other sheets, predictions on other sheets, and any model weights.
Using a class name other maps already use (e.g. `water`) is exactly what you want; a near miss
(`waters`) is flagged because it would become a separate class.

--already-mended (with --as semi_auto) records that you have already corrected the layer, so
feedback and the parcel step treat it as mended even though it predates edit tracking.

Usage:
    python mark_manual.py --sheet SHEET_ID                      (lists layers that could be marked)
    python mark_manual.py --sheet SHEET_ID --layers swamp water
    python mark_manual.py --sheet SHEET_ID --layers water --as semi_auto [--already-mended]
                          [--force] [--mended | --gpkg PATH] [--ignore-open]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "steps"))   # shared helpers
from common import (classify_name, known_classes, load_config, model_predicted,   # noqa: E402
                    rel_to_root, resolve_output_gpkg)
from geopackage import (MANUAL_TAG, SEMI_AUTO_TAG, apply_safety_flags, declare_edited,   # noqa: E402
                        ensure_not_open, layer_edit_state, layer_epsg, layer_origin,
                        stamp_layer, vector_layers, write_patch_grid)
from annotation_helpers import MANUAL_LAYER_TAG   # noqa: E402
from vectorise_common import sheet_georef   # noqa: E402


# -- What the toolkit already holds for this class ----------------------------------

def label_for(layer: str, boundary_label: str) -> str:
    """The annotation label a layer's training data is filed under."""
    return boundary_label if layer.lower() in ("boundaries", "boundary") else layer


def _count_pngs(d: Path) -> int:
    return sum(1 for _ in d.glob("*.png")) if d.is_dir() else 0


def training_report(cfg: dict, sheet: str, layer: str, boundary_label: str) -> list[str]:
    """Plain-English lines about training data / weights that already exist for this class."""
    paths = cfg["paths"]
    label = label_for(layer, boundary_label)
    ann_root = ROOT / paths["annotations"] / label
    json_dir = ROOT / paths["annotations"] / "labelme_json" / sheet

    here = {p.stem for p in (ann_root / sheet / "masks").glob("*.png")} if (ann_root / sheet / "masks").is_dir() else set()
    exported_from_layer = set()
    if json_dir.is_dir():
        for jp in json_dir.glob("*.json"):
            try:
                shapes = [s for s in json.loads(jp.read_text()).get("shapes", []) if s.get("label") == label]
            except (OSError, json.JSONDecodeError):
                continue
            if any(s.get("description") != MANUAL_LAYER_TAG for s in shapes):
                here.add(jp.stem)                                   # drawn by hand (or pre-drawn) in labelme
            elif shapes:
                exported_from_layer.add(jp.stem)
    elsewhere = {}
    if ann_root.is_dir():
        for d in sorted(ann_root.iterdir()):
            if d.is_dir() and d.name not in (sheet, "feedback") and _count_pngs(d / "masks"):
                elsewhere[d.name] = _count_pngs(d / "masks")
    pred_root = ROOT / paths["predictions"] / ("boundaries" if label == boundary_label else label)
    predicted_on = [d.name for d in sorted(pred_root.iterdir())
                    if d.is_dir() and d.name != sheet and any(d.glob("*.png"))] if pred_root.is_dir() else []

    base = ROOT / paths["models_base"]
    if label == boundary_label:
        weights = "solid-line U-Net base weights" if (base / "solid_lines_unet").exists() else None
    elif label == "dashed":
        weights = "dashed-line U-Net base weights" if (base / "dashed_unet").exists() else None
    else:
        has = (base / "MapSAM" / label).is_dir() or any((ROOT / paths["models_finetuned"]).rglob(f"mapsam_{label}*_best.pth"))
        weights = f"MapSAM adapter '{label}'" if has else None

    lines = []
    lines.append(f"annotations on this sheet : {len(here)} patch(es) drawn by hand - kept, still used for training"
                 if here else "annotations on this sheet : none")
    if exported_from_layer:
        lines.append(f"                            + {len(exported_from_layer)} patch(es) already exported from this layer")
    lines.append("annotations on other sheets: " + (", ".join(f"{k} ({v})" for k, v in elsewhere.items())
                                                    if elsewhere else "none"))
    lines.append("predicted on other sheets : " + (", ".join(predicted_on) if predicted_on else "none"))
    lines.append("model weights             : " + (weights if weights else "none yet for this class"))
    return lines


# -- Editing the layer -------------------------------------------------------------------

def source_values(gpkg: Path, layer: str) -> dict[str, int] | None:
    """{value: count} of the layer's `source` field, or None if the field does not exist."""
    con = sqlite3.connect(gpkg, timeout=5)
    try:
        if "source" not in [r[1] for r in con.execute(f"PRAGMA table_info([{layer}])")]:
            return None
        return {(v or ""): n for v, n in con.execute(f"SELECT source, count(*) FROM [{layer}] GROUP BY source")}
    finally:
        con.close()


def set_source(gpkg: Path, layer: str, value: str, overwrite_all: bool) -> None:
    """Add the text field `source` if missing and fill it. Done through GDAL (like QGIS does) so
    the GeoPackage's spatial-index triggers keep working. Empty values are filled; values that
    are already set are only replaced with overwrite_all."""
    from osgeo import ogr
    ogr.UseExceptions()
    ds = ogr.Open(str(gpkg), 1)
    if ds is None:
        sys.exit(f"Could not open {gpkg} for editing (is another program writing to it?)")
    try:
        lyr = ds.GetLayerByName(layer)
        if lyr.GetLayerDefn().GetFieldIndex("source") < 0:
            lyr.CreateField(ogr.FieldDefn("source", ogr.OFTString))
        where = "" if overwrite_all else " WHERE source IS NULL OR source = ''"
        ds.ExecuteSQL(f"UPDATE \"{layer}\" SET source = '{value}'{where}")
        ds.FlushCache()
    finally:
        ds = None


# -- Driver ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Mark a GeoPackage layer as manual or semi_auto.")
    ap.add_argument("--sheet", required=True, help="Sheet ID")
    ap.add_argument("--layers", nargs="+", default=None, help="Layer name(s) to mark (omit to list candidates)")
    ap.add_argument("--as", dest="kind", choices=[MANUAL_TAG, SEMI_AUTO_TAG], default=MANUAL_TAG,
                    help="manual = you made it by hand (default); semi_auto = model output you mend")
    ap.add_argument("--already-mended", action="store_true",
                    help="With --as semi_auto: you have already corrected this layer, so treat it as edited")
    ap.add_argument("--force", action="store_true",
                    help="Mark even if the layer looks like model output / has other source values / the CRS differs")
    ap.add_argument("--ignore-open", action="store_true", help="Carry on even if QGIS has the file open (risky)")
    target = ap.add_mutually_exclusive_group()
    target.add_argument("--mended", action="store_true", help="Use the hand-corrected GeoPackage in paths.outputs_mended")
    target.add_argument("--gpkg", default=None, help="Explicit GeoPackage path (overrides --mended and the default)")
    args = ap.parse_args()
    apply_safety_flags(args)
    if args.already_mended and args.kind != SEMI_AUTO_TAG:
        ap.error("--already-mended only makes sense with --as semi_auto")

    cfg, sheet = load_config(), args.sheet
    boundary_label = cfg.get("annotation", {}).get("boundary_label", "boundary")
    gpkg = resolve_output_gpkg(sheet, cfg, args.gpkg, args.mended)
    if not gpkg.exists():
        sys.exit(f"GeoPackage not found: {gpkg}\nIt is created by a vectorise step or by import-manual.")
    layers = vector_layers(gpkg)

    # -- no --layers: show what could be marked ---------------------------------------
    if not args.layers:
        print(f"Layers in {rel_to_root(gpkg)}:")
        todo = 0
        for name in layers:
            origin, state = layer_origin(gpkg, name), layer_edit_state(gpkg, name)
            print(f"  {name:<20} origin: {origin:<10} state: {state}")
            if origin == "untagged" and state == "untracked":
                todo += 1
                if model_predicted(cfg, sheet, name):
                    print(f"      the model predicted this layer:  python run.py mark-manual --sheet {sheet} "
                          f"--layers {name} --as semi_auto   (add --already-mended if you have corrected it)")
                else:
                    print(f"      looks hand-made:  python run.py mark-manual --sheet {sheet} --layers {name}")
        if not todo:
            print("\nNothing needs marking.")
        return

    wanted = {n.lower(): n for n in layers}
    missing = [n for n in args.layers if n.lower() not in wanted]
    if missing:
        sys.exit(f"Not a vector layer in {gpkg.name}: {', '.join(missing)}\nAvailable: {', '.join(layers) or '(none)'}")
    names = [wanted[n.lower()] for n in args.layers]

    # -- georeferencing, for the CRS check and the Patch_Grid rebuild -------------------------
    georef = sheet_georef(sheet, cfg)
    map_epsg = None
    if georef["has_georef"]:
        from pyproj import CRS
        crs = georef["crs"]
        map_epsg = (CRS.from_wkt(crs.to_wkt()) if hasattr(crs, "to_wkt") else CRS.from_user_input(crs)).to_epsg()

    classes = known_classes(cfg)
    print(f"Sheet      : {sheet}")
    print(f"GeoPackage : {rel_to_root(gpkg)}")
    print(f"Marking as : {args.kind}\n")

    ensure_not_open(gpkg)                                     # we are about to edit the file

    # -- check everything first, so a refusal changes nothing -------------------------------
    plan = []
    for name in names:
        origin, state = layer_origin(gpkg, name), layer_edit_state(gpkg, name)
        problems = []
        if origin == args.kind and not args.already_mended:
            plan.append((name, origin, state, problems, False))     # already marked: nothing to check or change
            continue
        if args.kind == MANUAL_TAG:
            if state in ("unchanged", "edited") and not args.force:
                problems.append("the toolkit wrote this layer (model output). If you mended it, it is semi_auto and "
                                "needs no marking; if you really redrew it by hand, add --force")
            if origin == SEMI_AUTO_TAG and not args.force:
                problems.append("its features are tagged semi_auto (model output); add --force to call it manual")
            lay_epsg = layer_epsg(gpkg, name)
            if map_epsg and lay_epsg and lay_epsg != map_epsg and not args.force:
                problems.append(f"its CRS is EPSG:{lay_epsg} but the map is EPSG:{map_epsg} - reproject the layer in QGIS "
                                f"first (Export > Save Features As), otherwise export would place it wrongly")
        else:
            if origin == MANUAL_TAG and not args.force:
                problems.append("its features are tagged manual; add --force to call it semi_auto")
        vals = source_values(gpkg, name) or {}
        others = {k: v for k, v in vals.items() if k and k != args.kind}
        if others and not args.force:
            problems.append("the 'source' field already holds " + ", ".join(f"'{k}' ({v})" for k, v in others.items())
                            + "; --force replaces them")
        plan.append((name, origin, state, problems, bool(others)))

    bad = [(n, p) for n, _, _, p, _ in plan if p]
    if bad:
        for n, p in bad:
            print(f"  REFUSED  {n}")
            for line in p:
                print(f"           - {line}")
        sys.exit("\nNothing was changed.")

    # -- do it -----------------------------------------------------------------------
    marked_manual = False
    for name, origin, state, _, replace_others in plan:
        if origin == args.kind and not args.already_mended:
            print(f"  {name:<20} already marked '{args.kind}' - nothing to change")
            continue
        set_source(gpkg, name, args.kind, overwrite_all=replace_others)
        n = (source_values(gpkg, name) or {}).get(args.kind, 0)
        extra = ""
        if args.kind == MANUAL_TAG:
            stamp_layer(gpkg, name)                          # baseline, so later edits to your layer are noticed
            marked_manual = True
        elif args.already_mended:
            declare_edited(gpkg, name)
            extra = "  [recorded as already mended]"
        print(f"  {name:<20} source = '{args.kind}' on {n:,} feature(s){extra}")
        kind, close = classify_name(label_for(name, boundary_label), classes)
        if kind == "existing":
            print(f"  {'':<20} class '{label_for(name, boundary_label)}' is already used by the toolkit")
        elif kind == "near":
            print(f"  {'':<20} ! looks like the existing class '{close}'. This is a SEPARATE class "
                  f"'{name}' - rename the layer in QGIS to '{close}' if you meant that one.")
        else:
            print(f"  {'':<20} new class (nothing in the toolkit uses this name yet)")
        for line in training_report(cfg, sheet, name, boundary_label):
            print(f"  {'':<20} {line}")

    meta_path = ROOT / cfg["paths"]["patches"] / "metadata" / f"{sheet}_patches.csv"
    if marked_manual:
        if meta_path.exists():
            write_patch_grid(gpkg, meta_path, georef["transform"], georef["crs"], georef["has_georef"], sheet, cfg)
        else:
            print("  Patch_Grid not updated (no patch metadata - run patchify first).")

    if args.kind == MANUAL_TAG:
        print(
            f"\nNext step: the toolkit now treats {', '.join(names)} as hand-made on this sheet (no prediction needed).\n"
            f"  To turn some of its patches into training data for other maps:\n"
            f"    python run.py pick-patches --sheet {sheet} --name manual\n"
            f"    python run.py export-manual --sheet {sheet} --select\n"
            f"  Patches you have already annotated by hand for the same feature are left as they are."
        )
    else:
        print(
            f"\nNext step: {', '.join(names)} will be treated as model output that you mend. Edits you make in QGIS are\n"
            f"  detected automatically (a re-run of its vectorise step will refuse to overwrite them)."
            + ("" if args.already_mended else
               f"\n  If you have ALREADY corrected it, re-run with --already-mended so feedback and parcels use it.")
        )


if __name__ == "__main__":
    main()
