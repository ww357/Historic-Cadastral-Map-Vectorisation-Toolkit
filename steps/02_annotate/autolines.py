"""
Pre-draw boundary linestrips on patches so labelme opens with them already there.

Instead of tracing every boundary by hand, this finds dark thin lines with a
classical (no-ML) filter, skeletonises them, and writes each line as a labelme
`linestrip` with the boundary label. In labelme you then just delete the wrong
lines (and add any that were missed) instead of drawing everything from scratch.

Method, per patch:
  1. Black top-hat      - keeps dark features thinner than --tophat-radius, which
                          flattens uneven paper tone and removes big dark areas.
  2. Hysteresis thresh. - strong ink (> --high) plus the weaker pixels (> --low)
                          connected to it; recovers faint stretches of a line.
  3. Map-area mask      - drops anything outside the map (neatline, margins).
  4. Skeletonise, remove skeleton from very ink-dense areas (--max-density: tree
     and woodland symbols), trace (skan), then drop clutter:
       - components shorter than --min-length or smaller than --min-extent
         (text, numerals and specks skeletonise into short stubs),
       - short spurs and small loops hanging off a junction (--spur-length,
         --loop-length) - orchard/tree symbols and attached numerals.
  5. Douglas-Peucker simplify (--simplify), write labelme JSON.

Patches that already have a labelme JSON are SKIPPED, so hand-made work is never
overwritten. With --overwrite only previously auto-drawn shapes are replaced and
your own shapes are kept.

Reads   : data/patches/images/<SHEET_ID>/*.png
          data/map_area_masks/<SHEET_ID>/  (optional; used if present)
          data/patches/metadata/<SHEET_ID>_patches.csv  (to place the mask)
Writes  : data/annotations/labelme_json/<SHEET_ID>/<patch_id>.json
          (--preview writes red-overlay PNGs to data/autolines_preview/<SHEET_ID>/
           instead and touches no JSON)

Tuning: run with --preview --limit 8 first, look at the overlays, adjust
--low/--high (more/less ink), --min-length (more/less clutter removed), then run
without --preview. Defaults live under `autolines:` in config.yaml.

Usage:
    python autolines.py --sheet SHEET_ID [--preview] [--limit N] [--patch ID]
                        [--select] [--patches ID ...]
                        [--overwrite] [--low F] [--high F] [--min-length PX] ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skan import Skeleton
from skimage.color import rgb2gray
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import approximate_polygon
from skimage.morphology import black_tophat, disk, skeletonize
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "steps"))   # shared helpers
from common import find_mask, load_config, selection_file   # noqa: E402
from annotation_helpers import write_labelme_json   # noqa: E402

AUTO_TAG = "autolines"   # stored in each shape's `description` so re-runs can tell them apart

DEFAULTS = {
    "tophat_radius": 7,      # px; must exceed half the widest line you want to keep
    "low":           0.08,   # hysteresis low threshold  (0-1 darkness above local paper)
    "high":          0.20,   # hysteresis high threshold
    "min_size":      20,     # px; drop ink blobs smaller than this before skeletonising
    "min_length":    50,     # px; drop line networks shorter than this (text, specks)
    "min_extent":    40,     # px; drop networks whose longest side is under this
    "spur_length":   25,     # px; drop short branches dangling off a junction
    "loop_length":   60,     # px; drop small closed loops hanging off a junction (tree symbols)
    "simplify":      1.5,    # px; Douglas-Peucker tolerance
    "density_window": 25,    # px; window for measuring local ink density
    "max_density":   0.40,   # drop skeleton in areas denser than this (tree/woodland symbols); 1 = off
    "mask_erode":    6,      # px; pull the map-area mask in so the neatline is not traced
}


# -- Map-area mask ---------------------------------------------------------------

def load_mask(mask_path: Path) -> np.ndarray:
    """Boolean map-area mask (True = inside). Mirrors patchify.load_mask."""
    Image.MAX_IMAGE_PIXELS = None   # large masks exceed PIL's default limit
    if mask_path.suffix.lower() == ".png":
        arr = np.array(Image.open(mask_path))
        if arr.ndim == 3:
            if arr.shape[2] == 4 and (arr[:, :, 3] == 0).mean() > 0.01:
                return arr[:, :, 3] > 0
            return arr[:, :, 0] > 0
        return arr > 0
    import rasterio   # only needed for GeoTIFF masks
    with rasterio.open(mask_path) as src:
        arr = src.read(4) if src.count == 4 else src.read(1)
    return arr > 0


def read_offsets(meta_csv: Path) -> dict[str, tuple[int, int]]:
    """patch_id -> (col_off, row_off) from the patchify metadata CSV."""
    with open(meta_csv, newline="") as f:
        return {r["patch_id"]: (int(r["col_off"]), int(r["row_off"]))
                for r in csv.DictReader(f)}


def patch_valid_region(mask: np.ndarray, col_off: int, row_off: int,
                       w: int, h: int, erode: int) -> np.ndarray:
    """Map-area mask for one patch, eroded by `erode` px. The crop is taken with a
    margin first so erosion does not bite into patch edges that continue into the map."""
    m = erode
    r0, c0 = max(row_off - m, 0), max(col_off - m, 0)
    r1, c1 = min(row_off + h + m, mask.shape[0]), min(col_off + w + m, mask.shape[1])
    crop = mask[r0:r1, c0:c1]
    if erode > 0:
        crop = ndi.binary_erosion(crop, iterations=erode, border_value=1)
    out = np.zeros((h, w), dtype=bool)
    sub = crop[row_off - r0: row_off - r0 + h, col_off - c0: col_off - c0 + w]
    out[:sub.shape[0], :sub.shape[1]] = sub
    return out


# -- Detection -------------------------------------------------------------------

def detect_lines(img: np.ndarray, valid: np.ndarray | None, p: dict) -> list[np.ndarray]:
    """
    Return polylines as (N, 2) arrays of (row, col) pixel coordinates.
    `img` is the patch as HxW or HxWx3 uint8.
    """
    gray = rgb2gray(img[..., :3]) if img.ndim == 3 else img.astype(float) / 255.0

    # Dark, thin structures stand out; broad dark areas and paper tone cancel out.
    ink = black_tophat(gray, disk(int(p["tophat_radius"])))
    binary = apply_hysteresis_threshold(ink, p["low"], p["high"])
    if valid is not None:
        binary &= valid

    # Drop tiny specks before skeletonising (cheaper, and avoids skeleton noise).
    lab, n = ndi.label(binary, structure=np.ones((3, 3)))
    if n:
        sizes = np.bincount(lab.ravel())
        keep = sizes >= p["min_size"]
        keep[0] = False
        binary = keep[lab]
    if not binary.any():
        return []

    skel = skeletonize(binary)

    # Orchard / woodland symbols are far inkier than line work: cut the skeleton out of
    # very dense areas so those glyph clusters fall apart and get filtered below.
    if p["max_density"] < 1:
        density = ndi.uniform_filter(binary.astype(np.float32), size=int(p["density_window"]))
        skel &= density <= p["max_density"]

    skel = _component_filter(skel, p)
    skel = _prune(skel, p)           # tree symbols / numerals hang off lines as spurs and loops
    skel = _component_filter(skel, p)
    if not skel.any():
        return []

    lines: list[np.ndarray] = []
    sk, skel = _trace(skel)
    if sk is None:
        return []
    for i in range(sk.n_paths):
        coords = sk.path_coordinates(i)
        if len(coords) < 4:                       # junction-to-junction crumbs
            continue
        pts = approximate_polygon(coords.astype(float), tolerance=p["simplify"])
        if len(pts) >= 2:
            lines.append(pts)
    return lines


def _trace(skel: np.ndarray):
    """
    (skan Skeleton, cleaned skeleton) - or (None, empty) if there is nothing to trace.
    skan raises ValueError on an empty image or one where no pixel has a neighbour,
    and pruning can leave exactly that (e.g. a Y of three short arms is erased down
    to its single junction pixel), so isolated pixels are dropped first.
    """
    nbr = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    skel = skel & (nbr > 0)
    if not skel.any():
        return None, skel
    return Skeleton(skel, keep_images=False), skel


def _component_filter(skel: np.ndarray, p: dict) -> np.ndarray:
    """Drop skeleton networks that are too short or too small - text, numerals, specks."""
    lab, n = ndi.label(skel, structure=np.ones((3, 3)))
    if n == 0:
        return skel
    counts = np.bincount(lab.ravel(), minlength=n + 1)
    keep = np.zeros(n + 1, dtype=bool)
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        extent = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
        keep[i] = counts[i] >= p["min_length"] and extent >= p["min_extent"]
    return keep[lab]


def _prune(skel: np.ndarray, p: dict, rounds: int = 3) -> np.ndarray:
    """
    Remove short spurs (endpoint-to-junction branches) and small closed loops that
    start and end at the same junction. Repeated, because removing a spur can expose
    the next one along (a tree glyph is usually a spur-with-a-loop-on-it).
    """
    skel = skel.copy()
    for _ in range(rounds):
        sk, skel = _trace(skel)
        if sk is None:
            break
        # Neighbour count per skeleton pixel: 1 = endpoint, >=3 = junction.
        deg = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8),
                           mode="constant") - 1
        deg[~skel] = 0
        changed = False
        for i in range(sk.n_paths):
            c = sk.path_coordinates(i).astype(int)
            n = len(c)
            d0, d1 = deg[tuple(c[0])], deg[tuple(c[-1])]
            if d0 == 1 and d1 >= 3 and n < p["spur_length"]:
                erase = c[:-1]                    # keep the junction pixel itself
            elif d1 == 1 and d0 >= 3 and n < p["spur_length"]:
                erase = c[1:]
            elif d0 >= 3 and np.abs(c[0] - c[-1]).max() <= 2 and n < p["loop_length"]:
                erase = c[1:-1]                   # small loop back onto the same junction
            else:
                continue
            skel[erase[:, 0], erase[:, 1]] = False
            changed = True
        if not changed:
            break
    return skel


# -- labelme JSON ----------------------------------------------------------------

def make_shape(pts_rc: np.ndarray, label: str) -> dict:
    return {
        "label": label,
        "points": [[float(c), float(r)] for r, c in pts_rc],   # labelme is (x, y)
        "group_id": None,
        "description": AUTO_TAG,
        "shape_type": "linestrip",
        "flags": {},
    }


def draw_preview(img: Image.Image, lines: list[np.ndarray], out: Path) -> None:
    rgb = img.convert("RGB")
    d = ImageDraw.Draw(rgb)
    for pts in lines:
        d.line([(float(c), float(r)) for r, c in pts], fill=(255, 0, 0), width=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out)


# -- Driver ----------------------------------------------------------------------

def sample_patches(patches: list[Path], n: int, mask, offsets, erode: int) -> list[Path]:
    """
    n patches spread evenly across the sheet. Sorted order starts at the sheet's top
    edge, where patches are mostly margin, so `first N` is a poor preview. With a mask,
    only patches at least half inside the map area are considered.
    """
    if len(patches) <= n:
        return patches
    pool = patches
    if mask is not None:
        cover = []
        for pp in patches:
            if pp.stem in offsets:
                w, h = Image.open(pp).size            # header only, no pixel decode
                c, r = offsets[pp.stem]
                cover.append(float(patch_valid_region(mask, c, r, w, h, erode).mean()))
            else:
                cover.append(1.0)
        pool = [pp for pp, c in zip(patches, cover) if c >= 0.5]
        if len(pool) < n:                             # sparse sheet: best-covered n
            ranked = sorted(zip(cover, patches), key=lambda t: -t[0])[:n]
            pool = sorted(pp for _, pp in ranked)
    idx = sorted(set(np.linspace(0, len(pool) - 1, n).round().astype(int)))
    return [pool[i] for i in idx]


def resolve_chosen(sheet_id: str, cfg: dict, use_saved: bool, ids: list[str] | None) -> list[str] | None:
    """Patch ids picked with --select (list saved by pick_patches.py) and/or --patches.
    Short ids like r0012_c0031 are expanded with the sheet name. None = no selection given."""
    if not use_saved and not ids:
        return None
    chosen: list[str] = []
    if use_saved:
        f = selection_file(cfg, sheet_id)
        if not f.exists():
            sys.exit(f"--select: no saved selection for '{sheet_id}' ({f}).\n"
                     f"Make one first:  python run.py pick-patches --sheet {sheet_id}")
        chosen += [ln.strip() for ln in f.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    chosen += list(ids or [])
    return [c if c.startswith(sheet_id) else f"{sheet_id}_{c}" for c in chosen]


def autolines(sheet_id: str, p: dict, label: str, preview: bool, overwrite: bool,
              limit: int | None, only_patch: str | None, use_mask: bool,
              chosen: list[str] | None = None) -> None:
    cfg         = load_config()
    patches_dir = ROOT / cfg["paths"]["patches"] / "images" / sheet_id
    json_dir    = ROOT / cfg["paths"]["annotations"] / "labelme_json" / sheet_id
    prev_dir    = ROOT / "data" / "autolines_preview" / sheet_id
    meta_csv    = ROOT / cfg["paths"]["patches"] / "metadata" / f"{sheet_id}_patches.csv"

    if not patches_dir.exists():
        sys.exit(f"Patches not found: {patches_dir}  - run 01_patchify first.")

    patches = sorted(patches_dir.glob("*.png"))
    if only_patch:
        patches = [x for x in patches if x.stem == only_patch]
        if not patches:
            sys.exit(f"Patch '{only_patch}' not found in {patches_dir}")
    chosen_note = ""
    if chosen is not None:
        have = {x.stem for x in patches}
        missing = [c for c in chosen if c not in have]
        patches = [x for x in patches if x.stem in set(chosen)]
        if not patches:
            sys.exit("None of the chosen patches exist in " + str(patches_dir))
        chosen_note = f"  (chosen: {len(patches)}" + (f", {len(missing)} not found: {missing[:3]}" if missing else "") + ")"

    # Map-area mask: optional, but needs the patchify metadata to line up with patches.
    mask = offsets = None
    mask_note = "none (whole patch searched)"
    if use_mask:
        mask_path = find_mask(ROOT / cfg["paths"]["masks"] / sheet_id, sheet_id)
        if mask_path and meta_csv.exists():
            mask, offsets = load_mask(mask_path), read_offsets(meta_csv)
            mask_note = f"{mask_path.name}, eroded {p['mask_erode']}px"
        elif mask_path:
            mask_note = f"found {mask_path.name} but {meta_csv.name} is missing - not used"

    if limit is not None and not only_patch:   # with a selection, --limit samples within it
        patches = sample_patches(patches, limit, mask, offsets, int(p["mask_erode"]))
        sample_note = f"  (--limit: {len(patches)} spread across the sheet)"
    else:
        sample_note = ""
    print(f"Sheet      : {sheet_id}")
    print(f"Patches    : {len(patches)}  ({patches_dir}){chosen_note}{sample_note}")
    print(f"Mask       : {mask_note}")
    print(f"Label      : {label}")
    print(f"Mode       : {'PREVIEW -> ' + str(prev_dir) if preview else 'write labelme JSON -> ' + str(json_dir)}")
    print(f"Params     : low={p['low']} high={p['high']} tophat={p['tophat_radius']} "
          f"min_length={p['min_length']} spur={p['spur_length']} simplify={p['simplify']}\n")

    done = skipped = empty = failed = total_lines = 0
    counts: dict[str, int] = {}
    for img_path in tqdm(patches, unit="patch"):
        if limit is not None and done >= limit:
            break
        json_path = json_dir / f"{img_path.stem}.json"

        keep_shapes: list[dict] = []
        if not preview and json_path.exists():
            if not overwrite:
                skipped += 1
                continue
            try:                                  # keep hand-made shapes, replace auto ones
                old = json.loads(json_path.read_text())
                keep_shapes = [s for s in old.get("shapes", [])
                               if s.get("description") != AUTO_TAG]
            except (OSError, json.JSONDecodeError):
                skipped += 1
                continue

        pil = Image.open(img_path)
        arr = np.array(pil)
        h, w = arr.shape[:2]

        valid = None
        if mask is not None and img_path.stem in offsets:
            col_off, row_off = offsets[img_path.stem]
            valid = patch_valid_region(mask, col_off, row_off, w, h, int(p["mask_erode"]))

        try:
            lines = detect_lines(arr, valid, p)
        except Exception as e:                    # one odd patch must not abort the sheet
            failed += 1
            tqdm.write(f"  ! {img_path.stem}: {type(e).__name__}: {e} - skipped")
            continue
        done += 1
        total_lines += len(lines)
        counts[img_path.stem] = len(lines)

        if preview:
            draw_preview(pil, lines, prev_dir / f"{img_path.stem}.png")
        elif lines:
            write_labelme_json(json_path, img_path,
                               [make_shape(pts, label) for pts in lines],
                               w, h, keep_shapes)
        else:
            empty += 1                            # nothing found: leave the patch untouched

    print(f"\nProcessed {done} patch(es): {total_lines} line(s) drawn"
          + (f", {empty} with no lines (no JSON written)" if empty and not preview else ""))
    if failed:
        print(f"WARNING: {failed} patch(es) raised an error and were skipped (messages above).")
    if preview and counts:
        print("Lines per patch: " + "  ".join(f"{k.rsplit('_', 2)[-2]}_{k.rsplit('_', 1)[-1]}={v}"
                                              for k, v in counts.items()))
    if done >= 4 and sum(v == 0 for v in counts.values()) >= done / 2:
        print("\nHint: at least half of these patches returned no lines. If the overlays show "
              "ink being missed, this scan is probably fainter than the tuning sheets - try "
              "--low 0.04 --high 0.12. If those patches really are blank margin, ignore this.")
    if skipped:
        print(f"Skipped {skipped} patch(es) that already have a labelme JSON "
              f"(use --overwrite to refresh the auto-drawn lines only)")

    if preview:
        print(f"\nOverlays in {prev_dir}/  (red = detected lines)."
              f"\nAdjust --low/--high/--min-length until the red lines look right, then re-run without --preview.")
    else:
        print(
            f"\nNext step: review in labelme - delete wrong lines, add missed ones, Save\n"
            f"  conda activate maptools\n"
            f"  python run.py annotate --sheet {sheet_id}"
        )


if __name__ == "__main__":
    cfg_defaults = {**DEFAULTS, **(load_config().get("autolines") or {})}

    ap = argparse.ArgumentParser(
        description="Pre-draw boundary linestrips (threshold + skeleton) as labelme JSON.")
    ap.add_argument("--sheet", required=True, help="Sheet ID")
    ap.add_argument("--preview", action="store_true",
                    help="Write red-overlay PNGs for tuning; write no labelme JSON")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only N patches, spread evenly across the sheet (well inside the map area)")
    ap.add_argument("--patch", default=None, help="Process a single patch ID")
    ap.add_argument("--select", action="store_true",
                    help="Only the patches chosen with `run.py pick-patches` (data/patches/selections/<sheet>.txt)")
    ap.add_argument("--patches", nargs="+", metavar="ID", default=None,
                    help="Only these patch ids (full, or short like r0012_c0031); can be combined with --select")
    ap.add_argument("--overwrite", action="store_true",
                    help="Refresh auto-drawn lines in existing JSONs (hand-made shapes are kept)")
    ap.add_argument("--no-mask", action="store_true", help="Ignore the map-area mask")
    ap.add_argument("--label", default=None,
                    help="Label for the drawn lines (default: annotation.boundary_label)")
    for key, default in cfg_defaults.items():
        ap.add_argument(f"--{key.replace('_', '-')}", dest=key, type=type(default), default=default)
    args = ap.parse_args()

    params = {k: getattr(args, k) for k in cfg_defaults}
    boundary = args.label or load_config()["annotation"].get("boundary_label", "boundary")
    chosen = resolve_chosen(args.sheet, load_config(), args.select, args.patches)
    autolines(args.sheet, params, boundary, args.preview, args.overwrite,
              args.limit, args.patch, not args.no_mask, chosen)
