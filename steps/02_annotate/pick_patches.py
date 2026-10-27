"""
Visually choose which patches `autolines` should draw lines on.

Opens one Matplotlib window showing the whole sheet as a mosaic of its real patches,
laid out by their grid row/column. Pick the patches you want pre-drawn lines on (a
representative handful is usually enough); autolines then runs on just those, so you
are not left deleting lines from patches you never meant to annotate.

    Left-click            toggle one patch
    Left-drag a box       select every patch in the box
    Right-click / drag    deselect (one patch / everything in the box)
    Mouse wheel           zoom in/out at the cursor      (toolbar pan/zoom also work)
    a                     select all        c    clear all
    Enter                 save and close    Esc  close WITHOUT saving

Green = selected.  Orange outline = that patch already has a labelme JSON.
Blue outline (only with --name manual) = the patch contains features of a manual layer.

Saved to  data/patches/selections/<SHEET_ID>.txt  (one patch id per line); an existing
selection is loaded back in, so you can re-run this to adjust it.

Usage:
    python pick_patches.py --sheet SHEET_ID [--thumb 40]
Then:
    python run.py autolines --sheet SHEET_ID --select --preview

For your hand-made layers:  pick_patches.py --sheet SHEET_ID --name manual  (then export-manual).
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "steps"))   # shared helpers
from common import load_config, selection_file, rel_to_root   # noqa: E402

CELL_RE = re.compile(r"_r(\d+)_c(\d+)$")      # patch ids end in _r<row>_c<col>
SELECTED_RGBA = (0.10, 0.80, 0.30, 0.45)
DRAG_PIXELS = 4                                # press-release closer than this is a click


def parse_cell(stem: str) -> tuple[int, int] | None:
    m = CELL_RE.search(stem)
    return (int(m.group(1)), int(m.group(2))) if m else None


def build_mosaic(files: list[Path], cells: list[tuple[int, int]], thumb: int):
    """Thumbnails placed by grid position. Returns (image, r0, c0, r1, c1)."""
    r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
    c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
    mosaic = np.full(((r1 - r0 + 1) * thumb, (c1 - c0 + 1) * thumb, 3), 232, np.uint8)  # empty = light grey
    for f, (r, c) in tqdm(list(zip(files, cells)), desc="Building overview", unit="patch"):
        with Image.open(f) as im:
            tile = np.asarray(im.convert("RGB").resize((thumb, thumb), Image.BOX))
        mosaic[(r - r0) * thumb:(r - r0 + 1) * thumb, (c - c0) * thumb:(c - c0 + 1) * thumb] = tile
    return mosaic, r0, c0, r1, c1


class Picker:
    """Selection state + mouse/keyboard handling for the mosaic window."""

    def __init__(self, fig, ax, stems: list[str], cells: list[tuple[int, int]],
                 selected: set[str], done: set[str], marked: set[str] = frozenset()):
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
        self.fig, self.ax = fig, ax
        self.stems, self.cells = stems, cells
        self.at = {cell: i for i, cell in enumerate(cells)}            # (row, col) -> index
        self.sel = np.array([s in selected for s in stems], dtype=bool)
        edge = np.array([(1.0, 0.55, 0.0, 1.0) if s in done else (0.1, 0.4, 1.0, 1.0) if s in marked
                         else (0, 0, 0, 0.18) for s in stems])
        self.coll = PatchCollection([Rectangle((c, r), 1, 1) for r, c in cells],
                                    facecolor="none", edgecolor=edge,
                                    linewidths=[1.6 if (s in done or s in marked) else 0.4 for s in stems])
        ax.add_collection(self.coll)
        self.band = None            # rubber-band artist while dragging
        self.press = None           # (x, y, pixel_x, pixel_y, button)
        self.saved = False
        for ev, fn in (("button_press_event", self.on_press), ("motion_notify_event", self.on_motion),
                       ("button_release_event", self.on_release), ("key_press_event", self.on_key),
                       ("scroll_event", self.on_scroll)):
            fig.canvas.mpl_connect(ev, fn)
        self.refresh()

    # -- state ---------------------------------------------------------------
    def refresh(self) -> None:
        face = np.zeros((len(self.stems), 4))
        face[self.sel] = SELECTED_RGBA
        self.coll.set_facecolor(face)
        self.ax.set_title(f"{int(self.sel.sum())} of {len(self.stems)} patches selected   "
                          f"[click toggle | drag box | right = remove | Enter save | Esc cancel]",
                          fontsize=10)
        self.fig.canvas.draw_idle()

    def cell_at(self, x, y):
        if x is None or y is None:
            return None
        return self.at.get((int(np.floor(y)), int(np.floor(x))))

    def set_box(self, x0, y0, x1, y1, value: bool) -> int:
        """Set every patch whose centre lies inside the box (data coords); returns how many."""
        lo_x, hi_x, lo_y, hi_y = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        hit = [i for i, (r, c) in enumerate(self.cells)
               if lo_x <= c + 0.5 <= hi_x and lo_y <= r + 0.5 <= hi_y]
        self.sel[hit] = value
        return len(hit)

    def chosen(self) -> list[str]:
        return sorted(s for s, on in zip(self.stems, self.sel) if on)

    # -- events --------------------------------------------------------------
    def _panning(self) -> bool:
        tb = getattr(self.fig.canvas, "toolbar", None)
        return bool(str(getattr(tb, "mode", "")))          # toolbar pan/zoom active -> hands off

    def on_press(self, ev) -> None:
        if ev.inaxes is not self.ax or self._panning() or ev.button not in (1, 3):
            return
        self.press = (ev.xdata, ev.ydata, ev.x, ev.y, ev.button)

    def on_motion(self, ev) -> None:
        if self.press is None or ev.inaxes is not self.ax or ev.xdata is None:
            return
        x0, y0, px, py, button = self.press
        if abs(ev.x - px) + abs(ev.y - py) < DRAG_PIXELS:
            return
        from matplotlib.patches import Rectangle
        if self.band is None:
            colour = "tab:green" if button == 1 else "tab:red"
            self.band = Rectangle((x0, y0), 0, 0, fill=True, alpha=0.25, color=colour, lw=1.5, zorder=5)
            self.ax.add_patch(self.band)
        self.band.set_bounds(min(x0, ev.xdata), min(y0, ev.ydata), abs(ev.xdata - x0), abs(ev.ydata - y0))
        self.fig.canvas.draw_idle()

    def on_release(self, ev) -> None:
        if self.press is None:
            return
        x0, y0, px, py, button = self.press
        self.press = None
        dragged = self.band is not None
        if dragged:
            self.band.remove()
            self.band = None
            if ev.xdata is not None and ev.ydata is not None:
                self.set_box(x0, y0, ev.xdata, ev.ydata, value=(button == 1))
        elif ev.inaxes is self.ax:                                    # plain click
            i = self.cell_at(ev.xdata, ev.ydata)
            if i is not None:
                self.sel[i] = (not self.sel[i]) if button == 1 else False
        self.refresh()

    def on_key(self, ev) -> None:
        import matplotlib.pyplot as plt
        if ev.key == "a":
            self.sel[:] = True
            self.refresh()
        elif ev.key == "c":
            self.sel[:] = False
            self.refresh()
        elif ev.key in ("enter", "return"):
            self.saved = True
            plt.close(self.fig)
        elif ev.key == "escape":
            plt.close(self.fig)

    def on_scroll(self, ev) -> None:
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        k = 1 / 1.3 if ev.button == "up" else 1.3
        (xa, xb), (ya, yb) = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.set_xlim(ev.xdata + (xa - ev.xdata) * k, ev.xdata + (xb - ev.xdata) * k)
        self.ax.set_ylim(ev.ydata + (ya - ev.ydata) * k, ev.ydata + (yb - ev.ydata) * k)
        self.fig.canvas.draw_idle()


def patches_with_manual_features(cfg: dict, sheet_id: str) -> set[str]:
    """Patch ids whose Patch_Grid row lists a manual layer (from the sheet's output GeoPackage)."""
    gpkg = ROOT / cfg["paths"]["outputs"] / f"{sheet_id}.gpkg"
    if not gpkg.exists():
        return set()
    con = sqlite3.connect(gpkg)
    try:
        return {r[0] for r in con.execute(
            "SELECT patch_id FROM Patch_Grid WHERE manual_features IS NOT NULL AND manual_features != ''")}
    except sqlite3.Error:
        return set()                                   # Patch_Grid has no manual columns yet
    finally:
        con.close()


def pick(sheet_id: str, thumb: int | None, name: str | None = None) -> None:
    cfg = load_config()
    patches_dir = ROOT / cfg["paths"]["patches"] / "images" / sheet_id
    json_dir = ROOT / cfg["paths"]["annotations"] / "labelme_json" / sheet_id
    if not patches_dir.exists():
        sys.exit(f"Patches not found: {patches_dir}  - run patchify first.")

    files, cells = [], []
    for f in sorted(patches_dir.glob("*.png")):
        cell = parse_cell(f.stem)
        if cell:
            files.append(f)
            cells.append(cell)
    if not files:
        sys.exit(f"No patches named <sheet>_r<row>_c<col>.png in {patches_dir}")

    span = max(max(r for r, _ in cells) - min(r for r, _ in cells),
               max(c for _, c in cells) - min(c for _, c in cells)) + 1
    thumb = thumb or int(np.clip(2400 // span, 24, 96))      # keep the overview ~2400px across
    stems = [f.stem for f in files]

    out = selection_file(cfg, sheet_id, name)
    selected = set()
    if out.exists():
        selected = {ln.strip() for ln in out.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
        print(f"Loaded existing selection: {len(selected & set(stems))} patch(es) from {rel_to_root(out)}")
    done = {p.stem for p in json_dir.glob("*.json")} if json_dir.exists() else set()
    marked = patches_with_manual_features(cfg, sheet_id) if name == "manual" else set()

    print(f"Sheet : {sheet_id}  |  {len(files)} patches on a {span}-cell-wide grid  |  thumbnails {thumb}px")
    mosaic, r0, c0, r1, c1 = build_mosaic(files, cells, thumb)

    try:                                                      # same backend as draw_mask.py (works under WSLg)
        matplotlib.use("TkAgg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    for k in ("keymap.back", "keymap.all_axes"):              # free up 'c' (clear) and 'a' (select all)
        if k in plt.rcParams:                                 # keymap names differ between matplotlib versions
            plt.rcParams[k] = [x for x in plt.rcParams[k] if x not in ("a", "c")]

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(mosaic, extent=(c0, c1 + 1, r1 + 1, r0), interpolation="nearest")
    ax.set_xlim(c0, c1 + 1)
    ax.set_ylim(r1 + 1, r0)
    ax.set_xlabel("grid column"); ax.set_ylabel("grid row")
    picker = Picker(fig, ax, stems, cells, selected, done, marked)
    print("Window open: click / drag to select, Enter to save, Esc to cancel.")
    plt.tight_layout()
    plt.show()                                                # blocks until the window closes

    if not picker.saved:
        print("Closed without saving - selection unchanged.")
        return
    chosen = picker.chosen()
    if not chosen:
        sys.exit("Nothing selected - not saved (select at least one patch, or run autolines without --select).")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(chosen) + "\n")
    print(f"\nSaved {len(chosen)} patch(es) -> {rel_to_root(out)}")
    if name == "manual":
        print(
            f"\nNext step: export these patches of your manual layers as training annotations\n"
            f"  conda activate maptools\n"
            f"  python run.py export-manual --sheet {sheet_id} --select\n"
            f"  python run.py export-masks --sheet {sheet_id}"
        )
        return
    print(
        f"\nNext step: draw lines on just these patches (check the overlays first)\n"
        f"  conda activate maptools\n"
        f"  python run.py autolines --sheet {sheet_id} --select --preview\n"
        f"  python run.py autolines --sheet {sheet_id} --select"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pick patches for autolines in a window.")
    ap.add_argument("--sheet", required=True, help="Sheet ID")
    ap.add_argument("--name", default=None, choices=["manual"],
                    help="Save under a separate name. 'manual' = choose patches of your manual layers "
                         "for export-manual (kept apart from the autolines selection)")
    ap.add_argument("--thumb", type=int, default=None,
                    help="Thumbnail size in px (default: sized so the overview is ~2400px across)")
    args = ap.parse_args()
    pick(args.sheet, args.thumb, args.name)
