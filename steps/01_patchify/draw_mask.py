"""
Interactively draw the map-area mask for a sheet directly in this script.

Opens a downsampled preview of the source GeoTIFF in a Matplotlib window.
Click to place polygon vertices that outline the map area; press Enter or
double-click to close and save the mask.

The polygon is scaled back to full resolution and rendered as a filled
binary PNG — identical in format to a hand-drawn mask from any other tool.

Saved to:
    data/map_area_masks/<SHEET_ID>/<SHEET_ID>.png

patchify.py --mask picks this up automatically.

Usage:
    python draw_mask.py --sheet SHEET_ID [--preview-size 1500]

Controls:
    Left-click      Add a vertex
    Right-click     Remove last vertex
    Enter / Return  Close polygon and save
    Escape          Quit without saving
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


# ---------------------------------------------------------------------------
# Image loading (downsampled for display)
# ---------------------------------------------------------------------------

def load_preview(tif_path: Path, preview_size: int) -> tuple[np.ndarray, int, int, float]:
    """
    Load the source GeoTIFF as a downsampled RGB/L uint8 preview.

    Returns
    -------
    preview     : (H, W) or (H, W, 3) uint8 array
    full_w      : original image width in pixels
    full_h      : original image height in pixels
    scale       : preview / full  (< 1.0)
    """
    with rasterio.open(tif_path) as src:
        full_w, full_h = src.width, src.height
        scale = min(preview_size / full_w, preview_size / full_h, 1.0)
        out_w = max(1, int(full_w * scale))
        out_h = max(1, int(full_h * scale))

        # Read at reduced resolution using rasterio's out parameter
        if src.count >= 3:
            data = src.read(
                [1, 2, 3],
                out=np.empty((3, out_h, out_w), dtype=np.uint8),
                resampling=rasterio.enums.Resampling.average,
            )
            preview = np.moveaxis(data, 0, -1)          # (H, W, 3)
        else:
            data = src.read(
                1,
                out=np.empty((out_h, out_w), dtype=np.uint8),
                resampling=rasterio.enums.Resampling.average,
            )
            preview = data                               # (H, W)

    return preview, full_w, full_h, scale


# ---------------------------------------------------------------------------
# Interactive polygon drawing
# ---------------------------------------------------------------------------

class PolygonDrawer:
    """
    Matplotlib-based interactive polygon editor.

    Stores vertices in *preview* pixel coordinates.
    Pressing Enter (or the toolbar close button) finalises.
    """

    VERTEX_COLOR  = "#ff3300"
    LINE_COLOR    = "#ff3300"
    FILL_COLOR    = "#ff330033"   # semi-transparent fill
    CURSOR_COLOR  = "#ffffff88"

    def __init__(self, ax: plt.Axes, fig: plt.Figure):
        self.ax     = ax
        self.fig    = fig
        self.xs: list[float] = []
        self.ys: list[float] = []
        self.done   = False
        self.saved  = False

        # Live artists
        self._line,  = ax.plot([], [], color=self.LINE_COLOR,  lw=1.5, zorder=3)
        self._verts, = ax.plot([], [], "o", color=self.VERTEX_COLOR, ms=5, zorder=4)
        self._fill   = None
        self._cursor_line = None   # ghost line from last vertex to cursor

        # Instructions overlay (top-left)
        self._help = ax.text(
            0.01, 0.99,
            "Left-click: add vertex\n"
            "Right-click: undo last\n"
            "Enter: save & close\n"
            "Esc: quit",
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=8,
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.55),
            zorder=10,
        )

        self._cid_click  = fig.canvas.mpl_connect("button_press_event",  self._on_click)
        self._cid_key    = fig.canvas.mpl_connect("key_press_event",      self._on_key)
        self._cid_motion = fig.canvas.mpl_connect("motion_notify_event",  self._on_motion)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_click(self, event):
        if self.done or event.inaxes is not self.ax:
            return
        if event.button == 1:          # left-click → add vertex
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self._update()
        elif event.button == 3:        # right-click → undo last vertex
            if self.xs:
                self.xs.pop()
                self.ys.pop()
                self._update()

    def _on_key(self, event):
        if event.key in ("enter", "return"):
            self._finalise(save=True)
        elif event.key == "escape":
            self._finalise(save=False)

    def _on_motion(self, event):
        """Draw a ghost segment from the last vertex to the cursor."""
        if self.done or event.inaxes is not self.ax or not self.xs:
            return
        if self._cursor_line is None:
            self._cursor_line, = self.ax.plot([], [], "--",
                                              color=self.LINE_COLOR,
                                              lw=1, alpha=0.5, zorder=2)
        self._cursor_line.set_data(
            [self.xs[-1], event.xdata],
            [self.ys[-1], event.ydata],
        )
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _update(self):
        n = len(self.xs)
        if n == 0:
            self._line.set_data([], [])
            self._verts.set_data([], [])
            self._clear_fill()
        else:
            # Close the loop visually if ≥ 3 vertices
            xs_loop = self.xs + ([self.xs[0]] if n >= 3 else [])
            ys_loop = self.ys + ([self.ys[0]] if n >= 3 else [])
            self._line.set_data(xs_loop, ys_loop)
            self._verts.set_data(self.xs, self.ys)
            if n >= 3:
                self._draw_fill()
            else:
                self._clear_fill()
        self.fig.canvas.draw_idle()

    def _draw_fill(self):
        self._clear_fill()
        from matplotlib.patches import Polygon as MplPolygon
        poly_pts = list(zip(self.xs, self.ys))
        self._fill = mpatches.Polygon(
            poly_pts,
            closed=True,
            facecolor=self.FILL_COLOR,
            edgecolor="none",
            zorder=2,
        )
        self.ax.add_patch(self._fill)

    def _clear_fill(self):
        if self._fill is not None:
            self._fill.remove()
            self._fill = None

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------

    def _finalise(self, save: bool):
        self.done  = True
        self.saved = save
        # Disconnect events
        for cid in (self._cid_click, self._cid_key, self._cid_motion):
            self.fig.canvas.mpl_disconnect(cid)
        if self._cursor_line is not None:
            self._cursor_line.set_data([], [])
        plt.close(self.fig)

    def get_polygon(self) -> list[tuple[float, float]] | None:
        """Return list of (x, y) preview-space coords, or None if cancelled."""
        if self.saved and len(self.xs) >= 3:
            return list(zip(self.xs, self.ys))
        return None


# ---------------------------------------------------------------------------
# Mask rendering
# ---------------------------------------------------------------------------

def render_mask(
    polygon_px: list[tuple[float, float]],
    scale: float,
    full_w: int,
    full_h: int,
) -> np.ndarray:
    """
    Scale polygon coords back to full resolution and render a binary mask.

    Parameters
    ----------
    polygon_px  : vertices in preview-space pixel coords (x, y)
    scale       : preview_size / full_size ratio
    full_w/h    : full-resolution image dimensions

    Returns
    -------
    uint8 numpy array (full_h, full_w), 255 = inside, 0 = outside
    """
    # Scale up to full resolution
    full_poly = [(int(round(x / scale)), int(round(y / scale))) for x, y in polygon_px]

    img = Image.new("L", (full_w, full_h), 0)
    ImageDraw.Draw(img).polygon(full_poly, fill=255)
    return np.array(img)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def draw_mask(sheet_id: str, preview_size: int):
    cfg = load_config()

    raw_path = ROOT / cfg["paths"]["raw"]   / sheet_id / f"{sheet_id}.tif"
    out_dir  = ROOT / cfg["paths"]["masks"] / sheet_id
    out_path = out_dir / f"{sheet_id}.png"

    if not raw_path.exists():
        sys.exit(
            f"Source image not found: {raw_path}\n"
            f"Place the GeoTIFF at data/raw/{sheet_id}/{sheet_id}.tif and try again."
        )

    print(f"Sheet        : {sheet_id}")
    print(f"Source       : {raw_path}")
    print(f"Preview size : {preview_size}px (longest edge)")
    print(f"Will save to : {out_path}")
    print()

    # ------------------------------------------------------------------
    # Load preview
    # ------------------------------------------------------------------
    print("Loading preview image…")
    preview, full_w, full_h, scale = load_preview(raw_path, preview_size)
    print(f"Full size    : {full_w} × {full_h} px")
    print(f"Preview size : {preview.shape[1]} × {preview.shape[0]} px  "
          f"(scale = {scale:.4f})\n")

    # ------------------------------------------------------------------
    # Open Matplotlib window
    # ------------------------------------------------------------------
    # Use a non-blocking backend that works in WSL (TkAgg) and native Windows
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass   # fall through to whatever backend is available

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title(f"Draw Map-Area Mask — {sheet_id}")
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    cmap = "gray" if preview.ndim == 2 else None
    ax.imshow(preview, cmap=cmap, interpolation="bilinear", origin="upper")
    ax.set_title(
        f"{sheet_id}   ({full_w}×{full_h} px, shown at {scale*100:.0f}%)\n"
        "Outline the MAP AREA — exclude margins, legends, north arrows etc.",
        color="white", fontsize=9, pad=6,
    )
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    drawer = PolygonDrawer(ax, fig)
    plt.tight_layout()
    plt.show()   # blocks until window is closed

    # ------------------------------------------------------------------
    # Save result
    # ------------------------------------------------------------------
    polygon = drawer.get_polygon()
    if polygon is None:
        print("Cancelled — no mask saved.")
        sys.exit(0)

    print(f"Polygon has {len(polygon)} vertices.")
    print("Rendering full-resolution mask…")
    mask = render_mask(polygon, scale, full_w, full_h)

    # Sanity check: mask must cover at least 1% of image
    coverage = mask.sum() / 255 / (full_w * full_h)
    if coverage < 0.01:
        print(
            "Warning: mask covers only {:.2f}% of image — polygon may be too small. "
            "Re-run to try again.".format(coverage * 100)
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(out_path)

    print(f"\nSaved → {out_path}")
    print(f"  Map-area coverage: {coverage * 100:.1f}% of image")
    print("\nRun patchify.py --sheet {} --mask  to use this mask.".format(sheet_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw map-area mask interactively in a Matplotlib window."
    )
    parser.add_argument("--sheet",        required=True,
                        help="Sheet ID (must match data/raw/<SHEET_ID>/<SHEET_ID>.tif)")
    parser.add_argument("--preview-size", type=int, default=1500,
                        help="Longest edge of the preview image in pixels (default: 1500)")
    args = parser.parse_args()
    draw_mask(args.sheet, args.preview_size)
