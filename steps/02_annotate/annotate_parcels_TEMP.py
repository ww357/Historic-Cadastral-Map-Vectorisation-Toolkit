"""
Launch labelme to annotate parcel patches for SAM fine-tuning.

Patches are 1024px PNGs created by steps/01_patchify/parcels/parcel_patchify.py.
Draw filled polygons labelled "parcel" around each land parcel.
labelme saves JSON files alongside the PNGs in the same directory.

You don't need to annotate every patch — 30-100 well-drawn parcels covering a
variety of sizes and boundary types is enough to start fine-tuning.

Usage:
    python steps/02_annotate/annotate_parcels.py --sheet SHEET_ID

After annotating, export masks and fine-tune:
    python steps/02_annotate/export_masks.py --sheet SHEET_ID \\
        --patches-dir data/patches/parcel/SHEET_ID \\
        --json-dir    data/patches/parcel/SHEET_ID

    python steps/03_finetune/parcels/train.py --sheet SHEET_ID
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.exists():
        sys.exit(f"config.yaml not found at {p}")
    return yaml.safe_load(p.read_text())


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except FileNotFoundError:
        return False


def to_windows_path(p: Path) -> str:
    s = str(p)
    if s.startswith("/mnt/"):
        drive = s[5]
        rest  = s[6:].replace("/", "\\")
        return f"{drive.upper()}:{rest}"
    return s


def annotate_parcels(sheet_id: str):
    cfg         = load_config()
    patches_dir = ROOT / "data" / "patches" / "parcel" / sheet_id
    config_path = Path(__file__).parent / "labelme_config_parcels.yaml"

    if not patches_dir.exists():
        sys.exit(
            f"Patches not found: {patches_dir}\n"
            f"Run first:  python steps/01_patchify/parcels/parcel_patchify.py --sheet {sheet_id}"
        )

    existing = list(patches_dir.glob("*.json"))
    patch_count = len(list(patches_dir.glob("*.png")))

    print(f"Sheet      : {sheet_id}")
    print(f"Patches    : {patches_dir}  ({patch_count} PNGs)")
    if existing:
        print(f"Resuming   : {len(existing)} patches already annotated")
    print()
    print("In labelme:")
    print("  Press P to draw a polygon — trace around each land parcel.")
    print("  Type 'parcel' as the label (or press Enter if it auto-fills).")
    print("  Annotate as many parcels as you can see on each patch.")
    print("  30-100 well-drawn examples across several patches is enough to start.")
    print()

    if is_wsl():
        win_patches = to_windows_path(patches_dir)
        win_config  = to_windows_path(config_path)
        cmd = [
            "powershell.exe", "-Command",
            f'labelme "{win_patches}" --config "{win_config}"',
        ]
        print("Launching Windows labelme via PowerShell...")
    else:
        if shutil.which("labelme") is None:
            sys.exit("labelme not found. Install it with:  pip install labelme")
        cmd = [
            "labelme",
            str(patches_dir),
            "--config", str(config_path),
        ]

    subprocess.run(cmd)

    completed = list(patches_dir.glob("*.json"))
    print(f"\nSession ended. {len(completed)} patch(es) annotated.")
    print()
    print("Next steps:")
    print(f"  python steps/02_annotate/export_masks.py --sheet {sheet_id} \\")
    print(f"      --patches-dir data/patches/parcel/{sheet_id} \\")
    print(f"      --json-dir    data/patches/parcel/{sheet_id}")
    print()
    print(f"  python steps/03_finetune/parcels/train.py --sheet {sheet_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Open parcel patches in labelme for SAM fine-tuning annotation."
    )
    parser.add_argument("--sheet", required=True, help="Sheet ID")
    args = parser.parse_args()
    annotate_parcels(args.sheet)
