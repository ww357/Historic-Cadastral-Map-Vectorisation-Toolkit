"""
Launch labelme to annotate patches from a map sheet.

All feature classes are annotated in a single labelme session:
  boundary  → linestrip  — land parcel boundary lines
  water     → polygon    — water features
  building  → polygon    — building footprints
  damage    → polygon    — document damage / symbology cover
  <text>    → polygon    — label = the actual text string

Annotations are saved as JSON files to:
  data/annotations/labelme_json/<SHEET_ID>/

Run export_masks.py after annotating to convert JSON → binary masks.

Usage:
    python annotate.py --sheet SHEET_ID
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
    """Convert a WSL /mnt/c/... path to a Windows C:\... path."""
    s = str(p)
    if s.startswith("/mnt/"):
        drive = s[5]                          # e.g. 'c'
        rest  = s[6:].replace("/", "\\")      # strip /mnt/c, flip slashes
        return f"{drive.upper()}:{rest}"
    return s


def annotate(sheet_id: str):
    cfg         = load_config()
    patches_dir = ROOT / cfg["paths"]["patches"] / "images" / sheet_id
    json_dir    = ROOT / cfg["paths"]["annotations"] / "labelme_json" / sheet_id
    config_path = Path(__file__).parent / "labelme_config.yaml"

    if not patches_dir.exists():
        sys.exit(f"Patches not found: {patches_dir}  — run 01_patchify first.")

    json_dir.mkdir(parents=True, exist_ok=True)

    existing = list(json_dir.glob("*.json"))
    print(f"Sheet      : {sheet_id}")
    print(f"Patches    : {patches_dir}")
    print(f"JSON output: {json_dir}")
    if existing:
        print(f"Resuming   : {len(existing)} patches already annotated")
    print()
    print("In labelme:")
    print("  Draw linestrip  (boundaries) — Edit > Shortcuts to assign Ctrl+L")
    print("  Draw polygon    (water / building / damage / text)")
    print("  For text polygons, type the actual text as the label.")
    print()

    if is_wsl():
        # Launch the native Windows labelme via PowerShell to avoid WSLg Qt issues
        win_patches = to_windows_path(patches_dir)
        win_json    = to_windows_path(json_dir)
        win_config  = to_windows_path(config_path)
        cmd = [
            "powershell.exe", "-Command",
            f'labelme "{win_patches}" --output "{win_json}" --config "{win_config}"',
        ]
        print("Launching Windows labelme via PowerShell...")
    else:
        if shutil.which("labelme") is None:
            sys.exit("labelme not found. Install it with:  pip install labelme")
        cmd = [
            "labelme",
            str(patches_dir),
            "--output", str(json_dir),
            "--config", str(config_path),
        ]

    subprocess.run(cmd)

    completed = list(json_dir.glob("*.json"))
    print(f"\nSession ended. {len(completed)} patch(es) annotated.")
    print(f"Run export_masks.py --sheet {sheet_id} to generate binary masks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open map patches in labelme for annotation.")
    parser.add_argument("--sheet", required=True, help="Sheet ID")
    args = parser.parse_args()
    annotate(args.sheet)
