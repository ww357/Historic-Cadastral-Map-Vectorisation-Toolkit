"""
Launch labelme to annotate patches from a map sheet.

All feature classes are annotated in a single labelme session:
  boundary  -> linestrip  - land parcel boundary lines
  water     -> polygon    - water features
  building  -> polygon    - building footprints
  damage    -> polygon    - document damage / symbology cover
  <text>    -> polygon    - label = the actual text string

Annotations are saved as JSON files to:
  data/annotations/labelme_json/<SHEET_ID>/

Run export_masks.py after annotating to convert JSON -> binary masks.

Usage:
    python annotate.py --sheet SHEET_ID
"""

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "steps"))   # shared helpers
from common import load_config   # noqa: E402

AUTO_TAG = "autolines"   # description tag autolines.py puts on the shapes it draws


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except FileNotFoundError:
        return False


def to_windows_path(p: Path) -> str:
    r"""Convert a WSL /mnt/c/... path to a Windows C:\... path."""
    s = str(p)
    if s.startswith("/mnt/"):
        drive = s[5]                          # e.g. 'c'
        rest  = s[6:].replace("/", "\\")      # strip /mnt/c, flip slashes
        return f"{drive.upper()}:{rest}"
    return s


def annotate(sheet_id: str, plain: bool = False):
    cfg         = load_config()
    patches_dir = ROOT / cfg["paths"]["patches"] / "images" / sheet_id
    json_dir    = ROOT / cfg["paths"]["annotations"] / "labelme_json" / sheet_id
    config_path = Path(__file__).parent / "labelme_config.yaml"
    launcher    = Path(__file__).parent / "labelme_launch.py"   # labelme + drag-select

    if not patches_dir.exists():
        sys.exit(f"Patches not found: {patches_dir}  - run 01_patchify first.")

    json_dir.mkdir(parents=True, exist_ok=True)

    existing = list(json_dir.glob("*.json"))
    n_auto = sum(AUTO_TAG in p.read_text() for p in existing)
    print(f"Sheet      : {sheet_id}")
    print(f"Patches    : {patches_dir}")
    print(f"JSON output: {json_dir}")
    if existing:
        print(f"Resuming   : {len(existing)} patches have a JSON"
              + (f" ({n_auto} pre-drawn by autolines - review them: delete wrong lines, then Save)"
                 if n_auto else " already annotated"))
    elif patches_dir.exists():
        print("Tip        : autolines.py can pre-draw boundary lines to edit instead of tracing from scratch")
    print()
    print("In labelme:")
    print("  Draw linestrip  (boundaries) - Edit > Shortcuts to assign Ctrl+L")
    print("  Draw polygon    (water / building / damage / text)")
    print("  For text polygons, type the actual text as the label.")
    if not plain:
        print("  Clearing lines: drag a box on empty background (left->right = fully inside,")
        print("  right->left = touching), Ctrl+drag adds, then Delete. Ctrl+Z undoes.")
    print()

    if is_wsl():
        # Launch the native Windows labelme via PowerShell to avoid WSLg Qt issues.
        # The drag-select launcher needs the Windows python that has labelme installed;
        # if `python -c "import labelme"` fails there, fall back to the plain command.
        win_patches = to_windows_path(patches_dir)
        win_json    = to_windows_path(json_dir)
        win_config  = to_windows_path(config_path)
        args = f'"{win_patches}" --output "{win_json}" --config "{win_config}"'
        if plain:
            ps = f"labelme {args}"
        else:
            ps = (f'python -c "import labelme" 2>$null; '
                  f'if ($LASTEXITCODE -eq 0) {{ python "{to_windows_path(launcher)}" {args} }} '
                  f'else {{ Write-Host "labelme_launch: no Windows python with labelme - plain labelme"; '
                  f'labelme {args} }}')
        cmd = ["powershell.exe", "-Command", ps]
        print("Launching Windows labelme via PowerShell...")
    else:
        use_launcher = not plain and importlib.util.find_spec("labelme") is not None
        if not use_launcher and shutil.which("labelme") is None:
            sys.exit("labelme not found. Install it with:  pip install labelme")
        base = [sys.executable, str(launcher)] if use_launcher else ["labelme"]
        cmd = base + [
            str(patches_dir),
            "--output", str(json_dir),
            "--config", str(config_path),
        ]

    subprocess.run(cmd)

    completed = list(json_dir.glob("*.json"))
    print(f"\nSession ended. {len(completed)} patch(es) annotated.")
    print(
        f"\nNext step: export the labelme JSON to binary masks\n"
        f"  conda activate maptools\n"
        f"  python run.py export-masks --sheet {sheet_id}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open map patches in labelme for annotation.")
    parser.add_argument("--sheet", required=True, help="Sheet ID")
    parser.add_argument("--plain", action="store_true",
                        help="Start plain labelme without the drag-select launcher")
    args = parser.parse_args()
    annotate(args.sheet, args.plain)
