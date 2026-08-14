"""
Download the base model weights into models/base/ so the toolkit is ready to run.

Each model's weights live in a separate HuggingFace repo whose contents mirror
the destination folder, so fetching is one predictable step per model.  The plain
SAM ViT-B checkpoint is NOT re-hosted — it is Meta's public, unmodified file and
is pulled straight from their official URL.

    python steps/fetch_weights.py                 # fetch everything missing
    python steps/fetch_weights.py --only sam      # just one model
    python steps/fetch_weights.py --force         # re-download even if present
    python steps/fetch_weights.py --list          # show the plan without downloading

Needs `huggingface_hub` for the HF repos (pip install huggingface_hub); the SAM
checkpoint uses only the standard library.  Runs in any conda environment.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── Edit these to match YOUR HuggingFace repos ─────────────────────────────────
# key         : short name for --only
# repo/url    : HF repo id, or a direct download URL for a single file
# dest        : folder the files land in (repo contents are mirrored into it)
# note        : shown in --list
WEIGHTS = {
    "unet": {
        "repo": "ww357/Improved-Linear-U-Net",
        "dest": "models/base/unet",
        "note": "boundary U-Net base weights (*.weights.h5)",
    },
    "dashed": {
        "repo": "ww357/DashedLineUNet",
        "dest": "models/base/dashed",
        "note": "dashed-line U-Net base weights (*.h5)",
    },
    "mapsam": {
        "repo": "ww357/MapSAM-Tithe-Map-Features",
        "dest": "models/base/MapSAM",
        "note": "DoRA feature adapters, one subfolder per feature (water/house/...)",
    },
    "text": {
        "repo": "ww357/MapTextPipeline-Rumsey",
        "dest": "models/base/MapTextPipeline",
        "note": "MapTextPipeline Rumsey weights (rumsey-finetune.pth)",
    },
    # Meta's public SAM ViT-B checkpoint — fetched from source, never re-hosted.
    "sam": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "dest": "models/base/MapSAM/original_weights",
        "note": "generic SAM ViT-B backbone (357 MB, Meta official - not our weights)",
    },
}


def _has_files(dest: Path) -> bool:
    """True if dest already holds at least one weight file."""
    if not dest.is_dir():
        return False
    return any(dest.rglob("*.pth")) or any(dest.rglob("*.h5"))


def _download_url(url: str, dest_file: Path) -> None:
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")

    def hook(count, block, total):
        if total > 0:
            pct = min(100, count * block * 100 // total)
            print(f"\r  {pct:3d}%  ({total // (1024*1024)} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest_file, reporthook=hook)
    print(f"\r  -> {dest_file.relative_to(ROOT)}            ")


def _download_repo(repo_id: str, dest: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "huggingface_hub is required for HF repos:\n"
            "  pip install huggingface_hub\n"
            "(The SAM checkpoint — --only sam — does not need it.)"
        )
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {repo_id}")
    snapshot_download(repo_id=repo_id, local_dir=str(dest),
                      local_dir_use_symlinks=False)
    print(f"  -> {dest.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download base model weights into models/base/.")
    ap.add_argument("--only", choices=list(WEIGHTS), default=None,
                    help="Fetch just one model (default: all missing).")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if weights are already present.")
    ap.add_argument("--list", action="store_true",
                    help="Show what would be fetched, and what already exists, then exit.")
    args = ap.parse_args()

    items = {args.only: WEIGHTS[args.only]} if args.only else WEIGHTS

    if args.list:
        print(f"\n{'model':<8} {'status':<9} destination")
        print("-" * 64)
        for key, spec in items.items():
            dest = ROOT / spec["dest"]
            status = "present" if _has_files(dest) else "MISSING"
            print(f"{key:<8} {status:<9} {spec['dest']}")
            print(f"{'':<18} {spec['note']}")
        print()
        return

    for key, spec in items.items():
        dest = ROOT / spec["dest"]
        print(f"\n[{key}]  {spec['note']}")
        if _has_files(dest) and not args.force:
            print(f"  already present in {spec['dest']} — skipping (use --force to refetch)")
            continue
        if "url" in spec:
            _download_url(spec["url"], dest / Path(spec["url"]).name)
        else:
            _download_repo(spec["repo"], dest)

    print("\nDone. Check models/base/ against the layout in the README.")


if __name__ == "__main__":
    main()
