"""
Single entry point for every pipeline step.

    python run.py <command> [--sheet SHEET] [any other flags...]

Saves having to remember script paths; all flags after the command are passed
straight through to the underlying script, so anything the script accepts works
here unchanged.

This CANNOT switch conda environment for you — a running process cannot change
its own interpreter. What it does instead is check the active environment before
launching and stop with the exact activate command if it is wrong, which turns a
confusing ImportError deep in a script into a one-line fix. Use --force-env to
skip that check (e.g. if your environments are named differently).

    python run.py list                     # every command, with its environment
    python run.py status --sheet SHEET     # how far this sheet has progressed
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# command -> (script path relative to repo root, required conda env)
# Environments are per CONTEXT.md: maptools = geospatial, lines = TensorFlow
# U-Net (solid + dashed), polygons = PyTorch (MapSAM / text / parcels).
COMMANDS: dict[str, tuple[str, str | None]] = {
    # 01 patchify
    "reproject":          ("steps/01_patchify/reproject.py",              "maptools"),
    "draw-mask":          ("steps/01_patchify/draw_mask.py",              "maptools"),
    "patchify":           ("steps/01_patchify/patchify.py",               "maptools"),
    # 02 annotate
    "annotate":           ("steps/02_annotate/annotate.py",               "maptools"),
    "export-masks":       ("steps/02_annotate/export_masks.py",           "maptools"),
    # 03 finetune
    "train-lines":        ("steps/03_finetune/lines/train.py",            "lines"),
    "evaluate-lines":     ("steps/03_finetune/lines/evaluate.py",         "lines"),
    "dashed-masks":       ("steps/03_finetune/dashed/gaussian_masks.py",  "lines"),
    "train-dashed":       ("steps/03_finetune/dashed/train.py",           "lines"),
    "train-polygons":     ("steps/03_finetune/polygons/train.py",         "polygons"),
    # 04 predict
    "predict-lines":      ("steps/04_predict/lines/predict.py",           "lines"),
    "predict-dashed":     ("steps/04_predict/dashed/predict.py",          "lines"),
    "predict-polygons":   ("steps/04_predict/polygons/predict.py",        "polygons"),
    "predict-text":       ("steps/04_predict/text/predict.py",            "polygons"),
    "predict-parcels":    ("steps/04_predict/parcels/predict.py",         "polygons"),
    # 05 vectorise
    "vectorise-lines":    ("steps/05_vectorise/lines/vectorise.py",       "maptools"),
    "vectorise-dashed":   ("steps/05_vectorise/dashed/vectorise.py",      "maptools"),
    "vectorise-polygons": ("steps/05_vectorise/polygons/vectorise.py",    "maptools"),
    "vectorise-text":     ("steps/05_vectorise/text/vectorise.py",        "maptools"),
    "vectorise-parcels":  ("steps/05_vectorise/parcels/vectorise.py",     "polygons"),
    # 06 feedback
    "prepare-lines":      ("steps/06_feedback/lines/prepare.py",          "maptools"),
    "feedback-lines":     ("steps/06_feedback/lines/train.py",            "lines"),
    "prepare-polygons":   ("steps/06_feedback/polygons/prepare.py",       "polygons"),
    "feedback-polygons":  ("steps/06_feedback/polygons/train.py",         "polygons"),
    # utility — run in any environment
    "status":             ("steps/status.py",                             None),
    "fetch-weights":      ("steps/fetch_weights.py",                      None),
}

# Printed by `list`, grouped so the pipeline order is obvious.
GROUPS = [
    ("01 patchify",  ["reproject", "draw-mask", "patchify"]),
    ("02 annotate",  ["annotate", "export-masks"]),
    ("03 finetune",  ["train-lines", "train-dashed", "dashed-masks",
                      "train-polygons", "evaluate-lines"]),
    ("04 predict",   ["predict-lines", "predict-dashed", "predict-polygons",
                      "predict-text", "predict-parcels"]),
    ("05 vectorise", ["vectorise-lines", "vectorise-dashed", "vectorise-polygons",
                      "vectorise-text", "vectorise-parcels"]),
    ("06 feedback",  ["prepare-lines", "feedback-lines",
                      "prepare-polygons", "feedback-polygons"]),
    ("utility",      ["fetch-weights", "status"]),
]


def current_env() -> str | None:
    """Active conda environment name, or None if not in one."""
    name = os.environ.get("CONDA_DEFAULT_ENV")
    if name:
        return name
    prefix = os.environ.get("CONDA_PREFIX")
    return Path(prefix).name if prefix else None


def print_list() -> None:
    print("\nUsage:  python run.py <command> [--sheet SHEET] [flags...]\n")
    for title, names in GROUPS:
        print(f"  {title}")
        for n in names:
            script, env = COMMANDS[n]
            print(f"    {n:<20} {env or 'any':<9} {script}")
        print()
    print("All flags after the command are passed through to the script.\n"
          "Example:  python run.py predict-parcels --sheet Selworthy "
          "--extent boundaries dashed\n")


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help", "list"):
        print_list()
        sys.exit(0 if argv else 1)

    command, passthrough = argv[0], argv[1:]
    if command not in COMMANDS:
        close = [c for c in COMMANDS if command in c or c.startswith(command[:4])]
        print(f"Unknown command: {command}", file=sys.stderr)
        if close:
            print(f"Did you mean: {', '.join(sorted(close))}", file=sys.stderr)
        print("Run 'python run.py list' to see all commands.", file=sys.stderr)
        sys.exit(1)

    script_rel, needed_env = COMMANDS[command]
    script = ROOT / script_rel
    if not script.exists():
        sys.exit(f"Script not found: {script}")

    # Environment guard — catches the most common failure in a 3-env pipeline
    # before it turns into an ImportError halfway through a script.
    force = "--force-env" in passthrough
    passthrough = [a for a in passthrough if a != "--force-env"]
    active = current_env()
    if needed_env and not force and active != needed_env:
        print(
            f"'{command}' needs the '{needed_env}' environment "
            f"(active: {active or 'none'}).\n\n"
            f"  conda activate {needed_env}\n"
            f"  python run.py {command} {' '.join(passthrough)}\n\n"
            f"Pass --force-env to run anyway.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Same interpreter, so the script runs in whatever env is active.
    raise SystemExit(subprocess.run([sys.executable, str(script), *passthrough]).returncode)


if __name__ == "__main__":
    main()
