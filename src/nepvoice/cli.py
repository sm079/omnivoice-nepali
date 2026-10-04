"""``nepvoice``: Nepali TTS data preparation and OmniVoice fine-tuning.

    nepvoice run INPUT_DIR --work dataset/preprocessed1 --to manifest     data preparation
    nepvoice run --work dataset/preprocessed1 --run models/run1 --from splits   training
    nepvoice run INPUT_DIR --work ... --run ...                            everything
    nepvoice run --work ... --from select --to export                      re-run part of it
    nepvoice merge models/run1 --out merged/run1     the run's adapter merged into a standalone model
    nepvoice config                                  print the resolved configuration
    nepvoice speakers                                report the speaker registry

Settings come from ``--config`` TOML files (later ones win) and ``--set section.key=value``.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

from . import config as config_mod
from .pipeline import STEPS, TRAIN_STEPS, run, select_steps
from .train.trainer import TrainingStateError


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--work", type=Path, default=Path("work"), help="work directory (default: ./work)")
    p.add_argument("--config", type=Path, action="append", default=[], metavar="TOML",
                   help="configuration file (repeatable; later files override earlier ones)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", dest="overrides",
                   help="override one setting, e.g. --set select.pad=0.5 (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nepvoice", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the pipeline (all steps by default)",
                       description=f"Steps: {', '.join(STEPS)}.")
    r.add_argument("input", type=Path, nargs="?", help="folder of recordings with transcripts (needed for ingest)")
    _common(r)
    r.add_argument("--run", type=Path, default=None, metavar="DIR",
                   help="training run directory, e.g. models/run1 (needed for the training steps)")
    r.add_argument("--from", dest="start", choices=STEPS, help="first step (default: ingest)")
    r.add_argument("--to", dest="stop", choices=STEPS, help="last step (default: package)")
    r.add_argument("--only", help="comma-separated steps to run, e.g. select,export")

    c = sub.add_parser("config", help="print the resolved configuration as TOML")
    _common(c)

    s = sub.add_parser("speakers", help="report the speaker registry of the work directory")
    _common(s)
    s.add_argument("--limit", type=int, default=15)

    m = sub.add_parser("merge", help="merge a packaged run's LoRA adapter into a standalone model")
    m.add_argument("run", type=Path, help="packaged run directory, e.g. models/run1")
    m.add_argument("--out", type=Path, required=True, help="output folder for the merged model")
    m.add_argument("--checkpoint", default=None, help="another saved step, e.g. checkpoint-2000 (default: final)")
    m.add_argument("--base-model", default=None, help="default: the adapter's base model")
    return parser


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    if hasattr(sys.stdout, "reconfigure"):  # Devanagari text on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    if args.command == "merge":
        from .train.trainer import merge

        try:
            print(f"merged model: {merge(args.run, args.out, args.base_model, args.checkpoint)}")
        except (FileNotFoundError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0
    try:
        cfg = config_mod.load(args.config, args.overrides)
    except config_mod.ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    if args.command == "config":
        print(config_mod.to_toml(cfg))
        return 0

    if args.command == "speakers":
        from .prep import speakers

        path = args.work / "speakers.json"
        if not path.exists():
            print(f"{path} not found; run the speakers step first", file=sys.stderr)
            return 1
        print(speakers.report(json.loads(path.read_text(encoding="utf-8")), args.limit))
        return 0

    try:
        only = [s.strip() for s in args.only.split(",")] if args.only else None
        steps = select_steps(args.start, args.stop, only)
        if args.run is None and args.stop is None and only is None:
            steps = [s for s in steps if s not in TRAIN_STEPS]  # no run directory: data preparation only
            if not steps:
                raise ValueError("training steps need a run directory: --run models/<name>")
        if "ingest" in steps and args.input is None:
            if args.start is None and only is None:
                steps.remove("ingest")  # nothing new to register: continue from the work dir
                if not (args.work / "recordings").exists():
                    raise ValueError("give an input folder: nepvoice run INPUT_DIR")
            else:
                raise ValueError("the ingest step needs an input folder: nepvoice run INPUT_DIR")
        run(args.work, cfg, steps, args.input, args.run)
    except (ValueError, FileNotFoundError, TrainingStateError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
