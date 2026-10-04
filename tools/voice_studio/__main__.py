"""``python -m tools.voice_studio``: compare the base and fine-tuned models by ear (blind A/B)."""

import argparse
from pathlib import Path

from tools.tts_bench.bench import parse_models, resolve_models

from .server import serve


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m tools.voice_studio", description=__doc__)
    p.add_argument("--work", type=Path, required=True,
                   help="preprocessed dataset (its dev voices are offered as references)")
    p.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                   help="model to compare (repeatable): Hub id, merged model folder or packaged run "
                        "(merged into <out>/models/NAME); default: the base model")
    p.add_argument("--bench", type=Path, default=None, help="benchmark folder to show, e.g. bench/bench1")
    p.add_argument("--out", type=Path, required=True, help="folder for rounds, references and merged models")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    models = resolve_models(parse_models(args.model), args.out)
    serve(models, args.work, args.out, args.bench, args.host, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
