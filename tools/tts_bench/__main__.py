"""``python -m tools.tts_bench``: benchmark TTS models on held-out Nepali recordings.

Every model says the same texts in the same reference voices, with one voice prompt and
seed per item; outputs are scored for intelligibility (CER/WER with IndicConformer),
voice similarity (SIM) and naturalness (UTMOS). Generated audio and scores are kept in
``--out``, so re-running only fills in what is missing.
"""

import argparse
import json
import sys
from pathlib import Path

from . import bench


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m tools.tts_bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", type=Path, default=None,
                   help="preprocessed dataset whose dev split becomes the test set (unless <out>/testset.json exists)")
    p.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                   help="model to benchmark (repeatable): Hub id, merged model folder or packaged run "
                        "(merged into <out>/models/NAME); default: the base model")
    p.add_argument("--out", type=Path, required=True, help="benchmark folder, e.g. bench/bench1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-generate", action="store_true", help="only score what was generated before")
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out = args.out
    try:
        models = bench.parse_models(args.model)
    except ValueError as e:
        p.error(str(e))
    out.mkdir(parents=True, exist_ok=True)
    ts_path = out / "testset.json"
    if ts_path.exists():
        testset = json.loads(ts_path.read_text(encoding="utf-8"))
    else:
        if args.work is None:
            p.error(f"{ts_path} does not exist yet: give --work to build the test set")
        testset = bench.build_testset(args.work)
        ts_path.write_text(json.dumps(testset, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"test set: {len(testset['refs'])} reference clips x {len(testset['texts'])} texts", flush=True)
    (out / "models.json").write_text(json.dumps(models, indent=1), encoding="utf-8")
    if not args.skip_generate:
        bench.generate(bench.resolve_models(models, out), testset, out, args.seed)
    rows = bench.score(list(models), testset, out)
    summary = bench.summarize(rows)
    bench.write_report(summary, out)
    for group in summary["groups"]:
        print(bench.format_table(summary, group))
    print(json.dumps(summary["paired_vs_base"], indent=1))


if __name__ == "__main__":
    main()
