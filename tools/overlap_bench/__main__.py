"""``python -m tools.overlap_bench RECORDING_ID``: measure how many hidden second voices leak into clips.

Builds a synthetic session from a processed recording's own clean clips, mixes short
snippets of a second speaker under the main one at a range of levels, runs both
detection models on it and reports, per selection configuration, how many snippets end
up inside a selected clip and how much clean speech survives.
"""

import argparse
import sys
from pathlib import Path

from nepvoice.workspace import find_workspaces

from . import bench


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m tools.overlap_bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", help="id of a processed recording with at least two speakers")
    p.add_argument("--work", type=Path, default=Path("work"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--minutes", type=float, default=15.0)
    args = p.parse_args()
    source = find_workspaces(args.work).get(args.recording)
    if source is None or not source.segments.exists():
        sys.exit(f"{args.recording} has not been processed under {args.work}")
    out = args.work / "overlap_bench" / f"{args.recording}_seed{args.seed}"
    report = bench.run(source, out, args.seed, args.minutes)
    for name, r in report["configs"].items():
        print(f"{name:32s} leaks={r['leaks']:3d}/{report['num_insertions']}  kept={r['kept_untouched_frac']:.3f}")
    print(f"full report: {out / 'report.json'}")


if __name__ == "__main__":
    main()
