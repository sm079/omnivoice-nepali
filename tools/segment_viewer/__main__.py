"""``python -m tools.segment_viewer --work WORK [WORK ...]``: open the segment viewer.

Pick a recording (grouped by work directory) from the dropdown.
"""

import argparse
from pathlib import Path

from .server import serve


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m tools.segment_viewer", description=__doc__)
    p.add_argument("--work", type=Path, nargs="+", required=True, metavar="WORK",
                   help="pipeline work directories, e.g. dataset/preprocessed1")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    serve(args.work, args.host, args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
