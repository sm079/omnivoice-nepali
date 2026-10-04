"""OmniVoice's audio-token extraction, with its per-clip peak normalisation made optional.

``omnivoice.scripts.extract_audio_tokens`` scales every clip so its peak sits at 0.9 and
has no option to turn that off. That throws away the clip's level, so loudness-normalised
clips come out as loud as their peaks allow, and the model learns to speak that loud.

Usage: ``python -m nepvoice.train.extract_tokens [--no-peak-normalize] <extract_audio_tokens args>``
"""

from __future__ import annotations

import sys
from functools import partial


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    peak_normalize = "--no-peak-normalize" not in args
    args = [a for a in args if a not in ("--no-peak-normalize", "--peak-normalize")]

    from omnivoice.data.dataset import JsonlDatasetReader
    from omnivoice.scripts import extract_audio_tokens as script

    # The reader is built in this process; DataLoader workers receive it pickled, with the
    # flag already set on the instance.
    script.JsonlDatasetReader = partial(JsonlDatasetReader, normalize_audio=peak_normalize)
    sys.argv = [script.__file__, *args]
    script.main()


if __name__ == "__main__":
    main()
