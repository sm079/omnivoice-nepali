"""Reference voices for cloning, and reproducible generation."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf

MAX_REF_SECONDS = 15.0
MIN_REF_SECONDS = 10.0


def seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_reference(path: str, out_dir: Path) -> tuple[str, str | None]:
    """Mono reference of at most ``MAX_REF_SECONDS``; returns ``(path, note)``.

    OmniVoice clones from a few seconds of speech; a long upload only makes the prompt
    (and the transcription) longer. Longer clips are cut at the quietest point between
    ``MIN_REF_SECONDS`` and ``MAX_REF_SECONDS`` so the cut falls in a pause, not a word.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    duration = len(audio) / sr
    if duration <= MAX_REF_SECONDS:
        return path, None
    hop = int(0.02 * sr)
    lo, hi = int(MIN_REF_SECONDS * sr), int(MAX_REF_SECONDS * sr)
    frames = audio[lo:hi][: (hi - lo) // hop * hop].reshape(-1, hop)
    cut = lo + int(np.argmin((frames ** 2).mean(axis=1))) * hop + hop // 2
    refs = out_dir / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    trimmed = refs / f"{time.strftime('%Y%m%d-%H%M%S')}_ref.wav"
    sf.write(str(trimmed), audio[:cut], sr)
    return str(trimmed), f"Reference was {duration:.1f} s; using the first {cut / sr:.1f} s (cut at a pause)."
