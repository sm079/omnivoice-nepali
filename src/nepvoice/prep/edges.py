"""Trim breaths from clip edges.

An inhale at the end of a clip is really the start of the next phrase, cut off; at
the start it becomes part of the voice prompt OmniVoice trains on. Both teach a TTS
model to breathe at utterance boundaries.

Signature of a breath at a clip edge, scanning inward from the edge:

    [edge] ... breath noise ... \\_ energy dip _/ [unvoiced tail attached to speech] voiced speech

* breath noise: unvoiced and noise-like (spectral flatness at least ``FLATNESS_MIN``),
  between ``NOISE_LO_DB`` and ``NOISE_HI_DB`` relative to the speech level, lasting at
  least ``breath_min`` seconds;
* energy dip between the noise and the speech: below ``breath_silence_db`` and at least
  ``breath_dip_db`` under the loudest part of the noise. This separates a breath from a word-final
  consonant release (च, छ, स), which is unvoiced and noisy too but flows straight out
  of the voiced sound without a dip.

The edge is moved to the bottom of the dip (the speech and its decay stay whole),
plus up to ``breath_keep`` seconds of the dip's silence, never reaching into the breath.
Only non-speech is removed, so the caption text stays as it is.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..audio import SAMPLE_RATE

HOP = SAMPLE_RATE // 100  # 10 ms analysis frames
SCAN = 1.2  # seconds examined at each edge
NOISE_LO_DB = -48.0
NOISE_HI_DB = -12.0
FLATNESS_MIN = 0.05  # voiced speech sits around 0.001-0.01, breath around 0.1-0.2


def _voicing_and_flatness(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    _, voiced, _ = librosa.pyin(
        x, fmin=70, fmax=450, sr=SAMPLE_RATE, frame_length=640, hop_length=HOP, center=True
    )
    flatness = librosa.feature.spectral_flatness(y=x, n_fft=640, hop_length=HOP, center=True)[0]
    return voiced, flatness[: len(voiced)]


def breath_at_edge(x: np.ndarray, level_db: float, cfg) -> int | None:
    """Samples to cut from the *start* of ``x`` (an edge-first view of the clip), or ``None``.

    ``x`` begins at the clip edge and runs inward; callers reverse the tail.
    """
    x = x[: int(SCAN * SAMPLE_RATE)]
    if len(x) < int(0.4 * SAMPLE_RATE):
        return None
    voiced, flatness = _voicing_and_flatness(x)
    n = len(voiced)
    frames = np.pad(x, (0, max(0, n * HOP - len(x))))[: n * HOP].reshape(n, HOP)
    rel = 10 * np.log10((frames**2).mean(axis=1) + 1e-10) - level_db

    first_voiced = int(np.argmax(voiced)) if voiced.any() else n
    if first_voiced == n:
        return None  # no speech found near this edge: leave it alone
    region = slice(0, first_voiced)
    noisy = (rel[region] > NOISE_LO_DB) & (rel[region] < NOISE_HI_DB) & (flatness[region] >= FLATNESS_MIN)
    min_breath = int(round(cfg.breath_min * 100))
    keep = int(round(cfg.breath_keep * 100))

    # Innermost qualifying breath wins: everything from the edge up to its dip goes.
    cut = None
    i = 0
    while i < first_voiced:
        if not noisy[i]:
            i += 1
            continue
        j = i
        while j < first_voiced and noisy[j]:
            j += 1
        if j - i >= min_breath:
            # Energy dip between this noise run and the voiced speech.
            between = rel[j:first_voiced + 1]
            m = j + int(np.argmin(between))
            dip_ok = rel[m] < cfg.breath_silence_db and rel[m] <= rel[i:j].max() - cfg.breath_dip_db
            if dip_ok:
                # Everything from the dip inward (the speech and its decay) stays. Keep up
                # to ``keep`` more frames toward the edge, but only while still silent.
                c = m
                while c > j and m - c < keep and rel[c - 1] < cfg.breath_silence_db:
                    c -= 1
                cut = c
        i = j
    return None if cut is None else cut * HOP


def trim_breaths(segments: list, audio16k: np.ndarray, level_db: float, cfg, verbose: bool = True) -> list[tuple]:
    """``(segment, trimmed_head_s, trimmed_tail_s)`` for each segment, breaths cut from both edges."""
    out = []
    for i, seg in enumerate(segments, 1):
        a, b = int(seg.start * SAMPLE_RATE), int(seg.end * SAMPLE_RATE)
        clip = audio16k[a:b]
        tail = breath_at_edge(clip[::-1], level_db, cfg) or 0
        head = breath_at_edge(clip, level_db, cfg) or 0
        if head or tail:
            seg = replace(seg, start=seg.start + head / SAMPLE_RATE, end=seg.end - tail / SAMPLE_RATE)
        out.append((seg, head / SAMPLE_RATE, tail / SAMPLE_RATE))
        if verbose and (i % 50 == 0 or i == len(segments)):
            print(f"breath trim {i}/{len(segments)}", flush=True)
    return out
