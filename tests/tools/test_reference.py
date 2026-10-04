"""Trimming long reference voices for the comparison UI."""

from pathlib import Path

import numpy as np
import soundfile as sf

from tools.voice_studio.reference import (
    MAX_REF_SECONDS,
    MIN_REF_SECONDS,
    prepare_reference,
)

SR = 24000


def test_short_reference_is_used_as_is(tmp_path: Path):
    f = tmp_path / "short.wav"
    sf.write(f, np.random.default_rng(0).normal(0, 0.1, 8 * SR).astype(np.float32), SR)
    assert prepare_reference(str(f), tmp_path) == (str(f), None)


def test_long_reference_is_cut_at_a_pause(tmp_path: Path):
    rng = np.random.default_rng(0)
    speech = rng.normal(0, 0.1, 40 * SR).astype(np.float32)
    speech[int(12.5 * SR): int(12.8 * SR)] = 0.0  # the only pause in the cut window
    stereo = np.stack([speech, speech], axis=1)
    f = tmp_path / "long.wav"
    sf.write(f, stereo, SR)
    path, note = prepare_reference(str(f), tmp_path)
    audio, sr = sf.read(path)
    assert audio.ndim == 1 and sr == SR
    assert MIN_REF_SECONDS <= len(audio) / sr <= MAX_REF_SECONDS
    assert 12.5 <= len(audio) / sr <= 12.8
    assert "40.0 s" in note
