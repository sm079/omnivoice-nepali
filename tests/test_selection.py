"""Behaviour of the overlap-free selection on small synthetic signals (10 ms frames)."""

import numpy as np
import pytest

from nepvoice.prep.selection import (
    SelectConfig,
    padded_foreign,
    pause_masks,
    select_segments,
    select_whole,
)

T = 3000  # 30 s
SPEECH_DB, QUIET_DB = -20.0, -70.0


def make(own_spans, other_spans=(), quiet_spans=()):
    """Speaker 0 talks in ``own_spans``; Nemotron 'other' (speaker 1) is high in ``other_spans``."""
    diar = np.zeros((T, 2), np.float32)
    for a, b in own_spans:
        diar[a:b, 0] = 1.0
    signals = {name: np.zeros((T, 2), np.float32) for name in ("nemotron_other", "diarizen_multi", "diarizen_other")}
    for a, b, v in other_spans:
        signals["nemotron_other"][a:b, 0] = v
    energy = np.full(T, QUIET_DB, np.float32)
    for a, b in own_spans:
        energy[a:b] = SPEECH_DB
    for a, b in quiet_spans:
        energy[a:b] = QUIET_DB
    return diar, signals, energy


def select(diar, signals, energy, **cfg):
    cfg = SelectConfig(min_speaker_seconds=0.0, **cfg)
    return [s for s in select_segments(diar, {}, None, cfg, signals, energy) if s.speaker == 0]


def covers(segs, a, b):
    return any(s.start < b / 100 and a / 100 < s.end for s in segs)


def test_clean_monologue_is_kept():
    diar, sig, en = make([(100, 600), (630, 1200)])
    segs = select(diar, sig, en)
    assert segs and sum(s.duration for s in segs) > 10


def test_faint_second_voice_is_never_kept():
    # A barely-detected interjection (0.04, just above the 0.03 threshold) mid-monologue.
    diar, sig, en = make([(100, 1500)], other_spans=[(700, 720, 0.04)])
    segs = select(diar, sig, en)
    assert not covers(segs, 700 - 30, 720 + 30)  # the event and its 0.3 s pad stay out


def test_below_threshold_signal_is_ignored():
    diar, sig, en = make([(100, 600), (630, 1200)], other_spans=[(300, 320, 0.02)])
    assert covers(select(diar, sig, en), 300, 320)


def test_smear_tail_across_a_pause_is_not_padded_into_speech():
    # Strong event, then a pause, then a weak tail that is only the event fading out.
    diar, sig, en = make([(100, 500), (535, 1200)], other_spans=[(450, 500, 0.9), (535, 545, 0.05)],
                         quiet_spans=[(500, 535)])
    segs = select(diar, sig, en)
    assert covers(segs, 560, 1100)
    assert not covers(segs, 450, 500)


def test_isolated_detection_in_a_pause_is_padded_in_every_direction():
    # Regression: a faint voice detected only inside a pause (it is what ends the pause)
    # must still get its pad, or the rest of that voice leaks into the next clip.
    diar, sig, en = make([(100, 500), (520, 1200)], other_spans=[(515, 520, 0.04)], quiet_spans=[(500, 520)])
    segs = select(diar, sig, en)
    assert not covers(segs, 520, 550)


def test_pause_masks_need_energy_in_acoustic_mode():
    diar, _, _ = make([(100, 500)])
    with pytest.raises(ValueError):
        pause_masks(diar, None, SelectConfig())
    assert pause_masks(diar, None, SelectConfig(cut_mode="speaker_gap")) == (None, None)


def test_padded_foreign_without_barrier_is_plain_dilation():
    _, sig, _ = make([(100, 500)], other_spans=[(300, 301, 0.5)])
    mask = padded_foreign(sig, SelectConfig(), 0, None)
    assert mask[270] and mask[330] and not mask[260] and not mask[340]


# ---- whole-file selection (plain transcripts) ----

def whole(own_spans, other_spans=(), **cfg):
    diar, sig, _ = make(own_spans, other_spans)
    return select_whole(diar, {}, SelectConfig(**cfg), sig)


def test_clean_file_is_one_clip_trimmed_to_its_speech():
    seg, reason = whole([(100, 600), (630, 900)])
    assert reason == "kept" and seg.speaker == 0
    assert (seg.start, seg.end) == (0.9, 9.1)  # speech 1.0-9.0 s plus the 0.1 s edge margin


def test_any_second_voice_rejects_the_file():
    seg, reason = whole([(100, 900)], other_spans=[(500, 505, 0.04)])
    assert seg is None and reason == "second voice detected"


def test_whole_file_length_limits():
    assert whole([(100, 150)])[1].startswith("too short")
    assert whole([(100, 2900)])[1].startswith("too long")
    assert whole([])[1] == "no speech detected"
