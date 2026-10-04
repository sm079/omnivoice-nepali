"""Loudness normalisation, fades and breath trimming."""

import numpy as np
import pyloudnorm

from nepvoice.audio import SAMPLE_RATE, fade_and_pad, normalize_loudness, normalize_peak, true_peak_dbfs
from nepvoice.prep.edges import breath_at_edge
from nepvoice.prep.selection import SelectConfig

SR = 24000
RNG = np.random.default_rng(0)


def speechlike(seconds, amp, sr=SR):
    """Amplitude-modulated harmonics: speech-like level dynamics."""
    t = np.arange(int(seconds * sr)) / sr
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 3 * t)
    return (amp * env * sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 6))).astype(np.float32)


def lufs(x, sr=SR):
    return pyloudnorm.Meter(sr).integrated_loudness(x.astype(np.float64))


def test_loud_clip_is_brought_down_to_target():
    x = speechlike(4.0, 0.9)
    y, r = normalize_loudness(x, SR, -23.0, -1.0)
    assert r.input_lufs > -15 and not r.peak_limited
    assert abs(lufs(y) - (-23.0)) < 0.1
    assert abs(r.output_lufs - (-23.0)) < 1e-6


def test_gain_stops_at_the_peak_ceiling():
    x = speechlike(4.0, 0.01)
    x[SR] = 0.5  # one sharp peak leaves little headroom
    y, r = normalize_loudness(x, SR, -16.0, -1.0)
    assert r.peak_limited and r.output_lufs < -16.0
    assert true_peak_dbfs(y) <= -1.0 + 1e-6


def test_silence_and_tiny_clips_are_left_alone():
    for x in (np.zeros(SR, np.float32), speechlike(0.2, 0.5)):
        y, r = normalize_loudness(x, SR)
        assert np.array_equal(y, x) and r.gain_db == 0.0
        assert r.to_dict()["input_lufs"] is None


def test_peak_normalisation_matches_omnivoice():
    x = speechlike(2.0, 0.2)
    y, gain_db = normalize_peak(x, 0.9)
    assert abs(np.abs(y).max() - 0.9) < 1e-6 and abs(gain_db - 20 * np.log10(0.9 / np.abs(x).max())) < 1e-4
    silent = np.zeros(SR, np.float32)
    assert np.array_equal(normalize_peak(silent)[0], silent) and normalize_peak(silent)[1] == 0.0


def test_fade_and_pad():
    x = np.ones(1000, np.float32)
    y = fade_and_pad(x, 1000, fade_ms=10, pad_ms=5)
    assert len(y) == 1010 and y[0] == 0 and y[5] == 0 and abs(y[505] - 1) < 1e-6


# ---- breath trimming (16 kHz analysis) ----

def voiced(seconds, f0=140.0, amp=0.3):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))).astype(np.float32)


def noise(seconds, amp):
    return (amp * RNG.standard_normal(int(seconds * SAMPLE_RATE))).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), np.float32) + 1e-5


LEVEL = 10 * np.log10(np.mean(voiced(1.0) ** 2))


def test_breath_after_a_dip_is_trimmed():
    clip = np.concatenate([voiced(1.5), silence(0.04), noise(0.3, 0.02)])
    cut = breath_at_edge(clip[::-1], LEVEL, SelectConfig())
    assert cut is not None
    assert 0.3 <= cut / SAMPLE_RATE <= 0.36  # the breath goes, the speech stays


def test_consonant_release_without_a_dip_is_kept():
    clip = np.concatenate([voiced(1.5), noise(0.15, 0.05)])
    assert breath_at_edge(clip[::-1], LEVEL, SelectConfig()) is None
