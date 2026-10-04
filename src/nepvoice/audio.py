"""Audio decoding, analysis and level normalisation."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000  # model input rate (diarization, overlap detection, embeddings)
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".webm", ".mka", ".mp4")


def to_wav(src: Path, dst: Path, sample_rate: int | None = SAMPLE_RATE, mono: bool = True) -> Path:
    """Decode any ffmpeg-readable file to PCM WAV. ``sample_rate=None`` keeps the source rate.

    Writes to a temporary name first, so an interrupted decode never leaves a truncated
    file that later stages would take for a finished one.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".partial.wav")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
    if mono:
        cmd += ["-ac", "1"]
    if sample_rate is not None:
        cmd += ["-ar", str(sample_rate)]
    cmd += ["-c:a", "pcm_s16le", str(tmp)]
    subprocess.run(cmd, check=True)
    tmp.replace(dst)
    return dst


def frame_energy_db(audio: np.ndarray, hop: int = SAMPLE_RATE // 100) -> np.ndarray:
    """Mean-square energy in dB for consecutive ``hop``-sample frames (10 ms at 16 kHz)."""
    n = -(-len(audio) // hop)
    padded = np.zeros(n * hop, dtype=np.float32)
    padded[: len(audio)] = audio
    return 10 * np.log10((padded.reshape(n, hop) ** 2).mean(axis=1) + 1e-10).astype(np.float32)


def load_mono(path: Path, expected_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a mono float32 waveform and check its sample rate."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != expected_sr:
        raise ValueError(f"{path} is {sr} Hz, expected {expected_sr} Hz")
    return audio.mean(axis=1)


def fade_and_pad(audio: np.ndarray, sr: int, fade_ms: float, pad_ms: float) -> np.ndarray:
    """Raised-cosine fade in/out, then digital silence on both ends."""
    n = min(int(sr * fade_ms / 1000), len(audio) // 2)
    if n > 0:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))).astype(audio.dtype)
        audio = audio.copy()
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    p = int(sr * pad_ms / 1000)
    if p > 0:
        audio = np.concatenate([np.zeros(p, audio.dtype), audio, np.zeros(p, audio.dtype)])
    return audio


# ---- levels ----

def normalize_peak(audio: np.ndarray, level: float = 0.9) -> tuple[np.ndarray, float]:
    """Scale ``audio`` so its sample peak is ``level``, as OmniVoice's training data loader does (0.9).

    Returns the scaled audio and the gain in dB. Silence is returned as is.
    """
    x = np.asarray(audio, dtype=np.float32)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak == 0:
        return x, 0.0
    return (x * (level / peak)).astype(np.float32), float(20 * np.log10(level / peak))


def true_peak_dbfs(audio: np.ndarray, oversample: int = 4) -> float:
    """Peak level of the 4x oversampled signal (approximates BS.1770 true peak)."""
    from scipy.signal import resample_poly

    if len(audio) == 0:
        return float("-inf")
    peak = float(np.abs(resample_poly(audio.astype(np.float64), oversample, 1)).max())
    return float(20 * np.log10(peak)) if peak > 0 else float("-inf")


@dataclass
class LoudnessResult:
    input_lufs: float  # integrated loudness before (-inf for silence / too short to measure)
    gain_db: float  # gain that was applied
    output_lufs: float
    peak_limited: bool  # the gain was lowered to respect the peak ceiling

    def to_dict(self) -> dict:
        r = lambda x: None if not np.isfinite(x) else round(float(x), 2)  # noqa: E731
        return {"input_lufs": r(self.input_lufs), "gain_db": r(self.gain_db),
                "output_lufs": r(self.output_lufs), "peak_limited": self.peak_limited}


def normalize_loudness(
    audio: np.ndarray, sr: int, target_lufs: float = -27.0, peak_dbfs: float = -1.0
) -> tuple[np.ndarray, LoudnessResult]:
    """Scale ``audio`` to ``target_lufs`` integrated loudness, never letting the true peak exceed ``peak_dbfs``.

    Only a single gain is applied (no compression or limiting), so the clip's dynamics are
    untouched. If reaching the target would push peaks over the ceiling, the gain stops
    at the ceiling and the clip ends up somewhat quieter than the target. Audio that
    cannot be measured (silent, or shorter than one 400 ms gating block) is returned as is.
    """
    import pyloudnorm

    x = np.asarray(audio, dtype=np.float32)
    if len(x) < int(0.4 * sr) or not np.any(x):
        return x, LoudnessResult(float("-inf"), 0.0, float("-inf"), False)
    measured = float(pyloudnorm.Meter(sr).integrated_loudness(x.astype(np.float64)))
    if not np.isfinite(measured):
        return x, LoudnessResult(measured, 0.0, measured, False)
    gain = target_lufs - measured
    headroom = peak_dbfs - true_peak_dbfs(x)
    limited = bool(gain > headroom)
    if limited:
        gain = headroom
    y = (x * 10 ** (gain / 20)).astype(np.float32)
    return y, LoudnessResult(measured, float(gain), measured + float(gain), limited)
