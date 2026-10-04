"""Shared fixtures: a synthetic recording and stand-ins for the diarization and overlap models."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SR = 48000
SECONDS = 40


def speech(seconds: float, amp: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 3 * t)
    return (amp * env * sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 6))).astype(np.float32)


def _make_recording(root: Path, name: str, amp: float) -> None:
    """Speaker 0 talks in 6 s phrases separated by 0.5 s of silence; transcript words every 0.5 s."""
    audio = np.zeros(SECONDS * SR, np.float32)
    events, t = [], 1.0
    while t + 6 < SECONDS - 1:
        audio[int(t * SR): int((t + 6) * SR)] = speech(6, amp)
        events.append({"tStartMs": int(t * 1000), "dDurationMs": 6000,
                       "segs": [{"utf8": f"शब्द{i}", "tOffsetMs": i * 500} for i in range(12)]})
        t += 6.5
    root.mkdir(parents=True, exist_ok=True)
    sf.write(root / f"{name}.flac", audio, SR)
    (root / f"{name}.json3").write_text(json.dumps({"events": events}, ensure_ascii=False), encoding="utf-8")


def _fake_models(ws) -> None:
    """Diarization: speaker 0 active wherever there is audio; no overlap anywhere."""
    audio, sr = sf.read(ws.audio_path, dtype="float32")
    frames = len(audio) * 100 // sr
    active = np.abs(audio[: frames * sr // 100].reshape(frames, -1)).max(axis=1) > 0
    diar = np.zeros((frames, 8), np.float32)
    diar[:, 0] = active
    np.save(ws.diar, diar)
    t20 = frames // 2
    counts = np.zeros((t20, 3), np.float32)
    counts[:, 1] = 1.0
    np.savez(ws.osd, counts=counts, local=np.zeros((1, t20, 4), np.float16), window_starts=np.array([0]))



@pytest.fixture()
def make_recording():
    """``make_recording(folder, name, amp)``: write ``<name>.flac`` and a matching json3 transcript."""
    return _make_recording


@pytest.fixture()
def fake_models():
    """``fake_models(ws)``: write diarization/overlap outputs that describe the synthetic audio."""
    return _fake_models
