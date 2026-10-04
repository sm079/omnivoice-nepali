"""Build the timeline payload the viewer renders for one processed recording."""

from __future__ import annotations

import base64
import json
from dataclasses import fields
from pathlib import Path
from urllib.parse import quote

import numpy as np
import soundfile as sf

from nepvoice.prep.process import load_energy, load_outputs
from nepvoice.prep.selection import (
    HOP,
    SelectConfig,
    _runs,
    foreign_signals,
    padded_foreign,
    pause_masks,
)
from nepvoice.workspace import Workspace

CURVE_HOP = 0.02  # seconds per curve sample sent to the browser
PEAKS_PER_SECOND = 100
BROWSER_AUDIO = {".wav", ".flac", ".mp3", ".m4a", ".webm", ".ogg", ".opus"}


def _b64_u8(x: np.ndarray) -> str:
    """Quantize values in [0, 1] to bytes and base64 them."""
    return base64.b64encode(np.clip(np.round(x * 255), 0, 255).astype(np.uint8).tobytes()).decode()


def _max_pool(x: np.ndarray, factor: int) -> np.ndarray:
    n = len(x) // factor * factor
    head = x[:n].reshape(-1, factor).max(axis=1)
    return np.concatenate([head, [x[n:].max()]]) if n < len(x) else head


def _intervals(mask: np.ndarray, hop: float = HOP) -> list[list[float]]:
    return [[round(a * hop, 2), round(b * hop, 2)] for a, b in _runs(mask)]


def _peaks(wav: Path) -> tuple[str, float]:
    """Max-abs waveform envelope at PEAKS_PER_SECOND, normalized to the clip peak."""
    audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    audio = np.abs(audio.mean(axis=1))
    env = _max_pool(audio, sr // PEAKS_PER_SECOND)
    env /= max(float(np.percentile(env, 99.9)), 1e-6)
    return _b64_u8(np.sqrt(np.clip(env, 0, 1))), len(audio) / sr  # sqrt: easier to see quiet speech


def _load_config(ws: Workspace) -> SelectConfig:
    cfg = SelectConfig()
    if ws.summary.exists():
        saved = json.loads(ws.summary.read_text(encoding="utf-8")).get("config", {})
        names = {f.name for f in fields(SelectConfig)}
        cfg = SelectConfig(**{k: v for k, v in saved.items() if k in names})
    return cfg


def playable_audio(ws: Workspace) -> Path:
    """The original recording if browsers can play it, else the 16 kHz model input."""
    source = ws.audio_path
    return source if source.suffix.lower() in BROWSER_AUDIO and source.exists() else ws.ensure_wav16k()


def build_payload(ws: Workspace, work: Path, key: str | None = None, media: str = "/media/") -> dict:
    """``key`` addresses the recording in API URLs (default: its id); clip URLs start with ``media``."""
    diar, osd = load_outputs(ws)
    cfg = _load_config(ws)
    signals = foreign_signals(diar, osd)
    K = diar.shape[1]
    pool = int(round(CURVE_HOP / HOP))
    _, barrier = pause_masks(diar, load_energy(ws), cfg)

    rows = ws.read_segments()
    speakers_file = work / "speakers.json"
    global_ids = (
        json.loads(speakers_file.read_text(encoding="utf-8"))["local_to_global"] if speakers_file.exists() else {}
    )
    speakers = []
    for k in range(K):
        active = diar[:, k] > cfg.speech_threshold
        seconds = float(active.sum() * HOP)
        if seconds == 0:
            continue
        rejected = padded_foreign(signals, cfg, k, barrier)
        name = f"{ws.id}_spk{k}"
        speakers.append({
            "index": k,
            "name": name,
            "label": f"Speaker {k}" + (f" · {global_ids[name]}" if name in global_ids else ""),
            "speech_seconds": round(seconds, 1),
            "target": seconds >= max(cfg.min_speaker_seconds, cfg.min_duration),
            "activity": _intervals(active),
            "rejected": _intervals(rejected & active),
            "prob": _b64_u8(_max_pool(diar[:, k], pool)),
            "other_prob": _b64_u8(_max_pool(signals["diarizen_other"][:, k], pool)),
        })

    num_active = (diar > cfg.speech_threshold).sum(axis=1)
    p_multi = signals["diarizen_multi"][:, 0]
    overlap = {
        "nemotron": _intervals(num_active >= 2),
        "diarizen": _intervals(p_multi > 0.5),
        "any": _intervals((num_active >= 2) | (p_multi > 0.5)),
        "p_multi": _b64_u8(_max_pool(p_multi, pool)),
    }

    peaks, duration = _peaks(ws.ensure_wav16k())
    clip_url = lambda r: media + quote(  # noqa: E731
        (ws.clips / f"{r['id']}.flac").resolve().relative_to(work.resolve()).as_posix())
    return {
        "recording_id": ws.id,
        "duration": duration,
        "audio_url": f"/api/audio/{quote(key or ws.id, safe='')}",
        "meta": ws.meta(),
        "curve_hop": CURVE_HOP,
        "peaks": peaks,
        "peaks_per_second": PEAKS_PER_SECOND,
        "speakers": speakers,
        "overlap": overlap,
        "segments": [
            {k: r[k] for k in ("id", "speaker", "start", "end", "duration", "text", "max_nemotron_other",
                               "max_diarizen_multi", "max_diarizen_other")}
            | {"clip_url": clip_url(r)}
            | {k: r[k] for k in ("trimmed_head", "trimmed_tail") if k in r}
            for r in rows
        ],
        "config": {f.name: getattr(cfg, f.name) for f in fields(SelectConfig)},
        "summary": json.loads(ws.summary.read_text(encoding="utf-8")) if ws.summary.exists() else {},
    }
