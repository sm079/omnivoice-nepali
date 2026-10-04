"""Per-recording processing on cached model outputs: run the models, select segments, export clips."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import soundfile as sf

from ..audio import fade_and_pad, frame_energy_db, load_mono, normalize_loudness, normalize_peak
from ..config import ExportConfig
from ..transcripts import load as load_transcript
from ..workspace import Workspace
from .selection import SelectConfig

EXPORT_MARKER = ".export.json"


# ---- models ----

def run_diarization(ws: Workspace) -> None:
    from . import diarize

    np.save(ws.diar, diarize.diarize(ws.ensure_wav16k()))


def run_overlap_detection(ws: Workspace) -> None:
    from . import osd

    np.savez_compressed(ws.osd, **osd.detect_overlap(ws.ensure_wav16k()))


def load_outputs(ws: Workspace) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    diar = np.load(ws.diar)
    with np.load(ws.osd) as z:
        osd = {k: z[k] for k in z.files}
    return diar, osd


def load_energy(ws: Workspace) -> np.ndarray:
    """10 ms frame energy (dB) of the 16 kHz audio, cached next to the model outputs."""
    if not ws.energy.exists():
        np.save(ws.energy, frame_energy_db(load_mono(ws.ensure_wav16k())))
    return np.load(ws.energy)


# ---- selection ----

def selection_done(ws: Workspace, cfg: SelectConfig) -> bool:
    if not (ws.segments.exists() and ws.summary.exists()):
        return False
    summary = json.loads(ws.summary.read_text(encoding="utf-8"))
    return summary.get("config") == json.loads(json.dumps(asdict(cfg)))


def select(ws: Workspace, cfg: SelectConfig, verbose: bool = False) -> list:
    """Write ``segments.jsonl`` and ``summary.json`` for ``ws``; return the segments."""
    from .selection import select_segments, select_whole

    diar, osd = load_outputs(ws)
    transcript = load_transcript(ws.transcript_path)
    energy = load_energy(ws)
    extra: dict = {}
    if transcript.timed:
        segments = select_segments(diar, osd, transcript.words, cfg, energy_db=energy)
    else:
        seg, reason = select_whole(diar, osd, cfg)
        if seg is not None:
            seg.text, seg.num_words = transcript.text, len(transcript.text.split())
        segments = [seg] if seg is not None and seg.num_words >= cfg.min_words else []
        extra["whole_file"] = reason if seg is None or segments else "no transcript words"

    # Ids are assigned before any trimming, so a segment keeps its number whether or not
    # breaths are trimmed (or it is dropped for becoming too short).
    ids = [f"{ws.id}_{i:05d}" for i in range(len(segments))]
    trims: list[tuple[float, float]] | None = None
    if cfg.trim_breaths and segments:
        from .edges import trim_breaths
        from .selection import speech_level_db

        level = speech_level_db(np.resize(energy, len(diar)), diar, cfg.speech_threshold)
        result = trim_breaths(segments, load_mono(ws.ensure_wav16k()), level, cfg, verbose)
        keep = [i for i, t in enumerate(result) if t[0].duration >= cfg.min_duration]
        extra["breaths_trimmed"] = {
            "head": [ids[i] for i in keep if result[i][1] > 0],
            "tail": [ids[i] for i in keep if result[i][2] > 0],
            "dropped_too_short": [ids[i] for i in range(len(result)) if i not in set(keep)],
        }
        ids = [ids[i] for i in keep]
        segments = [result[i][0] for i in keep]
        trims = [(result[i][1], result[i][2]) for i in keep]

    with ws.segments.open("w", encoding="utf-8") as f:
        for i, s in enumerate(segments):
            row = {"id": ids[i], "recording_id": ws.id, **s.to_dict(), "speaker": f"{ws.id}_spk{s.speaker}"}
            if trims is not None:
                row["trimmed_head"], row["trimmed_tail"] = round(trims[i][0], 3), round(trims[i][1], 3)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(diar) * 0.01
    kept = sum(s.duration for s in segments)
    per_speaker: dict[str, float] = {}
    for s in segments:
        key = f"{ws.id}_spk{s.speaker}"
        per_speaker[key] = per_speaker.get(key, 0.0) + s.duration
    summary = {
        "recording_id": ws.id,
        "audio_seconds": round(total, 1),
        "kept_seconds": round(kept, 1),
        "kept_fraction": round(kept / total, 4) if total else 0,
        "num_segments": len(segments),
        "per_speaker_seconds": {k: round(v, 1) for k, v in sorted(per_speaker.items())},
        **extra,
        "config": asdict(cfg),
    }
    ws.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return segments


# ---- export ----

def export_done(ws: Workspace, cfg: ExportConfig) -> bool:
    """True if the current ``segments.jsonl`` was already exported with ``cfg``."""
    marker = ws.clips / EXPORT_MARKER
    if not (ws.segments.exists() and marker.exists()):
        return False
    exported = json.loads(marker.read_text(encoding="utf-8"))
    return exported.get("params") == asdict(cfg) and exported.get("segments_mtime") == ws.segments.stat().st_mtime_ns


def export(ws: Workspace, cfg: ExportConfig, keep_source_wav: bool = False) -> int:
    """Cut every selected segment out of the full-quality source audio as FLAC.

    Per clip: cut, resample to ``cfg.sample_rate``, normalise loudness or peak, then fade and pad.
    The full-rate WAV (several hundred MB per hour of audio) is deleted afterwards unless
    ``keep_source_wav``; it is decoded again from the source whenever needed.
    """
    rows = ws.read_segments()
    ws.clips.mkdir(exist_ok=True)
    (ws.clips / EXPORT_MARKER).unlink(missing_ok=True)
    for old in ws.clips.glob("*.flac"):
        old.unlink()
    loudness: dict[str, dict] = {}
    if rows:
        source_wav = ws.ensure_source_wav()
        sr = sf.info(str(source_wav)).samplerate
        out_sr = cfg.sample_rate or sr
        with sf.SoundFile(str(source_wav)) as src:
            for row in rows:
                src.seek(int(row["start"] * sr))
                audio = src.read(int((row["end"] - row["start"]) * sr), dtype="float32")
                if out_sr != sr:
                    import librosa

                    audio = librosa.resample(audio, orig_sr=sr, target_sr=out_sr)
                if cfg.loudness_normalize:
                    audio, result = normalize_loudness(audio, out_sr, cfg.loudness_lufs, cfg.peak_dbfs)
                    loudness[row["id"]] = result.to_dict()
                elif cfg.peak_normalize:
                    audio, gain_db = normalize_peak(audio, cfg.peak_normalize)
                    loudness[row["id"]] = {"gain_db": round(gain_db, 2)}
                if cfg.fade_ms or cfg.pad_ms:
                    audio = fade_and_pad(audio, out_sr, cfg.fade_ms, cfg.pad_ms)
                sf.write(str(ws.clips / f"{row['id']}.flac"), audio, out_sr)
    marker = {
        "params": asdict(cfg),
        "segments_mtime": ws.segments.stat().st_mtime_ns,
        "num_clips": len(rows),
        "loudness": loudness,
    }
    (ws.clips / EXPORT_MARKER).write_text(json.dumps(marker), encoding="utf-8")
    if not keep_source_wav:
        ws.source_wav.unlink(missing_ok=True)
    return len(rows)
