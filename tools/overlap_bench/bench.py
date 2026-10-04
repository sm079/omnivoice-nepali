"""Synthetic backchannel benchmark: how many hidden second voices leak into selected clips?

Builds a session out of a processed recording's own clean segments: turns of the
primary speaker A alternate with turns of speaker B, and short snippets of B
(backchannel-sized, at a range of levels relative to A) are mixed *under* A's
turns. Both models then run on the synthetic session exactly as in production.

Reported per detector configuration:

* ``leaks``: inserted snippets that end up (even partly) inside a selected clip.
  This must be zero.
* ``recall`` per snippet length and level for each individual detector signal.
* ``kept``: fraction of A's untouched speech that survives selection (the cost).
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import soundfile as sf

from nepvoice.audio import SAMPLE_RATE, frame_energy_db, load_mono
from nepvoice.prep.selection import (
    SelectConfig,
    foreign_signals,
    select_segments,
)
from nepvoice.workspace import Workspace

SNIPPET_SECONDS = (0.15, 0.3, 0.5, 1.0, 2.0)
SNIPPET_GAINS_DB = (0.0, -6.0, -12.0, -18.0, -24.0)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def _speechy_crop(clip: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Pick the most energetic of a few random crops so the snippet actually contains speech."""
    best, best_rms = None, -1.0
    for _ in range(8):
        s = rng.integers(0, max(1, len(clip) - length))
        c = clip[s : s + length]
        if len(c) == length and _rms(c) > best_rms:
            best, best_rms = c, _rms(c)
    fade = min(len(best) // 4, int(0.01 * SAMPLE_RATE))
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    best = best.copy()
    best[:fade] *= ramp
    best[-fade:] *= ramp[::-1]
    return best


def build_session(segments_path: Path, wav16k: Path, out_dir: Path, seed: int = 0, minutes: float = 15.0) -> dict:
    rng = np.random.default_rng(seed)
    rows = [json.loads(line) for line in segments_path.read_text(encoding="utf-8").splitlines() if line]
    by_spk: dict[str, list[dict]] = {}
    for r in rows:
        by_spk.setdefault(r["speaker"], []).append(r)
    speakers = sorted(by_spk, key=lambda s: -sum(r["duration"] for r in by_spk[s]))
    a_name, b_name = speakers[0], speakers[1]

    audio = load_mono(wav16k)
    cut = lambda r: audio[int(r["start"] * SAMPLE_RATE) : int(r["end"] * SAMPLE_RATE)]  # noqa: E731
    a_clips = [cut(r) for r in by_spk[a_name] if r["duration"] >= 4.0]
    b_clips = [cut(r) for r in by_spk[b_name]]
    rng.shuffle(a_clips)
    rng.shuffle(b_clips)

    gap = np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32)
    pieces, a_turns, insertions = [], [], []
    pos, ai, bi = 0, 0, 0
    target = minutes * 60 * SAMPLE_RATE
    while pos < target and ai < len(a_clips):
        b = b_clips[bi % len(b_clips)]
        bi += 1
        pieces += [b, gap]
        pos += len(b) + len(gap)

        turn = []
        while sum(map(len, turn)) < 20 * SAMPLE_RATE and ai < len(a_clips):
            turn.append(a_clips[ai])
            ai += 1
        turn = np.concatenate(turn).copy()
        a_rms = _rms(turn)
        # Insert snippets at least 3 s apart, away from the turn edges.
        t = int(rng.uniform(1.5, 3.0) * SAMPLE_RATE)
        while t < len(turn) - 2.5 * SAMPLE_RATE:
            length = int(rng.choice(SNIPPET_SECONDS) * SAMPLE_RATE)
            gain_db = float(rng.choice(SNIPPET_GAINS_DB))
            snip = _speechy_crop(b_clips[bi % len(b_clips)], length, rng)
            bi += 1
            snip = snip * (a_rms / _rms(snip)) * 10 ** (gain_db / 20)
            turn[t : t + length] += snip
            insertions.append(
                {"start": (pos + t) / SAMPLE_RATE, "end": (pos + t + length) / SAMPLE_RATE,
                 "length": length / SAMPLE_RATE, "gain_db": gain_db}
            )
            t += length + int(rng.uniform(3.0, 6.0) * SAMPLE_RATE)
        a_turns.append({"start": pos / SAMPLE_RATE, "end": (pos + len(turn)) / SAMPLE_RATE})
        pieces += [turn, gap]
        pos += len(turn) + len(gap)

    session = np.concatenate(pieces)
    peak = np.abs(session).max()
    if peak > 0.99:
        session *= 0.99 / peak
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / "audio_16k.wav"), session, SAMPLE_RATE, subtype="PCM_16")
    truth = {"primary": a_name, "other": b_name, "a_turns": a_turns, "insertions": insertions}
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    return truth


def _interval_frames(start: float, end: float, hop: float = 0.01) -> slice:
    return slice(int(start / hop), int(np.ceil(end / hop)))


def evaluate(out_dir: Path, configs: dict[str, SelectConfig]) -> dict:
    truth = json.loads((out_dir / "truth.json").read_text(encoding="utf-8"))
    diar = np.load(out_dir / "diarization.npy")
    energy = frame_energy_db(load_mono(out_dir / "audio_16k.wav"))
    with np.load(out_dir / "osd.npz") as z:
        osd = {k: z[k] for k in z.files}

    # Primary speaker = Nemotron channel most active during A's turns.
    a_mask = np.zeros(len(diar), dtype=bool)
    for t in truth["a_turns"]:
        a_mask[_interval_frames(t["start"], t["end"])] = True
    k = int((diar[a_mask] > 0.5).sum(axis=0).argmax())

    ins = truth["insertions"]
    # A's speech far (>1 s) from any insertion: everything kept there is legitimately clean.
    untouched = a_mask.copy()
    for i in ins:
        untouched[_interval_frames(i["start"] - 1.0, i["end"] + 1.0)] = False
    untouched &= diar[:, k] > 0.5

    report = {"primary_channel": k, "num_insertions": len(ins), "configs": {}, "signal_recall": {}}
    for reduce in ("max", "mean"):
        signals = foreign_signals(diar, osd, reduce=reduce)
        for name, sig in signals.items():
            for th in (0.05, 0.1, 0.2, 0.3, 0.5):
                hit = [bool((sig[_interval_frames(i["start"], i["end"]), k] > th).any()) for i in ins]
                fp = float((sig[untouched, k] > th).mean())
                buckets = {}
                for i, h in zip(ins, hit):
                    key = f"{i['length']:.2f}s@{i['gain_db']:+.0f}dB"
                    buckets.setdefault(key, []).append(h)
                report["signal_recall"][f"{name}[{reduce}]>{th}"] = {
                    "recall": round(float(np.mean(hit)), 3),
                    "flagged_clean_frac": round(fp, 3),
                    "misses": {b: f"{len(v) - sum(v)}/{len(v)}" for b, v in sorted(buckets.items()) if not all(v)},
                }

        for cname, cfg in configs.items():
            segs = [s for s in select_segments(diar, osd, None, cfg, signals, energy) if s.speaker == k]
            leaked = [
                i for i in ins if any(s.start < i["end"] and i["start"] < s.end for s in segs)
            ]
            kept = np.zeros(len(diar), dtype=bool)
            for s in segs:
                kept[_interval_frames(s.start, s.end)] = True
            report["configs"][f"{cname}[{reduce}]"] = {
                "leaks": len(leaked),
                "leaked": [f"{i['length']:.2f}s@{i['gain_db']:+.0f}dB" for i in leaked],
                "kept_untouched_frac": round(float(kept[untouched].mean()), 3),
                "segments": len(segs),
                "config": asdict(cfg),
            }
    return report


def default_configs() -> dict[str, SelectConfig]:
    base = SelectConfig(min_speaker_seconds=0.0)
    off = dict(nemotron_other_threshold=None, diarizen_multi_threshold=None, diarizen_other_threshold=None)
    return {
        "nemotron_only@0.15": replace(base, **(off | {"nemotron_other_threshold": 0.15})),
        "diarizen_multi_only@0.2": replace(base, **(off | {"diarizen_multi_threshold": 0.2})),
        "diarizen_other_only@0.2": replace(base, **(off | {"diarizen_other_threshold": 0.2})),
        "union@0.2": replace(
            base, nemotron_other_threshold=0.15, diarizen_multi_threshold=0.2, diarizen_other_threshold=0.2
        ),
        "speaker_gap": replace(base, cut_mode="speaker_gap"),
        "default": base,
    }


def run(source: Workspace, out_dir: Path, seed: int = 0, minutes: float = 15.0) -> dict:
    """Build (once) and evaluate a synthetic session from the processed workspace ``source``."""
    from nepvoice.prep import diarize, osd

    if not (out_dir / "truth.json").exists():
        build_session(source.segments, source.ensure_wav16k(), out_dir, seed, minutes)
    wav = out_dir / "audio_16k.wav"
    if not (out_dir / "diarization.npy").exists():
        np.save(out_dir / "diarization.npy", diarize.diarize(wav))
    if not (out_dir / "osd.npz").exists():
        np.savez_compressed(out_dir / "osd.npz", **osd.detect_overlap(wav))
    report = evaluate(out_dir, default_configs())
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report
