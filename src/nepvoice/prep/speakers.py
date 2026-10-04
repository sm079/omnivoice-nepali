"""Recognise the same person across recordings, so frequent speakers can be capped.

1. Every selected clip gets a speaker embedding (SpeechBrain ECAPA-TDNN, 192-d),
   cached per recording in ``speaker_emb.npz``.
2. Each (recording, diarized speaker) gets a voiceprint: the duration-weighted mean of
   its clip embeddings, after dropping clips that disagree with the rest (likely
   diarization slips). Averaging many clips makes voiceprints far more reliable than
   single clips.
3. Voiceprints from all recordings are clustered (average linkage on cosine similarity)
   into global speakers. Voiceprints from the same recording may merge too: diarization
   occasionally splits one person into two channels.

Clustering holds an ``n x n`` distance matrix, so it comfortably handles ~10k
voiceprints (a few thousand recordings) in memory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..audio import SAMPLE_RATE, load_mono
from ..workspace import Workspace, find_workspaces

MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
GENDER_MODEL = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
MAX_CLIP_SECONDS = 8.0  # longer clips are centre-cropped; plenty for a stable embedding


@dataclass
class SpeakerConfig:
    # Voiceprints at least this cosine-similar are the same person. On long conversational
    # recordings, the same person in different recordings scored 0.93-0.97 while different
    # people reached up to 0.71 (0.6 merged five pairs of different people).
    same_speaker_similarity: float = 0.8
    # Clips this far from their own voiceprint are left out of it (and flagged).
    clip_outlier_similarity: float = 0.3
    # Voiceprints need this much speech to be trusted for cross-recording matching. A
    # 12.8 s voiceprint still matched its speaker at 0.83-0.89. Lower it for collections
    # of short single-utterance files, which otherwise never get matched.
    min_voiceprint_seconds: float = 10.0


# ---- per-recording embeddings ----

@lru_cache(maxsize=1)
def _model(device: str):
    from speechbrain.inference.speaker import EncoderClassifier

    return EncoderClassifier.from_hparams(source=MODEL_ID, run_opts={"device": device}, savedir=None)


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def embed_clips(audio: np.ndarray, spans: list[tuple[float, float]], batch_size: int = 32) -> np.ndarray:
    """L2-normalised embeddings ``(N, 192)`` for ``(start, end)`` spans of 16 kHz ``audio``."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _model(device)
    max_len = int(MAX_CLIP_SECONDS * SAMPLE_RATE)
    pieces = []
    for start, end in spans:
        x = audio[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]
        if len(x) > max_len:
            off = (len(x) - max_len) // 2
            x = x[off: off + max_len]
        pieces.append(x)

    out = np.zeros((len(pieces), 192), np.float32)
    order = np.argsort([len(p) for p in pieces])  # batch similar lengths together
    for b in range(0, len(order), batch_size):
        idx = order[b: b + batch_size]
        longest = max(len(pieces[i]) for i in idx)
        wavs = np.zeros((len(idx), longest), np.float32)
        lens = np.zeros(len(idx), np.float32)
        for j, i in enumerate(idx):
            wavs[j, : len(pieces[i])] = pieces[i]
            lens[j] = len(pieces[i]) / longest
        with torch.inference_mode():
            emb = model.encode_batch(torch.from_numpy(wavs).to(device), torch.from_numpy(lens).to(device))
        out[idx] = emb.squeeze(1).float().cpu().numpy()
    return _normalize(out)


def clip_pitch(audio: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
    """Median F0 (Hz) of the louder frames in the middle 4 s of each span (NaN if unclear)."""
    import librosa

    out = np.full(len(spans), np.nan, np.float32)
    for i, (start, end) in enumerate(spans):
        mid, half = (start + end) / 2, min(2.0, (end - start) / 2)
        x = audio[int((mid - half) * SAMPLE_RATE): int((mid + half) * SAMPLE_RATE)]
        if len(x) < SAMPLE_RATE // 2:
            continue
        f0 = librosa.yin(x, fmin=60, fmax=400, sr=SAMPLE_RATE, frame_length=1024, hop_length=256)
        rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=256)[0][: len(f0)]
        loud = rms >= np.percentile(rms, 50)  # voiced speech; skips pauses and unvoiced noise
        if loud.sum() >= 5:
            out[i] = float(np.median(f0[: len(loud)][loud]))
    return out


def embed_recording(ws: Workspace) -> Path:
    """Compute (or reuse) clip embeddings and pitch for a recording's current ``segments.jsonl``."""
    path = ws.speaker_emb
    mtime = ws.segments.stat().st_mtime_ns
    if path.exists():
        with np.load(path) as z:
            if int(z["segments_mtime"]) == mtime:
                return path
    rows = ws.read_segments()
    spans = [(r["start"], r["end"]) for r in rows]
    audio = load_mono(ws.ensure_wav16k()) if rows else np.zeros(0, np.float32)
    np.savez(
        path,
        ids=np.array([r["id"] for r in rows]),
        local_speakers=np.array([r["speaker"] for r in rows]),
        durations=np.array([r["duration"] for r in rows], np.float32),
        embeddings=embed_clips(audio, spans) if rows else np.zeros((0, 192), np.float32),
        f0=clip_pitch(audio, spans) if rows else np.zeros(0, np.float32),
        segments_mtime=np.int64(mtime),
    )
    return path


def voice_from_probability(p_female: float) -> str:
    if not np.isfinite(p_female):
        return "unknown"
    return "female" if p_female >= 0.7 else "male" if p_female <= 0.3 else "unclear"


@lru_cache(maxsize=1)
def _gender_classifier(device: int):
    from transformers import pipeline as hf_pipeline

    return hf_pipeline("audio-classification", model=GENDER_MODEL, device=device)


def recording_gender(ws: Workspace, clips_per_speaker: int = 8, window: float = 6.0) -> dict[str, float]:
    """P(female) per diarized speaker of a recording, from a wav2vec2 voice-gender classifier
    run on the middle of that speaker's longest clips; cached in ``speaker_gender.json``."""
    path = ws.speaker_gender
    mtime = ws.segments.stat().st_mtime_ns
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("segments_mtime") == mtime:
            return cached["p_female"]
    rows = ws.read_segments()
    by_speaker: dict[str, list[dict]] = {}
    for r in rows:
        by_speaker.setdefault(r["speaker"], []).append(r)
    result: dict[str, float] = {}
    if rows:
        import torch

        clf = _gender_classifier(0 if torch.cuda.is_available() else -1)
        female = next(label for label in clf.model.config.id2label.values() if label.lower() == "female")
        audio = load_mono(ws.ensure_wav16k())
        for name, spk_rows in by_speaker.items():
            probs = []
            for r in sorted(spk_rows, key=lambda r: -r["duration"])[:clips_per_speaker]:
                mid, half = (r["start"] + r["end"]) / 2, min(window, r["duration"]) / 2
                x = audio[int((mid - half) * SAMPLE_RATE): int((mid + half) * SAMPLE_RATE)]
                scores = clf({"raw": x, "sampling_rate": SAMPLE_RATE}, top_k=None)
                probs.append(next(s["score"] for s in scores if s["label"] == female))
            result[name] = round(float(np.mean(probs)), 3)
    path.write_text(json.dumps({"segments_mtime": mtime, "model": GENDER_MODEL, "p_female": result}), encoding="utf-8")
    return result


# ---- across recordings ----

def voiceprints(work: Path, cfg: SpeakerConfig) -> tuple[list[dict], dict[str, float]]:
    """One voiceprint per (recording, local speaker), plus each clip's similarity to its own."""
    prints, clip_sim = [], {}
    for rid, ws in find_workspaces(work).items():
        if not ws.speaker_emb.exists():
            continue
        gender = recording_gender(ws)
        with np.load(ws.speaker_emb) as z:
            ids, spk, dur, emb, f0 = z["ids"], z["local_speakers"], z["durations"], z["embeddings"], z["f0"]
        for name in sorted(set(spk.tolist())):
            m = spk == name
            e, d = emb[m], dur[m]
            centre = _normalize((e * d[:, None]).sum(0))
            sims = e @ centre
            keep = sims >= cfg.clip_outlier_similarity
            if keep.any():
                centre = _normalize((e[keep] * d[keep, None]).sum(0))
                sims = e @ centre
            clip_sim.update(zip(ids[m].tolist(), sims.round(3).tolist()))
            good = sims >= cfg.clip_outlier_similarity
            pitches = f0[m][good]
            prints.append({
                "local_speaker": name,
                "recording_id": rid,
                "seconds": float(d[good].sum()),
                "clips": int(m.sum()),
                "f0": float(np.nanmedian(pitches)) if np.isfinite(pitches).any() else float("nan"),
                "p_female": gender.get(name, float("nan")),
                "centroid": centre,
            })
    return prints, clip_sim


def cluster(prints: list[dict], cfg: SpeakerConfig) -> list[int]:
    """Global speaker index per voiceprint, ordered by total speech (0 = most speech)."""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(prints)
    if n == 0:
        return []
    trusted = [i for i, p in enumerate(prints) if p["seconds"] >= cfg.min_voiceprint_seconds]
    labels = list(range(n))  # voiceprints with too little speech stay on their own
    if len(trusted) > 1:
        c = np.stack([prints[i]["centroid"] for i in trusted]).astype(np.float64)
        tree = linkage(c, method="average", metric="cosine")
        groups = fcluster(tree, t=1.0 - cfg.same_speaker_similarity, criterion="distance")
        for j, g in zip(trusted, groups):
            labels[j] = n + int(g)  # disjoint from the singleton labels
    # Compact, ordered by total speech (largest first).
    totals: dict[int, float] = {}
    for i, lab in enumerate(labels):
        totals[lab] = totals.get(lab, 0.0) + prints[i]["seconds"]
    rank = {lab: r for r, lab in enumerate(sorted(totals, key=lambda k: -totals[k]))}
    return [rank[lab] for lab in labels]


def _stable_ids(prints: list[dict], labels: list[int], previous: dict[str, str]) -> list[str]:
    """Global id per voiceprint, reusing ids from the previous run wherever possible.

    Each new cluster takes the previous id held by most of its speech (if not already
    claimed by a bigger cluster); clusters with no history get fresh numbers. This keeps
    ids stable as recordings are added, even though clusters are recomputed every time.
    """
    clusters: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(i)
    order = sorted(clusters, key=lambda c: -sum(prints[i]["seconds"] for i in clusters[c]))
    taken: set[str] = set()
    assigned: dict[int, str] = {}
    for c in order:
        votes: dict[str, float] = {}
        for i in clusters[c]:
            old = previous.get(prints[i]["local_speaker"])
            if old:
                votes[old] = votes.get(old, 0.0) + prints[i]["seconds"] + 1e-3
        for old in sorted(votes, key=lambda k: -votes[k]):
            if old not in taken:
                assigned[c] = old
                taken.add(old)
                break
    used = [int(g[1:]) for g in set(previous.values()) | taken if g[1:].isdigit()]
    next_id = max(used, default=0) + 1
    for c in order:
        if c not in assigned:
            assigned[c] = f"S{next_id:04d}"
            next_id += 1
    return [assigned[lab] for lab in labels]


def build(work: Path, cfg: SpeakerConfig | None = None) -> dict:
    """Cluster all cached embeddings under ``work``; write ``speakers.json`` and ``voiceprints.npz``.

    ``speakers.json`` is the speaker registry: per global id its total speech, recordings,
    member voiceprints, median pitch, and P(female) from a voice classifier with a voice
    class. ``voiceprints.npz`` keeps every voiceprint vector so future recordings can be
    matched without recomputing anything.
    """
    cfg = cfg or SpeakerConfig()
    out_file = Path(work) / "speakers.json"
    previous = {}
    if out_file.exists():
        previous = json.loads(out_file.read_text(encoding="utf-8")).get("local_to_global", {})

    prints, clip_sim = voiceprints(work, cfg)
    ids = _stable_ids(prints, cluster(prints, cfg), previous)

    speakers: dict[str, dict] = {}
    for p, gid in zip(prints, ids):
        s = speakers.setdefault(gid, {"seconds": 0.0, "clips": 0, "recordings": [], "members": [], "_p": []})
        s["seconds"] += p["seconds"]
        s["clips"] += p["clips"]
        s["recordings"].append(p["recording_id"])
        s["members"].append(p["local_speaker"])
        s["_p"].append(p)
    for s in speakers.values():
        members = s.pop("_p")
        s["seconds"] = round(s["seconds"], 1)
        s["recordings"] = sorted(set(s["recordings"]))
        w = np.array([m["seconds"] for m in members])
        f0 = np.array([m["f0"] for m in members])
        ok = np.isfinite(f0) & (w > 0)
        s["f0_median_hz"] = round(float(np.average(f0[ok], weights=w[ok])), 1) if ok.any() else None
        pf = np.array([m["p_female"] for m in members])
        okf = np.isfinite(pf) & (w > 0)
        s["p_female"] = round(float(np.average(pf[okf], weights=w[okf])), 3) if okf.any() else None
        s["voice"] = voice_from_probability(s["p_female"] if s["p_female"] is not None else float("nan"))

    result = {
        "config": asdict(cfg),
        "note": "voice is an estimate (wav2vec2 gender classifier)",
        "speakers": dict(sorted(speakers.items())),
        "local_to_global": {p["local_speaker"]: gid for p, gid in zip(prints, ids)},
        "clip_similarity": clip_sim,
    }
    out_file.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez(
        out_file.with_name("voiceprints.npz"),
        local_speakers=np.array([p["local_speaker"] for p in prints]),
        global_ids=np.array(ids),
        recording_ids=np.array([p["recording_id"] for p in prints]),
        seconds=np.array([p["seconds"] for p in prints], np.float32),
        f0=np.array([p["f0"] for p in prints], np.float32),
        centroids=np.stack([p["centroid"] for p in prints]) if prints else np.zeros((0, 192), np.float32),
    )
    return result


def report(result: dict, limit: int = 15) -> str:
    """The largest speakers and the speech per voice class, as printable text."""
    spk = result["speakers"]
    total = sum(s["seconds"] for s in spk.values()) or 1.0
    n_rec = len({r for s in spk.values() for r in s["recordings"]})
    lines = [f"{len(spk)} speakers across {n_rec} recordings; largest:"]
    for gid, s in sorted(spk.items(), key=lambda kv: -kv[1]["seconds"])[:limit]:
        lines.append(
            f"  {gid}  {s['seconds'] / 3600:6.2f} h  {s['seconds'] / total:6.1%}  {len(s['recordings']):3d} recordings"
            f"  {s.get('voice', '?')}"
        )
    voices: dict[str, float] = {}
    for s in spk.values():
        voices[s.get("voice", "unknown")] = voices.get(s.get("voice", "unknown"), 0.0) + s["seconds"]
    lines.append("by voice (classifier estimate): "
                 + ", ".join(f"{v} {h / 3600:.1f} h" for v, h in sorted(voices.items())))
    return "\n".join(lines)
