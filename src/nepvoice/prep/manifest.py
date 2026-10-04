"""Combine every recording's exported clips into OmniVoice training manifests.

Each line follows OmniVoice's raw JSONL format, ``{"id", "audio_path", "text",
"language_id"}``, plus ``duration``, ``speaker`` and ``recording_id`` for bookkeeping
(OmniVoice carries extra fields along unused). ``audio_path`` is absolute, since
OmniVoice opens it as given. With speaker recognition (``speakers.json``), ``speaker``
is a global id shared across recordings and ``local_speaker`` keeps the per-recording label.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import ManifestConfig
from ..workspace import find_workspaces

LANGUAGE_ID = "npi"  # OmniVoice's id for Nepali (docs/lang_id_name_map.tsv)


def dataset_dir(work: Path) -> Path:
    return Path(work) / "dataset"


def collect(work: Path, min_duration: float = 0.0, max_duration: float = float("inf")) -> tuple[list[dict], dict]:
    """All exported clips under ``work`` that have text and fit the duration range."""
    rows: list[dict] = []
    skipped = {"no_clip_file": 0, "no_text": 0, "duration": 0}
    for rid, ws in find_workspaces(work).items():
        for r in ws.read_segments():
            audio = ws.clips / f"{r['id']}.flac"
            if not audio.exists():
                skipped["no_clip_file"] += 1
            elif not r.get("text", "").strip():
                skipped["no_text"] += 1
            elif not min_duration <= r["duration"] <= max_duration:
                skipped["duration"] += 1
            else:
                rows.append({
                    "id": r["id"],
                    "audio_path": str(audio.resolve()),
                    "text": r["text"].strip(),
                    "language_id": LANGUAGE_ID,
                    "duration": r["duration"],
                    "speaker": r["speaker"],
                    "recording_id": rid,
                })
    return rows, skipped


def keep_speakers(rows: list[dict], only: dict[str, list[int]]) -> tuple[list[dict], int]:
    """Drop clips of recordings in ``only`` whose speaker is not listed; returns kept rows and dropped count."""
    allowed = {rid: {f"{rid}_spk{k}" for k in keep} for rid, keep in only.items()}
    kept = [r for r in rows if r["recording_id"] not in allowed or r["speaker"] in allowed[r["recording_id"]]]
    return kept, len(rows) - len(kept)


def apply_speaker_ids(rows: list[dict], speakers: dict) -> None:
    """Replace per-recording speaker labels with global ids from ``speakers.json`` (in place)."""
    mapping, sims = speakers["local_to_global"], speakers.get("clip_similarity", {})
    for r in rows:
        r["local_speaker"] = r["speaker"]
        r["speaker"] = mapping.get(r["speaker"], r["speaker"])
        if r["id"] in sims:
            r["speaker_similarity"] = sims[r["id"]]


def cap_per_speaker(rows: list[dict], max_hours: float, seed: int = 0) -> tuple[list[dict], dict]:
    """Keep at most ``max_hours`` per speaker, spread evenly over that speaker's recordings.

    Clips are taken round-robin across the speaker's recordings (in random order within
    each), so a frequent speaker's kept hours come from as many recordings as possible rather than
    from the first few. A clip that would take the speaker over the cap is skipped, so
    the cap is never exceeded.
    """
    rng = random.Random(seed)
    by_speaker: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        by_speaker.setdefault(r["speaker"], {}).setdefault(r["recording_id"], []).append(r)
    budget = max_hours * 3600
    kept, capped = [], {}
    for spk, recs in by_speaker.items():
        total = sum(r["duration"] for v in recs.values() for r in v)
        if total <= budget:
            kept += [r for v in recs.values() for r in v]
            continue
        queues = [rng.sample(v, len(v)) for _, v in sorted(recs.items())]
        rng.shuffle(queues)
        used = 0.0
        while any(queues):
            for q in queues:
                if q:
                    r = q.pop()
                    if used + r["duration"] <= budget:
                        kept.append(r)
                        used += r["duration"]
        capped[spk] = {"available_hours": round(total / 3600, 2), "kept_hours": round(used / 3600, 2)}
    order = {id(r): i for i, r in enumerate(rows)}
    kept.sort(key=lambda r: order[id(r)])
    return kept, capped


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write(work: Path, cfg: ManifestConfig, speakers: dict | None = None) -> dict:
    """Write ``<work>/dataset/train.jsonl`` and ``dev.jsonl``; return stats.

    With ``speakers`` (the contents of ``speakers.json``), clips carry global speaker ids
    and ``cfg.max_hours_per_speaker`` caps how much any one person contributes. The dev
    split holds out whole *recordings*, so no dev clip shares a recording with training.
    """
    rows, skipped = collect(work, cfg.min_duration, cfg.max_duration)
    rows, skipped["other_speaker"] = keep_speakers(rows, cfg.only_speakers)
    capped = {}
    if speakers is not None:
        apply_speaker_ids(rows, speakers)
        if cfg.max_hours_per_speaker > 0:
            rows, capped = cap_per_speaker(rows, cfg.max_hours_per_speaker, cfg.seed)
    recordings = sorted({r["recording_id"] for r in rows})
    random.Random(cfg.seed).shuffle(recordings)
    n_dev = round(len(recordings) * cfg.dev_fraction)
    if cfg.dev_fraction > 0 and len(recordings) > 1:
        n_dev = max(1, n_dev)
    dev_recordings = set(recordings[:n_dev])

    splits = {
        "train": [r for r in rows if r["recording_id"] not in dev_recordings],
        "dev": [r for r in rows if r["recording_id"] in dev_recordings],
    }
    out = dataset_dir(work)
    stats = {"skipped": skipped, "recordings": len(recordings), "capped_speakers": capped}
    for name, split in splits.items():
        path = out / f"{name}.jsonl"
        _write_jsonl(path, split)
        stats[name] = {
            "file": str(path),
            "clips": len(split),
            "hours": round(sum(r["duration"] for r in split) / 3600, 2),
            "speakers": len({r["speaker"] for r in split}),
        }
    return stats
