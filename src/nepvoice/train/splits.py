"""Turn the prepared dataset into balanced OmniVoice manifests and a data config.

``dataset/train.jsonl`` is split by the voice class the speaker registry estimates per
speaker. Female voices are often a small share of the speech, so their clips go to a
separate manifest that the data config repeats (``female_repeat``), oversampling them
during training without duplicating any audio. Without a registry, everything is "main".
"""

from __future__ import annotations

import json
from pathlib import Path

from .paths import TrainPaths

SPLITS = ("train_main", "train_female", "dev")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_manifests(train_jsonl: Path, dev_jsonl: Path, speakers_json: Path | None, paths: TrainPaths) -> dict:
    """Write ``manifests/{train_main,train_female,dev}.jsonl``; return per-split stats."""
    voices = {}
    if speakers_json is not None and speakers_json.exists():
        registry = json.loads(speakers_json.read_text(encoding="utf-8"))["speakers"]
        voices = {gid: s.get("voice", "unknown") for gid, s in registry.items()}
    train = read_jsonl(train_jsonl)
    if not train:
        raise ValueError(f"no training clips in {train_jsonl}; check the prep stages' output")
    splits = {
        "train_main": [r for r in train if voices.get(r["speaker"]) != "female"],
        "train_female": [r for r in train if voices.get(r["speaker"]) == "female"],
        "dev": read_jsonl(dev_jsonl),
    }
    missing = [r["audio_path"] for rows in splits.values() for r in rows if not Path(r["audio_path"]).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} clips listed but missing on disk, e.g. {missing[0]}")
    stats = {}
    for name, rows in splits.items():
        _write_jsonl(paths.manifest(name), rows)
        stats[name] = {
            "clips": len(rows),
            "hours": round(sum(r["duration"] for r in rows) / 3600, 2),
            "speakers": len({r["speaker"] for r in rows}),
        }
    return stats


def nonempty_splits(paths: TrainPaths) -> list[str]:
    return [s for s in SPLITS if read_jsonl(paths.manifest(s))]


def write_data_config(paths: TrainPaths, female_repeat: int = 2) -> Path:
    """OmniVoice data config over the tokenized shards (``tokens/<split>/data.lst``); empty splits are left out."""
    present = set(nonempty_splits(paths))
    entry = lambda split, repeat: {  # noqa: E731
        "language_id": "npi", "manifest_path": [str(paths.token_list(split).resolve())], "repeat": repeat,
    }
    config = {"train": [], "dev": []}
    if "train_main" in present:
        config["train"].append(entry("train_main", 1))
    if "train_female" in present:
        config["train"].append(entry("train_female", female_repeat))
    if "dev" in present:
        config["dev"].append(entry("dev", 1))
    paths.data_config.parent.mkdir(parents=True, exist_ok=True)
    paths.data_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return paths.data_config
