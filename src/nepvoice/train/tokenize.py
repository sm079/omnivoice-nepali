"""Convert manifests to OmniVoice's tokenized WebDataset shards (Higgs-audio-v2, 24 kHz)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .paths import TrainPaths
from .splits import nonempty_splits

TOKENIZER = "eustlb/higgs-audio-v2-tokenizer"
_DRIVE = re.compile(r"^(?:file:/*)?([A-Za-z]:[\\/].*)$")


def windows_safe_manifest(lst: Path, base: Path | None = None) -> bool:
    """Make ``data.lst`` usable from anywhere; returns True if it changed.

    Relative shard and label paths (the tokenizer runs inside the split's folder, see
    :func:`tokenize_split`) are resolved against ``base``. Windows shard paths are then
    written as ``file:C:/...`` URLs: WebDataset parses ``C:\\...`` as a URL with scheme
    ``c`` and fails with "no gopen handler defined". ``file:C:/...`` opens the same file
    through both of WebDataset's file code paths (``file:///C:/...`` does not: one of
    them strips only ``file://`` and leaves ``/C:/...``). The label column stays a plain
    path (OmniVoice opens it with ``open``), and OmniVoice keys its tar-to-label map by
    the same string WebDataset reports, so they stay in sync.
    """
    lines = lst.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        parts = line.split(" ")
        if base is not None and len(parts) >= 2:
            for i in (0, 1):
                if not parts[i].startswith("file:") and not Path(parts[i]).is_absolute():
                    parts[i] = str((base / parts[i]).resolve())
        m = _DRIVE.match(parts[0]) if parts else None
        if m:
            parts[0] = "file:" + m.group(1).replace("\\", "/")
        out.append(" ".join(parts))
    if out != lines:
        lst.write_text("\n".join(out) + "\n", encoding="utf-8")
        return True
    return False


def _fingerprint(manifest: Path, peak_normalize: bool) -> dict:
    return {
        "manifest_sha1": hashlib.sha1(manifest.read_bytes()).hexdigest(),
        "peak_normalize": peak_normalize,
        "tokenizer": TOKENIZER,
    }


def tokenize_split(
    paths: TrainPaths, split: str, peak_normalize: bool = True, jobs_per_gpu: int = 2, loader_workers: int = 4
) -> Path:
    """Tokenize ``manifests/<split>.jsonl`` into ``tokens/<split>/``.

    Skipped when the shards were made from the same manifest with the same settings;
    redone from scratch when either changed.
    """
    src = paths.manifest(split)
    dst = paths.tokens / split
    lst = paths.token_list(split)
    stamp = dst / "source.json"
    want = _fingerprint(src, peak_normalize)
    if lst.exists() and lst.stat().st_size > 0 and stamp.exists() and json.loads(stamp.read_text()) == want:
        windows_safe_manifest(lst)
        print(f"[{split}] already tokenized ({lst})", flush=True)
        return lst
    if dst.exists():
        shutil.rmtree(dst)
    print(f"[{split}] tokenizing {src}", flush=True)
    dst.mkdir(parents=True)
    # Output paths are relative to the split folder (the working directory below): the
    # shard writer, like WebDataset's reader, takes "C:\..." for a URL scheme.
    cmd = [
        sys.executable, "-m", "nepvoice.train.extract_tokens",
        "--input_jsonl", str(src.resolve()),
        "--tar_output_pattern", "audios/shard-%06d.tar",
        "--jsonl_output_pattern", "txts/shard-%06d.jsonl",
        "--tokenizer_path", TOKENIZER,
        "--nj_per_gpu", str(jobs_per_gpu),
        "--loader_workers", str(loader_workers),
        "--skip_errors",
        "--shuffle", "True",
    ]
    if not peak_normalize:
        cmd.append("--no-peak-normalize")
    subprocess.run(cmd, check=True, cwd=dst)
    if not lst.exists():
        raise RuntimeError(f"tokenizer finished without writing {lst}")
    windows_safe_manifest(lst, base=dst)
    stamp.write_text(json.dumps(want), encoding="utf-8")
    return lst


def tokenize_all(
    paths: TrainPaths, peak_normalize: bool = True, jobs_per_gpu: int = 2, loader_workers: int = 4
) -> None:
    splits = nonempty_splits(paths)
    for i, split in enumerate(splits, 1):
        tokenize_split(paths, split, peak_normalize, jobs_per_gpu, loader_workers)
        print(f"tokenize split {i}/{len(splits)} done", flush=True)
