"""Register an input folder of recordings and transcripts as workspaces.

Input layout (files pair up by name; subfolders are searched too)::

    input/
      talk-01.webm         audio in any format ffmpeg reads, any length and sample rate
      talk-01.json3        transcript: .json3 / .srt / .vtt (timed) or .txt (whole file)
      talk-01.json         optional metadata, e.g. {"title": ...}

Each recording's workspace sits in the same subfolders as the recording itself
(``input/a/talk-01.webm`` -> ``<work>/recordings/a/talk-01/``) and is named by the file
name alone, so file names must be unique across the input folder. Reorganising the input
folders moves the workspaces along, keeping everything computed so far.

Timed transcripts let a long recording be cut into many clips. A ``.txt`` transcript
covers its whole file, which is then kept or rejected as one clip. Recordings without a
transcript are skipped.

Metadata is kept in the workspace's ``source.json``; its ``title`` is shown in the segment
viewer.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .audio import AUDIO_EXTS
from .config import IngestConfig
from .transcripts import TIMED_EXTS, TRANSCRIPT_EXTS
from .workspace import Workspace, find_workspaces, recordings_dir

_UNSAFE = re.compile(r"[^\w.\-]+")

# Files that depend only on the audio; everything else in a workspace also depends on the transcript.
AUDIO_CACHES = ("audio_16k.wav", "audio_source.wav", "diarization.npy", "osd.npz", "energy.npy")


def recording_id(audio: Path) -> str:
    """Folder-safe id from the file name: ``a/b/talk 1.mp3`` -> ``talk_1``."""
    return _UNSAFE.sub("_", audio.stem).strip("._") or "_"


@dataclass
class Recording:
    id: str
    audio: Path
    transcript: Path
    meta: dict
    subdir: Path = Path()  # the recording's folder relative to the input root


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    no_transcript: list[str] = field(default_factory=list)
    not_in_input: list[str] = field(default_factory=list)

    @property
    def ids(self) -> list[str]:
        return sorted(self.added + self.updated + self.unchanged)

    def summary(self) -> str:
        return (f"{len(self.ids)} recordings ({len(self.added)} new, {len(self.updated)} changed, "
                f"{len(self.unchanged)} unchanged); {len(self.no_transcript)} skipped without a transcript")


def scan(input_dir: Path, cfg: IngestConfig | None = None) -> tuple[list[Recording], list[Path]]:
    """Recordings with a transcript, and audio files without one."""
    cfg = cfg or IngestConfig()
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input folder not found: {input_dir}")
    pattern = "**/*" if cfg.recursive else "*"
    audio_files = sorted(p for p in input_dir.glob(pattern) if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    found, missing, ids = [], [], {}
    for audio in audio_files:
        transcript = next((t for ext in TRANSCRIPT_EXTS if (t := audio.with_suffix(ext)).is_file()), None)
        if transcript is None:
            missing.append(audio)
            continue
        rid = recording_id(audio)
        if rid in ids:
            raise ValueError(f"{audio} and {ids[rid]} map to the same recording id {rid!r}; rename one")
        ids[rid] = audio
        meta_file = audio.with_suffix(".json")
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
        if not isinstance(meta, dict):
            raise ValueError(f"{meta_file} must hold a JSON object")
        found.append(Recording(rid, audio.resolve(), transcript.resolve(), meta, audio.parent.relative_to(input_dir)))
    return found, missing


def _fingerprint(path: Path) -> dict:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _clear(ws: Workspace, keep: tuple[str, ...]) -> None:
    for p in ws.dir.iterdir():
        if p.name in keep or p.name == "source.json":
            continue
        if p.is_dir() and next(p.rglob("source.json"), None):  # another recording's workspace (input/a/ next to a.*)
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()


def _relocate(old: Workspace, new: Path, root: Path) -> None:
    """Move a workspace to where its recording now sits in the input folder."""
    new.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old.dir), str(new))
    parent = old.dir.parent
    while parent != root and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent


def register(rec: Recording, work: Path, existing: dict[str, Workspace] | None = None) -> tuple[Workspace, str]:
    """Create or refresh the workspace for ``rec``; returns it with "added", "updated" or "unchanged".

    ``existing`` (id -> workspace) lets a workspace that is somewhere else under
    ``recordings/`` move to the recording's current subfolder instead of starting over."""
    root = recordings_dir(work)
    ws = Workspace(root / rec.subdir / rec.id)
    old_ws = (existing or {}).get(rec.id)
    if old_ws is not None and old_ws.dir.resolve() != ws.dir.resolve():
        _relocate(old_ws, ws.dir, root)
    ws.dir.mkdir(parents=True, exist_ok=True)
    text = rec.transcript.read_bytes()
    info = {
        "id": rec.id,
        "audio": str(rec.audio),
        "audio_fingerprint": _fingerprint(rec.audio),
        "transcript_source": str(rec.transcript),
        "transcript_file": "transcript" + rec.transcript.suffix.lower(),
        "transcript_kind": "timed" if rec.transcript.suffix.lower() in TIMED_EXTS else "plain",
        "transcript_sha1": hashlib.sha1(text).hexdigest(),
        "meta": rec.meta,
    }
    status = "added"
    if ws.source_info.exists():
        old = ws.source()
        if old == info:
            return ws, "unchanged"
        status = "updated"
        if old.get("audio_fingerprint") != info["audio_fingerprint"]:  # a moved file keeps its caches
            _clear(ws, keep=())
        elif {k: old.get(k) for k in ("transcript_sha1", "transcript_file")} != \
                {k: info[k] for k in ("transcript_sha1", "transcript_file")}:
            _clear(ws, keep=AUDIO_CACHES)
    (ws.dir / info["transcript_file"]).write_bytes(text)
    ws.source_info.write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    return ws, status


def ingest(input_dir: Path, work: Path, cfg: IngestConfig | None = None) -> IngestReport:
    recordings, missing = scan(input_dir, cfg)
    report = IngestReport(no_transcript=[str(p) for p in missing])
    existing = find_workspaces(work)
    for i, rec in enumerate(recordings, 1):
        _, status = register(rec, work, existing)
        getattr(report, status).append(rec.id)
        if i % 25 == 0 or i == len(recordings):
            print(f"ingest {i}/{len(recordings)}", flush=True)
    current = {r.id for r in recordings}
    report.not_in_input = sorted(set(find_workspaces(work)) - current)
    return report
