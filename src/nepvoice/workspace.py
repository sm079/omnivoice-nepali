"""Work-directory layout and per-recording workspaces.

::

    <work>/
      recordings/<sub>/<id>/
                           one workspace per input recording (model caches, segments, clips),
                           in the same subfolders as the recording in the input folder
      speakers.json        speaker registry across recordings (+ voiceprints.npz)
      dataset/             train.jsonl, dev.jsonl: OmniVoice manifests of every kept clip
      failures.jsonl       recordings that failed a stage, with the error

Source audio is not copied: ``source.json`` records where it is, and the 16 kHz model
input is decoded from it. A workspace whose source file changed is reset on ingest.
"""

from __future__ import annotations

import json
from pathlib import Path

from .audio import to_wav

RECORDINGS = "recordings"


class Workspace:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.id = self.dir.name

    # ---- inputs ----
    @property
    def source_info(self) -> Path:
        return self.dir / "source.json"

    def source(self) -> dict:
        return json.loads(self.source_info.read_text(encoding="utf-8"))

    @property
    def audio_path(self) -> Path:
        """The original recording (outside the work directory)."""
        return Path(self.source()["audio"])

    @property
    def transcript_path(self) -> Path:
        return self.dir / self.source()["transcript_file"]

    @property
    def timed(self) -> bool:
        return self.source()["transcript_kind"] == "timed"

    def meta(self) -> dict:
        """Optional descriptive metadata from the input (e.g. a title)."""
        return self.source().get("meta", {})

    # ---- derived files ----
    wav16k = property(lambda self: self.dir / "audio_16k.wav")
    source_wav = property(lambda self: self.dir / "audio_source.wav")
    diar = property(lambda self: self.dir / "diarization.npy")
    osd = property(lambda self: self.dir / "osd.npz")
    energy = property(lambda self: self.dir / "energy.npy")
    segments = property(lambda self: self.dir / "segments.jsonl")
    summary = property(lambda self: self.dir / "summary.json")
    clips = property(lambda self: self.dir / "clips")
    speaker_emb = property(lambda self: self.dir / "speaker_emb.npz")
    speaker_gender = property(lambda self: self.dir / "speaker_gender.json")

    def ensure_wav16k(self) -> Path:
        """16 kHz mono WAV for the models, decoded from the source when missing."""
        if not self.wav16k.exists():
            to_wav(self.audio_path, self.wav16k)
        return self.wav16k

    def ensure_source_wav(self) -> Path:
        """Full-rate mono WAV that clips are cut from, decoded from the source when missing."""
        if not self.source_wav.exists():
            to_wav(self.audio_path, self.source_wav, sample_rate=None)
        return self.source_wav

    def read_segments(self) -> list[dict]:
        if not self.segments.exists():
            return []
        return [json.loads(line) for line in self.segments.read_text(encoding="utf-8").splitlines() if line]

    def __repr__(self) -> str:
        return f"Workspace({self.dir})"


def recordings_dir(work: Path) -> Path:
    return Path(work) / RECORDINGS


def find_workspaces(work: Path) -> dict[str, Workspace]:
    """Every ingested recording under ``work`` (at any depth of ``recordings/``), by id."""
    root = recordings_dir(work)
    if not root.exists():
        return {}
    found: dict[str, Workspace] = {}
    for info in sorted(root.rglob("source.json")):
        ws = Workspace(info.parent)
        if ws.id in found:
            raise ValueError(f"recording id {ws.id!r} is used twice: {found[ws.id].dir} and {ws.dir}")
        found[ws.id] = ws
    return dict(sorted(found.items()))
