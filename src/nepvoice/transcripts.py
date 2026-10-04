"""Transcripts: timed captions (json3, SRT, WebVTT) as words, or a plain text file.

A *timed* transcript lets long recordings be cut into clips, each with the words
spoken inside it. A *plain* transcript (``.txt``) describes a whole file, so that file
is kept or rejected as one clip.

Word timings are approximate either way: json3 gives each word a start time,
SRT/VTT only time whole cues, so words inside a cue get times proportional to their length.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Preferred first when several transcripts share a recording's name.
TIMED_EXTS = (".json3", ".srt", ".vtt")
PLAIN_EXTS = (".txt",)
TRANSCRIPT_EXTS = TIMED_EXTS + PLAIN_EXTS

# A word ends where the next one starts, capped so a long pause isn't absorbed into it.
MAX_WORD_DURATION = 1.0


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    words: list[Word] | None  # None for a plain transcript

    @property
    def timed(self) -> bool:
        return self.words is not None


def load(path: Path) -> Transcript:
    ext = path.suffix.lower()
    raw = path.read_text(encoding="utf-8-sig")
    if ext == ".json3":
        words = parse_json3(raw)
    elif ext in (".srt", ".vtt"):
        words = cues_to_words(parse_cues(raw))
    elif ext in PLAIN_EXTS:
        return Transcript(" ".join(raw.split()), None)
    else:
        raise ValueError(f"unsupported transcript format: {path.name}")
    return Transcript(" ".join(w.text for w in words), words)


def _finish(raw: list[tuple[float, float, str]]) -> list[Word]:
    """``(start, latest_end, text)`` -> words ending at the next start (capped)."""
    raw.sort(key=lambda w: w[0])
    words = []
    for i, (start, limit, text) in enumerate(raw):
        next_start = raw[i + 1][0] if i + 1 < len(raw) else limit
        end = min(next_start, limit, start + MAX_WORD_DURATION)
        words.append(Word(start, max(end, start + 0.05), text))
    return words


def parse_json3(text: str) -> list[Word]:
    """json3 timed-text captions; each word has a start time, ends with its event at the latest."""
    data = json.loads(text)
    raw: list[tuple[float, float, str]] = []
    for event in data.get("events", []):
        segs = event.get("segs")
        if not segs:
            continue
        t0 = event["tStartMs"] / 1000
        event_end = t0 + event.get("dDurationMs", 0) / 1000
        for seg in segs:
            for word in seg.get("utf8", "").replace(">>", "").split():
                raw.append((t0 + seg.get("tOffsetMs", 0) / 1000, event_end, word))
    return _finish(raw)


_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})"
_CUE = re.compile(rf"{_TIME}\s*-->\s*{_TIME}")
_TAG = re.compile(r"<[^>]*>")


def _seconds(h: str | None, m: str, s: str, ms: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_cues(text: str) -> list[tuple[float, float, str]]:
    """``(start, end, text)`` cues of an SRT or WebVTT file.

    Markup and inline timestamps are removed. Auto-generated VTT often repeats the
    previous caption line at the top of every cue (a "rolling" display); lines that
    repeat the previous cue are dropped so no word is counted twice.
    """
    cues: list[tuple[float, float, str]] = []
    prev_lines: list[str] = []
    for block in re.split(r"\r?\n\s*\r?\n", text):
        lines = block.strip().splitlines()
        for i, line in enumerate(lines):
            m = _CUE.search(line)
            if not m:
                continue
            g = m.groups()
            start, end = _seconds(*g[:4]), _seconds(*g[4:])
            body = [_TAG.sub("", ln).strip() for ln in lines[i + 1:]]
            body = [ln for ln in body if ln]
            new = [ln for ln in body if ln not in prev_lines]
            if body:
                prev_lines = body
            if new and end > start:
                cues.append((start, end, " ".join(new)))
            break
    return cues


def cues_to_words(cues: list[tuple[float, float, str]]) -> list[Word]:
    """Spread each cue's duration over its words in proportion to their length."""
    raw: list[tuple[float, float, str]] = []
    for start, end, text in cues:
        words = text.split()
        total = sum(len(w) + 1 for w in words)
        t = start
        for w in words:
            raw.append((t, end, w))
            t += (end - start) * (len(w) + 1) / total
    return _finish(raw)


def text_between(words: list[Word], start: float, end: float) -> tuple[str, int, int]:
    """Words whose start falls in ``[start, end)``, plus the count of words cut by either boundary."""
    inside = [w for w in words if start <= w.start < end]
    cut = sum(1 for w in words if w.start < start < w.end) + sum(1 for w in inside if w.end > end + 0.2)
    return " ".join(w.text for w in inside), len(inside), cut
