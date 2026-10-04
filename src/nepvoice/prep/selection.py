"""Fuse diarization and overlap detection into strictly single-speaker segments.

Every detector contributes a per-frame, per-target-speaker signal saying how
likely it is that someone *other* than the target is audible. A frame is
*foreign* to speaker ``k`` when **any** signal crosses its threshold (a union,
not an agreement):

* ``nemotron_other``: the strongest Nemotron channel other than ``k``;
* ``diarizen_multi``: DiariZen's P(two or more simultaneous speakers);
* ``diarizen_other``: DiariZen's local speakers, matched per window to the
  Nemotron speakers, taking the strongest one *not* matched to ``k``. This catches
  short interjections that Nemotron misses or folds into ``k``'s channel.

Foreign frames are dilated by ``pad`` seconds on both sides. What remains are
*clean runs*, which are then cut into clips of ``min_duration``..``max_duration``
seconds. A clip may only start or end at a *cut point*:

* ``cut_mode="acoustic"`` (default): a real pause in the waveform (energy at
  least ``pause_db`` below the speech level for ``min_pause``) or a gap in
  ``k``'s Nemotron activity. Speech that runs into a foreign zone is trimmed back
  to the nearest pause, so a backchannel in the middle of a monologue only costs
  the phrases around it.
* ``cut_mode="speaker_gap"``: only gaps in ``k``'s Nemotron activity. Nemotron's
  activity is smoothed over phrase pauses, so any utterance touching a foreign
  zone is dropped whole.

Thresholds sit far below 0.5: the goal is zero overlap in the kept data, and
throwing away some clean speech is an acceptable cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment

from ..transcripts import Word, text_between

HOP = 0.01  # analysis grid, seconds (Nemotron's native frame rate)
OSD_HOP_RATIO = 2  # DiariZen frames are 20 ms


@dataclass
class SelectConfig:
    speech_threshold: float = 0.5
    nemotron_other_threshold: float = 0.03
    diarizen_multi_threshold: float = 0.03
    diarizen_other_threshold: float = 0.03
    pad: float = 0.3
    min_duration: float = 2.0
    max_duration: float = 20.0
    split_pause: float = 0.2
    max_internal_pause: float = 1.0
    edge_margin: float = 0.1
    min_words: int = 1
    # Channels with less speech than this are treated as spurious: they still count as
    # foreign for everyone else, but never yield segments of their own.
    min_speaker_seconds: float = 30.0
    cut_mode: str = "acoustic"
    # Acoustic pause: energy this many dB below the speech level (90th percentile of
    # energy while anyone speaks), lasting at least ``min_pause`` seconds. 150 ms keeps
    # cuts out of long consonant closures inside words.
    pause_db: float = -30.0
    min_pause: float = 0.15
    # A foreign fragment this short and weak (peak signal / threshold), separated from a
    # detected event only by a pause, is treated as that event's fade-out (acoustic mode).
    blur_max_duration: float = 0.2
    blur_max_strength: float = 8.0
    # Optional breath trimming at clip edges (see edges.py).
    trim_breaths: bool = False
    breath_silence_db: float = -30.0
    breath_dip_db: float = 8.0
    breath_min: float = 0.12
    breath_keep: float = 0.06

    def thresholds(self) -> dict[str, float]:
        return {
            "nemotron_other": self.nemotron_other_threshold,
            "diarizen_multi": self.diarizen_multi_threshold,
            "diarizen_other": self.diarizen_other_threshold,
        }


@dataclass
class Segment:
    speaker: int
    start: float
    end: float
    text: str = ""
    num_words: int = 0
    cut_words: int = 0
    max_nemotron_other: float = 0.0
    max_diarizen_multi: float = 0.0
    max_diarizen_other: float = 0.0
    mean_speaker_prob: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        return {k: round(float(v), 3) if isinstance(v, float) else v for k, v in d.items()}


def upsample(x: np.ndarray, num_frames: int) -> np.ndarray:
    """Repeat 20 ms frames onto the 10 ms grid, padding or trimming to ``num_frames``."""
    up = np.repeat(x, OSD_HOP_RATIO, axis=0)
    if len(up) < num_frames:
        up = np.concatenate([up, np.repeat(up[-1:], num_frames - len(up), axis=0)])
    return up[:num_frames]


def diarizen_other_activity(
    diar: np.ndarray, local: np.ndarray, window_starts: np.ndarray, reduce: str = "max"
) -> np.ndarray:
    """Per-frame activity of DiariZen speakers that are *not* each global Nemotron speaker.

    Returns ``(T20, K)`` at DiariZen's 20 ms hop: entry ``[t, k]`` is the strongest local
    speaker active at ``t`` that the window-level matching did not assign to ``k``.
    Local speakers with no Nemotron counterpart count as "other" for every ``k``.
    ``reduce`` combines overlapping windows: ``"max"`` (most aggressive) or ``"mean"``.
    """
    num_windows, num_frames, num_local = local.shape
    num_global = diar.shape[1]
    T20 = int(window_starts[-1]) + num_frames
    d20 = diar[: T20 * OSD_HOP_RATIO]
    d20 = np.pad(d20, ((0, T20 * OSD_HOP_RATIO - len(d20)), (0, 0)))
    d20 = d20.reshape(T20, OSD_HOP_RATIO, num_global).max(axis=1)

    out = np.zeros((T20, num_global), dtype=np.float32)
    norm = np.zeros((T20, 1), dtype=np.float32)
    for w in range(num_windows):
        f0 = int(window_starts[w])
        loc = local[w].astype(np.float32)  # (F, L)
        glob = d20[f0 : f0 + num_frames]  # (F, K)
        score = loc.T @ glob  # (L, K) soft co-activity
        rows, cols = linear_sum_assignment(-score)
        assigned = np.full(num_local, -1)
        for r, c in zip(rows, cols):
            if score[r, c] > 0.5:  # at least ~0.5 s of shared activity in the window
                assigned[r] = c
        for k in range(num_global):
            mask = assigned != k
            other = loc[:, mask].max(axis=1) if mask.any() else np.zeros(num_frames, np.float32)
            if reduce == "max":
                np.maximum(out[f0 : f0 + num_frames, k], other, out=out[f0 : f0 + num_frames, k])
            else:
                out[f0 : f0 + num_frames, k] += other
        norm[f0 : f0 + num_frames] += 1
    if reduce != "max":
        out /= np.maximum(norm, 1)
    return out


def foreign_signals(diar: np.ndarray, osd: dict[str, np.ndarray], reduce: str = "max") -> dict[str, np.ndarray]:
    """All detector signals on the 10 ms grid, each shaped ``(T, K)`` (one column per target speaker)."""
    T, K = diar.shape
    others = np.stack([np.delete(diar, k, axis=1).max(axis=1) for k in range(K)], axis=1)
    multi = upsample(osd["counts"][:, 2], T)
    dz_other = upsample(diarizen_other_activity(diar, osd["local"], osd["window_starts"], reduce), T)
    return {
        "nemotron_other": others,
        "diarizen_multi": np.repeat(multi[:, None], K, axis=1),
        "diarizen_other": dz_other,
    }


def foreign_mask(signals: dict[str, np.ndarray], thresholds: dict[str, float], k: int) -> np.ndarray:
    """Union of every enabled detector for target speaker ``k`` (threshold ``None`` disables one)."""
    mask = np.zeros(next(iter(signals.values())).shape[0], dtype=bool)
    for name, th in thresholds.items():
        if th is not None:
            mask |= signals[name][:, k] > th
    return mask


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` index ranges where ``mask`` is True."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _split_long(start: int, end: int, own: np.ndarray, max_len: int, min_pause: int) -> list[tuple[int, int]]:
    """Recursively split ``[start, end)`` at its longest internal pause until each part fits ``max_len``."""
    if end - start <= max_len:
        return [(start, end)]
    gaps = [(b - a, a, b) for a, b in _runs(~own[start:end]) if a > 0 and b < end - start and b - a >= min_pause]
    if not gaps:
        return []  # no safe cut point without slicing through speech
    _, a, b = max(gaps)
    return _split_long(start, start + a, own, max_len, min_pause) + _split_long(
        start + b, end, own, max_len, min_pause
    )


def _cuts_speaker_gap(r0: int, r1: int, own: np.ndarray, f: dict) -> list[tuple[int, int, int, int]]:
    """``(s0, s1, c0, c1)`` speech and clip bounds, cutting only at gaps in ``k``'s Nemotron activity."""
    margin, max_len, max_pause, split_pause = f["margin"], f["max_len"], f["max_pause"], f["split_pause"]
    # Speech running into either edge of a clean run started or ends inside a
    # foreign zone, so cutting there would slice through a word: drop it.
    islands = [(r0 + a, r0 + b) for a, b in _runs(own[r0:r1]) if a >= margin and (r1 - r0) - b >= margin]
    if not islands:
        return []
    # Merge islands separated by short pauses into utterances.
    groups: list[list[int]] = [list(islands[0])]
    for a, b in islands[1:]:
        gap = a - groups[-1][1]
        if gap > max_pause or (b - groups[-1][0] > max_len and gap >= split_pause):
            groups.append([a, b])
        else:
            groups[-1][1] = b
    return [
        (s0, s1, s0 - margin, s1 + margin)
        for g0, g1 in groups
        for s0, s1 in _split_long(g0, g1, own, max_len, split_pause)
    ]


def _cuts_acoustic(
    r0: int, r1: int, own: np.ndarray, quiet: np.ndarray, barrier: np.ndarray, f: dict
) -> list[tuple[int, int, int, int]]:
    """``(s0, s1, c0, c1)`` speech and clip bounds, cutting at acoustic pauses or gaps in ``k``'s activity.

    A run that ends where a pause begins (or starts where one ends) has a valid
    boundary there even though the pause itself is foreign: the clip stops right as
    the audio falls silent, without taking any flagged frame.
    """
    margin, max_len, max_pause = f["margin"], f["max_len"], f["max_pause"]
    n = r1 - r0
    left_at_pause = r0 > 0 and barrier[r0 - 1]
    right_at_pause = r1 < len(barrier) and barrier[r1]
    own_r, quiet_r = own[r0:r1], quiet[r0:r1]
    cut = np.zeros(n, dtype=bool)
    for a, b in _runs(quiet_r):
        if b - a >= f["min_pause"]:
            cut[a:b] = True
    for a, b in _runs(~own_r):
        if b - a >= f["split_pause"]:
            cut[a:b] = True
    # A run edge is a valid boundary only if k is already silent (or quiet) there.
    for a, b in _runs(quiet_r | ~own_r):
        if (a == 0 or b == n) and b - a >= margin:
            cut[a:b] = True

    # Chunks: stretches between cut regions. One touching a run edge continues into a
    # foreign zone, so its true start/end is unknown: drop it.
    chunks = [
        (a, b)
        for a, b in _runs(~cut)
        if (a > 0 or left_at_pause) and (b < n or right_at_pause) and b - a <= max_len and own_r[a:b].mean() >= 0.5
    ]

    out = []
    i = 0
    while i < len(chunks):
        j = i
        while (
            j + 1 < len(chunks)
            and chunks[j + 1][0] - chunks[j][1] <= max_pause
            and chunks[j + 1][1] - chunks[i][0] <= max_len
        ):
            j += 1
        s0, s1 = chunks[i][0], chunks[j][1]
        # Extend into the surrounding pauses by up to ``margin``, never past them.
        prev_end = chunks[i - 1][1] if i > 0 else 0
        next_start = chunks[j + 1][0] if j + 1 < len(chunks) else n
        lead = min(margin, (s0 - prev_end) // 2 if i > 0 else s0)
        trail = min(margin, (next_start - s1) // 2 if j + 1 < len(chunks) else n - s1)
        out.append((r0 + s0, r0 + s1, r0 + s0 - lead, r0 + s1 + trail))
        i = j + 1
    return out


def speech_level_db(energy_db: np.ndarray, diar: np.ndarray, speech_threshold: float = 0.5) -> float:
    """Typical loud-speech energy: the 90th percentile of energy while anyone is speaking."""
    active = diar.max(axis=1) > speech_threshold
    return float(np.percentile(energy_db[: len(diar)][active[: len(energy_db)]], 90))


def pause_masks(
    diar: np.ndarray, energy_db: np.ndarray | None, cfg: SelectConfig
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """``(quiet, barrier)`` on the 10 ms grid: frames well below the speech level, and the
    quiet runs long enough to count as pauses. Both are ``None`` in ``speaker_gap`` mode."""
    if cfg.cut_mode == "speaker_gap":
        return None, None
    if cfg.cut_mode != "acoustic":
        raise ValueError(f"Unknown cut_mode {cfg.cut_mode!r}")
    if energy_db is None:
        raise ValueError("cut_mode='acoustic' needs energy_db (see audio.frame_energy_db)")
    T = len(diar)
    energy = np.full(T, -100.0, dtype=np.float32)
    energy[: min(T, len(energy_db))] = energy_db[:T]
    quiet = energy < speech_level_db(energy, diar, cfg.speech_threshold) + cfg.pause_db
    barrier = np.zeros(T, dtype=bool)
    min_pause = max(1, int(round(cfg.min_pause / HOP)))
    for a, b in _runs(quiet):
        if b - a >= min_pause:
            barrier[a:b] = True
    return quiet, barrier


def _trim_blur_at_pauses(
    foreign: np.ndarray, signals: dict[str, np.ndarray], cfg: SelectConfig, k: int, barrier: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove detector smear around pauses from the raw foreign mask.

    Detector outputs fade in and out over a few frames, so a strongly detected event
    right next to a pause smears into it and can leave a short, weak fragment on the
    far side. A pause directly bordered by a strong detection *outside* it is a
    "smear pause". Then:

    * weak frames inside it are cleared (strong ones stay foreign: a soft interjection
      can be quiet enough to pass for silence);
    * a weak fragment (at most ``blur_max_duration`` long, peak at most
      ``blur_max_strength`` times the threshold) on its far side is cleared.

    Detections in or around any other pause are left alone: an isolated faint
    detection still counts. Returns the trimmed mask and the smear-pause frames.
    """
    max_len = int(round(cfg.blur_max_duration / HOP))
    # Strength of each frame relative to its threshold (1.0 = just foreign).
    strength = np.zeros(len(foreign), dtype=np.float32)
    for name, th in cfg.thresholds().items():
        if th is not None:
            np.maximum(strength, signals[name][:, k] / th, out=strength)

    outside = foreign & ~barrier
    runs = _runs(outside)
    weak = [b - a <= max_len and strength[a:b].max() <= cfg.blur_max_strength for a, b in runs]
    strong_end = {b for (a, b), w in zip(runs, weak) if not w}
    strong_start = {a for (a, b), w in zip(runs, weak) if not w}

    out = foreign.copy()
    smear = np.zeros(len(foreign), dtype=bool)
    pause_by_end, pause_by_start = {}, {}
    for pa, pb in _runs(barrier):
        if pa in strong_end or pb in strong_start:
            smear[pa:pb] = True
            out[pa:pb] &= strength[pa:pb] > cfg.blur_max_strength
            pause_by_end[pb] = pa
            pause_by_start[pa] = pb

    for (a, b), w in zip(runs, weak):
        if w and (a in pause_by_end or b in pause_by_start):
            out[a:b] = False
    return out, smear


def padded_foreign(
    signals: dict[str, np.ndarray], cfg: SelectConfig, k: int, barrier: np.ndarray | None = None
) -> np.ndarray:
    """Frames where a voice other than ``k`` may be audible, including the safety pad.

    The pad absorbs blurry detection edges in time. With a ``barrier`` (acoustic mode):

    * the pad around detections outside pauses stops at pauses, so a pause stays
      usable as a cut point right next to an interjection;
    * detections inside a smear pause (the fading edge of an event just outside it)
      get no pad of their own, since the event's own pad already covers its side;
    * detections inside any other pause get the full pad in every direction: the
      voice may be what ends the pause.
    """
    foreign = foreign_mask(signals, cfg.thresholds(), k)
    pad = int(round(cfg.pad / HOP))
    if barrier is None:
        return binary_dilation(foreign, iterations=pad) if pad else foreign
    foreign, smear = _trim_blur_at_pauses(foreign, signals, cfg, k, barrier)
    if not pad:
        return foreign
    outside = binary_dilation(foreign & ~barrier, iterations=pad, mask=~barrier)
    in_pause = binary_dilation(foreign & barrier & ~smear, iterations=pad)
    return foreign | outside | in_pause


def select_segments(
    diar: np.ndarray,
    osd: dict[str, np.ndarray],
    words: list[Word] | None = None,
    cfg: SelectConfig | None = None,
    signals: dict[str, np.ndarray] | None = None,
    energy_db: np.ndarray | None = None,
) -> list[Segment]:
    cfg = cfg or SelectConfig()
    T, num_speakers = diar.shape
    signals = signals if signals is not None else foreign_signals(diar, osd)

    sec = lambda s: max(1, int(round(s / HOP)))  # noqa: E731
    min_len = sec(cfg.min_duration)
    frames = {
        "max_len": sec(cfg.max_duration),
        "split_pause": sec(cfg.split_pause),
        "max_pause": sec(cfg.max_internal_pause),
        "margin": sec(cfg.edge_margin),
        "min_pause": sec(cfg.min_pause),
    }

    quiet, barrier = pause_masks(diar, energy_db, cfg)

    segments: list[Segment] = []
    for k in range(num_speakers):
        own = diar[:, k] > cfg.speech_threshold
        if own.sum() * HOP < max(cfg.min_speaker_seconds, cfg.min_duration):
            continue
        foreign = padded_foreign(signals, cfg, k, barrier)

        for r0, r1 in _runs(~foreign):
            if cfg.cut_mode == "acoustic":
                cuts = _cuts_acoustic(r0, r1, own, quiet, barrier, frames)
            else:
                cuts = _cuts_speaker_gap(r0, r1, own, frames)
            for s0, s1, c0, c1 in cuts:
                if c1 - c0 < min_len:
                    continue
                seg = Segment(
                    speaker=k,
                    start=c0 * HOP,
                    end=c1 * HOP,
                    max_nemotron_other=float(signals["nemotron_other"][c0:c1, k].max()),
                    max_diarizen_multi=float(signals["diarizen_multi"][c0:c1, k].max()),
                    max_diarizen_other=float(signals["diarizen_other"][c0:c1, k].max()),
                    mean_speaker_prob=float(diar[s0:s1, k].mean()),
                )
                if words is not None:
                    seg.text, seg.num_words, seg.cut_words = text_between(words, seg.start, seg.end)
                    if seg.num_words < cfg.min_words:
                        continue
                segments.append(seg)

    segments.sort(key=lambda s: s.start)
    return segments


def select_whole(
    diar: np.ndarray,
    osd: dict[str, np.ndarray],
    cfg: SelectConfig | None = None,
    signals: dict[str, np.ndarray] | None = None,
) -> tuple[Segment | None, str]:
    """One clip spanning all of a recording's speech, or ``None`` with the reason it was rejected.

    Used for recordings whose transcript covers the whole file (no timings to cut by). The
    recording is kept only if no detector flags a second voice anywhere within ``pad`` of
    the main speaker's speech. The clip runs from the first to the last speech frame,
    widened by ``edge_margin`` on both sides (within the file).
    """
    cfg = cfg or SelectConfig()
    T = diar.shape[0]
    signals = signals if signals is not None else foreign_signals(diar, osd)
    active = diar > cfg.speech_threshold
    if not active.any():
        return None, "no speech detected"
    k = int(active.sum(axis=0).argmax())
    idx = np.flatnonzero(active[:, k])
    margin = int(round(cfg.edge_margin / HOP))
    c0, c1 = max(0, int(idx[0]) - margin), min(T, int(idx[-1]) + 1 + margin)
    duration = (c1 - c0) * HOP
    if duration < cfg.min_duration:
        return None, f"too short ({duration:.1f} s)"
    if duration > cfg.max_duration:
        return None, f"too long ({duration:.1f} s > max_duration)"
    pad = int(round(cfg.pad / HOP))
    if foreign_mask(signals, cfg.thresholds(), k)[max(0, c0 - pad): c1 + pad].any():
        return None, "second voice detected"
    seg = Segment(
        speaker=k,
        start=c0 * HOP,
        end=c1 * HOP,
        max_nemotron_other=float(signals["nemotron_other"][c0:c1, k].max()),
        max_diarizen_multi=float(signals["diarizen_multi"][c0:c1, k].max()),
        max_diarizen_other=float(signals["diarizen_other"][c0:c1, k].max()),
        mean_speaker_prob=float(diar[c0:c1, k][active[c0:c1, k]].mean()),
    )
    return seg, "kept"
