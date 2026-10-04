"""Cross-recording speaker matching and per-speaker capping, on synthetic voiceprints."""

import numpy as np

from nepvoice.prep.manifest import cap_per_speaker
from nepvoice.prep.speakers import (
    SpeakerConfig,
    _stable_ids,
    cluster,
    voice_from_probability,
)

RNG = np.random.default_rng(0)


def voice(base, noise=0.25):
    v = base + noise * RNG.standard_normal(base.shape)
    return v / np.linalg.norm(v)


def prints(spec):
    """spec: list of (true person, recording, seconds)."""
    people = {p: RNG.standard_normal(192) for p, _, _ in spec}
    return [
        {"local_speaker": f"{v}_spk{i}", "recording_id": v, "seconds": s, "clips": 10, "centroid": voice(people[p])}
        for i, (p, v, s) in enumerate(spec)
    ]


def test_same_person_across_recordings_merges_and_others_stay_apart():
    spec = [("main", "v1", 900), ("a", "v1", 400), ("main", "v2", 800), ("a", "v2", 300),
            ("main", "v3", 700), ("b", "v3", 500)]
    labels = cluster(prints(spec), SpeakerConfig())
    main = {labels[i] for i, (p, _, _) in enumerate(spec) if p == "main"}
    a = {labels[i] for i, (p, _, _) in enumerate(spec) if p == "a"}
    assert len(main) == 1 and len(a) == 1 and len(set(labels)) == 3
    assert labels[0] == 0  # "main" has the most speech, so gets the first id


def test_too_little_speech_is_never_matched():
    spec = [("main", "v1", 900), ("main", "v2", 5)]
    labels = cluster(prints(spec), SpeakerConfig(min_voiceprint_seconds=20))
    assert labels[0] != labels[1]


def test_cap_spreads_a_speaker_over_recordings():
    rows = [{"id": f"{v}_{i}", "speaker": "S0001", "recording_id": v, "duration": 10.0}
            for v in ("a", "b", "c", "d") for i in range(100)]
    rows += [{"id": f"g_{i}", "speaker": "S0002", "recording_id": "a", "duration": 10.0} for i in range(5)]
    kept, capped = cap_per_speaker(rows, max_hours=0.5, seed=1)  # 1800 s = 180 clips
    frequent = [r for r in kept if r["speaker"] == "S0001"]
    assert len(frequent) == 180
    per_recording = {v: sum(r["recording_id"] == v for r in frequent) for v in "abcd"}
    assert max(per_recording.values()) - min(per_recording.values()) <= 1
    assert sum(r["speaker"] == "S0002" for r in kept) == 5  # under the cap: untouched
    assert set(capped) == {"S0001"}
    assert sum(r["duration"] for r in frequent) <= 1800


def test_cap_is_never_exceeded():
    rows = [{"id": f"c{i}", "speaker": "S0001", "recording_id": "ab"[i % 2], "duration": 7.0} for i in range(100)]
    kept, capped = cap_per_speaker(rows, max_hours=100 / 3600)  # 100 s: 14 clips of 7 s fit, a 15th would not
    assert len(kept) == 14 and sum(r["duration"] for r in kept) <= 100 and "S0001" in capped


def test_stable_ids_survive_new_recordings():
    prints = [{"local_speaker": n, "seconds": s} for n, s in
              [("v1_spk0", 100), ("v1_spk1", 300), ("v2_spk0", 120), ("v3_spk0", 500)]]
    # one person = v1_spk0 + v2_spk0 (cluster 0); v1_spk1 (1); a new, larger v3_spk0 (2)
    labels = [0, 1, 0, 2]
    previous = {"v1_spk0": "S0002", "v1_spk1": "S0001", "v2_spk0": "S0002"}
    assert _stable_ids(prints, labels, previous) == ["S0002", "S0001", "S0002", "S0003"]


def test_voice_classes():
    assert voice_from_probability(0.9) == "female"
    assert voice_from_probability(0.1) == "male"
    assert voice_from_probability(0.5) == "unclear"
    assert voice_from_probability(float("nan")) == "unknown"
