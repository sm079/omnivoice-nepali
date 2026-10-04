"""Transcript formats: json3, SRT, WebVTT and plain text."""

import json
from pathlib import Path

from nepvoice.transcripts import cues_to_words, load, parse_cues, text_between


def test_json3_words_end_at_the_next_word(tmp_path: Path):
    data = {"events": [
        {"tStartMs": 1000, "dDurationMs": 3000, "segs": [{"utf8": "नमस्ते"}, {"utf8": " साथी", "tOffsetMs": 600}]},
        {"tStartMs": 5000, "dDurationMs": 1000, "segs": [{"utf8": ">> ठीक"}]},
        {"tStartMs": 6000, "dDurationMs": 10},  # no segs
    ]}
    p = tmp_path / "a.json3"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    t = load(p)
    assert t.timed and t.text == "नमस्ते साथी ठीक"
    w = t.words
    assert (w[0].start, w[0].end) == (1.0, 1.6)
    assert w[1].end == 2.6  # capped at MAX_WORD_DURATION, not running to the next event
    assert w[2].start == 5.0 and w[2].text == "ठीक"


def test_srt_cues_spread_over_words(tmp_path: Path):
    p = tmp_path / "a.srt"
    p.write_text("1\n00:00:01,000 --> 00:00:03,000\nएक दुई\n\n2\n00:00:04,500 --> 00:00:05,000\n<i>तीन</i>\n",
                 encoding="utf-8")
    t = load(p)
    assert [w.text for w in t.words] == ["एक", "दुई", "तीन"]
    assert t.words[0].start == 1.0 and 1.0 < t.words[1].start < 3.0
    assert t.words[2].start == 4.5
    assert text_between(t.words, 0.5, 3.5)[:2] == ("एक दुई", 2)


def test_vtt_rolling_captions_are_not_doubled():
    vtt = ("WEBVTT\n\n"
           "00:00:01.000 --> 00:00:02.000\nएक<00:00:01.500><c> दुई</c>\n\n"
           "00:00:02.000 --> 00:00:03.000\nएक दुई\nतीन\n\n"
           "01:00:03.000 --> 01:00:04.000\nतीन\nचार\n")
    cues = parse_cues(vtt)
    assert [c[2] for c in cues] == ["एक दुई", "तीन", "चार"]
    assert cues[2][0] == 3603.0
    assert [w.text for w in cues_to_words(cues)] == ["एक", "दुई", "तीन", "चार"]


def test_plain_text_has_no_timings(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("﻿नमस्ते,\n  साथी। \n", encoding="utf-8")
    t = load(p)
    assert not t.timed and t.text == "नमस्ते, साथी।"
