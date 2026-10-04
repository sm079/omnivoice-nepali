"""Voice Studio API without loading any TTS model."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from tools.voice_studio.server import Studio, create_app


@pytest.fixture()
def studio(tmp_path: Path):
    work = tmp_path / "work" / "dataset"
    work.mkdir(parents=True)
    clip = tmp_path / "clip.wav"
    sf.write(clip, np.random.default_rng(0).normal(0, 0.1, 8 * 16000).astype(np.float32), 16000)
    row = lambda i, spk: {"id": f"c{i}", "audio_path": str(clip), "text": "नमस्ते", "duration": 8.0, "speaker": spk}  # noqa: E731
    dev = "\n".join(json.dumps(row(i, s)) for i, s in enumerate(["S1", "S2"]))
    (work / "dev.jsonl").write_text(dev, encoding="utf-8")
    (work / "train.jsonl").write_text(json.dumps(row(9, "S2")), encoding="utf-8")
    s = Studio({"base": "x", "run1": "y"}, tmp_path / "work", tmp_path / "out", tmp_path / "bench")
    return s


def test_config_and_library(studio):
    c = TestClient(create_app(studio))
    cfg = c.get("/api/config").json()
    assert [m["name"] for m in cfg["models"]] == ["base", "run1"]
    assert cfg["ready"] is False
    lib = cfg["library"]
    assert [x["speaker"] for x in lib] == ["S1", "S2"]  # unseen voices first
    assert lib[0]["seen"] is False and lib[1]["seen"] is True
    assert "path" not in lib[0]
    assert c.get(lib[0]["url"]).status_code == 200
    ref = c.post("/api/reference/library", json={"clip_id": "c0"}).json()
    assert ref["text"] == "नमस्ते" and c.get(ref["url"]).status_code == 200


def test_generate_refuses_until_models_are_loaded(studio):
    c = TestClient(create_app(studio))
    r = c.post("/api/generate", json={"ref_id": "nope", "text": "नमस्ते"})
    assert r.status_code == 503


def test_blind_vote_reveals_and_tallies(studio):
    studio.rounds["r1"] = {
        "round_id": "r1", "time": "", "ref_id": "x", "ref_source": "library", "ref_text": "", "text": "t",
        "seed": 0, "blind": True, "prompt_time": 0.1, "vote": None,
        "slots": [{"slot": "A", "model": "run1", "seconds": 1, "gen_time": 1},
                  {"slot": "B", "model": "base", "seconds": 1, "gen_time": 1}],
    }
    c = TestClient(create_app(studio))
    hidden = c.get("/api/rounds").json()[0]
    assert all(s["model"] is None for s in hidden["slots"])  # names stay hidden until the vote
    out = c.post("/api/vote", json={"round_id": "r1", "best": "A"}).json()
    assert [s["model"] for s in out["round"]["slots"]] == ["run1", "base"]
    assert out["tally"]["models"]["run1"]["wins"] == 1 and out["tally"]["models"]["base"]["wins"] == 0
    assert out["tally"]["pairwise"]["run1>base"]["wins"] == 1
    assert c.post("/api/vote", json={"round_id": "r1", "best": "B"}).status_code == 400  # one vote per round
