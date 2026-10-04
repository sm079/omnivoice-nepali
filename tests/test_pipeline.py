"""The CPU part of the pipeline end to end, on a synthetic recording with stand-in model outputs.

ingest -> select -> export (with loudness normalisation) -> manifest -> splits, checking the
files each step hands to the next. Diarization and overlap detection are replaced by
arrays describing the synthetic audio, so no model is downloaded.
"""

import json
from pathlib import Path

import numpy as np
import pyloudnorm
import pytest
import soundfile as sf

from nepvoice import config
from nepvoice.ingest import ingest
from nepvoice.pipeline import STEPS, run, select_steps
from nepvoice.prep import manifest, process
from nepvoice.train import splits
from nepvoice.train.paths import TrainPaths
from nepvoice.workspace import find_workspaces


@pytest.fixture()
def prepared(tmp_path: Path, make_recording, fake_models):
    inbox, work = tmp_path / "in", tmp_path / "work"
    make_recording(inbox, "loud", amp=0.9)
    make_recording(inbox, "quiet", amp=0.02)
    ingest(inbox, work)
    cfg = config.load([], ["select.min_speaker_seconds=5", "manifest.dev_fraction=0.5", "manifest.speaker_id=false",
                           "export.loudness_normalize=true"])
    for ws in find_workspaces(work).values():
        fake_models(ws)
    return work, cfg


def test_select_export_manifest_splits(prepared):
    work, cfg = prepared
    for ws in find_workspaces(work).values():
        segments = process.select(ws, cfg.select)
        assert len(segments) >= 2 and all(s.text.startswith("शब्द") and s.duration <= 20 for s in segments)
        assert process.selection_done(ws, cfg.select)
        assert process.export(ws, cfg.export) == len(segments)
        assert process.export_done(ws, cfg.export)
        assert not ws.source_wav.exists()  # deleted after export by default

        for clip in sorted(ws.clips.glob("*.flac")):
            audio, sr = sf.read(clip, dtype="float32")
            assert sr == 24000
            loudness = pyloudnorm.Meter(sr).integrated_loudness(audio.astype(np.float64))
            assert abs(loudness - cfg.export.loudness_lufs) < 0.5, (clip.name, loudness)

    stats = manifest.write(work, cfg.manifest)
    assert stats["recordings"] == 2 and stats["train"]["clips"] and stats["dev"]["clips"]
    train_rows = splits.read_jsonl(work / "dataset" / "train.jsonl")
    dev_rows = splits.read_jsonl(work / "dataset" / "dev.jsonl")
    assert {r["recording_id"] for r in train_rows}.isdisjoint({r["recording_id"] for r in dev_rows})
    assert all(Path(r["audio_path"]).is_absolute() and r["language_id"] == "npi" for r in train_rows)

    paths = TrainPaths.of(work)
    split_stats = splits.split_manifests(work / "dataset" / "train.jsonl", work / "dataset" / "dev.jsonl", None, paths)
    assert split_stats["train_female"]["clips"] == 0  # no registry: no voice classes
    data_config = json.loads(splits.write_data_config(paths).read_text())
    assert len(data_config["train"]) == 1 and len(data_config["dev"]) == 1  # empty split left out


def test_only_listed_speakers_of_a_recording_are_kept(prepared):
    work, cfg = prepared
    for ws in find_workspaces(work).values():
        process.select(ws, cfg.select)
        process.export(ws, cfg.export)
    rows, _ = manifest.collect(work)
    absent = 7  # a speaker number the synthetic "loud" recording does not have
    assert f"loud_spk{absent}" not in {r["speaker"] for r in rows}
    only = config.load([], ["manifest.dev_fraction=0", "manifest.speaker_id=false",
                            f"manifest.only_speakers.loud=[{absent}]"]).manifest
    stats = manifest.write(work, only)
    kept = splits.read_jsonl(work / "dataset" / "train.jsonl")
    assert kept and {r["recording_id"] for r in kept} == {"quiet"}  # other recordings keep everyone
    assert stats["skipped"]["other_speaker"] == sum(r["recording_id"] == "loud" for r in rows) > 0

    two = [{"recording_id": "a", "speaker": "a_spk0"}, {"recording_id": "a", "speaker": "a_spk2"},
           {"recording_id": "b", "speaker": "b_spk0"}]
    assert manifest.keep_speakers(two, {"a": [2]}) == (two[1:], 1)


def test_changed_export_settings_trigger_a_new_export(prepared):
    work, cfg = prepared
    ws = find_workspaces(work)["loud"]
    process.select(ws, cfg.select)
    process.export(ws, cfg.export)
    louder = config.load([], ["export.loudness_lufs=-18"]).export
    assert not process.export_done(ws, louder)


def test_default_export_is_peak_normalised_like_omnivoice(prepared):
    work, _ = prepared
    ws = find_workspaces(work)["loud"]
    cfg = config.load()
    process.select(ws, cfg.select)
    assert process.export(ws, cfg.export) > 0
    for clip in ws.clips.glob("*.flac"):
        audio, _ = sf.read(str(clip), dtype="float32")
        assert abs(np.abs(audio).max() - 0.9) < 1e-3, clip.name


def test_run_writes_the_resolved_config_and_steps_are_ordered(prepared):
    work, cfg = prepared
    run(work, cfg, ["select", "export"])
    assert config.load([work / "config.toml"]) == cfg
    assert all(ws.segments.exists() for ws in find_workspaces(work).values())
    assert select_steps("select", "manifest") == ["select", "export", "embed", "speakers", "manifest"]
    assert select_steps(only=["package", "ingest"]) == ["ingest", "package"]
    assert select_steps() == list(STEPS)
    with pytest.raises(ValueError):
        select_steps("train", "select")


def test_training_steps_need_a_run_directory_and_skip_a_packaged_run(prepared, tmp_path: Path, capsys):
    work, cfg = prepared
    with pytest.raises(ValueError, match="--run"):
        run(work, cfg, ["splits"])
    done = tmp_path / "run1"
    done.mkdir()
    (done / "adapter_model.safetensors").write_bytes(b"x")
    (done / "adapter_config.json").write_text("{}")
    run(work, cfg, ["splits", "tokenize", "train"], run_dir=done)
    assert "packaged run" in capsys.readouterr().out
    assert not (done / "scratch").exists() and not (work / "config.toml").exists()
