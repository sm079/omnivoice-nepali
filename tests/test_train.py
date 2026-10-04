"""Training preparation: balanced manifests, data config, shard paths, checkpoint choice."""

import json
from pathlib import Path

import pytest

from nepvoice import config
from nepvoice.train.paths import TrainPaths
from nepvoice.train.splits import split_manifests, write_data_config
from nepvoice.train.tokenize import windows_safe_manifest
from nepvoice.train.trainer import (
    TrainingStateError,
    base_model_path,
    choose_checkpoint,
    merge,
    package,
    run_identity,
)


def _jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def test_split_by_voice_and_config(tmp_path: Path):
    clip = tmp_path / "a.flac"
    clip.write_bytes(b"x")
    row = lambda i, spk: {"id": f"c{i}", "audio_path": str(clip), "text": "नमस्ते", "duration": 4.0, "speaker": spk}  # noqa: E731
    _jsonl(tmp_path / "train.jsonl", [row(0, "S0001"), row(1, "S0002"), row(2, "S0003")])
    _jsonl(tmp_path / "dev.jsonl", [row(3, "S0002")])
    (tmp_path / "speakers.json").write_text(json.dumps({"speakers": {
        "S0001": {"voice": "male"}, "S0002": {"voice": "female"}, "S0003": {"voice": "unclear"}}}))

    paths = TrainPaths(tmp_path / "w")
    stats = split_manifests(tmp_path / "train.jsonl", tmp_path / "dev.jsonl", tmp_path / "speakers.json", paths)
    assert stats["train_main"]["clips"] == 2  # male + unclear
    assert stats["train_female"]["clips"] == 1
    assert stats["dev"]["clips"] == 1

    cfg = json.loads(write_data_config(paths, female_repeat=3).read_text())
    assert [e["repeat"] for e in cfg["train"]] == [1, 3]
    assert all(e["language_id"] == "npi" for e in cfg["train"] + cfg["dev"])


def test_missing_audio_is_an_error(tmp_path: Path):
    _jsonl(tmp_path / "train.jsonl", [{"id": "c0", "audio_path": str(tmp_path / "gone.flac"),
                                       "text": "x", "duration": 4.0, "speaker": "S0001"}])
    _jsonl(tmp_path / "dev.jsonl", [])
    with pytest.raises(FileNotFoundError):
        split_manifests(tmp_path / "train.jsonl", tmp_path / "dev.jsonl", None, TrainPaths(tmp_path / "w"))


def test_drive_paths_become_file_urls_and_rewrite_is_idempotent(tmp_path: Path):
    lst = tmp_path / "data.lst"
    lst.write_text(
        r"C:\data\tokens\a\shard-000000.tar C:\data\tokens\a\txt.jsonl 12 99.5" "\n"
        r"file:///C:/data/tokens/a/shard-000001.tar C:\data\tokens\a\txt1.jsonl 3 7.0" "\n"
        "/home/x/shard.tar /home/x/t.jsonl 1 1.0\n",
        encoding="utf-8",
    )
    assert windows_safe_manifest(lst)
    lines = lst.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(" ")[0] == "file:C:/data/tokens/a/shard-000000.tar"
    assert lines[0].split(" ")[1] == r"C:\data\tokens\a\txt.jsonl"  # label path untouched
    assert lines[1].split(" ")[0] == "file:C:/data/tokens/a/shard-000001.tar"
    assert lines[2].startswith("/home/x/shard.tar")  # POSIX paths untouched
    assert not windows_safe_manifest(lst)


def test_relative_shard_paths_become_absolute(tmp_path: Path):
    lst = tmp_path / "data.lst"
    lst.write_text("audios/shard-000000.tar txts/shard-000000.jsonl 3 7.0", encoding="utf-8")
    assert windows_safe_manifest(lst, base=tmp_path)
    tar, label, *_ = lst.read_text(encoding="utf-8").split(" ")
    assert tar.endswith("audios/shard-000000.tar") and ("file:" in tar or tar.startswith("/"))
    assert Path(label) == (tmp_path / "txts" / "shard-000000.jsonl").resolve()


def test_checkpoint_choice(tmp_path: Path):
    paths = TrainPaths(tmp_path)
    for step in (500, 1000, 1500):
        (paths.exp / f"checkpoint-{step}").mkdir(parents=True)
    (paths.exp / "checkpoint-notes").mkdir()
    cfg = config.load().train
    assert [p.name for p in paths.checkpoints()] == ["checkpoint-500", "checkpoint-1000", "checkpoint-1500"]
    assert choose_checkpoint(cfg, paths).name == "checkpoint-1500"
    cfg.final_checkpoint = "best"
    scores = {"base": 4.6, "checkpoint-500": 4.5, "checkpoint-1000": 4.4, "checkpoint-1500": 4.45}
    assert choose_checkpoint(cfg, paths, scores).name == "checkpoint-1000"
    with pytest.raises(TrainingStateError):
        choose_checkpoint(cfg, paths, None)
    cfg.final_checkpoint = "checkpoint-500"
    assert choose_checkpoint(cfg, paths).name == "checkpoint-500"


def test_run_identity_ignores_schedule_but_not_hyperparameters(tmp_path: Path):
    paths = TrainPaths(tmp_path)
    base = run_identity(config.load().train, paths)
    assert run_identity(config.load([], ["train.omnivoice.steps=6000", "train.omnivoice.save_steps=100"]).train,
                        paths) == base
    assert run_identity(config.load([], ["train.omnivoice.learning_rate=2e-4"]).train, paths) != base
    assert run_identity(config.load([], ["train.female_repeat=3"]).train, paths) != base


def test_package_keeps_the_adapters_and_deletes_scratch(tmp_path: Path):
    paths = TrainPaths(tmp_path / "run1")
    cache = "C:\\hf\\hub\\models--k2-fsa--OmniVoice\\snapshots\\" + "c5" * 20
    for step in (500, 1000):
        ckpt = paths.exp / f"checkpoint-{step}"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter_model.safetensors").write_bytes(str(step).encode())
        (ckpt / "optimizer.bin").write_bytes(b"state")
        (ckpt / "adapter_config.json").write_text(json.dumps({"r": 16, "base_model_name_or_path": cache}))
    paths.manifests.mkdir(parents=True)

    cfg = config.load().train
    assert package(cfg, paths, {"base": 4.6, "checkpoint-1000": 4.4}) == paths.adapter
    assert sorted(p.name for p in paths.root.iterdir()) == [
        "adapter_config.json", "adapter_model.safetensors", "checkpoint-500.safetensors", "eval.json"]
    assert paths.adapter.read_bytes() == b"1000" and paths.checkpoint_file(500).read_bytes() == b"500"
    adapter_cfg = json.loads(paths.adapter_config.read_text())
    assert adapter_cfg == {"r": 16, "base_model_name_or_path": "k2-fsa/OmniVoice", "revision": "c5" * 20}
    assert paths.packaged and not paths.scratch.exists()


def _packaged(run: Path, revision: str | None = None) -> Path:
    run.mkdir(parents=True)
    (run / "adapter_model.safetensors").write_bytes(b"3000")
    (run / "checkpoint-500.safetensors").write_bytes(b"500")
    cfg = {"r": 16, "base_model_name_or_path": "org/base"} | ({"revision": revision} if revision else {})
    (run / "adapter_config.json").write_text(json.dumps(cfg))
    return run


def test_merge_checks_the_checkpoint_name(tmp_path: Path):
    run = _packaged(tmp_path / "run1")
    with pytest.raises(ValueError, match="checkpoint-2000"):
        merge(run, tmp_path / "out", checkpoint="2000")
    with pytest.raises(FileNotFoundError, match="checkpoint-500"):
        merge(run, tmp_path / "out", checkpoint="checkpoint-3000")
    with pytest.raises(FileNotFoundError, match="not a packaged run"):
        merge(tmp_path, tmp_path / "out")


def test_merge_uses_the_pinned_base_revision(monkeypatch):
    calls = []
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda repo, revision: calls.append((repo, revision)) or f"/cache/{repo}@{revision}")
    assert base_model_path({"base_model_name_or_path": "org/base", "revision": "abc"}) == "/cache/org/base@abc"
    assert base_model_path({"base_model_name_or_path": "org/base"}) == "org/base"
    assert calls == [("org/base", "abc")]


def test_package_closes_log_files_left_open_in_scratch(tmp_path: Path):
    import logging

    paths = TrainPaths(tmp_path / "run1")
    ckpt = paths.exp / "checkpoint-10"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_model.safetensors").write_bytes(b"10")
    (ckpt / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "org/base"}))
    handler = logging.FileHandler(paths.exp / "train.log")
    logging.getLogger("omnivoice.test").addHandler(handler)
    assert not paths.packaged  # adapters not yet written
    package(config.load().train, paths)
    assert paths.packaged and handler not in logging.getLogger("omnivoice.test").handlers


def test_a_run_with_scratch_left_is_not_packaged(tmp_path: Path):
    paths = TrainPaths(tmp_path)
    paths.adapter.write_bytes(b"x")
    paths.adapter_config.write_text("{}")
    paths.scratch.mkdir()
    assert not paths.packaged
