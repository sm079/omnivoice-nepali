"""Layered configuration: files, overrides, validation, and the documented defaults."""

from pathlib import Path

import pytest

from nepvoice import config
from nepvoice.config import Config, ConfigError

CONFIGS = Path(__file__).parent.parent / "configs"


def test_default_toml_documents_the_code_defaults():
    assert config.load([CONFIGS / "default.toml"]) == Config()


def test_every_shipped_config_loads():
    for path in CONFIGS.glob("*.toml"):
        config.load([path])


def test_files_layer_and_overrides_win(tmp_path: Path):
    a = tmp_path / "a.toml"
    a.write_text("[select]\npad = 0.5\n[train.omnivoice]\nsteps = 100\n")
    b = tmp_path / "b.toml"
    b.write_text("[select]\npad = 0.7\n")
    cfg = config.load([a, b], ["export.loudness_lufs=-20", "train.final_checkpoint=best"])
    assert cfg.select.pad == 0.7
    assert cfg.train.omnivoice["steps"] == 100
    assert cfg.train.omnivoice["lora_r"] == 16  # untouched keys of the table stay
    assert cfg.export.loudness_lufs == -20.0 and isinstance(cfg.export.loudness_lufs, float)
    assert cfg.train.final_checkpoint == "best"


def test_defaults_are_not_shared_between_configs():
    a = config.load([], ["train.omnivoice.steps=5"])
    assert a.train.omnivoice["steps"] == 5
    assert Config().train.omnivoice["steps"] == 3000


@pytest.mark.parametrize("override", ["selct.pad=1", "select.padd=1", "select.trim_breaths=1", "select.pad=fast",
                                      "manifest.only_speakers.x=2"])
def test_mistakes_are_errors(override):
    with pytest.raises(ConfigError):
        config.load([], [override])


def test_detectors_can_be_switched_off():
    cfg = config.load([], ['select.nemotron_other_threshold="off"'])
    assert cfg.select.nemotron_other_threshold is None
    assert cfg.select.thresholds()["nemotron_other"] is None


def test_resolved_config_round_trips(tmp_path: Path):
    cfg = config.load([], ['select.diarizen_multi_threshold="off"', "train.omnivoice.steps=7"])
    out = tmp_path / "resolved.toml"
    out.write_text(config.to_toml(cfg), encoding="utf-8")
    assert config.load([out]) == cfg


def test_recording_ids_that_are_not_bare_toml_keys_round_trip(tmp_path: Path):
    cfg = config.load([], ['manifest.only_speakers={"talk.v2" = [1], "नमस्ते" = [0, 2]}'])
    path = tmp_path / "config.toml"
    path.write_text(config.to_toml(cfg), encoding="utf-8")
    assert config.load([path]) == cfg
