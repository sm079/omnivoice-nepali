"""Typed pipeline configuration, loaded from layered TOML files plus ``section.key=value`` overrides.

Every tunable of data preparation and training lives in one :class:`Config`. Defaults
are defined here (``configs/default.toml`` documents them); a TOML file only needs the
keys it changes, later files override earlier ones, and ``--set`` overrides come last.
Unknown sections or keys are errors, so a typo never silently falls back to a default.
"""

from __future__ import annotations

import copy
import re
import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from .prep.selection import SelectConfig
from .prep.speakers import SpeakerConfig


class ConfigError(ValueError):
    pass


@dataclass
class IngestConfig:
    # Look for recordings in subfolders of the input folder too.
    recursive: bool = True


@dataclass
class PrepConfig:
    # Parallel recordings in the CPU-bound select and export stages.
    cpu_workers: int = 3
    # Redo selection and export even when cached with identical settings.
    force: bool = False
    # Keep the full-rate WAV after export (it is decoded again from the source when needed).
    keep_source_wav: bool = False


@dataclass
class ExportConfig:
    # Clip sample rate; 0 keeps the source rate. OmniVoice's audio tokenizer works at 24 kHz.
    sample_rate: int = 24000
    # Scale each clip so its sample peak is this level (0 = off), as OmniVoice's own data
    # loader does with 0.9.
    peak_normalize: float = 0.9
    # Instead (takes precedence): normalise to ``loudness_lufs`` integrated loudness
    # (ITU-R BS.1770), lowering the gain where needed to keep the true peak <= ``peak_dbfs``.
    loudness_normalize: bool = False
    loudness_lufs: float = -27.0
    peak_dbfs: float = -1.0
    # Raised-cosine fade at both ends, and digital silence added before and after each clip.
    fade_ms: float = 0.0
    pad_ms: float = 0.0


@dataclass
class ManifestConfig:
    min_duration: float = 3.0
    max_duration: float = 30.0
    # Share of recordings held out (whole) as the dev set.
    dev_fraction: float = 0.02
    seed: int = 0
    # Cap any one person at this many hours, spread across their recordings; 0 = no cap.
    max_hours_per_speaker: float = 0.0
    # Recognise speakers across recordings (global ids, voice classes for balancing).
    speaker_id: bool = True
    # Per recording id, the only speakers to keep, by index as in the segment viewer's
    # "Speaker N" (e.g. {"talk-01" = [2]}); other recordings keep everyone.
    only_speakers: dict[str, list[int]] = field(default_factory=dict)


def default_omnivoice() -> dict[str, Any]:
    """OmniVoice ``TrainingConfig`` fields: its LoRA recipe, sized for a single small GPU."""
    return {
        "llm_name_or_path": "Qwen/Qwen3-0.6B",
        "audio_vocab_size": 1025,
        "audio_mask_id": 1024,
        "num_audio_codebook": 8,
        "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
        "drop_cond_ratio": 0.1,
        "prompt_ratio_range": [0.0, 0.3],
        "mask_ratio_range": [0.0, 1.0],
        "language_ratio": 0.8,
        "use_pinyin_ratio": 0.0,
        "instruct_ratio": 0.0,
        "only_instruct_ratio": 0.0,
        "resume_from_checkpoint": None,
        "init_from_checkpoint": "k2-fsa/OmniVoice",
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_bias": "none",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_modules_to_save": ["audio_embeddings", "audio_heads"],
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "steps": 3000,
        "seed": 42,
        "lr_scheduler_type": "cosine",
        "warmup_type": "ratio",
        "warmup_ratio": 0.03,
        "warmup_steps": 0,
        "attn_implementation": "sdpa",
        "max_sample_tokens": 2000,
        "min_sample_tokens": 50,
        "max_batch_size": 16,
        "batch_tokens": 2048,
        "gradient_accumulation_steps": 4,
        "num_workers": 2,
        "mixed_precision": "bf16",
        "allow_tf32": True,
        "logging_steps": 25,
        "eval_steps": 250,
        "save_steps": 500,
        "keep_last_n_checkpoints": -1,
    }


@dataclass
class TrainConfig:
    # How often each clip of a female voice is seen per pass (they are often a small share
    # of conversational recordings); 1 = no oversampling.
    female_repeat: int = 2
    # Recompute activations in the backward pass: much less GPU memory for a bit more compute.
    # Without it a small GPU spills into system memory and runs several times slower.
    gradient_checkpointing: bool = True
    # OmniVoice's tokenizer peak-normalises every clip to 0.9, as upstream training does.
    # Turn off to train on the exported levels (e.g. loudness-normalised clips).
    peak_normalize: bool = True
    tokenize_jobs_per_gpu: int = 2
    tokenize_loader_workers: int = 4
    # Dev loss of the base model and every checkpoint, all on the same random masks.
    evaluate: bool = True
    eval_seed: int = 0
    # Checkpoint kept as the run's final adapter (adapter_model.safetensors): "last",
    # "best" (lowest dev loss; needs evaluate) or a checkpoint name such as "checkpoint-2000".
    final_checkpoint: str = "last"
    omnivoice: dict[str, Any] = field(default_factory=default_omnivoice)


@dataclass
class Config:
    ingest: IngestConfig = field(default_factory=IngestConfig)
    prep: PrepConfig = field(default_factory=PrepConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    speakers: SpeakerConfig = field(default_factory=SpeakerConfig)
    manifest: ManifestConfig = field(default_factory=ManifestConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)


# SelectConfig thresholds accept None (detector disabled); TOML has no null, so "off" is used.
_NULLABLE = {("select", "nemotron_other_threshold"), ("select", "diarizen_multi_threshold"),
             ("select", "diarizen_other_threshold")}


def _coerce(section: str, key: str, value: Any, default: Any) -> Any:
    if (section, key) in _NULLABLE and value in (None, "off", False):
        return None
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{section}.{key} must be true or false, got {value!r}")
        return value
    if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(default, (int, float, str)) and not isinstance(value, type(default)):
        raise ConfigError(f"{section}.{key} must be {type(default).__name__}, got {value!r}")
    return value


def _apply(cfg: Config, data: dict, origin: str) -> None:
    for section, values in data.items():
        target = getattr(cfg, section, None)
        if not is_dataclass(target):
            valid = ", ".join(f.name for f in fields(Config))
            raise ConfigError(f"{origin}: unknown section [{section}]; valid: {valid}")
        if not isinstance(values, dict):
            raise ConfigError(f"{origin}: [{section}] must be a table")
        names = {f.name for f in fields(target)}
        for key, value in values.items():
            if key not in names:
                raise ConfigError(f"{origin}: unknown key {section}.{key}; valid: {', '.join(sorted(names))}")
            current = getattr(target, key)
            if isinstance(current, dict):  # train.omnivoice: merged key by key
                if not isinstance(value, dict):
                    raise ConfigError(f"{origin}: {section}.{key} must be a table")
                current.update(value)
            else:
                setattr(target, key, _coerce(section, key, value, current))


def parse_override(text: str) -> dict:
    """``"select.pad=0.5"`` -> ``{"select": {"pad": 0.5}}``. Values are TOML; bare words are strings."""
    if "=" not in text:
        raise ConfigError(f"override {text!r} is not section.key=value")
    path, raw = (s.strip() for s in text.split("=", 1))
    parts = path.split(".")
    if len(parts) < 2:
        raise ConfigError(f"override {text!r} needs a section, e.g. select.pad=0.5")
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw
    out: dict = {}
    node = out
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    return out


def load(files: list[Path] | tuple[Path, ...] = (), overrides: list[str] | tuple[str, ...] = ()) -> Config:
    cfg = Config()
    for path in files:
        try:
            data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{path}: {e}") from e
        _apply(cfg, data, str(path))
    for text in overrides:
        _apply(cfg, parse_override(text), f"--set {text}")
    for rid, keep in cfg.manifest.only_speakers.items():
        if not isinstance(keep, list) or not all(isinstance(k, int) and not isinstance(k, bool) for k in keep):
            raise ConfigError(f"manifest.only_speakers.{rid} must be a list of speaker numbers, got {keep!r}")
    return cfg


def to_toml(cfg: Config) -> str:
    """The resolved configuration as TOML (``None`` thresholds written as ``"off"``)."""

    def value(v: Any) -> str:
        if v is None:
            return '"off"'
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if isinstance(v, float) and v == float("inf"):
            return "inf"
        if isinstance(v, list):
            return "[" + ", ".join(value(x) for x in v) + "]"
        return repr(v)

    def key(k: str) -> str:  # bare TOML keys are ASCII letters, digits, "_" and "-" only
        return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else value(k)

    lines: list[str] = []
    for section, values in copy.deepcopy(cfg.to_dict()).items():
        nested = {k: v for k, v in values.items() if isinstance(v, dict)}
        lines.append(f"[{section}]")
        lines += [f"{k} = {value(v)}" for k, v in values.items() if k not in nested]
        lines.append("")
        for k, table in nested.items():
            lines.append(f"[{section}.{k}]")
            lines += [f"{key(kk)} = {value(vv)}" for kk, vv in table.items() if vv is not None]
            lines.append("")
    return "\n".join(lines)
