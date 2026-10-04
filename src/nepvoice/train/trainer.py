"""Single-GPU LoRA training, evaluation and merging with OmniVoice's own trainer.

Model, data pipeline and training loop all come from ``omnivoice.training``. Added here:

* optional gradient checkpointing, for GPUs with little memory;
* a fix that lets DataLoader workers start on Windows (``spawn`` cannot pickle a lambda);
* resuming: re-running continues from the newest checkpoint, and a finished run is
  skipped. A run is only resumed with the settings and data it started with;
* dev loss of the base model and every checkpoint on the same random masks;
* packaging a finished run as its LoRA adapters alone (the optimizer state, token shards
  and other scratch files are deleted), and merging an adapter into a standalone model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..config import TrainConfig
from .paths import ADAPTER, ADAPTER_CONFIG, TrainPaths

# Fields that change how often things happen, not what is learned: a run may be resumed
# with different values (``steps`` can be raised to train longer).
_RESUMABLE = {"steps", "resume_from_checkpoint", "logging_steps", "eval_steps", "save_steps",
              "keep_last_n_checkpoints", "num_workers"}


class TrainingStateError(RuntimeError):
    pass


def _make_picklable(loader) -> None:
    """Swap OmniVoice's ``length_fn=lambda s: s["length"]`` for an equivalent picklable one.

    DataLoader workers are started with ``spawn`` on Windows (and macOS), which pickles the
    dataset; a lambda cannot be pickled, so the first batch fails with "Can't get local
    object 'build_dataloaders.<locals>.<lambda>'".
    """
    from operator import itemgetter

    dataset = getattr(loader, "dataset", None)
    fn = getattr(dataset, "length_fn", None)
    if fn is not None and getattr(fn, "__name__", "") == "<lambda>":
        assert fn({"length": 7}) == 7, "unexpected length_fn; update _make_picklable"
        dataset.length_fn = itemgetter("length")


def _llm(model):
    """The Qwen3 backbone, whether or not the model is wrapped by PEFT."""
    inner = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    return inner.llm


def run_identity(cfg: TrainConfig, paths: TrainPaths) -> str:
    """Hash of everything that defines what a run learns: hyperparameters and token shards."""
    stamps = sorted(str(p.read_text(encoding="utf-8")) for p in paths.tokens.glob("*/source.json"))
    data = {
        "omnivoice": {k: v for k, v in cfg.omnivoice.items() if k not in _RESUMABLE},
        "female_repeat": cfg.female_repeat,
        "tokens": stamps,
    }
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def _omnivoice_config(cfg: TrainConfig, paths: TrainPaths, resume: Path | None = None):
    from omnivoice.training.config import TrainingConfig

    values = dict(cfg.omnivoice)
    values["resume_from_checkpoint"] = str(resume) if resume else None
    path = paths.exp / "omnivoice_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")
    config = TrainingConfig.from_json(str(path))
    config.output_dir = str(paths.exp)
    config.data_config = str(paths.data_config)
    return config


def _build(config, gradient_checkpointing: bool = False):
    from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
    from omnivoice.training.trainer import OmniTrainer

    model, tokenizer = build_model_and_tokenizer(config)
    if gradient_checkpointing:
        _llm(model).gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train_loader, eval_loader = build_dataloaders(config, tokenizer)
    for loader in (train_loader, eval_loader):
        if loader is not None:
            _make_picklable(loader)
    return OmniTrainer(model=model, config=config, train_dataloader=train_loader,
                       eval_dataloader=eval_loader, tokenizer=tokenizer)


def train(cfg: TrainConfig, paths: TrainPaths) -> Path:
    """Train (or resume, or skip a finished run); returns the final checkpoint."""
    identity = run_identity(cfg, paths)
    run_file = paths.exp / "run.json"
    checkpoints = paths.checkpoints()
    if checkpoints:
        previous = json.loads(run_file.read_text(encoding="utf-8")).get("identity") if run_file.exists() else None
        if previous != identity:
            raise TrainingStateError(
                f"{paths.exp} holds checkpoints of a run with different settings or data. "
                "Delete that folder to start over, or use another --run directory."
            )
        last = checkpoints[-1]
        if int(last.name.split("-")[-1]) >= cfg.omnivoice["steps"]:
            print(f"training already finished ({last.name})", flush=True)
            return last
        print(f"resuming from {last.name}", flush=True)
    resume = checkpoints[-1] if checkpoints else None

    paths.exp.mkdir(parents=True, exist_ok=True)
    run_file.write_text(json.dumps({
        "identity": identity,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "omnivoice": cfg.omnivoice,
    }, indent=2), encoding="utf-8")
    _build(_omnivoice_config(cfg, paths, resume), cfg.gradient_checkpointing).train()
    return paths.checkpoints()[-1]


def evaluate(cfg: TrainConfig, paths: TrainPaths) -> dict[str, float]:
    """Dev loss of the base model and every checkpoint, each under the same seed; cached in ``eval.json``.

    OmniVoice's loss masks a random share of each clip, so a single pass is noisy;
    reseeding before each model makes every model see the same masks and prompts, which
    makes the numbers comparable. With LoRA's zero-initialised adapters, the freshly
    built model *is* the base model.
    """
    import random

    import numpy as np
    import torch

    data_config = json.loads(paths.data_config.read_text(encoding="utf-8"))
    if not data_config.get("dev"):
        print("no dev set: skipping evaluation (set manifest.dev_fraction > 0)", flush=True)
        return {}
    key = f"{run_identity(cfg, paths)}-seed{cfg.eval_seed}"
    cache = json.loads(paths.eval_results.read_text(encoding="utf-8")) if paths.eval_results.exists() else {}
    results: dict[str, float] = cache.get(key, {})
    todo = [None, *paths.checkpoints()]
    for i, ckpt in enumerate(todo, 1):
        name = "base" if ckpt is None else ckpt.name
        if name in results:
            continue
        trainer = _build(_omnivoice_config(cfg, paths))
        if ckpt is not None:
            trainer.load_checkpoint(str(ckpt))
        random.seed(cfg.eval_seed)
        np.random.seed(cfg.eval_seed)
        torch.manual_seed(cfg.eval_seed)
        results[name] = round(trainer.evaluate()["eval/loss"], 4)
        print(f"evaluate {i}/{len(todo)} {name}: dev loss {results[name]}", flush=True)
        paths.eval_results.write_text(json.dumps({key: results}, indent=2), encoding="utf-8")
        del trainer
        torch.cuda.empty_cache()
    return results


def choose_checkpoint(cfg: TrainConfig, paths: TrainPaths, eval_results: dict[str, float] | None = None) -> Path:
    checkpoints = paths.checkpoints()
    if not checkpoints:
        raise TrainingStateError(f"no checkpoints in {paths.exp}; run the train step first")
    choice = cfg.final_checkpoint
    if choice == "last":
        return checkpoints[-1]
    if choice == "best":
        scored = {c.name: eval_results[c.name] for c in checkpoints if eval_results and c.name in eval_results}
        if not scored:
            raise TrainingStateError("final_checkpoint = \"best\" needs dev-loss results (train.evaluate = true)")
        return paths.exp / min(scored, key=scored.get)
    named = paths.exp / choice
    if not named.is_dir():
        raise TrainingStateError(f"final_checkpoint {choice!r}: no such folder in {paths.exp}")
    return named


def _portable_base(recorded: str, fallback: str) -> tuple[str, str | None]:
    """Hub id and revision for a base model path recorded from the local Hugging Face cache."""
    m = re.search(r"models--([^\\/]+)--([^\\/]+)[\\/]snapshots[\\/]([0-9a-f]{40})", recorded)
    if m:
        return f"{m[1]}/{m[2]}", m[3]
    return (fallback or recorded), None


def _close_log_files(root: Path) -> None:
    """Close logging handlers writing under ``root`` (OmniVoice's trainer leaves ``train.log`` open,
    which Windows refuses to delete)."""
    root = root.resolve()
    loggers = [logging.getLogger(), *(lg for lg in logging.Logger.manager.loggerDict.values()
                                      if isinstance(lg, logging.Logger))]
    for lg in loggers:
        for h in list(lg.handlers):
            if isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve().is_relative_to(root):
                h.close()
                lg.removeHandler(h)


def package(cfg: TrainConfig, paths: TrainPaths, eval_results: dict[str, float] | None = None) -> Path:
    """Keep only the LoRA adapters of a finished run and delete its scratch files.

    The chosen checkpoint (``final_checkpoint``) becomes ``adapter_model.safetensors``, the
    other saved steps ``checkpoint-<step>.safetensors``; they share one ``adapter_config.json``
    whose base model is the Hub id (and revision) rather than a local cache path.
    """
    checkpoints = paths.checkpoints()
    if not checkpoints:
        raise TrainingStateError(f"no checkpoints in {paths.exp}; run the train step first")
    final = choose_checkpoint(cfg, paths, eval_results)
    missing = [c.name for c in checkpoints if not (c / ADAPTER).exists()]
    if missing or not (final / ADAPTER_CONFIG).exists():
        where = ", ".join(missing) or final.name
        raise TrainingStateError(f"no LoRA adapter in {where}; only LoRA runs can be packaged")
    for ckpt in checkpoints:
        dst = paths.adapter if ckpt == final else paths.checkpoint_file(int(ckpt.name.split("-")[-1]))
        shutil.copy2(ckpt / ADAPTER, dst)
    adapter_cfg = json.loads((final / ADAPTER_CONFIG).read_text(encoding="utf-8"))
    base, revision = _portable_base(adapter_cfg.get("base_model_name_or_path") or "",
                                    cfg.omnivoice.get("init_from_checkpoint") or "")
    adapter_cfg["base_model_name_or_path"] = base
    if revision:
        adapter_cfg["revision"] = revision
    paths.adapter_config.write_text(json.dumps(adapter_cfg, indent=2) + "\n", encoding="utf-8")
    if eval_results:
        paths.eval_file.write_text(json.dumps(eval_results, indent=2), encoding="utf-8")
    _close_log_files(paths.scratch)
    shutil.rmtree(paths.scratch)
    return paths.adapter


def base_model_path(adapter_cfg: dict) -> str:
    """The adapter's base model: its pinned Hub revision when recorded, else the id as is."""
    base, revision = adapter_cfg["base_model_name_or_path"], adapter_cfg.get("revision")
    if not revision or Path(base).exists():
        return base
    from huggingface_hub import snapshot_download

    return snapshot_download(base, revision=revision)


def merge(run: Path, out: Path, base_model: str | None = None, checkpoint: str | None = None) -> Path:
    """Merge a packaged run's adapter into its base model as a standalone model in ``out``.

    ``checkpoint`` (e.g. "checkpoint-2000") picks another saved step than the final adapter.
    The base model defaults to the revision the run was trained from. Skipped if ``out``
    already holds this merge.
    """
    paths = TrainPaths.of(run)
    if not paths.packaged:
        raise FileNotFoundError(f"{run} is not a packaged run (no {paths.adapter.name} + {paths.adapter_config.name})")
    weights = paths.adapter
    if checkpoint:
        if not re.fullmatch(r"checkpoint-\d+", checkpoint):
            raise ValueError(f"--checkpoint must look like checkpoint-2000, got {checkpoint!r}")
        weights = paths.checkpoint_file(int(checkpoint.split("-")[-1]))
        if not weights.exists():
            saved = sorted(p.stem for p in paths.root.glob("checkpoint-*.safetensors"))
            raise FileNotFoundError(f"no {weights.name} in {run} (the final step is {paths.adapter.name}; "
                                    f"others: {', '.join(saved) or 'none'})")
    adapter_cfg = json.loads(paths.adapter_config.read_text(encoding="utf-8"))
    want = {"adapter": str(weights.resolve()), "base_model": base_model or adapter_cfg["base_model_name_or_path"]}
    if not base_model and adapter_cfg.get("revision"):
        want["revision"] = adapter_cfg["revision"]
    out = Path(out)
    stamp = out / "merged_from.json"
    if stamp.exists() and json.loads(stamp.read_text(encoding="utf-8")) == want:
        print(f"{out} is already {weights.name} merged", flush=True)
        return out
    base_model = base_model or base_model_path(adapter_cfg)
    with tempfile.TemporaryDirectory() as tmp:
        adapter_dir = Path(tmp)  # merge_lora reads <dir>/adapter_config.json + <dir>/adapter_model.safetensors
        shutil.copy2(paths.adapter_config, adapter_dir / ADAPTER_CONFIG)
        shutil.copy2(weights, adapter_dir / ADAPTER)
        subprocess.run(
            [sys.executable, "-m", "omnivoice.cli.merge_lora", "--base_model", base_model,
             "--lora_adapter", str(adapter_dir), "--output_dir", str(out)],
            check=True,
        )
    stamp.write_text(json.dumps(want, indent=1), encoding="utf-8")
    return out
