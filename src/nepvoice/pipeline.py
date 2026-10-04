"""The whole pipeline, from a folder of recordings and transcripts to a fine-tuned model.

Data preparation (``ingest`` .. ``manifest``) writes the work directory (e.g.
``dataset/preprocessed1``); training (``splits`` .. ``package``) reads it and writes a run
directory (e.g. ``models/run1``). Steps, in order (every step caches its output, so
re-running resumes where it stopped):

    ingest     register the input folder's recordings in the work directory
    diarize    who speaks when (Nemotron-3-Diarization)                      GPU
    osd        overlapped-speech detection (DiariZen)                        GPU
    select     overlap-free single-speaker segments with their transcript
    export     cut clips from the source audio, resample, normalise their level
    embed      speaker embeddings per clip                                   GPU
    speakers   recognise speakers across recordings (speakers.json)          GPU
    manifest   dataset/train.jsonl + dev.jsonl
    splits     balanced training manifests and OmniVoice data config
    tokenize   audio tokens for training                                     GPU
    train      LoRA fine-tuning                                              GPU
    evaluate   dev loss of the base model and every checkpoint               GPU
    package    keep only the LoRA adapters (+ config) in the run directory
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config, to_toml

PREP_STAGES = ("diarize", "osd", "select", "export", "embed")
PREP_STEPS = ("ingest", *PREP_STAGES, "speakers", "manifest")
TRAIN_STEPS = ("splits", "tokenize", "train", "evaluate", "package")
STEPS = (*PREP_STEPS, *TRAIN_STEPS)


def select_steps(start: str | None = None, stop: str | None = None, only: list[str] | None = None) -> list[str]:
    """Steps from ``start`` to ``stop`` (inclusive), or exactly ``only`` (in pipeline order)."""
    for name in [start, stop, *(only or [])]:
        if name is not None and name not in STEPS:
            raise ValueError(f"unknown step {name!r}; steps are: {', '.join(STEPS)}")
    if only:
        return [s for s in STEPS if s in only]
    i = STEPS.index(start) if start else 0
    j = STEPS.index(stop) if stop else len(STEPS) - 1
    if i > j:
        raise ValueError(f"step {start!r} comes after {stop!r}")
    return list(STEPS[i: j + 1])


def run(work: Path, cfg: Config, steps: list[str], input_dir: Path | None = None,
        run_dir: Path | None = None) -> None:
    """Run ``steps``; the resolved config is saved in the work directory (data preparation)
    and in the run directory (training)."""
    from .prep import stages
    from .train.paths import TrainPaths

    work = Path(work)
    training = [s for s in steps if s in TRAIN_STEPS]
    if training and run_dir is None:
        raise ValueError("training steps need a run directory: --run models/<name>")
    if any(s in PREP_STEPS for s in steps):
        work.mkdir(parents=True, exist_ok=True)
        (work / "config.toml").write_text(to_toml(cfg), encoding="utf-8")
    if training:
        paths = TrainPaths.of(run_dir)
        if paths.packaged:
            print(f"{run_dir} is a finished, packaged run: skipping {', '.join(training)} "
                  "(delete it or use another --run to train again)", flush=True)
            steps = [s for s in steps if s not in TRAIN_STEPS]
        else:
            paths.root.mkdir(parents=True, exist_ok=True)
            paths.config.write_text(to_toml(cfg), encoding="utf-8")
    for i, step in enumerate(steps, 1):
        print(f"=== step {i}/{len(steps)}: {step}", flush=True)
        if step == "ingest":
            _ingest(work, cfg, input_dir)
        elif step in PREP_STAGES:
            stages.run(step, None, work, cfg)
        elif step in TRAIN_STEPS:
            _STEP_FUNCTIONS[step](work, TrainPaths.of(run_dir), cfg)
        else:
            _STEP_FUNCTIONS[step](work, cfg)


# ---- steps ----

def _ingest(work: Path, cfg: Config, input_dir: Path | None) -> None:
    from .ingest import ingest

    if input_dir is None:
        raise ValueError("the ingest step needs an input folder")
    report = ingest(input_dir, work, cfg.ingest)
    print(report.summary(), flush=True)
    for path in report.no_transcript[:10]:
        print(f"  no transcript: {path}", flush=True)
    if report.not_in_input:
        print(f"note: {len(report.not_in_input)} recordings in {work} are no longer in the input folder "
              f"and are still used (delete their folders under recordings/ to drop them)", flush=True)
    if not report.ids:
        raise ValueError(f"no recordings with transcripts found in {input_dir}")


def _speakers(work: Path, cfg: Config) -> None:
    from .prep import speakers

    if not cfg.manifest.speaker_id:
        print("speaker recognition off (manifest.speaker_id = false)", flush=True)
        return
    print(speakers.report(speakers.build(work, cfg.speakers)), flush=True)


def _manifest(work: Path, cfg: Config) -> None:
    from .prep import manifest

    registry = None
    if cfg.manifest.speaker_id:
        path = work / "speakers.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run the speakers step (or set manifest.speaker_id = false)")
        registry = json.loads(path.read_text(encoding="utf-8"))
    stats = manifest.write(work, cfg.manifest, registry)
    print(json.dumps(stats, indent=2), flush=True)
    if not stats["train"]["clips"]:
        raise ValueError("the dataset is empty: no clips survived selection and export")


def _splits(work: Path, paths, cfg: Config) -> None:
    from .prep.manifest import dataset_dir
    from .train import splits

    registry = work / "speakers.json" if cfg.manifest.speaker_id else None
    data = dataset_dir(work)
    stats = splits.split_manifests(data / "train.jsonl", data / "dev.jsonl", registry, paths)
    print(json.dumps(stats, indent=2), flush=True)
    print(f"data config: {splits.write_data_config(paths, cfg.train.female_repeat)}", flush=True)


def _tokenize(work: Path, paths, cfg: Config) -> None:
    from .train.tokenize import tokenize_all

    t = cfg.train
    tokenize_all(paths, t.peak_normalize, t.tokenize_jobs_per_gpu, t.tokenize_loader_workers)


def _train(work: Path, paths, cfg: Config) -> None:
    from .train.trainer import train

    print(f"final checkpoint: {train(cfg.train, paths)}", flush=True)


def _evaluate(work: Path, paths, cfg: Config) -> None:
    from .train.trainer import evaluate

    if not cfg.train.evaluate:
        print("evaluation off (train.evaluate = false)", flush=True)
        return
    results = evaluate(cfg.train, paths)
    if results:
        print(json.dumps(results, indent=2), flush=True)


def _package(work: Path, paths, cfg: Config) -> None:
    from .train.trainer import package

    results = None
    if paths.eval_results.exists():
        results = next(iter(json.loads(paths.eval_results.read_text(encoding="utf-8")).values()), None)
    print(f"final adapter: {package(cfg.train, paths, results)} (scratch files deleted)", flush=True)


_STEP_FUNCTIONS = {
    "speakers": _speakers,
    "manifest": _manifest,
    "splits": _splits,
    "tokenize": _tokenize,
    "train": _train,
    "evaluate": _evaluate,
    "package": _package,
}
