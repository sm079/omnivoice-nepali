"""Benchmark TTS models on one fixed Nepali test set.

Test set (deterministic, from the held-out dev recordings of a preprocessed dataset):

* **voices** — every dev speaker with enough clean clips, two reference clips each. A
  speaker who also appears in training recordings is *seen*, the others *unseen*.
  Results are reported for both groups.
* **texts** — six written sentences plus held-out transcripts of real dev recordings.

Every model says every text in every voice, with the same voice prompt and seed. Outputs
are scored with :mod:`tools.tts_bench.metrics`. The real recordings behind the dev transcripts
are scored too ("real speech"): their CER is the ASR's own error floor and their UTMOS
what natural speech scores.

Models are generated one at a time (each alone on the GPU), then scored; generated audio
is kept, so re-running only fills in what is missing. A model given as a packaged training
run (``models/run1``) is first merged into ``<out>/models/<name>``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from nepvoice.prep.manifest import dataset_dir
from nepvoice.train.paths import TrainPaths

from .metrics import bootstrap_ci, cer, wer
from .texts import SENTENCES

SAMPLE_RATE = 24000
BASE_MODEL = "k2-fsa/OmniVoice"


def parse_models(specs: list[str]) -> dict[str, str]:
    """``NAME=PATH`` options (Hub id, merged model folder or packaged run); default: the base model."""
    bad = [m for m in specs if "=" not in m]
    if bad:
        raise ValueError(f"--model needs NAME=PATH, got {bad[0]!r}")
    return dict(m.split("=", 1) for m in specs) if specs else {"base": BASE_MODEL}


def resolve_models(models: dict[str, str], out_dir: Path) -> dict[str, str]:
    """Loadable model paths: packaged runs are merged into ``<out_dir>/models/<name>``."""
    from nepvoice.train.trainer import merge

    resolved = {}
    for name, path in models.items():
        if TrainPaths.of(Path(path)).packaged:
            resolved[name] = str(merge(Path(path), out_dir / "models" / name))
        else:
            resolved[name] = path
    return resolved


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_testset(work: Path, refs_per_voice: int = 2, n_dev_texts: int = 10) -> dict:
    """From the dev split of the preprocessed dataset ``work`` (its ``dataset/dev.jsonl``)."""
    data = dataset_dir(work)
    dev = _read(data / "dev.jsonl")
    if not dev:
        raise ValueError(f"{data / 'dev.jsonl'} is empty: the benchmark needs held-out recordings "
                         "(manifest.dev_fraction > 0)")
    trained = {r["speaker"] for r in _read(data / "train.jsonl")}
    speakers_json = Path(work) / "speakers.json"
    voices = {}
    if speakers_json.exists():
        registry = json.loads(speakers_json.read_text(encoding="utf-8"))["speakers"]
        voices = {g: s.get("voice") for g, s in registry.items()}

    by_spk: dict[str, list[dict]] = defaultdict(list)
    for r in dev:
        if r.get("text"):
            by_spk[r["speaker"]].append(r)

    refs, used = [], set()
    for spk, rows in sorted(by_spk.items()):
        cands = sorted((r for r in rows if 6.0 <= r["duration"] <= 12.0), key=lambda r: (-r["duration"], r["id"]))
        if len(cands) < refs_per_voice:
            continue
        step = max(1, len(cands) // refs_per_voice)
        for r in cands[::step][:refs_per_voice]:
            refs.append({"id": r["id"], "speaker": spk, "voice": voices.get(spk, "unknown"),
                         "seen": spk in trained, "audio_path": r["audio_path"], "text": r["text"]})
            used.add(r["id"])

    texts = [{"id": f"w{i:02d}", "kind": "written", "text": t} for i, t in enumerate(SENTENCES)]
    pool = sorted((r for r in dev if r["id"] not in used and r.get("text") and 4.0 <= r["duration"] <= 10.0),
                  key=lambda r: r["id"])
    step = max(1, len(pool) // max(1, n_dev_texts))
    for r in pool[::step][:n_dev_texts]:
        texts.append({"id": f"d_{r['id']}", "kind": "dev", "text": r["text"], "real_audio": r["audio_path"],
                      "speaker": r["speaker"]})
    return {"refs": refs, "texts": texts}


def generate(models: dict[str, str], testset: dict, out_dir: Path, seed: int = 0) -> None:
    """``<out_dir>/audio/<model>/<ref id>__<text id>.wav`` for every model, ref and text."""
    import random

    import torch
    from omnivoice import OmniVoice

    for name, path in models.items():
        d = out_dir / "audio" / name
        d.mkdir(parents=True, exist_ok=True)
        todo = [(r, t) for r in testset["refs"] for t in testset["texts"]
                if not (d / f"{r['id']}__{t['id']}.wav").exists()]
        if not todo:
            print(f"[{name}] all {len(testset['refs']) * len(testset['texts'])} outputs exist", flush=True)
            continue
        print(f"[{name}] loading {path}", flush=True)
        model = OmniVoice.from_pretrained(path, device_map="cuda", dtype=torch.float16).eval()
        prompts = {}
        for i, (ref, text) in enumerate(todo, 1):
            if ref["id"] not in prompts:
                prompts[ref["id"]] = model.create_voice_clone_prompt(ref_audio=ref["audio_path"], ref_text=ref["text"])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            wav = model.generate(text=text["text"], language="npi", voice_clone_prompt=prompts[ref["id"]])[0]
            sf.write(str(d / f"{ref['id']}__{text['id']}.wav"), np.asarray(wav, dtype=np.float32), SAMPLE_RATE)
            print(f"generate {name} {i}/{len(todo)}", flush=True)
        del model, prompts
        torch.cuda.empty_cache()


def score(models: list[str], testset: dict, out_dir: Path) -> list[dict]:
    """Score every generated file (and the real recordings); append to ``results.jsonl``."""
    from .asr import NepaliASR
    from .metrics import Scorer

    res_path = out_dir / "results.jsonl"
    done = {(r["model"], r["item"]) for r in _read(res_path)} if res_path.exists() else set()
    asr, scorer = NepaliASR(), Scorer()
    jobs = [(m, ref, t) for m in models for ref in testset["refs"] for t in testset["texts"]]
    refs_by_spk = defaultdict(list)
    for ref in testset["refs"]:
        refs_by_spk[ref["speaker"]].append(ref)
    real = [(t, ref) for t in testset["texts"] if t["kind"] == "dev" for ref in refs_by_spk.get(t["speaker"], [])[:1]]

    with res_path.open("a", encoding="utf-8") as f:
        total = len(jobs) + len(real)
        for i, (m, ref, t) in enumerate(jobs, 1):
            item = f"{ref['id']}__{t['id']}"
            if (m, item) in done:
                continue
            wav = out_dir / "audio" / m / f"{item}.wav"
            if not wav.exists():
                continue
            hyp = asr.transcribe(str(wav))
            row = {
                "model": m, "item": item, "ref": ref["id"], "text_id": t["id"], "text_kind": t["kind"],
                "voice": ref["voice"], "seen": ref["seen"], "text": t["text"], "asr": hyp,
                "cer": cer(t["text"], hyp), "wer": wer(t["text"], hyp),
                "sim": scorer.sim(str(wav), ref["audio_path"]), "utmos": scorer.utmos(str(wav)),
                "seconds": sf.info(str(wav)).duration,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"score {i}/{total}", flush=True)
        for j, (t, ref) in enumerate(real, 1):
            item = f"real__{t['id']}"
            if ("real speech", item) in done:
                continue
            hyp = asr.transcribe(t["real_audio"])
            row = {
                "model": "real speech", "item": item, "ref": ref["id"], "text_id": t["id"], "text_kind": "dev",
                "voice": ref["voice"], "seen": ref["seen"], "text": t["text"], "asr": hyp,
                "cer": cer(t["text"], hyp), "wer": wer(t["text"], hyp),
                # same speaker, different recording: what a perfect clone could score
                "sim": scorer.sim(t["real_audio"], ref["audio_path"]), "utmos": scorer.utmos(t["real_audio"]),
                "seconds": sf.info(t["real_audio"]).duration,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"score {len(jobs) + j}/{total}", flush=True)
    scorer.release()
    return _read(res_path)


METRICS = ("cer", "wer", "sim", "utmos")


def summarize(rows: list[dict], baseline: str = "base") -> dict:
    """Mean and 95% CI per model (overall and per voice group) plus paired differences vs ``baseline``."""
    groups = {
        "all": lambda r: True,
        "unseen voices": lambda r: not r["seen"],
        "seen voices": lambda r: r["seen"],
        "female voices": lambda r: r["voice"] == "female",
        "male voices": lambda r: r["voice"] == "male",
    }
    models = list(dict.fromkeys(r["model"] for r in rows))
    out = {"n_items": {}, "groups": {}, "paired_vs_" + baseline: {}}
    for gname, pred in groups.items():
        out["groups"][gname] = {}
        for m in models:
            sel = [r for r in rows if r["model"] == m and pred(r)]
            if sel:
                out["groups"][gname][m] = {k: bootstrap_ci([r[k] for r in sel]) for k in METRICS} | {"n": len(sel)}
    base = {r["item"]: r for r in rows if r["model"] == baseline}
    for m in models:
        if m in (baseline, "real speech"):
            continue
        pairs = [(r, base[r["item"]]) for r in rows if r["model"] == m and r["item"] in base]
        if not pairs:
            continue
        entry = {"n": len(pairs)}
        for k in METRICS:
            diff = [a[k] - b[k] for a, b in pairs]
            better = (lambda a, b: a < b) if k in ("cer", "wer") else (lambda a, b: a > b)
            entry[k] = {"diff": bootstrap_ci(diff),
                        "win_rate": float(np.mean([better(a[k], b[k]) for a, b in pairs]))}
        out["paired_vs_" + baseline][m] = entry
    return out


def write_report(summary: dict, out_dir: Path) -> Path:
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return path


def format_table(summary: dict, group: str = "all") -> str:
    rows = summary["groups"].get(group, {})
    lines = [f"{group}:", f"  {'model':14s} {'CER %':>13s} {'WER %':>13s} {'SIM':>13s} {'UTMOS':>13s}   n"]
    for m, s in rows.items():
        def f(k, scale=1.0, d=1):
            mean, lo, hi = s[k]
            return f"{mean * scale:.{d}f} ±{(hi - lo) / 2 * scale:.{d}f}"
        cells = f"{f('cer', 100):>13s} {f('wer', 100):>13s} {f('sim', 1, 3):>13s} {f('utmos', 1, 2):>13s}"
        lines.append(f"  {m:14s} {cells}   {s['n']}")
    return "\n".join(lines)
