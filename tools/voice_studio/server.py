"""Backend of the Voice Studio (FastAPI).

* ``/api/config``     models, reference-voice library, text presets
* ``/api/reference``  upload/record a voice or pick a library clip (long clips are trimmed)
* ``/api/transcribe`` Nepali transcript of a reference (IndicConformer, CPU)
* ``/api/generate``   the same text in the same voice with several models; optionally blind
* ``/api/vote``       pick the best output of a blind round; reveals which model was which
* ``/api/tally``      running blind-preference results
* ``/api/bench``      objective benchmark results and the audio behind them
* ``/media/...``      audio files

All models stay on the GPU together: they share one audio codec (their Higgs-audio
tokenizers are byte-identical), so each extra 0.6B model costs little memory. Each round
uses one voice prompt and one seed for every model, so only the weights differ.
"""

from __future__ import annotations

import json
import random
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from nepvoice.prep.manifest import dataset_dir
from tools.tts_bench.asr import NepaliASR
from tools.tts_bench.texts import SENTENCES

from .reference import prepare_reference, seed_all

STATIC = Path(__file__).parent / "static"
SAMPLE_RATE = 24000
PALETTE = ["#7c5cff", "#22c3a6", "#ff8a3d", "#e8577e", "#3fa7ff", "#c9b13c"]
DESCRIPTIONS = {"base": "k2-fsa/OmniVoice, no Nepali fine-tuning"}
LABELS = {"base": "OmniVoice base"}


class Studio:
    def __init__(self, models: dict[str, str], work: Path, out_dir: Path, bench_dir: Path | None):
        self.model_paths = models
        self.work, self.out_dir, self.bench_dir = work, out_dir, bench_dir
        self.data = dataset_dir(work)
        (out_dir / "refs").mkdir(parents=True, exist_ok=True)
        (out_dir / "rounds").mkdir(parents=True, exist_ok=True)
        self.models: dict = {}
        self.asr = NepaliASR()
        self.lock = threading.Lock()
        self.refs: dict[str, dict] = {}
        self.rounds: dict[str, dict] = {}
        self.library = self._library(work / "speakers.json")
        self._load_rounds()

    # ---- models ----
    def load_models(self) -> None:
        import torch
        from omnivoice import OmniVoice

        shared = None
        for name, path in self.model_paths.items():
            print(f"loading {name}: {path}", flush=True)
            m = OmniVoice.from_pretrained(path, device_map="cuda", dtype=torch.float16).eval()
            if shared is None:
                shared = m.audio_tokenizer
            else:
                old, m.audio_tokenizer = m.audio_tokenizer, shared
                del old
                torch.cuda.empty_cache()
            self.models[name] = m
        print("models ready", flush=True)

    # ---- reference voices ----
    def _library(self, speakers_json: Path) -> list[dict]:
        dev_path = self.data / "dev.jsonl"
        if not dev_path.exists():
            return []
        read = lambda p: [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]  # noqa: E731
        trained = set()
        if (self.data / "train.jsonl").exists():
            trained = {r["speaker"] for r in read(self.data / "train.jsonl")}
        voices = {}
        if speakers_json.exists():
            reg = json.loads(speakers_json.read_text(encoding="utf-8"))["speakers"]
            voices = {g: s.get("voice") for g, s in reg.items()}
        by_spk = defaultdict(list)
        for r in read(dev_path):
            if 5.0 <= r["duration"] <= 12.0 and r.get("text"):
                by_spk[r["speaker"]].append(r)
        lib = []
        for spk, rows in sorted(by_spk.items(), key=lambda kv: (kv[0] in trained, kv[0])):
            rows.sort(key=lambda r: -r["duration"])
            for r in rows[:6]:
                lib.append({"id": r["id"], "speaker": spk, "voice": voices.get(spk) or "unknown",
                            "seen": spk in trained, "duration": round(r["duration"], 1), "text": r["text"],
                            "path": r["audio_path"]})
        return lib

    def add_reference(self, src: str, source: str, text: str = "") -> dict:
        path, note = prepare_reference(src, self.out_dir)
        ref_id = uuid.uuid4().hex[:12]
        info = sf.info(path)
        self.refs[ref_id] = {"path": path, "note": note, "duration": round(info.duration, 1),
                             "source": source, "text": "" if note else text}
        public = {k: v for k, v in self.refs[ref_id].items() if k != "path"}
        return {"ref_id": ref_id, "url": f"/media/ref/{ref_id}", **public}

    # ---- rounds and votes ----
    def _load_rounds(self) -> None:
        for f in sorted((self.out_dir / "rounds").glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            self.rounds[r["round_id"]] = r

    def _save_round(self, r: dict) -> None:
        (self.out_dir / "rounds" / f"{r['round_id']}.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")

    def generate(self, ref_id: str, ref_text: str, text: str, seed: int, names: list[str], blind: bool) -> dict:
        import torch

        ref = self.refs.get(ref_id)
        if ref is None:
            raise HTTPException(404, "unknown reference; add it again")
        names = [n for n in names if n in self.models] or list(self.models)
        round_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        d = self.out_dir / "rounds" / round_id
        d.mkdir(parents=True, exist_ok=True)
        with self.lock:
            t = time.perf_counter()
            first = self.models[names[0]]
            prompt = first.create_voice_clone_prompt(ref_audio=ref["path"], ref_text=ref_text)
            prompt_time = time.perf_counter() - t
            outputs = []
            for name in names:
                seed_all(seed)
                t = time.perf_counter()
                wav = self.models[name].generate(text=text, language="npi", voice_clone_prompt=prompt)[0]
                torch.cuda.synchronize()
                gen_time = time.perf_counter() - t
                wav = np.asarray(wav, dtype=np.float32)
                sf.write(str(d / f"{name}.wav"), wav, SAMPLE_RATE)
                outputs.append({"model": name, "seconds": round(len(wav) / SAMPLE_RATE, 2),
                                "gen_time": round(gen_time, 2)})
        order = list(range(len(outputs)))
        if blind:
            random.shuffle(order)
        slots = []
        for slot_idx, i in enumerate(order):
            o = outputs[i]
            slots.append({"slot": chr(65 + slot_idx), **o})
        r = {"round_id": round_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "ref_id": ref_id,
             "ref_source": ref["source"], "ref_text": ref_text, "text": text, "seed": seed, "blind": blind,
             "prompt_time": round(prompt_time, 2), "slots": slots, "vote": None}
        self.rounds[round_id] = r
        self._save_round(r)
        return self.public_round(r)

    def public_round(self, r: dict) -> dict:
        hidden = r["blind"] and r["vote"] is None
        return {
            "round_id": r["round_id"], "blind": r["blind"], "revealed": not hidden, "vote": r["vote"],
            "text": r["text"], "prompt_time": r["prompt_time"], "time": r["time"],
            "slots": [{"slot": s["slot"], "url": f"/media/round/{r['round_id']}/{s['slot']}",
                       "seconds": s["seconds"], "gen_time": s["gen_time"],
                       "model": None if hidden else s["model"]} for s in r["slots"]],
        }

    def vote(self, round_id: str, best: str) -> dict:
        r = self.rounds.get(round_id)
        if r is None:
            raise HTTPException(404, "unknown round")
        if not r["blind"]:
            raise HTTPException(400, "votes are only counted in blind rounds")
        if r["vote"] is not None:
            raise HTTPException(400, "already voted")
        slots = {s["slot"]: s["model"] for s in r["slots"]}
        if best != "tie" and best not in slots:
            raise HTTPException(400, "unknown slot")
        r["vote"] = {"best_slot": best, "best_model": slots.get(best, "tie")}
        self._save_round(r)
        return {"round": self.public_round(r), "tally": self.tally()}

    def tally(self) -> dict:
        wins, played, ties, pair = defaultdict(int), defaultdict(int), 0, defaultdict(lambda: [0, 0])
        for r in self.rounds.values():
            if not r["blind"] or not r["vote"]:
                continue
            models = [s["model"] for s in r["slots"]]
            for m in models:
                played[m] += 1
            best = r["vote"]["best_model"]
            if best == "tie":
                ties += 1
                continue
            wins[best] += 1
            for m in models:
                if m != best:
                    pair[f"{best}>{m}"][0] += 1
                    pair[f"{m}>{best}"][1] += 1
        rounds = sum(1 for r in self.rounds.values() if r["blind"] and r["vote"])
        names = list(dict.fromkeys([*self.model_paths, *played]))  # configured models, plus any from older rounds
        return {"rounds": rounds, "ties": ties,
                "models": {m: {"wins": wins[m], "played": played[m],
                               "win_rate": round(wins[m] / played[m], 3) if played[m] else None} for m in names},
                "pairwise": {k: {"wins": v[0], "losses": v[1]} for k, v in pair.items()}}

    # ---- benchmark ----
    def bench(self) -> dict:
        if self.bench_dir is None or not (self.bench_dir / "summary.json").exists():
            return {"available": False}
        testset = json.loads((self.bench_dir / "testset.json").read_text(encoding="utf-8"))
        results = (self.bench_dir / "results.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(x) for x in results.splitlines() if x.strip()]
        by_item = defaultdict(dict)
        for r in rows:
            key = r["item"] if r["model"] != "real speech" else None
            if key:
                by_item[key][r["model"]] = {k: r[k] for k in ("cer", "wer", "sim", "utmos", "asr")}
        real = {r["text_id"]: r for r in rows if r["model"] == "real speech"}
        refs = {r["id"]: {k: v for k, v in r.items() if k != "audio_path"} for r in testset["refs"]}
        texts = {t["id"]: {k: v for k, v in t.items() if k != "real_audio"} for t in testset["texts"]}
        items = []
        for item, scores in sorted(by_item.items()):
            ref_id, text_id = item.split("__", 1)
            real_scores = {k: real[text_id][k] for k in ("cer", "utmos", "asr")} if text_id in real else None
            items.append({"item": item, "ref": ref_id, "text_id": text_id, "scores": scores, "real": real_scores})
        return {"available": True, "summary": json.loads((self.bench_dir / "summary.json").read_text(encoding="utf-8")),
                "models": json.loads((self.bench_dir / "models.json").read_text(encoding="utf-8")),
                "refs": refs, "texts": texts, "items": items}


class RefPick(BaseModel):
    clip_id: str


class TranscribeReq(BaseModel):
    ref_id: str


class GenerateReq(BaseModel):
    ref_id: str
    ref_text: str = ""
    text: str
    seed: int = 0
    models: list[str] = []
    blind: bool = False


class VoteReq(BaseModel):
    round_id: str
    best: str


def create_app(studio: Studio) -> FastAPI:
    app = FastAPI(title="Voice Studio")
    lib_by_id = {c["id"]: c for c in studio.library}

    @app.get("/api/config")
    def config():
        names = list(studio.model_paths)
        return {
            "models": [{"name": n, "label": LABELS.get(n, n), "description": DESCRIPTIONS.get(n, studio.model_paths[n]),
                        "color": PALETTE[i % len(PALETTE)]} for i, n in enumerate(names)],
            "library": [{k: v for k, v in c.items() if k != "path"} | {"url": f"/media/lib/{c['id']}"}
                        for c in studio.library],
            "presets": SENTENCES,
            "ready": bool(studio.models),
        }

    @app.post("/api/reference/upload")
    async def upload(file: UploadFile = File(...)):
        suffix = Path(file.filename or "voice.wav").suffix or ".wav"
        dst = studio.out_dir / "refs" / f"upload-{uuid.uuid4().hex[:8]}{suffix}"
        dst.write_bytes(await file.read())
        try:
            sf.info(str(dst))
        except Exception:
            # browsers record webm/opus; decode it to wav with ffmpeg
            import subprocess

            wav = dst.with_suffix(".wav")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(dst), str(wav)], check=True)
            dst = wav
        return studio.add_reference(str(dst), "upload")

    @app.post("/api/reference/library")
    def pick(req: RefPick):
        clip = lib_by_id.get(req.clip_id)
        if clip is None:
            raise HTTPException(404, "unknown clip")
        return studio.add_reference(clip["path"], "library", clip["text"])

    @app.post("/api/transcribe")
    def transcribe(req: TranscribeReq):
        ref = studio.refs.get(req.ref_id)
        if ref is None:
            raise HTTPException(404, "unknown reference")
        t = time.perf_counter()
        text = studio.asr.transcribe(ref["path"])
        return {"text": text, "seconds": round(time.perf_counter() - t, 2)}

    @app.post("/api/generate")
    def generate(req: GenerateReq):
        if not studio.models:
            raise HTTPException(503, "models are still loading")
        if not req.text.strip():
            raise HTTPException(400, "enter some Nepali text")
        ref_text = req.ref_text.strip()
        if not ref_text:
            ref = studio.refs.get(req.ref_id)
            if ref is None:
                raise HTTPException(404, "unknown reference")
            ref_text = studio.asr.transcribe(ref["path"])
        result = studio.generate(req.ref_id, ref_text, req.text.strip(), req.seed, req.models, req.blind)
        return result | {"ref_text": ref_text}

    @app.post("/api/vote")
    def vote(req: VoteReq):
        return studio.vote(req.round_id, req.best)

    @app.get("/api/tally")
    def tally():
        return studio.tally()

    @app.get("/api/rounds")
    def rounds():
        rs = sorted(studio.rounds.values(), key=lambda r: r["round_id"], reverse=True)[:30]
        return [studio.public_round(r) for r in rs]

    @app.get("/api/bench")
    def bench():
        return studio.bench()

    # ---- media ----
    def _file(path: Path):
        if not path.exists():
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/media/ref/{ref_id}")
    def media_ref(ref_id: str):
        ref = studio.refs.get(ref_id)
        if ref is None:
            raise HTTPException(404)
        return _file(Path(ref["path"]))

    @app.get("/media/lib/{clip_id}")
    def media_lib(clip_id: str):
        clip = lib_by_id.get(clip_id)
        if clip is None:
            raise HTTPException(404)
        return _file(Path(clip["path"]))

    @app.get("/media/round/{round_id}/{slot}")
    def media_round(round_id: str, slot: str):
        r = studio.rounds.get(round_id)
        if r is None:
            raise HTTPException(404)
        s = next((s for s in r["slots"] if s["slot"] == slot), None)
        if s is None:
            raise HTTPException(404)
        return _file(studio.out_dir / "rounds" / round_id / f"{s['model']}.wav")

    @app.get("/media/bench/{model}/{item}")
    def media_bench(model: str, item: str):
        if "/" in model or "\\" in model or ".." in item:
            raise HTTPException(400)
        return _file(studio.bench_dir / "audio" / model / f"{item}.wav")

    @app.get("/media/bench-ref/{ref_id}")
    def media_bench_ref(ref_id: str):
        testset = json.loads((studio.bench_dir / "testset.json").read_text(encoding="utf-8"))
        ref = next((r for r in testset["refs"] if r["id"] == ref_id), None)
        if ref is None:
            raise HTTPException(404)
        return _file(Path(ref["audio_path"]))

    @app.get("/media/bench-real/{text_id}")
    def media_bench_real(text_id: str):
        testset = json.loads((studio.bench_dir / "testset.json").read_text(encoding="utf-8"))
        t = next((t for t in testset["texts"] if t["id"] == text_id and t.get("real_audio")), None)
        if t is None:
            raise HTTPException(404)
        return _file(Path(t["real_audio"]))

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def serve(models: dict[str, str], work: Path, out_dir: Path, bench_dir: Path | None,
          host: str = "127.0.0.1", port: int = 7861, open_browser: bool = True) -> None:
    import uvicorn

    studio = Studio(models, work, out_dir, bench_dir)
    app = create_app(studio)
    threading.Thread(target=studio.load_models, daemon=True).start()
    if open_browser:
        import webbrowser

        threading.Timer(2.0, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
