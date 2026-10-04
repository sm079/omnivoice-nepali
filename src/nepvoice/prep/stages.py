"""Stage-by-stage processing of many recordings.

Each stage runs for every recording before the next starts, so a long batch makes
visible, checkpointed progress and each kind of resource is used on its own:

    diarize   GPU           one recording at a time, model loaded once
    osd       GPU           one recording at a time, model loaded once
    select    CPU           parallel (processes)
    export    CPU + disk    parallel (processes)
    embed     GPU           one recording at a time, model loaded once

Every stage skips work whose output is already cached, so re-running resumes. A
recording that fails is logged to ``<work>/failures.jsonl`` and skipped by later stages.
"""

from __future__ import annotations

import json
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..config import Config, ExportConfig
from ..workspace import Workspace, find_workspaces
from . import process
from .selection import SelectConfig

STAGES = ("diarize", "osd", "select", "export", "embed")


class MissingInput(Exception):
    """An earlier stage has not produced this recording's input (it failed or was skipped)."""


def _require(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            raise MissingInput(p.name)


# ---- per-recording stage functions (module level, so process pools can pickle them) ----

def do_diarize(ws: Workspace) -> str:
    if ws.diar.exists():
        return "cached"
    process.run_diarization(ws)
    return "done"


def do_osd(ws: Workspace) -> str:
    if ws.osd.exists():
        return "cached"
    process.run_overlap_detection(ws)
    return "done"


def do_select(ws_dir: Path, cfg: SelectConfig, force: bool) -> str:
    ws = Workspace(ws_dir)
    if not force and process.selection_done(ws, cfg):
        return "cached"
    _require(ws.diar, ws.osd)
    segments = process.select(ws, cfg)
    return f"{len(segments)} segments, {sum(s.duration for s in segments) / 60:.1f} min"


def do_export(ws_dir: Path, cfg: ExportConfig, keep_source_wav: bool, force: bool) -> str:
    ws = Workspace(ws_dir)
    _require(ws.segments)
    if not force and process.export_done(ws, cfg):
        return "cached"
    return f"{process.export(ws, cfg, keep_source_wav)} clips"


def do_embed(ws: Workspace) -> str:
    _require(ws.segments)
    from . import speakers

    speakers.embed_recording(ws)
    return "done"


# ---- driver ----

def _log_failure(work: Path, stage: str, rid: str, exc: BaseException, tb: str) -> None:
    work.mkdir(parents=True, exist_ok=True)
    with (work / "failures.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "recording_id": rid,
            "stage": stage,
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": tb,
        }) + "\n")


def run_stage(
    stage: str,
    ids: list[str],
    work: Path,
    jobs: dict[str, Callable[[], str]] | dict[str, tuple],
    workers: int = 1,
    processes: bool = False,
) -> dict[str, str]:
    """Run one stage over ``ids``. ``jobs`` maps id -> zero-arg callable (in this process)
    or id -> (function, args) (process pool). Prints ``<stage> i/N`` progress lines."""
    results: dict[str, str] = {}
    n = len(ids)
    counts = {"ok": 0, "skipped": 0, "failed": 0}

    def record(i: int, rid: str, outcome: str, kind: str) -> None:
        counts[kind] += 1
        results[rid] = outcome
        print(f"{stage} {i}/{n} [{rid}] {outcome}", flush=True)

    def failed(rid: str, e: BaseException) -> tuple[str, str]:
        if isinstance(e, MissingInput):
            return f"skipped (missing {e})", "skipped"
        _log_failure(work, stage, rid, e, "".join(traceback.format_exception(e, limit=6)))
        return f"FAILED {type(e).__name__}: {e}", "failed"

    if workers <= 1 or n <= 1:
        for i, rid in enumerate(ids, 1):
            job = jobs[rid]
            try:
                outcome, kind = (job() if callable(job) else job[0](*job[1])), "ok"
            except Exception as e:  # noqa: BLE001 - one bad recording must not stop the batch
                outcome, kind = failed(rid, e)
            record(i, rid, outcome, kind)
    else:
        pool_cls = ProcessPoolExecutor if processes else ThreadPoolExecutor
        with pool_cls(max_workers=workers) as pool:
            futures = {}
            for rid in ids:
                job = jobs[rid]
                futures[pool.submit(job[0], *job[1]) if not callable(job) else pool.submit(job)] = rid
            for i, fut in enumerate(as_completed(futures), 1):
                rid = futures[fut]
                try:
                    outcome, kind = fut.result(), "ok"
                except Exception as e:  # noqa: BLE001
                    outcome, kind = failed(rid, e)
                record(i, rid, outcome, kind)
    print(f"{stage}: {counts['ok']} ok, {counts['skipped']} skipped, {counts['failed']} failed", flush=True)
    return results


def run(stage: str, ids: list[str] | None, work: Path, cfg: Config) -> dict[str, str]:
    """Run one prep ``stage`` for ``ids`` (default: every ingested recording)."""
    index = find_workspaces(work)
    ids = sorted(index) if ids is None else [i for i in ids if i in index]
    p = cfg.prep
    if stage == "diarize":
        jobs = {i: (lambda ws=index[i]: do_diarize(ws)) for i in ids}
        return run_stage(stage, ids, work, jobs)
    if stage == "osd":
        jobs = {i: (lambda ws=index[i]: do_osd(ws)) for i in ids}
        return run_stage(stage, ids, work, jobs)
    if stage == "select":
        jobs = {i: (do_select, (index[i].dir, cfg.select, p.force)) for i in ids}
        return run_stage(stage, ids, work, jobs, workers=p.cpu_workers, processes=True)
    if stage == "export":
        jobs = {i: (do_export, (index[i].dir, cfg.export, p.keep_source_wav, p.force)) for i in ids}
        return run_stage(stage, ids, work, jobs, workers=p.cpu_workers, processes=True)
    if stage == "embed":
        jobs = {i: (lambda ws=index[i]: do_embed(ws)) for i in ids}
        return run_stage(stage, ids, work, jobs)
    raise ValueError(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
