"""Overlapped speech detection with the DiariZen WavLM-Conformer segmentation model.

DiariZen's end-to-end segmentation model predicts, for every 20 ms frame of a
16 s window, a distribution over *powerset* classes (which subset of up to four
local speakers is active). Summing the classes by subset size gives a calibrated
estimate of the instantaneous number of speakers, independent of who they are.

Only the segmentation network is run here; DiariZen's embedding and clustering
stages are skipped because overlap detection doesn't need speaker identities.

Outputs, at a 20 ms hop:

* ``counts`` ``(T, 3)``: P(no speech), P(one speaker), P(two or more speakers),
  aggregated over overlapping windows;
* ``local`` ``(num_windows, frames, 4)``: per-window soft activity of each of
  the window's local speakers, with ``window_starts`` (in frames) locating each
  window. Local speaker indices are only consistent within one window; they are
  matched to global speakers later (see ``selection.diarizen_other_activity``).
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..audio import SAMPLE_RATE, load_mono

MODEL_ID = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
FRAME_HOP = 0.02
_SAMPLES_PER_FRAME = int(FRAME_HOP * SAMPLE_RATE)  # WavLM conv stack has a total stride of 320


@lru_cache(maxsize=1)
def load_model(model_id: str = MODEL_ID):
    """Load once per batch, into CPU memory; ``detect_overlap`` moves it to the GPU while it runs."""
    import tomllib

    from huggingface_hub import snapshot_download
    from pyannote.audio import Model

    hub = Path(snapshot_download(model_id, allow_patterns=["config.toml", "pytorch_model.bin"]))
    config = tomllib.loads((hub / "config.toml").read_text(encoding="utf-8"))
    model = Model.from_pretrained(hub / "pytorch_model.bin", config=config, map_location="cpu")
    return model.eval(), config


def _powerset_counts(model) -> np.ndarray:
    """Number of active speakers encoded by each powerset class."""
    mapping = model.powerset.mapping.cpu().numpy()  # (num_classes, max_speakers)
    return mapping.sum(axis=1).astype(int)


def detect_overlap(
    wav_path: Path,
    model_id: str = MODEL_ID,
    step_ratio: float = 0.25,
    batch_size: int = 16,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(model_id)
    model.to(device)
    chunk_s = config["inference"]["args"]["seg_duration"]

    audio = load_mono(wav_path, SAMPLE_RATE)
    chunk = chunk_s * SAMPLE_RATE
    # Keep the step a whole number of frames so every window lands on the global frame grid.
    step = max(1, round(chunk * step_ratio / _SAMPLES_PER_FRAME)) * _SAMPLES_PER_FRAME
    num_total_frames = math.ceil(len(audio) / _SAMPLES_PER_FRAME)

    starts = list(range(0, max(1, len(audio) - chunk + step), step))
    if starts[-1] + chunk < len(audio):
        starts.append(len(audio) - chunk)

    counts = _powerset_counts(model)
    mapping = model.powerset.mapping.cpu().numpy().astype(np.float32)  # (classes, local speakers)
    num_frames = model.num_frames(chunk)
    local = np.zeros((len(starts), num_frames, mapping.shape[1]), dtype=np.float16)
    # Down-weight window edges, where the model sees little context (same idea as pyannote's aggregation).
    weight = np.hamming(num_frames).astype(np.float32) + 1e-3

    acc = np.zeros((num_total_frames + num_frames, 3), dtype=np.float64)
    norm = np.zeros(num_total_frames + num_frames, dtype=np.float64)

    use_amp = device == "cuda"
    for b in tqdm(range(0, len(starts), batch_size), desc="osd", unit="batch", mininterval=5):
        batch_starts = starts[b : b + batch_size]
        windows = np.zeros((len(batch_starts), 1, chunk), dtype=np.float32)
        for i, s in enumerate(batch_starts):
            piece = audio[s : s + chunk]
            windows[i, 0, : len(piece)] = piece
        with torch.inference_mode(), torch.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            log_probs = model(torch.from_numpy(windows).to(device))
        probs = log_probs.float().exp().cpu().numpy()  # (B, frames, classes)
        local[b : b + len(batch_starts)] = probs @ mapping  # soft multi-label speaker activity

        grouped = np.stack(
            [
                probs[..., counts == 0].sum(-1),
                probs[..., counts == 1].sum(-1),
                probs[..., counts >= 2].sum(-1),
            ],
            axis=-1,
        )
        for i, s in enumerate(batch_starts):
            f0 = s // _SAMPLES_PER_FRAME
            valid = min(num_frames, math.ceil((len(audio) - s) / _SAMPLES_PER_FRAME))
            acc[f0 : f0 + valid] += grouped[i, :valid] * weight[:valid, None]
            norm[f0 : f0 + valid] += weight[:valid]

    model.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    out = acc[:num_total_frames] / np.maximum(norm[:num_total_frames, None], 1e-8)
    return {
        "counts": out.astype(np.float32),
        "local": local,
        "window_starts": np.array(starts) // _SAMPLES_PER_FRAME,
    }
