"""Speaker diarization with NVIDIA Nemotron-3-Diarization (Sortformer).

Produces per-frame speaker activity probabilities of shape ``(T, 8)`` at a 10 ms hop.
Each column is one speaker, ordered by first appearance. Several columns can be
active in the same frame, which is how the model signals overlap.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ..audio import SAMPLE_RATE, load_mono

MODEL_ID = "nvidia/Nemotron-3-Diarization"
FRAME_HOP = 0.01


@lru_cache(maxsize=1)
def _load(model_id: str):
    """Load once per batch. The model waits in CPU memory between videos, so it never
    competes for GPU memory with the overlap detector."""
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForAudioFrameClassification.from_pretrained(model_id).eval()
    return processor, model


def diarize(wav_path: Path, model_id: str = MODEL_ID, device: str | None = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = _load(model_id)
    model.to(device)
    try:
        audio = load_mono(wav_path, SAMPLE_RATE)
        inputs = processor(audio, sampling_rate=SAMPLE_RATE).to(device, dtype=model.dtype)
        with torch.inference_mode():
            logits = model(**inputs).logits  # (1, T, 8)
        probs = logits.float().sigmoid()[0].cpu().numpy()
        del inputs, logits
    finally:
        model.to("cpu")
        if device == "cuda":
            torch.cuda.empty_cache()
    return probs.astype(np.float32)
