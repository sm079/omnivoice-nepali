"""Nepali ASR with AI4Bharat IndicConformer 600M (ONNX), run on the CPU.

Used to transcribe reference voices in the UI and to measure intelligibility (CER/WER)
of generated speech. On conversational Nepali it was far more accurate than Whisper
large-v3-turbo (which drifts into Hindi).
"""

from __future__ import annotations

import threading

import numpy as np
import soundfile as sf

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
SAMPLE_RATE = 16000


class NepaliASR:
    """Lazily loaded; thread-safe. The model is gated on Hugging Face (accept its terms once)."""

    def __init__(self, decoding: str = "rnnt"):
        self.decoding = decoding  # RNNT read slightly cleaner than CTC, at similar speed
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            from unittest import mock

            from transformers import AutoModel

            # Its remote code picks CUDA whenever torch sees a GPU, where it competes with the
            # TTS models for memory. On the CPU it still does ~10 s of speech per second.
            with mock.patch("torch.cuda.is_available", return_value=False):
                self._model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
        return self._model

    def transcribe_array(self, audio: np.ndarray, sr: int) -> str:
        import torch
        import torchaudio

        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32).reshape(-1))[None]
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        with self._lock:
            return str(self._load()(wav, "ne", self.decoding)).strip()

    def transcribe(self, path: str) -> str:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        return self.transcribe_array(audio.mean(axis=1), sr)
