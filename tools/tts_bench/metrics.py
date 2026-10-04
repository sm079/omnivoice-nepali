"""Objective TTS metrics: intelligibility, speaker similarity, naturalness.

* **CER / WER** — the generated speech is transcribed with IndicConformer and compared
  with the text it was asked to say (after normalising punctuation and spacing). CER is
  computed without spaces, since Nepali word boundaries are often written either way.
* **SIM** — cosine similarity between WavLM-Large ECAPA-TDNN speaker embeddings of the
  output and the reference voice (the SIM-o metric used in the OmniVoice paper).
* **UTMOS** — UTMOS22-strong, a predicted mean opinion score (1–5) for naturalness.

SIM and UTMOS use the evaluation models OmniVoice publishes (``k2-fsa/TTS_eval_models``).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

EVAL_REPO = "k2-fsa/TTS_eval_models"
_PUNCT = re.compile(r"[।॥.,!?;:'\"“”‘’()\[\]{}<>/\\|@#$%^&*_+=~`…—–-]")
_ZW = re.compile(r"[​-‍﻿]")


def normalize(text: str) -> str:
    """NFC, no zero-width characters, punctuation (incl. danda) to spaces, single spaces."""
    text = unicodedata.normalize("NFC", text or "")
    text = _ZW.sub("", text)
    text = _PUNCT.sub(" ", text)
    return " ".join(text.lower().split())


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = normalize(reference).replace(" ", "")
    hyp = normalize(hypothesis).replace(" ", "")
    return edit_distance(list(ref), list(hyp)) / max(1, len(ref))


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    return edit_distance(ref, hyp) / max(1, len(ref))


def _load_16k(path: str, max_seconds: float = 60.0):
    import soundfile as sf
    import torch
    import torchaudio

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.mean(axis=1))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav[: int(16000 * max_seconds)]


class Scorer:
    """Loads SIM and UTMOS models lazily; use on a free GPU (or CPU)."""

    def __init__(self, device: str | None = None):
        import torch

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._sim = None
        self._utmos = None

    def _sim_model(self):
        if self._sim is None:
            import torch
            from huggingface_hub import hf_hub_download
            from omnivoice.eval.models.ecapa_tdnn_wavlm import ECAPA_TDNN_WAVLM

            ckpt = hf_hub_download(EVAL_REPO, "speaker_similarity/wavlm_large_finetune.pth")
            ssl_pt = hf_hub_download(EVAL_REPO, "speaker_similarity/wavlm_large/wavlm_large.pt")
            hf_hub_download(EVAL_REPO, "speaker_similarity/wavlm_large/hubconf.py")
            # ECAPA_TDNN_WAVLM loads the hubconf from dirname(ssl_model_path): pass the folder
            # with a trailing separator so dirname() is the folder itself.
            ssl_dir = str(Path(ssl_pt).parent) + "/"
            model = ECAPA_TDNN_WAVLM(feat_dim=1024, channels=512, emb_dim=256, sr=16000, ssl_model_path=ssl_dir)
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state["model"], strict=False)
            self._sim = model.to(self.device).eval()
        return self._sim

    def _utmos_model(self):
        if self._utmos is None:
            import torch
            from huggingface_hub import hf_hub_download
            from omnivoice.eval.models.utmos import UTMOS22Strong

            model = UTMOS22Strong()
            model.load_state_dict(torch.load(hf_hub_download(EVAL_REPO, "mos/utmos22_strong_step7459_v1.pt"),
                                             map_location="cpu"))
            self._utmos = model.to(self.device).eval()
        return self._utmos

    @lru_cache(maxsize=4096)
    def embedding(self, path: str):
        import torch

        with torch.no_grad():
            return self._sim_model()([_load_16k(path).to(self.device)]).float().cpu()

    def sim(self, generated: str, reference: str) -> float:
        import torch

        a, b = self.embedding(generated), self.embedding(reference)
        return float(torch.nn.functional.cosine_similarity(a, b, dim=-1))

    def utmos(self, path: str) -> float:
        import torch

        with torch.no_grad():
            return float(self._utmos_model()(_load_16k(path).to(self.device)[None], 16000))

    def release(self) -> None:
        import torch

        self._sim = self._utmos = None
        self.embedding.cache_clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def bootstrap_ci(values, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% bootstrap interval."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
