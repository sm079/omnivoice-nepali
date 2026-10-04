"""Preset voices for the demo: synthetic Nepali voices made with OmniVoice's voice design.

For each voice description (gender, age, pitch), the base model speaks a Nepali sentence with
several seeds; every take is transcribed (IndicConformer) and scored (UTMOS), and the take with
the lowest character error rate (then highest UTMOS) becomes the voice prompt: its audio tokens,
its text and its loudness. The app uses them like any voice-cloning reference, so no real
person's voice is shipped.

  uv sync --extra tools
  python demo/tools/build_voices.py                 # -> demo/models/voices.json (+ voices/*.wav previews)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from tools.tts_bench.asr import NepaliASR  # noqa: E402
from tools.tts_bench.metrics import Scorer, cer  # noqa: E402

REPO = "k2-fsa/OmniVoice"
REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"

# id, label, instruct
VOICES = [
    ("asha", "Asha", "female, young adult, moderate pitch"),
    ("sita", "Sita", "female, middle-aged, low pitch"),
    ("maya", "Maya", "female, young adult, high pitch"),
    ("bikash", "Bikash", "male, young adult, moderate pitch"),
    ("hari", "Hari", "male, middle-aged, low pitch"),
    ("ram", "Ram", "male, elderly, moderate pitch"),
]
SENTENCES = [
    "नमस्कार, आज म तपाईंलाई एउटा छोटो कथा सुनाउन चाहन्छु।",
    "हिमालको काखमा बसेको यो सानो गाउँ साँच्चै सुन्दर छ।",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "models"))
    ap.add_argument("--takes", type=int, default=6, help="seeds per voice and sentence")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "voices"), exist_ok=True)

    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig

    model = OmniVoice.from_pretrained(REPO, revision=REVISION, dtype=torch.float32).to("cuda").eval()
    sr = model.sampling_rate
    asr = NepaliASR()
    scorer = Scorer()
    cfg = OmniVoiceGenerationConfig()
    voices = []
    for vid, label, instruct in VOICES:
        best = None
        for text in SENTENCES:
            for seed in range(args.takes):
                torch.manual_seed(seed)
                task = model._preprocess_all(text=text, language="npi", instruct=instruct)
                with torch.inference_mode():
                    tokens = model._generate_iterative(task, cfg)[0]
                    wav = model.audio_tokenizer.decode(tokens.unsqueeze(0)).audio_values[0, 0].cpu().numpy()
                path = os.path.join(out, "voices", "_take.wav")
                sf.write(path, wav, sr)
                err = cer(text, asr.transcribe_array(wav, sr))
                mos = scorer.utmos(path)
                key = (round(err, 3), -mos)
                print(f"{vid} seed {seed}: CER {err:.3f} UTMOS {mos:.2f}  {text[:20]}", flush=True)
                if best is None or key < best[0]:
                    best = (key, text, tokens.cpu().numpy().astype(np.int32), wav)
        (err, neg_mos), text, tokens, wav = best
        sf.write(os.path.join(out, "voices", f"{vid}.wav"), wav, sr)
        voices.append(
            {
                "id": vid,
                "label": label,
                "instruct": instruct,
                "text": text,
                "frames": int(tokens.shape[1]),
                "tokens": tokens.flatten().tolist(),
                "rms": float(np.sqrt(np.mean(wav**2))),
                "cer": err,
                "utmos": -neg_mos,
                "preview": f"voices/{vid}.wav",
            }
        )
        print(f"-> {vid}: CER {err:.3f} UTMOS {-neg_mos:.2f}", flush=True)
    os.remove(os.path.join(out, "voices", "_take.wav"))
    with open(os.path.join(out, "voices.json"), "w", encoding="utf-8") as f:
        json.dump({"version": 1, "voices": voices}, f, ensure_ascii=False)
    print(f"wrote {len(voices)} voices to {out}/voices.json")


if __name__ == "__main__":
    main()
