"""PyTorch/ONNX reference for the browser ASR (tools/check.html?asr=1): runs AI4Bharat's own
IndicConformer pipeline (TorchScript mel front end, ONNX encoder and CTC head) on a preset voice
and dumps its intermediates.

  python demo/tools/reference_asr.py --out demo/out/asr
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from unittest import mock

import numpy as np
import soundfile as sf
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=os.path.join(HERE, "..", "models"))
    ap.add_argument("--wav", help="speech to transcribe (default: a preset voice)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoModel

    with mock.patch("torch.cuda.is_available", return_value=False):
        m = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)
    wav, sr = sf.read(args.wav or os.path.join(args.models, "voices", "hari.wav"), dtype="float32", always_2d=True)
    x = torchaudio.functional.resample(torch.from_numpy(wav.mean(axis=1))[None], sr, 16000)
    with torch.no_grad():
        feats, lengths = m.models["preprocessor"](input_signal=x, length=torch.tensor([x.shape[-1]]))
        enc, enc_len = m.models["encoder"].run(
            ["outputs", "encoded_lengths"], {"audio_signal": feats.numpy(), "length": lengths.numpy()}
        )
        logprobs = m.models["ctc_decoder"].run(["logprobs"], {"encoder_output": enc})[0]
    mask = np.array(m.language_masks["ne"], dtype=bool)
    text = m(x, "ne", "ctc")
    x.numpy().astype(np.float32).tofile(os.path.join(args.out, "wav16.f32"))
    feats[0].T.numpy().astype(np.float32).tofile(os.path.join(args.out, "mel.f32"))  # [T, 80]
    enc[0].T.astype(np.float32).tofile(os.path.join(args.out, "enc.f32"))  # [T', 1024]
    logprobs[0][:, mask].astype(np.float32).tofile(os.path.join(args.out, "ctc.f32"))  # [T', 257] (pre-softmax)
    meta = {"T": int(feats.shape[2]), "L": int(enc.shape[2]), "text": str(text).strip()}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
    sys.stdout.reconfigure(encoding="utf-8")
    print(meta)


if __name__ == "__main__":
    main()
