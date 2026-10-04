"""PyTorch reference for checking the WebGPU engine (tools/check.html).

Runs the upstream OmniVoice model with the weights the web app runs (dequantized from the web
build, the LoRA adapter merged the way the app merges it) on fixed inputs, and dumps:

  meta.json          inputs: text, voice, packed sequence layout, token ids
  ref_tokens.i32     reference-voice audio tokens [8, Tr]
  state.i32          partially unmasked target tokens [8, T] (the decoding state of the step)
  hidden.f32         final-normed hidden states of the target rows, cond then uncond [2T, 1024]
  logits.f32         audio-head logits of those rows [2T, 8 * 1025]
  cfg_tok.i32        guided greedy token per (codebook, frame) [8, T]
  cfg_score.f32      its log-probability [8, T]
  codes.i32          tokens for the decoder check [8, Tc]
  audio.f32          the Higgs decoder's output for them (original fp32 codec)

Usage:
  python demo/tools/reference.py --precision int8 --out demo/out/dump_int8
  python demo/tools/reference.py --precision int8 --adapter local/models/run2/adapter_model.safetensors --out demo/out/dump_int8_run2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEXT = "नमस्ते, म नेपाली भाषामा बोल्न सक्छु। आज मौसम राम्रो छ।"
REF_TEXT = "यो एउटा छोटो उदाहरण वाक्य हो।"


def web_state_dict(path: str, adapter: str | None, alpha_over_r: float) -> dict[str, torch.Tensor]:
    """The backbone the app runs: dequantized web weights, LoRA merged in the stored space."""
    lora = safe_open(adapter, "pt") if adapter else None
    sd: dict[str, torch.Tensor] = {}
    with safe_open(path, "pt") as f:
        keys = list(f.keys())
        bases = sorted({k[: -len("comfy_quant")] for k in keys if k.endswith("comfy_quant")})
        for b in bases:
            t = {s: f.get_tensor(b + s) for s in ("weight", "weight_scale", "comfy_quant", "weight_s_rel", "weight_s_channel", "weight_codebook") if b + s in keys}
            meta = json.loads(bytes(t["comfy_quant"].tolist()).decode())
            if b == "llm.embed_tokens.":
                sd[b + "weight"] = t["weight"].float() * t["weight_scale"].float()
                continue
            # stored (rotated) row space as the GPU holds it
            if meta["format"] == "int8_tensorwise":
                w_rot = t["weight"].float() * t["weight_scale"].float()
            else:
                w_rot = quant.rotate(quant.dequantize_w4a8(t))
            name = "base_model.model." + b + "lora_A.weight"
            if lora is not None and name in lora.keys():
                a = quant.rotate(lora.get_tensor(name).float())  # A H
                bm = lora.get_tensor("base_model.model." + b + "lora_B.weight").float()
                w_rot = w_rot + alpha_over_r * (bm @ a)
                scale = (w_rot.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
                w_rot = (w_rot / scale).round().clamp(-128, 127) * scale
            sd[b + "weight"] = quant.rotate(w_rot)
        for k in keys:
            if any(k.startswith(b) for b in bases):
                continue
            v = f.get_tensor(k).float()
            name = "base_model.model." + k[: -len("weight")] + "lora_A.weight"
            if lora is not None and name in lora.keys():
                a = lora.get_tensor(name).float()
                bm = lora.get_tensor(name.replace("lora_A", "lora_B")).float()
                v = (v + alpha_over_r * (bm @ a)).to(torch.bfloat16).float()
            sd[k] = v
    if lora is not None:
        for k in ("audio_embeddings.weight", "audio_heads.weight"):
            sd[k] = lora.get_tensor("base_model.model." + k).to(torch.bfloat16).float()
    return sd


def dump(out: str, name: str, a) -> None:
    a = a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
    a.astype(np.float32 if name.endswith(".f32") else np.int32).tofile(os.path.join(out, name))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=os.path.join(HERE, "..", "models"))
    ap.add_argument("--precision", default="int8")
    ap.add_argument("--adapter", help="PEFT adapter .safetensors to merge (run1/run2)")
    ap.add_argument("--alpha-over-r", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from omnivoice.models.omnivoice import OmniVoice, _get_time_steps  # noqa: F401
    from transformers import AutoTokenizer, HiggsAudioV2TokenizerModel

    manifest = json.load(open(os.path.join(args.models, "manifest.json")))
    src = manifest["source"]
    from huggingface_hub import snapshot_download

    snap = snapshot_download(src["repo"], revision=src["revision"])
    model = OmniVoice.from_pretrained(snap, train=True, dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
    sd = web_state_dict(os.path.join(args.models, manifest["llm"][args.precision]["path"]), args.adapter, args.alpha_over_r)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert set(missing) <= {"codebook_layer_offsets"}, missing
    tok = AutoTokenizer.from_pretrained(os.path.join(args.models, "tokenizer"))
    model.text_tokenizer = tok

    # ---- one decoding step, packed exactly as OmniVoice._generate_iterative batches it
    Tr, T, C = 40, 60, 8
    ref_tokens = torch.randint(0, 1024, (C, Tr))
    state = torch.randint(0, 1024, (C, T))
    state[torch.rand(C, T) < 0.6] = 1024  # 60% still masked
    inp = model._prepare_inference_inputs(TEXT, T, REF_TEXT + "", ref_tokens, "npi", None, True)
    style = "<|denoise|><|lang_start|>npi<|lang_end|><|instruct_start|>None<|instruct_end|>"
    style_ids = tok(style, return_tensors="pt").input_ids[0].tolist()
    ids = inp["input_ids"].clone()
    c_len = ids.shape[2]
    ids[0, :, c_len - T:] = state.to(ids.device)
    batch_ids = torch.full((2, C, c_len), 1024, dtype=torch.long, device=dev)
    batch_mask = torch.zeros((2, c_len), dtype=torch.bool, device=dev)
    attn = torch.zeros((2, 1, c_len, c_len), dtype=torch.bool, device=dev)
    batch_ids[0] = ids[0]
    batch_mask[0] = inp["audio_mask"][0]
    attn[0, :, :, :] = True
    batch_ids[1, :, :T] = state
    batch_mask[1, :T] = True
    attn[1, :, :T, :T] = True
    pad = torch.arange(T, c_len, device=dev)
    attn[1, :, pad, pad] = True
    with torch.no_grad():
        emb = model._prepare_embed_inputs(batch_ids, batch_mask)
        hs = model.llm(inputs_embeds=emb, attention_mask=attn, return_dict=True)[0]
        hidden = torch.cat([hs[0, c_len - T:], hs[1, :T]])
        logits = model.audio_heads(hidden)
        lc = logits[:T].view(T, C, 1025).permute(1, 0, 2).unsqueeze(0)
        lu = logits[T:].view(T, C, 1025).permute(1, 0, 2).unsqueeze(0)

        class G:
            guidance_scale = 2.0
            class_temperature = 0.0

        pred, score = model._predict_tokens_with_scoring(lc, lu, G)
    text_ids = tok(f"<|text_start|>{REF_TEXT} {TEXT}<|text_end|>", add_special_tokens=False).input_ids
    meta = {
        "text": TEXT, "ref_text": REF_TEXT, "Tr": Tr, "T": T, "c_len": c_len, "style_ids": style_ids,
        "text_ids": text_ids, "precision": args.precision, "adapter": args.adapter, "guidance": 2.0,
    }
    assert len(style_ids) + len(text_ids) + Tr + T == c_len, (len(style_ids), len(text_ids), c_len)
    dump(args.out, "ref_tokens.i32", ref_tokens)
    dump(args.out, "state.i32", state)
    dump(args.out, "hidden.f32", hidden)
    dump(args.out, "logits.f32", logits)
    dump(args.out, "cfg_tok.i32", pred[0])
    dump(args.out, "cfg_score.f32", score[0])

    # ---- codec decoder on tokens with a plausible structure (encode a synthetic sweep)
    codec = HiggsAudioV2TokenizerModel.from_pretrained(os.path.join(snap, "audio_tokenizer")).to(dev).eval()
    sr = 24000
    t = torch.arange(int(sr * 1.6)) / sr
    wav = (0.3 * torch.sin(2 * torch.pi * (120 + 200 * t) * t) * (1 + 0.5 * torch.sin(2 * torch.pi * 3 * t))).float()
    wav = wav[: (wav.numel() // 960) * 960]
    with torch.no_grad():
        codes = codec.encode(wav.view(1, 1, -1).to(dev)).audio_codes[0]
        audio = codec.decode(codes.unsqueeze(0)).audio_values[0, 0]
    dump(args.out, "codes.i32", codes)
    dump(args.out, "audio.f32", audio)
    meta["Tc"] = int(codes.shape[1])
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {args.out}: c_len {c_len}, T {T}, codes {tuple(codes.shape)}, audio {audio.numel()} samples")


if __name__ == "__main__":
    main()
