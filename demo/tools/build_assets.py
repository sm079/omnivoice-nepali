"""One-time offline build of the web assets the demo downloads.

Reads the OmniVoice checkpoint (k2-fsa/OmniVoice, pinned revision) and writes:

  models/
    manifest.json                              components, file sizes, defaults
    omnivoice-llm.<bf16|int8|w4a8>.safetensors Qwen3 backbone, text embeddings, audio embeddings + heads
    higgs-decoder.safetensors                  codebooks folded through project_out and fc2, DAC decoder
    tokenizer/tokenizer.json (+ tokenizer_config.json)

Precisions of the backbone (the 196 attention/MLP linears; everything else stays bf16):
  bf16  original weights
  int8  per-channel int8 + ConvRot (Hadamard-rotated rows), text embedding table per-row int8
  w4a8  4-bit codebook weights on an int8 grid + ConvRot; the first and last layer int8

Codec decoder layout (all convs run as implicit GEMMs over time-major activations [T, C]):
  conv1d weight [Cout, Cin, k]     -> [Cout, k*Cin]          (index tap*Cin + ci)
  conv-transpose [Cin, Cout, 2s]   -> [s, Cout, 2*Cin]       (one GEMM per output phase, see codec.js)
  codebook c: project_out(embed_c) then fc2, folded into one [1024 codes, 256] table per codebook

Usage:
  python demo/tools/build_assets.py                         # every precision -> demo/models/
  python demo/tools/build_assets.py --llm int8 --out demo/models
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant  # noqa: E402

REPO = "k2-fsa/OmniVoice"
REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
N_LAYERS = 28

LINEAR_RE = re.compile(r"^llm\.layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")
UPSAMPLE = [8, 5, 4, 2, 3]


MODEL_LABELS = {
    "base": ("Base", "OmniVoice as released, without Nepali fine-tuning."),
    "run1": ("More natural v1", "Fine-tuned on single-speaker Nepali speech (run 1)."),
    "run2": ("More natural v2", "Fine-tuned on more Nepali speakers (run 2)."),
}


def adapter_entry(name: str, path: str, size: int, cfg: dict) -> dict:
    label, desc = MODEL_LABELS.get(name, (name, ""))
    # the file name in the browser cache must not collide across repos or revisions
    cache = f"adapter-{name}-{size}.safetensors"
    config = {"lora_alpha": cfg["lora_alpha"], "r": cfg["r"]}
    return {
        "id": name,
        "label": label,
        "desc": desc,
        "adapter": {"path": path, "size": size, "cacheName": cache, "config": config},
    }


def snapshot(src: str | None) -> str:
    if src:
        return src
    from huggingface_hub import snapshot_download

    return snapshot_download(REPO, revision=REVISION, allow_patterns=["*.json", "*.safetensors", "audio_tokenizer/*"])


def fmt_size(n: int) -> str:
    return f"{n / 2**30:.2f} GiB" if n > 2**30 else f"{n / 2**20:.1f} MiB"


def build_llm(src: str, out: str, mode: str, device: str) -> None:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(src, "pt") as f:
        keys = [k for k in f.keys() if k != "codebook_layer_offsets"]
        for i, k in enumerate(keys):
            t = f.get_tensor(k)
            base = k[: -len("weight")]
            m = LINEAR_RE.match(k)
            if mode != "bf16" and m and quant.can_convrot(t):
                layer = int(m.group(1))
                use_w4 = mode == "w4a8" and 0 < layer < N_LAYERS - 1
                q = (quant.quantize_w4a8 if use_w4 else quant.quantize_int8_convrot)(t.to(device))
            elif mode != "bf16" and k == "llm.embed_tokens.weight":
                q = quant.quantize_int8_rows(t.to(device))
            else:
                tensors[k] = t.to(torch.bfloat16)
                continue
            for suffix, v in q.items():
                tensors[base + suffix] = v.cpu()
            if i % 40 == 0:
                print(f"  llm {mode}: {i}/{len(keys)}", flush=True)
    save_file(tensors, out, metadata={"format": "pt", "omnivoice_web": "llm", "quant": mode})


def conv_rows(w: torch.Tensor) -> torch.Tensor:
    """[Cout, Cin, k] -> [Cout, k*Cin] with index tap*Cin + ci."""
    return w.permute(0, 2, 1).reshape(w.shape[0], -1).contiguous()


def conv_t_phases(w: torch.Tensor, stride: int) -> torch.Tensor:
    """ConvTranspose1d weight [Cin, Cout, 2s] (padding ceil(s/2)) -> [s, Cout, 2*Cin].

    Output t = q*s + r takes taps j0 + a*s (a in {0, 1}) from input q + (r + pad)//s - a,
    with j0 = (r + pad) % s. Phase r's GEMM weight row co is [W[:, co, j0], W[:, co, j0 + s]].
    """
    cin, cout, k = w.shape
    assert k == 2 * stride
    pad = (stride + 1) // 2
    out = torch.empty(stride, cout, 2 * cin)
    for r in range(stride):
        j0 = (r + pad) % stride
        out[r, :, :cin] = w[:, :, j0].T
        out[r, :, cin:] = w[:, :, j0 + stride].T
    return out.contiguous()


def build_decoder(src: str, out: str) -> None:
    t: dict[str, torch.Tensor] = {}
    with safe_open(src, "pt") as f:
        g = lambda k: f.get_tensor(k).float()  # noqa: E731
        # codebooks: x = sum_c fc2(project_out_c(embed_c[code_c])) + fc2.bias, folded per codebook
        fc2_w, fc2_b = g("fc2.weight"), g("fc2.bias")
        for c in range(8):
            p = f"quantizer.quantizers.{c}."
            proj = g(p + "codebook.embed") @ g(p + "project_out.weight").T + g(p + "project_out.bias")
            t[f"codebook.{c}"] = (proj @ fc2_w.T).contiguous()
        t["codebook.bias"] = fc2_b
        d = "acoustic_decoder."
        t["conv1.weight"] = conv_rows(g(d + "conv1.weight"))
        t["conv1.bias"] = g(d + "conv1.bias")
        for b, s in enumerate(UPSAMPLE):
            p = f"{d}block.{b}."
            q = f"block.{b}."
            t[q + "snake.alpha"] = g(p + "snake1.alpha").flatten()
            t[q + "up.weight"] = conv_t_phases(g(p + "conv_t1.weight"), s)
            t[q + "up.bias"] = g(p + "conv_t1.bias")
            for u in (1, 2, 3):
                r = f"{p}res_unit{u}."
                o = f"{q}res{u}."
                t[o + "snake1.alpha"] = g(r + "snake1.alpha").flatten()
                t[o + "conv1.weight"] = conv_rows(g(r + "conv1.weight"))
                t[o + "conv1.bias"] = g(r + "conv1.bias")
                t[o + "snake2.alpha"] = g(r + "snake2.alpha").flatten()
                t[o + "conv2.weight"] = conv_rows(g(r + "conv2.weight"))
                t[o + "conv2.bias"] = g(r + "conv2.bias")
        t["snake.alpha"] = g(d + "snake1.alpha").flatten()
        t["conv2.weight"] = conv_rows(g(d + "conv2.weight"))
        t["conv2.bias"] = g(d + "conv2.bias")
    # big matrices bf16; snake alphas, biases and the tiny output conv stay fp32
    for k, v in t.items():
        if v.dim() >= 2 and v.numel() > 4096 and not k.startswith("codebook."):
            t[k] = v.to(torch.bfloat16)
    save_file(t, out, metadata={"format": "pt", "omnivoice_web": "decoder"})


def build_encoder(src: str, out: str) -> None:
    """Encoder side of the codec, for voice cloning: HuBERT (semantic), its conv encoder, the DAC
    encoder (acoustic), the fusing fc and the residual quantizer. Same conv layout as the decoder;
    HuBERT's grouped positional conv has its weight norm folded and its groups' rows stacked."""
    t: dict[str, torch.Tensor] = {}
    with safe_open(src, "pt") as f:
        g = lambda k: f.get_tensor(k).float()  # noqa: E731
        h = "semantic_model."
        for i in range(7):
            t[f"hubert.conv.{i}.weight"] = conv_rows(g(f"{h}feature_extractor.conv_layers.{i}.conv.weight"))
        t["hubert.gn.weight"] = g(h + "feature_extractor.conv_layers.0.layer_norm.weight")
        t["hubert.gn.bias"] = g(h + "feature_extractor.conv_layers.0.layer_norm.bias")
        for s in ("weight", "bias"):
            t[f"hubert.fp_ln.{s}"] = g(f"{h}feature_projection.layer_norm.{s}")
            t[f"hubert.fp.{s}"] = g(f"{h}feature_projection.projection.{s}")
            t[f"hubert.ln.{s}"] = g(f"{h}encoder.layer_norm.{s}")
        p = h + "encoder.pos_conv_embed.conv."
        wg, wv = g(p + "parametrizations.weight.original0"), g(p + "parametrizations.weight.original1")
        w = wg * wv / wv.norm(dim=(0, 1), keepdim=True)  # weight norm over dim 2
        t["hubert.pos.weight"] = conv_rows(w)  # [768, 128 * 48], 16 groups of 48 rows
        t["hubert.pos.bias"] = g(p + "bias")
        for i in range(12):
            p = f"{h}encoder.layers.{i}."
            q = f"hubert.layers.{i}."
            a = p + "attention."
            t[q + "qkv.weight"] = torch.cat([g(a + f"{x}_proj.weight") for x in "qkv"])
            t[q + "qkv.bias"] = torch.cat([g(a + f"{x}_proj.bias") for x in "qkv"])
            for s in ("weight", "bias"):
                t[q + f"out.{s}"] = g(a + f"out_proj.{s}")
                t[q + f"ln.{s}"] = g(p + f"layer_norm.{s}")
                t[q + f"ff1.{s}"] = g(p + f"feed_forward.intermediate_dense.{s}")
                t[q + f"ff2.{s}"] = g(p + f"feed_forward.output_dense.{s}")
                t[q + f"fln.{s}"] = g(p + f"final_layer_norm.{s}")
        p = "encoder_semantic."
        t["sem.conv.weight"] = conv_rows(g(p + "conv.weight"))
        # strides [1, 1] and block_dilations [1, 1]: two blocks of two residual units and a conv
        for b in range(2):
            for u in range(2):
                for c in ("conv1", "conv2"):
                    t[f"sem.block.{b}.res{u}.{c}.weight"] = conv_rows(
                        g(p + f"conv_blocks.{b}.res_units.{u}.{c}.weight")
                    )
            t[f"sem.block.{b}.down.weight"] = conv_rows(g(p + f"conv_blocks.{b}.conv.weight"))
            t[f"sem.block.{b}.down.bias"] = g(p + f"conv_blocks.{b}.conv.bias")
        d = "acoustic_encoder."
        t["dac.conv1.weight"] = conv_rows(g(d + "conv1.weight"))
        t["dac.conv1.bias"] = g(d + "conv1.bias")
        for b in range(len(UPSAMPLE)):
            p = f"{d}block.{b}."
            q = f"dac.block.{b}."
            for u in (1, 2, 3):
                r, o = f"{p}res_unit{u}.", f"{q}res{u}."
                t[o + "snake1.alpha"] = g(r + "snake1.alpha").flatten()
                t[o + "conv1.weight"] = conv_rows(g(r + "conv1.weight"))
                t[o + "conv1.bias"] = g(r + "conv1.bias")
                t[o + "snake2.alpha"] = g(r + "snake2.alpha").flatten()
                t[o + "conv2.weight"] = conv_rows(g(r + "conv2.weight"))
                t[o + "conv2.bias"] = g(r + "conv2.bias")
            t[q + "snake.alpha"] = g(p + "snake1.alpha").flatten()
            t[q + "down.weight"] = conv_rows(g(p + "conv1.weight"))
            t[q + "down.bias"] = g(p + "conv1.bias")
        t["dac.snake.alpha"] = g(d + "snake1.alpha").flatten()
        t["dac.conv2.weight"] = conv_rows(g(d + "conv2.weight"))
        t["dac.conv2.bias"] = g(d + "conv2.bias")
        t["fc.weight"], t["fc.bias"] = g("fc.weight"), g("fc.bias")
        for c in range(8):
            p = f"quantizer.quantizers.{c}."
            t[f"rvq.{c}.in.weight"] = g(p + "project_in.weight")
            t[f"rvq.{c}.in.bias"] = g(p + "project_in.bias")
            t[f"rvq.{c}.embed"] = g(p + "codebook.embed")
            t[f"rvq.{c}.out.weight"] = g(p + "project_out.weight")
            t[f"rvq.{c}.out.bias"] = g(p + "project_out.bias")
    # big matrices bf16 (the quantizer stays fp32: its nearest-code search runs on the CPU)
    for k, v in t.items():
        if v.dim() >= 2 and v.numel() > 4096 and not k.startswith("rvq."):
            t[k] = v.to(torch.bfloat16)
    save_file(t, out, metadata={"format": "pt", "omnivoice_web": "encoder"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="local OmniVoice snapshot (default: download the pinned revision)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    ap.add_argument("--llm", nargs="+", default=["bf16", "int8", "w4a8"], choices=["bf16", "int8", "w4a8"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true", help="rebuild files that already exist")
    ap.add_argument(
        "--adapter",
        action="append",
        metavar="NAME=PATH",
        help="local PEFT adapter, copied to adapters/NAME.safetensors",
    )
    ap.add_argument(
        "--adapter-repo", metavar="REPO@REV", help="Hub repo with <name>.safetensors adapters (linked, not copied)"
    )
    args = ap.parse_args()

    src = snapshot(args.src)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "tokenizer"), exist_ok=True)

    manifest_path = os.path.join(out, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    manifest.update({"version": 1, "source": {"repo": REPO, "revision": REVISION}})
    manifest.setdefault("llm", {})

    for mode in args.llm:
        name = f"omnivoice-llm.{mode}.safetensors"
        path = os.path.join(out, name)
        if args.force or not os.path.exists(path):
            print(f"building {name}", flush=True)
            build_llm(os.path.join(src, "model.safetensors"), path, mode, args.device)
        manifest["llm"][mode] = {"path": name, "size": os.path.getsize(path)}
        print(f"  {name}: {fmt_size(os.path.getsize(path))}")

    name = "higgs-decoder.safetensors"
    path = os.path.join(out, name)
    if args.force or not os.path.exists(path):
        print(f"building {name}", flush=True)
        build_decoder(os.path.join(src, "audio_tokenizer", "model.safetensors"), path)
    manifest["decoder"] = {"path": name, "size": os.path.getsize(path)}
    print(f"  {name}: {fmt_size(os.path.getsize(path))}")

    name = "higgs-encoder.safetensors"
    path = os.path.join(out, name)
    if args.force or not os.path.exists(path):
        print(f"building {name}", flush=True)
        build_encoder(os.path.join(src, "audio_tokenizer", "model.safetensors"), path)
    manifest["encoder"] = {"path": name, "size": os.path.getsize(path)}
    print(f"  {name}: {fmt_size(os.path.getsize(path))}")

    for fn in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(os.path.join(src, fn), os.path.join(out, "tokenizer", fn))
    manifest["tokenizer"] = "tokenizer/tokenizer.json"

    # fine-tuned models: PEFT adapters merged in the browser, from a local run or the Hub
    models = [{"id": "base", "label": MODEL_LABELS["base"][0], "desc": MODEL_LABELS["base"][1]}]
    for spec in args.adapter or []:
        name, path = spec.split("=", 1)
        cfg = json.load(open(os.path.join(os.path.dirname(path), "adapter_config.json")))
        dst = os.path.join(out, "adapters", f"{name}.safetensors")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(path, dst)
        models.append(adapter_entry(name, f"adapters/{name}.safetensors", os.path.getsize(dst), cfg))
    if args.adapter_repo:
        from huggingface_hub import HfApi, hf_hub_download

        repo, rev = args.adapter_repo.split("@")
        info = HfApi().model_info(repo, revision=rev, files_metadata=True)
        cfg = json.load(open(hf_hub_download(repo, "adapter_config.json", revision=rev)))
        for s in info.siblings:
            if s.rfilename.endswith(".safetensors"):
                name = s.rfilename[: -len(".safetensors")]
                url = f"https://huggingface.co/{repo}/resolve/{info.sha}/{s.rfilename}"
                models.append(adapter_entry(name, url, s.size, cfg))
    if len(models) > 1 or "models" not in manifest:
        manifest["models"] = models

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
