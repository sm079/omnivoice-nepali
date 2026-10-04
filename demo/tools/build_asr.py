"""Web build of the Nepali ASR used to transcribe cloning references: AI4Bharat IndicConformer
600M (multilingual), its Conformer encoder and the CTC head restricted to Nepali.

The checkpoint is an ONNX export (gated: accept its terms on Hugging Face first). Its weights are
mapped back to module names through the graph's node names (MatMul weights are stored [in, out]
and unnamed). Writes:

  models/
    indicconformer-ne.<int8|bf16>.safetensors   encoder + Nepali CTC head
    indicconformer-ne.json                       mel front end (window, filterbank), vocabulary

Layout notes:
  - linears as [out, in]; int8 builds use per-channel int8 + ConvRot like the TTS backbone
  - subsampling convs NHWC; the projection after them is permuted to the NHWC flatten order
  - the 24 depthwise convs come with their BatchNorm already folded in by the export
  - folded: the input scale sqrt(1024) into the subsampling projection, the macaron 0.5 into each
    feed-forward's second linear; the conv module's GLU halves interleaved for a fused epilogue

  uv run --with onnx python demo/tools/build_asr.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant  # noqa: E402

REPO = "ai4bharat/indic-conformer-600m-multilingual"
LANG = "ne"


def assets_dir() -> str:
    from huggingface_hub import snapshot_download

    return os.path.join(snapshot_download(REPO), "assets")


def load_graph(path: str):
    import onnx
    from onnx import numpy_helper

    m = onnx.load(path, load_external_data=False)
    base = os.path.dirname(path)
    init = {}
    for t in m.graph.initializer:
        if t.data_location == onnx.TensorProto.EXTERNAL:
            loc = {e.key: e.value for e in t.external_data}
            raw = np.fromfile(os.path.join(base, loc["location"]), dtype=np.uint8)
            off = int(loc.get("offset", 0))
            n = int(loc.get("length", raw.size - off))
            dt = onnx.helper.tensor_dtype_to_np_dtype(t.data_type)
            init[t.name] = raw[off : off + n].view(dt).reshape(tuple(t.dims)).copy()
        else:
            init[t.name] = numpy_helper.to_array(t)
    consts = {}
    for n in m.graph.node:
        if n.op_type == "Constant" and n.attribute and n.attribute[0].type == onnx.AttributeProto.TENSOR:
            t = n.attribute[0].t
            if t.data_location == onnx.TensorProto.EXTERNAL:
                loc = {e.key: e.value for e in t.external_data}
                dt = onnx.helper.tensor_dtype_to_np_dtype(t.data_type)
                consts[n.output[0]] = np.fromfile(os.path.join(base, loc["location"]), dtype=dt).reshape(tuple(t.dims))
            else:
                consts[n.output[0]] = numpy_helper.to_array(t)
    return m.graph, init, consts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", help="the model's assets/ folder (default: from the Hugging Face cache)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    ap.add_argument("--precision", nargs="+", default=["int8"], choices=["int8", "bf16"])
    args = ap.parse_args()
    src = args.assets or assets_dir()
    out = os.path.abspath(args.out)

    graph, init, consts = load_graph(os.path.join(src, "encoder.onnx"))
    # weight of the node with this name prefix: the first initializer input with a weight's shape
    by_node = {}
    for n in graph.node:
        for i in n.input:
            if i in init and init[i].ndim >= 1:
                by_node.setdefault(n.name, []).append(i)

    def w(node: str, k: int = 0) -> np.ndarray:
        return init[by_node[node][k]]

    t: dict[str, np.ndarray] = {}
    t["pre.conv0.weight"] = init["pre_encode.conv.0.weight"].reshape(256, 9)  # [c, ky*3+kx] (cin 1)
    t["pre.conv0.bias"] = init["pre_encode.conv.0.bias"]
    for i, (dw, pw) in enumerate([(2, 3), (5, 6)]):
        t[f"pre.dw{i}.weight"] = init[f"pre_encode.conv.{dw}.weight"].reshape(256, 9)
        t[f"pre.dw{i}.bias"] = init[f"pre_encode.conv.{dw}.bias"]
        t[f"pre.pw{i}.weight"] = init[f"pre_encode.conv.{pw}.weight"].reshape(256, 256)
        t[f"pre.pw{i}.bias"] = init[f"pre_encode.conv.{pw}.bias"]
    proj = w("/pre_encode/out/MatMul").T  # [1024, 2560], input index c * 10 + f
    t["pre.out.weight"] = proj.reshape(1024, 256, 10).transpose(0, 2, 1).reshape(1024, 2560)  # -> f * 256 + c
    t["pre.out.bias"] = init["pre_encode.out.bias"]
    # the positional encoding scales its input by sqrt(d_model) = 32: folded into the projection
    t["pre.out.weight"] = t["pre.out.weight"] * 32.0
    t["pre.out.bias"] = t["pre.out.bias"] * 32.0
    pe_table = next(v for v in consts.values() if v.ndim == 3 and v.shape[1] > 9000)
    n_layers = 1 + max(int(n.name.split("/")[1].split(".")[1]) for n in graph.node if n.name.startswith("/layers."))
    for i in range(n_layers):
        p, q = f"/layers.{i}/", f"layers.{i}."
        for norm in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
            t[q + norm + ".weight"] = init[f"layers.{i}.{norm}.weight"]
            t[q + norm + ".bias"] = init[f"layers.{i}.{norm}.bias"]
        for ff in ("feed_forward1", "feed_forward2"):
            for lin in ("linear1", "linear2"):
                # macaron half step: x + 0.5 * ff(x), the 0.5 folded into linear2
                half = 0.5 if lin == "linear2" else 1.0
                t[q + f"{ff}.{lin}.weight"] = w(f"{p}{ff}/{lin}/MatMul").T * half
                t[q + f"{ff}.{lin}.bias"] = init[f"layers.{i}.{ff}.{lin}.bias"] * half
        a = p + "self_attn/"
        t[q + "attn.qkv.weight"] = np.concatenate([w(a + f"linear_{x}/MatMul").T for x in "qkv"])
        t[q + "attn.qkv.bias"] = np.concatenate([init[f"layers.{i}.self_attn.linear_{x}.bias"] for x in "qkv"])
        t[q + "attn.pos.weight"] = w(a + "linear_pos/MatMul").T
        t[q + "attn.out.weight"] = w(a + "linear_out/MatMul").T
        t[q + "attn.out.bias"] = init[f"layers.{i}.self_attn.linear_out.bias"]
        t[q + "attn.pos_bias_u"] = init[f"layers.{i}.self_attn.pos_bias_u"].reshape(-1)
        t[q + "attn.pos_bias_v"] = init[f"layers.{i}.self_attn.pos_bias_v"].reshape(-1)
        # GLU(a, b) = a * sigmoid(b) over the two halves: rows interleaved (a_0, b_0, a_1, ...)
        order = np.arange(2048).reshape(2, 1024).T.reshape(-1)
        t[q + "conv.pw1.weight"] = init[f"layers.{i}.conv.pointwise_conv1.weight"][:, :, 0][order]
        t[q + "conv.pw1.bias"] = init[f"layers.{i}.conv.pointwise_conv1.bias"][order]
        dw = by_node[p + "conv/depthwise_conv/Conv"]
        t[q + "conv.dw.weight"] = init[dw[0]][:, 0, :]  # [1024, 9], BatchNorm folded
        t[q + "conv.dw.bias"] = init[dw[1]]
        t[q + "conv.pw2.weight"] = init[f"layers.{i}.conv.pointwise_conv2.weight"][:, :, 0]
        t[q + "conv.pw2.bias"] = init[f"layers.{i}.conv.pointwise_conv2.bias"]

    # CTC head, Nepali rows only
    _, cinit, _ = load_graph(os.path.join(src, "ctc_decoder.onnx"))
    mask = np.array(json.load(open(os.path.join(src, "language_masks.json")))[LANG], dtype=bool)
    t["ctc.weight"] = cinit["language_ids"][mask, :, 0]
    t["ctc.bias"] = cinit["decoder_layers.0.bias"][mask]

    # mel front end constants
    pre = torch.jit.load(os.path.join(src, "preprocessor.ts"), map_location="cpu")
    c = pre.code_with_constants[1].const_mapping
    vocab = json.load(open(os.path.join(src, "vocab.json"), encoding="utf-8"))[LANG]
    center = (pe_table.shape[1] - 1) // 2
    front = {
        "sample_rate": 16000,
        "n_fft": 512,
        "hop": int(c["c0"]),
        "preemph": float(c["c2"]),
        "log_guard": float(c["c5"]),
        "std_eps": float(c["c6"]),
        "window": c["c3"].tolist(),
        "fb": c["c4"].T.contiguous().flatten().tolist(),  # [80, 257]
        "subsampling": 8,
        "layers": n_layers,
        "vocab": vocab,
        "blank": 256,  # BLANK_ID: the last entry of the Nepali subset
    }
    # relative positions T-1 .. -(T-1) are pe rows center-(T-1) .. center+(T-1); keep +-1500 (2 min)
    keep = 1500
    t["pos_emb"] = pe_table[0, center - keep : center + keep + 1]
    front["pos_center"] = keep

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "indicconformer-ne.json"), "w", encoding="utf-8") as f:
        json.dump(front, f, ensure_ascii=False)
    for prec in args.precision:
        tensors: dict[str, torch.Tensor] = {}
        for k, v in t.items():
            v = torch.from_numpy(np.ascontiguousarray(v)).float()
            is_linear = k.endswith(".weight") and v.dim() == 2 and k.startswith("layers.") and ".dw." not in k
            if prec == "int8" and is_linear and quant.can_convrot(v):
                for s, x in quant.quantize_int8_convrot(v).items():
                    tensors[k[: -len("weight")] + s] = x
            elif v.dim() == 2 and v.numel() > 65536 and k != "pos_emb":
                tensors[k] = v.to(torch.bfloat16)
            else:
                tensors[k] = v
        path = os.path.join(out, f"indicconformer-ne.{prec}.safetensors")
        save_file(tensors, path, metadata={"format": "pt", "omnivoice_web": "asr", "quant": prec})
        print(f"wrote {path}: {os.path.getsize(path) / 2**20:.1f} MiB")

    manifest_path = os.path.join(out, "manifest.json")
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
        manifest["asr"] = {
            "config": "indicconformer-ne.json",
            **{
                p: {
                    "path": f"indicconformer-ne.{p}.safetensors",
                    "size": os.path.getsize(os.path.join(out, f"indicconformer-ne.{p}.safetensors")),
                }
                for p in args.precision
            },
        }
        json.dump(manifest, open(manifest_path, "w"), indent=2)


if __name__ == "__main__":
    main()
