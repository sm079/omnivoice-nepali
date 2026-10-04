// OmniVoice's backbone: Qwen3-0.6B run bidirectionally (no causal mask), on input embeddings that
// mix text tokens and the sum of 8 audio-codebook embeddings, followed by the audio heads.
//
// One forward pass takes two packed sequences: the conditional one (style + text + reference
// audio + target) in rows [0, seg) and the unconditional one (target only) in rows [seg, L).
// Each attends only to itself and restarts its positions at 0, matching the batched
// cond/uncond forward of OmniVoice._generate_iterative.

import * as ops from "../gpu/ops.js";
import { mergeIntoLinear } from "./lora.js";

export const CFG = { layers: 28, hidden: 1024, heads: 16, kvHeads: 8, headDim: 128, ffn: 3072, theta: 1e6, codebooks: 8, vocab: 1025, mask: 1024 };

export function ropeTable(gpu, L) {
  const half = CFG.headDim / 2;
  const cs = new Float32Array(2 * L * half);
  for (let l = 0; l < L; l++) {
    for (let i = 0; i < half; i++) {
      const a = l * Math.fround(1 / Math.pow(CFG.theta, (2 * i) / CFG.headDim));
      cs[l * half + i] = Math.cos(a);
      cs[L * half + l * half + i] = Math.sin(a);
    }
  }
  const t = gpu.fromArray(cs, [cs.length]);
  t.sinOff = L * half;
  return t;
}

export class OmniLLM {
  // lora: parsed adapter (lora.js) merged into the weights while loading, or null for the base model
  static async load(gpu, st, { lora = null, onProgress } = {}) {
    const m = new OmniLLM(gpu, st);
    m.layers = [];
    for (let i = 0; i < CFG.layers; i++) {
      const p = `llm.layers.${i}.`;
      const a = p + "self_attn.";
      const L = {
        ln1: await st.vector(gpu, p + "input_layernorm.weight"),
        ln2: await st.vector(gpu, p + "post_attention_layernorm.weight"),
        qn: await st.vector(gpu, a + "q_norm.weight"),
        kn: await st.vector(gpu, a + "k_norm.weight"),
        qkv: await st.fused(gpu, [a + "q_proj.", a + "k_proj.", a + "v_proj."]),
        o: await st.linear(gpu, a + "o_proj."),
        gu: await st.fused(gpu, [p + "mlp.gate_proj.", p + "mlp.up_proj."], { interleave: true }),
        down: await st.linear(gpu, p + "mlp.down_proj."),
      };
      if (lora) for (const k of ["qkv", "o", "gu", "down"]) await mergeIntoLinear(gpu, L[k], lora);
      m.layers.push(L);
      onProgress?.((i + 1) / CFG.layers);
    }
    m.norm = await st.vector(gpu, "llm.norm.weight");
    if (lora) {
      m.audioTable = gpu.upload(lora.audioEmbeddings);
      m.heads = { kind: "bf16", N: CFG.codebooks * CFG.vocab, K: CFG.hidden, buf: gpu.upload(lora.audioHeads) };
    } else {
      m.audioTable = await st.bf16(gpu, "audio_embeddings.weight");
      m.heads = { kind: "bf16", N: CFG.codebooks * CFG.vocab, K: CFG.hidden, buf: await st.bf16(gpu, "audio_heads.weight") };
    }
    await gpu.sync();
    return m;
  }

  constructor(gpu, st) {
    this.gpu = gpu;
    this.st = st; // text embedding rows are gathered from the file on demand
  }

  // token ids -> Float32Array [n, hidden]
  textEmbeddings(ids) {
    return this.st.rows("llm.embed_tokens.weight", ids);
  }

  // x: [L, hidden] input embeddings, rows [0, seg) and [seg, L) are two independent sequences.
  // Returns the final-normed hidden states of rows [r0, r0 + n) (x is consumed).
  forward(x, { L, seg, cs, r0, n, onLayer }) {
    const gpu = this.gpu;
    const { hidden: D, heads: H, kvHeads: KH, headDim: HD } = CFG;
    for (const [i, l] of this.layers.entries()) {
      const qRot = ops.needsRotation(l.qkv);
      const h = ops.rmsnorm(gpu, x, l.ln1, D, { rotate: qRot });
      const qkv = ops.linear(gpu, h, l.qkv, { rotated: qRot });
      h.release();
      ops.qkNormRope(gpu, qkv, l.qn, l.kn, cs, L, H, KH, HD, seg);
      const a = gpu.empty([L, H * HD]);
      ops.attention(gpu, qkv, a, { r0: 0, L: seg, H, KH, D: HD });
      ops.attention(gpu, qkv, a, { r0: seg, L: L - seg, H, KH, D: HD });
      qkv.release();
      ops.linear(gpu, a, l.o, { out: x, resid: true });
      a.release();
      const fRot = ops.needsRotation(l.gu);
      const h2 = ops.rmsnorm(gpu, x, l.ln2, D, { rotate: fRot });
      const g = ops.linear(gpu, h2, l.gu, { rotated: fRot, act: "swiglu" });
      h2.release();
      ops.linear(gpu, g, l.down, { out: x, resid: true });
      g.release();
      onLayer?.(i);
    }
    const out = ops.rmsnorm(gpu, x, this.norm, D, { row0: r0, rows: n });
    x.release();
    return out;
  }

  // hidden [n, D] -> logits [n, 8 * 1025]
  logits(h) {
    return ops.linear(this.gpu, h, this.heads);
  }
}
