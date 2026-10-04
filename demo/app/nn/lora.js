// Fine-tuned models as PEFT LoRA adapters, merged into the backbone weights on the GPU while they
// load: W' = W + (alpha / r) * B A. The adapters also carry fully trained audio embeddings and
// audio heads (PEFT modules_to_save), which replace the base model's.
//
// Weights are stored fused (q/k/v stacked, gate/up interleaved), so one merge covers every part
// of a fused weight: A_cat stacks the parts' A matrices, and B_cat gives each part's rows its own
// columns (zeros elsewhere). int8 weights are stored ConvRot-rotated (W H), so A is rotated the
// same way (A H) and the update lands in the rotated space; the merged row is then requantized.

import { SafeTensors } from "../weights.js";
import * as ops from "../gpu/ops.js";

const PREFIX = "base_model.model.";

function f32ToBf16Bytes(f) {
  const u32 = new Uint32Array(f.buffer, f.byteOffset, f.length);
  const out = new Uint16Array(f.length);
  for (let i = 0; i < f.length; i++) {
    const u = u32[i];
    out[i] = (u + 0x7fff + ((u >>> 16) & 1)) >>> 16;
  }
  return new Uint8Array(out.buffer);
}

// x H per group of 256 (in place), the transform the kernels apply to activations
export function hadamardRows(data, rows, K) {
  for (let r = 0; r < rows; r++) {
    for (let g = 0; g < K; g += 256) {
      const base = r * K + g;
      for (let stride = 1; stride < 256; stride *= 4) {
        for (let t = 0; t < 64; t++) {
          const i0 = base + Math.floor(t / stride) * stride * 4 + (t % stride);
          const a = data[i0], b = data[i0 + stride], c = data[i0 + 2 * stride], e = data[i0 + 3 * stride];
          data[i0] = a + b + c - e;
          data[i0 + stride] = a + b - c + e;
          data[i0 + 2 * stride] = a - b + c + e;
          data[i0 + 3 * stride] = -a + b + c + e;
        }
      }
      for (let i = 0; i < 256; i++) data[base + i] *= 0.0625;
    }
  }
}

// blob: the adapter .safetensors; config: { lora_alpha, r } from adapter_config.json
export async function loadAdapter(blob, config) {
  const st = await SafeTensors.open(blob);
  const need = (k) => {
    if (!st.has(k)) throw new Error(`the adapter has no ${k}; is it an OmniVoice LoRA?`);
    return k;
  };
  const [emb, heads] = await Promise.all([
    st.f32(need(PREFIX + "audio_embeddings.weight")),
    st.f32(need(PREFIX + "audio_heads.weight")),
  ]);
  return {
    st,
    scale: config.lora_alpha / config.r,
    audioEmbeddings: f32ToBf16Bytes(emb),
    audioHeads: f32ToBf16Bytes(heads),
  };
}

export async function mergeIntoLinear(gpu, W, lora) {
  const parts = [];
  for (const p of W.parts) {
    const a = PREFIX + p.name + ".lora_A.weight";
    const b = PREFIX + p.name + ".lora_B.weight";
    if (!lora.st.has(a)) continue;
    const r = lora.st.info(a).shape[0];
    parts.push({ ...p, r, A: await lora.st.f32(a), B: await lora.st.f32(b) });
  }
  if (!parts.length) return W;
  const rt = parts.reduce((s, p) => s + p.r, 0);
  const A = new Float32Array(rt * W.K);
  const B = new Float32Array(W.N * rt);
  let c = 0;
  for (const p of parts) {
    A.set(p.A, c * W.K);
    for (let i = 0; i < p.n; i++) B.set(p.B.subarray(i * p.r, (i + 1) * p.r), (p.off + i * p.stride) * rt + c);
    c += p.r;
  }
  if (ops.needsRotation(W)) hadamardRows(A, rt, W.K);
  const Ab = gpu.upload(A);
  const Bb = gpu.upload(B);
  ops.mergeLora(gpu, W, Ab, Bb, rt, lora.scale);
  gpu.flush();
  Ab.destroy();
  Bb.destroy();
  return W;
}
