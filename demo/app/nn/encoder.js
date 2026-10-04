// Higgs Audio v2 tokenizer, encoder side: 24 kHz audio -> 8 codebook indices per 40 ms frame,
// for voice cloning (HiggsAudioV2TokenizerModel.encode).
//
//   semantic: 16 kHz audio (padded 160 each side) -> HuBERT, mean of its 13 hidden states, every
//             second frame -> conv encoder (conv, then two blocks of two ELU residual units + conv)
//   acoustic: DAC encoder on the 24 kHz audio (padded 480 each side when its length needs it)
//   [acoustic | semantic] -> fc -> 8-stage residual vector quantization (nearest code, on the CPU)

import * as ops from "../gpu/ops.js";
import { resample } from "../resample.js";

const STRIDES = [8, 5, 4, 2, 3];
const PAD = 480; // hop / 2

async function weight(gpu, st, base, { N, bias = true } = {}) {
  const t = st.info(base + ".weight");
  const kind = t.dtype === "BF16" ? "bf16" : "f32";
  const buf = kind === "bf16" ? await st.bf16(gpu, base + ".weight") : await st.vector(gpu, base + ".weight");
  const K = t.shape.slice(1).reduce((a, b) => a * b, 1);
  return { kind, N: N ?? t.shape[0], K, buf, bias: bias && st.has(base + ".bias") ? await st.vector(gpu, base + ".bias") : null };
}

const convOut = (L, k, s, p) => Math.floor((L + 2 * p - k) / s) + 1;

export class CodecEncoder {
  static async load(gpu, st, onProgress) {
    const e = new CodecEncoder(gpu);
    const vec = (n) => st.vector(gpu, n);
    e.hconv = [];
    for (let i = 0; i < 7; i++) e.hconv.push(await weight(gpu, st, `hubert.conv.${i}`));
    e.gn = [await vec("hubert.gn.weight"), await vec("hubert.gn.bias")];
    e.fpLn = [await vec("hubert.fp_ln.weight"), await vec("hubert.fp_ln.bias")];
    e.fp = await weight(gpu, st, "hubert.fp");
    e.pos = await weight(gpu, st, "hubert.pos");
    e.ln = [await vec("hubert.ln.weight"), await vec("hubert.ln.bias")];
    e.layers = [];
    for (let i = 0; i < 12; i++) {
      const p = `hubert.layers.${i}.`;
      e.layers.push({
        qkv: await weight(gpu, st, p + "qkv"), out: await weight(gpu, st, p + "out"),
        ln: [await vec(p + "ln.weight"), await vec(p + "ln.bias")],
        ff1: await weight(gpu, st, p + "ff1"), ff2: await weight(gpu, st, p + "ff2"),
        fln: [await vec(p + "fln.weight"), await vec(p + "fln.bias")],
      });
      onProgress?.((i + 1) / 16);
    }
    e.sem = {
      conv: await weight(gpu, st, "sem.conv"),
      blocks: [],
    };
    for (const b of [0, 1]) {
      const p = `sem.block.${b}.`;
      const res = [];
      for (const u of [0, 1]) res.push([await weight(gpu, st, `${p}res${u}.conv1`), await weight(gpu, st, `${p}res${u}.conv2`)]);
      e.sem.blocks.push({ res, down: await weight(gpu, st, p + "down") });
    }
    e.dac = { conv1: await weight(gpu, st, "dac.conv1"), blocks: [] };
    for (let b = 0; b < STRIDES.length; b++) {
      const p = `dac.block.${b}.`;
      const blk = { s: STRIDES[b], res: [] };
      for (const u of [1, 2, 3]) {
        const r = `${p}res${u}.`;
        blk.res.push({
          dil: [1, 3, 9][u - 1], snake1: await vec(r + "snake1.alpha"), conv1: await weight(gpu, st, r + "conv1"),
          snake2: await vec(r + "snake2.alpha"), conv2: await weight(gpu, st, r + "conv2"),
        });
      }
      blk.snake = await vec(p + "snake.alpha");
      blk.down = await weight(gpu, st, p + "down");
      e.dac.blocks.push(blk);
    }
    e.dac.snake = await vec("dac.snake.alpha");
    e.dac.conv2 = await weight(gpu, st, "dac.conv2");
    e.fc = await weight(gpu, st, "fc");
    e.rvq = [];
    for (let c = 0; c < 8; c++) {
      const p = `rvq.${c}.`;
      e.rvq.push({
        inW: await st.f32(p + "in.weight"), inB: await st.f32(p + "in.bias"), embed: await st.f32(p + "embed"),
        outW: await st.f32(p + "out.weight"), outB: await st.f32(p + "out.bias"),
      });
    }
    onProgress?.(1);
    return e;
  }

  constructor(gpu) {
    this.gpu = gpu;
  }

  // 16 kHz audio -> semantic features [ceil(T/2), 768] after the conv encoder
  semantic(wav16) {
    const gpu = this.gpu;
    const x = new Float32Array(wav16.length + 320);
    x.set(wav16, 160);
    let h = gpu.fromArray(x, [x.length, 1]);
    let L = x.length;
    const ks = [10, 3, 3, 3, 3, 2, 2], ss = [5, 2, 2, 2, 2, 2, 2];
    for (let i = 0; i < 7; i++) {
      const y = ops.conv1d(gpu, h, this.hconv[i], { tin: L, cin: i ? 512 : 1, k: ks[i], stride: ss[i], pad: 0, act: i ? "gelu" : "none" });
      h.release();
      h = y;
      L = convOut(L, ks[i], ss[i], 0);
      if (i === 0) ops.channelNormGelu(gpu, h, this.gn[0], this.gn[1], L, 512);
    }
    const n = ops.layernorm(gpu, h, this.fpLn[0], this.fpLn[1], 512);
    h.release();
    let z = ops.linear(gpu, n, this.fp);
    n.release();
    // positional conv (16 groups, kernel 128, last frame dropped) + GELU, added; then LayerNorm
    const pos = ops.conv1d(gpu, z, this.pos, { tin: L, cin: 48, k: 128, pad: 64, groups: 16, act: "gelu", tout: L });
    ops.axpy(gpu, pos, z);
    pos.release();
    h = ops.layernorm(gpu, z, this.ln[0], this.ln[1], 768);
    z.release();
    const sum = gpu.empty([L, 768]);
    ops.copy(gpu, h, sum, L * 768);
    for (const l of this.layers) {
      const qkv = ops.linear(gpu, h, l.qkv);
      const a = gpu.empty([L, 768]);
      ops.attention(gpu, qkv, a, { r0: 0, L, H: 12, KH: 12, D: 64 });
      qkv.release();
      ops.linear(gpu, a, l.out, { out: h, resid: true });
      a.release();
      const h1 = ops.layernorm(gpu, h, l.ln[0], l.ln[1], 768);
      h.release();
      const f = ops.linear(gpu, h1, l.ff1, { act: "gelu" });
      ops.linear(gpu, f, l.ff2, { out: h1, resid: true });
      f.release();
      h = ops.layernorm(gpu, h1, l.fln[0], l.fln[1], 768);
      h1.release();
      ops.axpy(gpu, h, sum);
    }
    h.release();
    // mean of the 13 states (folded into the first conv), every second frame (input stride)
    const Ts = Math.ceil(L / 2);
    const s0 = ops.conv1d(gpu, sum, this.sem.conv, { tin: Ts, cin: 768, k: 3, pad: 1, lda: 1536, alpha: 1 / 13 });
    const keep = (name, t) => { if (!this.keepFeatures) return; const c = gpu.empty(t.shape); ops.copy(gpu, t, c, t.size); this.stages[name] = c; };
    this.stages = {};
    if (this.keepFeatures) this.features = sum; // tools/check.html
    else sum.release();
    keep("conv", s0);
    let h2 = s0;
    for (const [b, blk] of this.sem.blocks.entries()) {
      for (const [u, [c1, c2]] of blk.res.entries()) {
        const r = ops.conv1d(gpu, h2, c1, { tin: Ts, cin: 768, k: 3, pad: 1, pre: "elu" });
        ops.conv1d(gpu, r, c2, { tin: Ts, cin: 768, k: 1, pad: 0, pre: "elu", into: h2 });
        r.release();
        if (b === 0) keep("res" + u, h2);
      }
      const y = ops.conv1d(gpu, h2, blk.down, { tin: Ts, cin: 768, k: 3, pad: 1 });
      h2.release();
      h2 = y;
    }
    const out = h2;
    return { t: out, T: Ts };
  }

  // DAC encoder output length for n input samples
  static acousticFrames(n) {
    let L = n;
    for (const s of STRIDES) L = convOut(L, 2 * s, s, Math.ceil(s / 2));
    return L;
  }

  acoustic(wav24, T) {
    const gpu = this.gpu;
    let x = wav24;
    if (CodecEncoder.acousticFrames(x.length) !== T) {
      x = new Float32Array(wav24.length + 2 * PAD);
      x.set(wav24, PAD);
    }
    const inp = gpu.fromArray(x, [x.length, 1]);
    let h = ops.conv1d(gpu, inp, this.dac.conv1, { tin: x.length, cin: 1, k: 7, pad: 3 });
    inp.release();
    let L = x.length, C = 64;
    for (const b of this.dac.blocks) {
      for (const r of b.res) {
        const t = ops.conv1d(gpu, h, r.conv1, { tin: L, cin: C, k: 7, dil: r.dil, pad: 3 * r.dil, snake: r.snake1 });
        ops.conv1d(gpu, t, r.conv2, { tin: L, cin: C, k: 1, pad: 0, snake: r.snake2, into: h });
        t.release();
      }
      const y = ops.conv1d(gpu, h, b.down, { tin: L, cin: C, k: 2 * b.s, stride: b.s, pad: Math.ceil(b.s / 2), snake: b.snake });
      h.release();
      h = y;
      L = convOut(L, 2 * b.s, b.s, Math.ceil(b.s / 2));
      C *= 2;
    }
    const out = ops.conv1d(gpu, h, this.dac.conv2, { tin: L, cin: C, k: 3, pad: 1, snake: this.dac.snake });
    h.release();
    return { t: out, T: L };
  }

  // wav24: Float32Array at 24 kHz (a multiple of 960 samples) -> Int32Array [8][T] codes
  async encode(wav24) {
    const gpu = this.gpu;
    const sem = this.semantic(resample(wav24, 24000, 16000));
    const ac = this.acoustic(wav24, sem.T);
    const T = Math.min(sem.T, ac.T);
    const cat = gpu.empty([T, 1024]);
    ops.copyCols(gpu, ac.t, cat, { rows: T, cols: 256, dstLd: 1024 });
    ops.copyCols(gpu, sem.t, cat, { rows: T, cols: 768, dstLd: 1024, dstOff: 256 });
    sem.t.release();
    ac.t.release();
    const emb = ops.linear(gpu, cat, this.fc);
    cat.release();
    const r = await gpu.read(emb, T * 1024);
    emb.release();
    return { codes: this.quantize(r, T), T };
  }

  // residual vector quantization on the CPU: per stage, project to 64 dims, nearest code, subtract
  quantize(res, T) {
    const codes = new Int32Array(8 * T);
    const p = new Float32Array(64);
    for (let c = 0; c < 8; c++) {
      const q = this.rvq[c];
      const n2 = new Float32Array(1024);
      for (let k = 0; k < 1024; k++) { let s = 0; for (let d = 0; d < 64; d++) s += q.embed[k * 64 + d] ** 2; n2[k] = s; }
      for (let t = 0; t < T; t++) {
        const x = res.subarray(t * 1024, (t + 1) * 1024);
        for (let d = 0; d < 64; d++) {
          let s = q.inB[d];
          const w = d * 1024;
          for (let i = 0; i < 1024; i++) s += q.inW[w + i] * x[i];
          p[d] = s;
        }
        let best = 0, bd = Infinity;
        for (let k = 0; k < 1024; k++) {
          let dot = 0;
          for (let d = 0; d < 64; d++) dot += p[d] * q.embed[k * 64 + d];
          const dist = n2[k] - 2 * dot;
          if (dist < bd) { bd = dist; best = k; }
        }
        codes[c * T + t] = best;
        for (let i = 0; i < 1024; i++) {
          let s = q.outB[i];
          for (let d = 0; d < 64; d++) s += q.outW[i * 64 + d] * q.embed[best * 64 + d];
          x[i] -= s;
        }
      }
    }
    return codes;
  }
}
