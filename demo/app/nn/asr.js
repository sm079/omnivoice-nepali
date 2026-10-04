// Nepali speech recognition for cloning references: AI4Bharat IndicConformer 600M (multilingual),
// its Conformer encoder and the CTC head restricted to Nepali, greedy CTC decoding.
//
//   log-mel features (CPU): pre-emphasis, 512-point STFT (hop 10 ms, 25 ms Hann), 80 mel bands,
//                           log, per-feature normalization over the clip
//   subsampling (8x): 3x3 stride-2 conv + ReLU, then twice depthwise 3x3 stride 2 + pointwise + ReLU,
//                     flattened and projected to 1024 (input scale folded in)
//   24 Conformer layers: x + ff1/2, x + relative-position attention (8 x 128), x + conv module
//                        (pointwise + GLU, depthwise k9 + SiLU, pointwise), x + ff2/2, LayerNorm
// tools/build_asr.py converts the ONNX export; tools/check.html compares with it.

import * as ops from "../gpu/ops.js";
import { grid } from "../gpu/device.js";
import * as K from "../gpu/kernels.js";

const D = 1024, H = 8, HD = 128;

// in-place radix-2 FFT of length n (power of two)
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const wr = Math.cos(ang * k), wi = Math.sin(ang * k);
        const a = i + k, b = a + len / 2;
        const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
    }
  }
}

// 16 kHz audio -> normalized log-mel features [T][80] (NeMo AudioToMelSpectrogramPreprocessor)
export function logMel(x, cfg) {
  const { n_fft: N, hop, preemph, log_guard: guard, std_eps: eps } = cfg;
  const bins = N / 2 + 1;
  const win = cfg.window; // 400 taps, centered in the 512-point frame
  const off = (N - win.length) >> 1;
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) y[i] = x[i] - preemph * (i ? x[i - 1] : 0);
  const pad = N / 2;
  const at = (i) => { // reflect padding
    let j = i - pad;
    if (j < 0) j = -j;
    if (j >= y.length) j = 2 * (y.length - 1) - j;
    return y[j];
  };
  const T = Math.floor(x.length / hop) + 1;
  const mels = cfg.fb.length / bins;
  const out = new Float32Array(T * mels);
  const re = new Float64Array(N), im = new Float64Array(N), pw = new Float64Array(bins);
  for (let t = 0; t < T; t++) {
    re.fill(0); im.fill(0);
    for (let i = 0; i < win.length; i++) re[off + i] = at(t * hop + off + i) * win[i];
    fft(re, im);
    for (let k = 0; k < bins; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < mels; m++) {
      let s = 0;
      const row = m * bins;
      for (let k = 0; k < bins; k++) s += cfg.fb[row + k] * pw[k];
      out[t * mels + m] = Math.log(s + guard);
    }
  }
  for (let m = 0; m < mels; m++) {
    let mean = 0;
    for (let t = 0; t < T; t++) mean += out[t * mels + m];
    mean /= T;
    let v = 0;
    for (let t = 0; t < T; t++) v += (out[t * mels + m] - mean) ** 2;
    const std = Math.sqrt(Math.max(v / Math.max(1, T - 1), 5.9604644775390625e-8));
    for (let t = 0; t < T; t++) out[t * mels + m] = (out[t * mels + m] - mean) / (std + eps);
  }
  return { feats: out, T, mels };
}

async function lin(gpu, st, base, { bias = true } = {}) {
  const t = st.info(base + ".weight");
  const W = t.dtype === "F32" ? { kind: "f32", N: t.shape[0], K: t.shape[1], buf: await st.vector(gpu, base + ".weight") } : await st.linear(gpu, base + ".");
  if (bias && st.has(base + ".bias")) W.bias = await st.vector(gpu, base + ".bias");
  return W;
}

export class NepaliASR {
  static async load(gpu, st, cfg, onProgress) {
    const a = new NepaliASR(gpu, cfg);
    const vec = (n) => st.vector(gpu, n);
    a.pre = {
      conv0: [await vec("pre.conv0.weight"), await vec("pre.conv0.bias")],
      dw: [0, 1].map(() => null), pw: [0, 1].map(() => null),
      out: await lin(gpu, st, "pre.out"),
    };
    for (const i of [0, 1]) {
      a.pre.dw[i] = [await vec(`pre.dw${i}.weight`), await vec(`pre.dw${i}.bias`)];
      a.pre.pw[i] = await lin(gpu, st, `pre.pw${i}`);
    }
    a.posEmb = await st.f32("pos_emb"); // [2c + 1, 1024], relative position c - row
    a.layers = [];
    for (let i = 0; i < cfg.layers; i++) {
      const p = `layers.${i}.`;
      const ln = async (n) => [await vec(p + n + ".weight"), await vec(p + n + ".bias")];
      a.layers.push({
        lnFf1: await ln("norm_feed_forward1"), ff1: [await lin(gpu, st, p + "feed_forward1.linear1"), await lin(gpu, st, p + "feed_forward1.linear2")],
        lnAtt: await ln("norm_self_att"), qkv: await lin(gpu, st, p + "attn.qkv"), pos: await lin(gpu, st, p + "attn.pos"),
        out: await lin(gpu, st, p + "attn.out"), u: await vec(p + "attn.pos_bias_u"), v: await vec(p + "attn.pos_bias_v"),
        lnConv: await ln("norm_conv"), pw1: await lin(gpu, st, p + "conv.pw1"),
        dw: [await vec(p + "conv.dw.weight"), await vec(p + "conv.dw.bias")], pw2: await lin(gpu, st, p + "conv.pw2"),
        lnFf2: await ln("norm_feed_forward2"), ff2: [await lin(gpu, st, p + "feed_forward2.linear1"), await lin(gpu, st, p + "feed_forward2.linear2")],
        lnOut: await ln("norm_out"),
      });
      onProgress?.((i + 1) / cfg.layers);
    }
    a.ctc = await lin(gpu, st, "ctc");
    return a;
  }

  constructor(gpu, cfg) {
    this.gpu = gpu;
    this.cfg = cfg;
  }

  conv3(x, w, b, Hh, Ww, C, dw, relu) {
    const gpu = this.gpu;
    const Ho = Math.floor((Hh - 1) / 2) + 1, Wo = Math.floor((Ww - 1) / 2) + 1;
    const y = gpu.empty([Ho * Wo, C]);
    const [nx, ny] = grid(Math.ceil((Ho * Wo * C) / 256));
    gpu.dispatch(K.conv3x3s2Shader(dw, relu), [x, w, b, y], [["u32", Hh], ["u32", Ww], ["u32", C], ["u32", Ho], ["u32", Wo], ["u32", nx]], [nx, ny], { name: "conv3x3s2" });
    return { y, Ho, Wo };
  }

  // features [T][80] -> encoder output [T/8][1024]
  encode(feats, T, mels) {
    const gpu = this.gpu;
    const x0 = gpu.fromArray(feats, [T, mels]);
    let { y, Ho, Wo } = this.conv3(x0, ...this.pre.conv0, T, mels, 256, false, true);
    x0.release();
    for (const i of [0, 1]) {
      const d = this.conv3(y, ...this.pre.dw[i], Ho, Wo, 256, true, false);
      y.release();
      ({ Ho, Wo } = d);
      y = ops.linear(gpu, d.y, this.pre.pw[i], { act: "relu" });
      d.y.release();
    }
    const L = Ho;
    let x = ops.linear(gpu, y, this.pre.out, { rows: L }); // [L, Wo * 256] -> [L, 1024]
    y.release();
    const keep = (k, t) => { if (!this.stages) return; const c = gpu.empty(t.shape); ops.copy(gpu, t, c, t.size); this.stages[k] = c; };
    keep("pre", x);

    // relative positional embeddings for distances L-1 .. -(L-1)
    const c = this.cfg.pos_center;
    if (L > c) throw new Error("This recording is too long to transcribe (over 2 minutes).");
    const pe = gpu.fromArray(this.posEmb.subarray((c - (L - 1)) * D, (c + L) * D), [2 * L - 1, D]);
    for (const l of this.layers) {
      // feed-forward 1 (half step folded into its weights)
      let h = ops.layernorm(gpu, x, ...l.lnFf1, D);
      let f = ops.linear(gpu, h, l.ff1[0], { act: "silu" });
      h.release();
      ops.linear(gpu, f, l.ff1[1], { out: x, resid: true });
      f.release();
      if (l === this.layers[0]) keep("ff1", x);
      // relative-position self-attention
      h = ops.layernorm(gpu, x, ...l.lnAtt, D);
      const qkv = ops.linear(gpu, h, l.qkv);
      h.release();
      const a = this.relAttention(qkv, ops.linear(gpu, pe, l.pos), l, L);
      qkv.release();
      ops.linear(gpu, a, l.out, { out: x, resid: true });
      a.release();
      if (l === this.layers[0]) keep("att", x);
      // convolution module
      h = ops.layernorm(gpu, x, ...l.lnConv, D);
      const g = ops.linear(gpu, h, l.pw1, { act: "glu" });
      h.release();
      const dwo = gpu.empty([L, D]);
      const [nx, ny] = grid(Math.ceil((L * D) / 256));
      gpu.dispatch(K.dwConvSiluShader(), [g, l.dw[0], l.dw[1], dwo], [["u32", L], ["u32", D], ["u32", 9], ["u32", nx]], [nx, ny], { name: "dwconv_silu" });
      g.release();
      ops.linear(gpu, dwo, l.pw2, { out: x, resid: true });
      dwo.release();
      if (l === this.layers[0]) keep("conv", x);
      // feed-forward 2
      h = ops.layernorm(gpu, x, ...l.lnFf2, D);
      f = ops.linear(gpu, h, l.ff2[0], { act: "silu" });
      h.release();
      ops.linear(gpu, f, l.ff2[1], { out: x, resid: true });
      f.release();
      const y2 = ops.layernorm(gpu, x, ...l.lnOut, D);
      x.release();
      x = y2;
      if (l === this.layers[0]) keep("out", x);
    }
    pe.release();
    return { x, L };
  }

  // softmax(((q + u) k^T + rel_shift((q + v) p^T)) / sqrt(d)) v, per head, as batched GEMMs
  relAttention(qkv, p, l, L) {
    const gpu = this.gpu;
    const qu = gpu.empty([L, D]), qv = gpu.empty([L, D]);
    let [nx, ny] = grid(Math.ceil((L * D) / 256));
    gpu.dispatch(K.qBiasShader(), [qkv, l.u, l.v, qu, qv], [["u32", L], ["u32", D], ["u32", nx]], [nx, ny], { name: "q_bias" });
    const L2 = 2 * L - 1;
    const ac = gpu.empty([H, L, L]);
    ops.matmul(gpu, { A: qu, W: { kind: "f32", buf: qkv.buf }, C: ac, batch: H, M: L, N: L, K: HD, lda: D, aBatch: HD, ldb: 3 * D, bOff: D, bBatch: HD, ldc: L, cBatch: L * L, name: "attn.ac" });
    const bd = gpu.empty([H, L, L2]);
    ops.matmul(gpu, { A: qv, W: { kind: "f32", buf: p.buf }, C: bd, batch: H, M: L, N: L2, K: HD, lda: D, aBatch: HD, ldb: D, bBatch: HD, ldc: L2, cBatch: L * L2, name: "attn.bd" });
    qu.release(); qv.release(); p.release();
    [nx, ny] = grid(H * L);
    gpu.dispatch(K.relSoftmaxShader(), [ac, bd], [["u32", H], ["u32", L], ["u32", nx], ["f32", 1 / Math.sqrt(HD)]], [nx, ny], { name: "attn.relsoftmax" });
    bd.release();
    const o = gpu.empty([L, D]);
    ops.matmul(gpu, { A: ac, W: { kind: "f32t", buf: qkv.buf }, C: o, batch: H, M: L, N: HD, K: L, lda: L, aBatch: L * L, ldb: 3 * D, bOff: 2 * D, bBatch: HD, ldc: D, cBatch: HD, name: "attn.pv" });
    ac.release();
    return o;
  }

  // 16 kHz audio -> text
  async transcribe(wav16) {
    const gpu = this.gpu;
    const { feats, T, mels } = logMel(wav16, this.cfg);
    const { x, L } = this.encode(feats, T, mels);
    const logits = ops.linear(gpu, x, this.ctc);
    x.release();
    const V = this.ctc.N;
    const lg = await gpu.read(logits, L * V);
    logits.release();
    // greedy CTC: best token per frame, merge repeats, drop blanks
    const blank = this.cfg.blank;
    let prev = -1;
    let text = "";
    for (let t = 0; t < L; t++) {
      let best = 0;
      for (let v = 1; v < V; v++) if (lg[t * V + v] > lg[t * V + best]) best = v;
      if (best !== prev && best !== blank) text += this.cfg.vocab[best];
      prev = best;
    }
    return { text: text.replace(/▁/g, " ").trim(), frames: L, logits: lg };
  }
}
