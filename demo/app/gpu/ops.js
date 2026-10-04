// Tensor ops built on the WGSL kernels. Activations are f32 Tensors (row-major).

import { grid } from "./device.js";
import * as K from "./kernels.js";

const mmParams = (o) => [
  ["u32", o.M], ["u32", o.N], ["u32", o.K], ["f32", o.alpha ?? 1],
  ["u32", o.lda ?? o.K], ["u32", o.aBatch ?? 0], ["u32", o.aOff ?? 0], ["u32", o.ldb ?? o.K],
  ["u32", o.bBatch ?? 0], ["u32", o.bDiv ?? 1], ["u32", o.bOff ?? 0], ["u32", o.ldc ?? o.N],
  ["u32", o.cBatch ?? 0], ["u32", o.cOff ?? 0], ["u32", o.bz ?? 0], ["u32", o.cin ?? 0],
  ["u32", o.tin ?? 0], ["u32", o.stride ?? 1], ["u32", o.dil ?? 1], ["u32", o.pad ?? 0],
];

export function matmul(gpu, { a = "rows", A, W, C, bias, act = "none", resid = false, snake, batch = 1, name, ...o }) {
  const b = W.kind;
  // big 128x128 tiles unless the problem is too small to fill them
  const R = o.M >= 256 && o.N >= 128 ? 8 : 4;
  const T = 16 * R;
  // vec4 loads need whole 8-wide k chunks and 4-aligned offsets/strides
  const al = (...xs) => xs.every((x) => (x ?? 0) % 4 === 0);
  const k8 = o.K % 8 === 0;
  const vecA = k8 && (a === "rows" ? al(o.lda ?? o.K, o.aOff, o.aBatch) : o.cin % 8 === 0 && al(o.aOff));
  const vecB = k8 && (b === "bf16" || b === "i8" || (b === "f32" && al(o.ldb ?? o.K, o.bOff, o.bBatch)));
  const code = K.matmulShader({ a, b, bias: !!bias, act, resid, R, vecA, vecB, snake: !!snake });
  const bufs = [A, W.buf, C];
  if (b === "i8" || b === "w4") bufs.push(W.scale);
  if (b === "w4") bufs.push(W.srel, W.codebook);
  if (bias) bufs.push(bias);
  if (snake) bufs.push(snake);
  const meta = { name: name || (a === "rows" ? `gemm.${b}` : `${a}.${b}`), flops: 2 * o.M * o.N * o.K * batch };
  gpu.dispatch(code, bufs, mmParams(o), [Math.ceil(o.N / T), Math.ceil(o.M / T), batch], meta);
}

export const needsRotation = (W) => W.kind === "i8" || W.kind === "w4";

// ConvRot input rotation (Hadamard over each group of 256 features).
export function rotate(gpu, x) {
  const y = gpu.empty(x.shape);
  const groups = x.size / 256;
  const [nx, ny] = grid(groups);
  gpu.dispatch(K.hadamardShader(), [x, y], [["u32", groups], ["u32", nx]], [nx, ny], { name: "hadamard" });
  return y;
}

// y = x @ W^T (+ bias) (act). `xr` is the pre-rotated input for quantized weights; when omitted
// it is computed here. opts.out + opts.resid: out += y (in place). opts.rows/rowOff: a row range of x.
export function linear(gpu, x, W, opts = {}) {
  const M = opts.rows ?? x.size / W.K;
  let input = x;
  let tmp = null;
  if (needsRotation(W) && !opts.rotated) input = opts.xr || (tmp = rotate(gpu, x));
  const swiglu = opts.act === "swiglu";
  const out = opts.out || gpu.empty([M, swiglu ? W.N / 2 : W.N]);
  matmul(gpu, {
    A: input, W, C: out, M, N: W.N, K: W.K, aOff: (opts.rowOff ?? 0) * W.K, ldc: swiglu ? W.N / 2 : W.N,
    bias: opts.bias || W.bias, act: opts.act, resid: !!opts.resid, name: opts.name,
  });
  if (tmp) tmp.release();
  return out;
}

// RMSNorm of `rows` rows of x starting at row0 (default: all) into a new [rows, cols] tensor;
// rotate: output ConvRot-rotated for a quantized linear.
export function rmsnorm(gpu, x, weight, cols, { eps = 1e-6, rotate = false, row0 = 0, rows } = {}) {
  const n = rows ?? x.size / cols - row0;
  const y = gpu.empty([n, cols]);
  const [nx, ny] = grid(n);
  gpu.dispatch(K.rmsnormShader(rotate), [x, weight, y], [["u32", n], ["u32", cols], ["u32", nx], ["f32", eps], ["u32", row0]], [nx, ny],
    { name: rotate ? "rmsnorm+rot" : "rmsnorm" });
  return y;
}

// In-place per-head RMSNorm + RoPE on the q and k heads of a fused QKV buffer [L, (H+2KH)*D].
// Rows from `seg` on restart their positions at 0 (second packed sequence).
export function qkNormRope(gpu, qkv, qn, kn, cs, L, H, KH, D, seg) {
  const [nx, ny] = grid(L * (H + KH));
  gpu.dispatch(K.qkNormRopeShader(D), [qkv, qn, kn, cs.buf],
    [["u32", L], ["u32", H], ["u32", KH], ["u32", (H + 2 * KH) * D], ["u32", cs.sinOff], ["u32", nx], ["f32", 1e-6], ["u32", seg]], [nx, ny],
    { name: "qk_norm_rope" });
}

// Bidirectional attention over rows [r0, r0 + L) of a fused QKV buffer, written into the same rows of `out`.
export function attention(gpu, qkv, out, { r0, L, H, KH, D }) {
  const ld = (H + 2 * KH) * D;
  gpu.dispatch(K.flashAttentionShader(D), [qkv, qkv, qkv, out],
    [["u32", L], ["u32", L], ["u32", ld], ["u32", ld], ["u32", ld], ["u32", H * D],
      ["u32", r0 * ld], ["u32", r0 * ld + H * D], ["u32", r0 * ld + (H + KH) * D], ["u32", r0 * H * D],
      ["f32", 1 / Math.sqrt(D)], ["u32", H / KH]],
    [Math.ceil(L / 64), H], { name: "attn.flash", flops: 4 * L * L * D * H });
}

// Y[row0 + r] = sum of the 8 codebook embeddings of ids[r] (table: bf16 [8 * vocab, D])
export function audioEmbed(gpu, table, ids, Y, { rows, D, vocab, row0 }) {
  const [nx, ny] = grid(rows);
  gpu.dispatch(K.audioEmbedShader(), [table, ids, Y], [["u32", rows], ["u32", D], ["u32", vocab], ["u32", row0], ["u32", nx]], [nx, ny], { name: "audio_embed" });
}

export function codebookSum(gpu, cb, bias, ids, T, D, codes = 1024) {
  const y = gpu.empty([T, D]);
  const [nx, ny] = grid(Math.ceil((T * D) / 256));
  gpu.dispatch(K.codebookSumShader(), [cb, bias, ids, y], [["u32", T], ["u32", D], ["u32", codes], ["u32", nx]], [nx, ny], { name: "codebook_sum" });
  return y;
}

// logits [2T, 8V] -> (tokens, scores) [8, T]
export function cfgScore(gpu, logits, tok, score, { T, V, mask, g }) {
  const [nx, ny] = grid(8 * T);
  gpu.dispatch(K.cfgScoreShader(), [logits, tok, score], [["u32", T], ["u32", V], ["u32", mask], ["f32", g], ["u32", nx]], [nx, ny], { name: "cfg_score" });
}

export function copy(gpu, src, dst, n, srcOff = 0, dstOff = 0) {
  const [nx, ny] = grid(Math.ceil(n / 256));
  gpu.dispatch(K.copyShader(), [src, dst], [["u32", n], ["u32", nx], ["u32", srcOff], ["u32", dstOff]], [nx, ny], { name: "copy" });
}

// 1D convolution over a time-major signal x [tin, cin] with weight W [cout, k*cin] (tap-major).
// snake: per-channel Snake alphas applied to the input on load. into: accumulate (residual).
export function conv1d(gpu, x, W, { tin, cin, k, stride = 1, dil = 1, pad, snake, into }) {
  const tout = Math.floor((tin + 2 * pad - dil * (k - 1) - 1) / stride) + 1;
  const out = into || gpu.empty([tout, W.N]);
  matmul(gpu, { a: "conv", A: x, W, C: out, M: tout, N: W.N, K: k * cin, cin, tin, stride, dil, pad, bias: W.bias, snake, resid: !!into });
  return out;
}

// Transposed conv (kernel 2s, stride s, padding ceil(s/2), output padding s % 2) of x [tin, cin]
// -> [tin*s, cout]; W holds one [cout, 2*cin] weight per output phase (rows z*cout..).
export function convT1d(gpu, x, W, { tin, cin, s, snake }) {
  const cout = W.N;
  const out = gpu.empty([tin * s, cout]);
  matmul(gpu, {
    a: "convT", A: x, W, C: out, M: tin, N: cout, K: 2 * cin, cin, tin, stride: s, pad: Math.ceil(s / 2),
    bz: cout, ldc: s * cout, cBatch: cout, batch: s, bias: W.bias, snake,
  });
  return out;
}

// LoRA merge into a weight in place (bf16, i8) or into a new int8 weight (w4).
// A: [r, K] f32 (pre-rotated for ConvRot weights), B: [N, r] f32 buffers.
export function mergeLora(gpu, W, A, B, r, scale) {
  const [nx, ny] = grid(W.N);
  const params = [["u32", W.N], ["u32", W.K], ["u32", r], ["u32", nx], ["f32", scale]];
  if (W.kind === "bf16") {
    gpu.dispatch(K.mergeShader("bf16"), [W.buf, A, B], params, [nx, ny], { name: "lora_merge" });
    return W;
  }
  if (W.kind === "i8") {
    gpu.dispatch(K.mergeShader("i8"), [W.buf, A, B, W.scale], params, [nx, ny], { name: "lora_merge" });
    return W;
  }
  const out = gpu.device.createBuffer({ size: W.N * W.K, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const os = gpu.device.createBuffer({ size: Math.ceil((W.N * 4) / 16) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpu.dispatch(K.mergeShader("w4"), [W.buf, A, B, W.scale, W.srel, W.codebook, out, os], params, [nx, ny], { name: "lora_merge" });
  return { ...W, kind: "i8", buf: out, scale: os, srel: null, codebook: null, codebookData: null, replaced: [W.buf, W.scale, W.srel, W.codebook] };
}
