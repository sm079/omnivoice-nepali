// WGSL kernel generators. Every kernel binds its parameters as a uniform at binding 0.
// Flash attention and the Hadamard butterflies come from Anima Studio's engine.

const cache = new Map();
const memo = (key, fn) => {
  if (!cache.has(key)) cache.set(key, fn());
  return cache.get(key);
};

export { matmulShader } from "./gemm.js";

// In-place y = x @ H on a workgroup array S of `cols` floats (cols % 256 == 0), per group of 256,
// H = kron(H4,H4,H4,H4)/16 (the ConvRot rotation): one 4-point butterfly per base-4 digit.
const butterflies = (threads) => `
  let quads = P.cols / 4u;
  var stride = 1u;
  for (var d = 0u; d < 4u; d++) {
    for (var q = t; q < quads; q += ${threads}u) {
      let g = q / 64u;
      let w = q % 64u;
      let i0 = g * 256u + (w / stride) * stride * 4u + w % stride;
      let a = S[i0]; let b = S[i0 + stride]; let c = S[i0 + 2u * stride]; let e = S[i0 + 3u * stride];
      S[i0] = a + b + c - e;
      S[i0 + stride] = a + b - c + e;
      S[i0 + 2u * stride] = a - b + c + e;
      S[i0 + 3u * stride] = -a + b + c + e;
    }
    workgroupBarrier();
    stride *= 4u;
  }`;

// RMSNorm over rows of `cols` (<= 3072), y = x * rsqrt(mean(x^2) + eps) * w, rows starting at
// row0. rotate: also apply the ConvRot Hadamard, so a quantized linear can consume y directly.
export const rmsnormShader = (rotate) => memo("rms" + rotate, () => /* wgsl */ `
struct Params { rows: u32, cols: u32, nx: u32, eps: f32, row0: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;
${rotate ? "var<workgroup> S: array<f32, 3072>;" : ""}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let row = wg.y * P.nx + wg.x;
  if (row >= P.rows) { return; }
  let src = (P.row0 + row) * P.cols;
  let dst = row * P.cols;
  var ss = 0.0;
  for (var c = t; c < P.cols; c += 256u) { let v = X[src + c]; ss += v * v; }
  red[t] = ss; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { red[t] = red[t] + red[t + s]; } workgroupBarrier(); }
  let r = inverseSqrt(red[0] / f32(P.cols) + P.eps);
  ${rotate ? `
  for (var c = t; c < P.cols; c += 256u) { S[c] = X[src + c] * r * Wt[c]; }
  workgroupBarrier();
  ${butterflies(256)}
  for (var c = t; c < P.cols; c += 256u) { Y[dst + c] = S[c] * 0.0625; }` : `
  for (var c = t; c < P.cols; c += 256u) { Y[dst + c] = X[src + c] * r * Wt[c]; }`}
}`);

// ConvRot: y = x @ H per contiguous group of 256 features, H = kron(H4,H4,H4,H4)/16 (symmetric).
export const hadamardShader = () => memo("had", () => /* wgsl */ `
struct Params { groups: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<workgroup> s: array<f32, 256>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let g = wg.y * P.nx + wg.x;
  if (g >= P.groups) { return; }
  let base = g * 256u;
  for (var j = 0u; j < 4u; j++) { s[t + 64u * j] = X[base + t + 64u * j]; }
  workgroupBarrier();
  var stride = 1u;
  for (var d = 0u; d < 4u; d++) {
    let lo = t % stride;
    let i0 = (t / stride) * stride * 4u + lo;
    let a = s[i0]; let b = s[i0 + stride]; let c = s[i0 + 2u * stride]; let e = s[i0 + 3u * stride];
    // regular H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]
    workgroupBarrier();
    s[i0] = a + b + c - e;
    s[i0 + stride] = a + b - c + e;
    s[i0 + 2u * stride] = a - b + c + e;
    s[i0 + 3u * stride] = -a + b + c + e;
    workgroupBarrier();
    stride *= 4u;
  }
  for (var j = 0u; j < 4u; j++) { Y[base + t + 64u * j] = s[t + 64u * j] * 0.0625; }
}`);

// Fused per-head RMSNorm + split-half RoPE, in place on the q and k heads of a fused QKV buffer
// X[L][(H + 2*KH)*D] (q heads, then k heads, then v). Rows from `seg` on belong to a second
// sequence whose positions restart at 0. One workgroup of D/2 threads per (row, q-or-k head).
export const qkNormRopeShader = (D) => memo("qknr" + D, () => /* wgsl */ `
struct Params { L: u32, H: u32, KH: u32, ld: u32, sinOff: u32, nx: u32, eps: f32, seg: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read> QN: array<f32>;
@group(0) @binding(3) var<storage, read> KN: array<f32>;
@group(0) @binding(4) var<storage, read> CS: array<f32>;
const HALF = ${D / 2}u;
var<workgroup> red: array<f32, ${D / 2}>;
@compute @workgroup_size(${D / 2})
fn main(@builtin(local_invocation_index) i: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let id = wg.y * P.nx + wg.x;
  let heads = P.H + P.KH;
  if (id >= P.L * heads) { return; }
  let l = id / heads;
  let h = id % heads;               // 0..H-1 = q heads, H.. = k heads
  let base = l * P.ld + h * ${D}u;
  let x1 = X[base + i];
  let x2 = X[base + i + HALF];
  red[i] = x1 * x1 + x2 * x2;
  workgroupBarrier();
  for (var s = HALF / 2u; s > 0u; s >>= 1u) { if (i < s) { red[i] = red[i] + red[i + s]; } workgroupBarrier(); }
  let r = inverseSqrt(red[0] / ${D}.0 + P.eps);
  var w1: f32; var w2: f32;
  if (h < P.H) { w1 = QN[i]; w2 = QN[i + HALF]; } else { w1 = KN[i]; w2 = KN[i + HALF]; }
  let y1 = x1 * r * w1;
  let y2 = x2 * r * w2;
  var pos = l;
  if (l >= P.seg) { pos = l - P.seg; }
  let c = CS[pos * HALF + i];
  let s = CS[P.sinOff + pos * HALF + i];
  X[base + i] = y1 * c - y2 * s;
  X[base + i + HALF] = y2 * c + y1 * s;
}`);

// Fused ("flash") attention, one head x 64 queries per workgroup of 128 threads, keys streamed in
// blocks of 32 with an online softmax; nothing proportional to Lq*Lk touches memory. Grouped-query
// attention: query head h reads kv head h / group. Head dim D in {64, 128}.
// 16 KB of workgroup memory, as a vec4 array SH[1024]:
//   [0, 512)    P    probabilities of the current key block, [32 keys][64 rows]
//   [512, 768)  Qs   Q chunk [16 dims][64 rows]        (phase 1)
//   [768, 896)  Ks   K chunk [16 dims][32 keys]        (phase 1), then row-sum partials
//   [896, 1024) row-max partials [64 rows][8]
//   [512, 1024) Vs   V chunk [32 keys][64 dims]         (phase 3, reuses the above)
// Thread (sr = t / 8, sc = t % 8) owns rows sr*4..+3, keys sc*4..+3 of the score tile and
// columns c*32 + sc*4..+3 (per 64-wide V chunk) of the output rows; all register arrays use
// constant indices only.
export const flashAttentionShader = (D) => memo("flash" + D, () => {
  const DC = D / 16; // phase-1 dim chunks
  const VC = D / 64; // phase-3 value chunks
  const NO = 2 * VC; // output vec4 columns per row
  const r4 = [0, 1, 2, 3];
  const o = [];
  for (let i = 0; i < 4; i++) for (let c = 0; c < NO; c++) o.push(`var o${i}_${c} = vec4<f32>();`);
  const rescale = [];
  for (let i = 0; i < 4; i++) for (let c = 0; c < NO; c++) rescale.push(`o${i}_${c} = o${i}_${c} * alpha.${"xyzw"[i]};`);
  const pvFma = (vc) => {
    const out = [];
    for (let i = 0; i < 4; i++) {
      for (let c = 0; c < 2; c++) out.push(`o${i}_${vc * 2 + c} += pv.${"xyzw"[i]} * v${c};`);
    }
    return out.join("\n        ");
  };
  const store = [];
  for (let i = 0; i < 4; i++) {
    store.push(`{ let q = q0 + sr * 4u + ${i}u; if (q < P.Lq) { let inv = 1.0 / l.${"xyzw"[i]}; let ob = q * P.ldo + P.oOff + h * ${D}u;`);
    for (let vc = 0; vc < VC; vc++) {
      for (let c = 0; c < 2; c++) {
        const col = `${vc * 64 + c * 32}u + sc * 4u`;
        store.push(`  { let v = o${i}_${vc * 2 + c} * inv; let b = ob + ${col}; O[b] = v.x; O[b + 1u] = v.y; O[b + 2u] = v.z; O[b + 3u] = v.w; }`);
      }
    }
    store.push("} }");
  }
  return /* wgsl */ `
struct Params { Lq: u32, Lk: u32, ldq: u32, ldk: u32, ldv: u32, ldo: u32, qOff: u32, kOff: u32, vOff: u32, oOff: u32, scale: f32, group: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
var<workgroup> SH: array<vec4<f32>, 1024>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let h = wg.y;
  let kvh = h / P.group;
  let q0 = wg.x * 64u;
  let sr = t / 8u;
  let sc = t % 8u;
  ${o.join("\n  ")}
  var m = vec4<f32>(-1e30);
  var l = vec4<f32>(0.0);

  for (var k0 = 0u; k0 < P.Lk; k0 += 32u) {
    // ---- phase 1: S = Q K^T for the 64 x 32 tile
    var s0 = vec4<f32>(); var s1 = vec4<f32>(); var s2 = vec4<f32>(); var s3 = vec4<f32>();
    for (var dc = 0u; dc < ${DC}u; dc++) {
      {
        // Q chunk: row t/2, dims (t%2)*8..+7 of this chunk
        let qr = t / 2u;
        let d8 = (t % 2u) * 8u;
        var a = vec4<f32>(); var b = vec4<f32>();
        if (q0 + qr < P.Lq) {
          let gi = ((q0 + qr) * P.ldq + P.qOff + h * ${D}u + dc * 16u + d8) >> 2u;
          a = Q[gi]; b = Q[gi + 1u];
        }
        let lane = qr & 3u;
        let col = qr >> 2u;
        SH[512u + (d8 + 0u) * 16u + col][lane] = a.x; SH[512u + (d8 + 1u) * 16u + col][lane] = a.y;
        SH[512u + (d8 + 2u) * 16u + col][lane] = a.z; SH[512u + (d8 + 3u) * 16u + col][lane] = a.w;
        SH[512u + (d8 + 4u) * 16u + col][lane] = b.x; SH[512u + (d8 + 5u) * 16u + col][lane] = b.y;
        SH[512u + (d8 + 6u) * 16u + col][lane] = b.z; SH[512u + (d8 + 7u) * 16u + col][lane] = b.w;
      }
      {
        // K chunk: key t/4, dims (t%4)*4..+3
        let kr = t / 4u;
        let d4 = (t % 4u) * 4u;
        var a = vec4<f32>();
        if (k0 + kr < P.Lk) { a = K[((k0 + kr) * P.ldk + P.kOff + kvh * ${D}u + dc * 16u + d4) >> 2u]; }
        let lane = kr & 3u;
        let col = kr >> 2u;
        SH[768u + (d4 + 0u) * 8u + col][lane] = a.x; SH[768u + (d4 + 1u) * 8u + col][lane] = a.y;
        SH[768u + (d4 + 2u) * 8u + col][lane] = a.z; SH[768u + (d4 + 3u) * 8u + col][lane] = a.w;
      }
      workgroupBarrier();
      for (var d = 0u; d < 16u; d++) {
        let qa = SH[512u + d * 16u + sr];
        let kb = SH[768u + d * 8u + sc];
        s0 += qa.x * kb; s1 += qa.y * kb; s2 += qa.z * kb; s3 += qa.w * kb;
      }
      workgroupBarrier();
    }

    // ---- phase 2: online softmax
    let kmask = vec4<f32>(select(vec4<f32>(0.0), vec4<f32>(-1e30),
      vec4<u32>(k0 + sc * 4u) + vec4<u32>(0u, 1u, 2u, 3u) >= vec4<u32>(P.Lk)));
    s0 = s0 * P.scale + kmask; s1 = s1 * P.scale + kmask; s2 = s2 * P.scale + kmask; s3 = s3 * P.scale + kmask;
    let pm = vec4<f32>(max(max(s0.x, s0.y), max(s0.z, s0.w)), max(max(s1.x, s1.y), max(s1.z, s1.w)),
                       max(max(s2.x, s2.y), max(s2.z, s2.w)), max(max(s3.x, s3.y), max(s3.z, s3.w)));
    ${r4.map((i) => `SH[896u + ((sr * 4u + ${i}u) * 8u + sc) / 4u][sc % 4u] = pm.${"xyzw"[i]};`).join("\n    ")}
    workgroupBarrier();
    var bm = vec4<f32>(-1e30);
    ${r4.map((i) => `{ let a = SH[896u + (sr * 4u + ${i}u) * 2u]; let b = SH[896u + (sr * 4u + ${i}u) * 2u + 1u];
      bm.${"xyzw"[i]} = max(max(max(a.x, a.y), max(a.z, a.w)), max(max(b.x, b.y), max(b.z, b.w))); }`).join("\n    ")}
    let mn = max(m, bm);
    let e0 = exp(s0 - mn.x); let e1 = exp(s1 - mn.y); let e2 = exp(s2 - mn.z); let e3 = exp(s3 - mn.w);
    let ps = vec4<f32>(dot(e0, vec4<f32>(1.0)), dot(e1, vec4<f32>(1.0)), dot(e2, vec4<f32>(1.0)), dot(e3, vec4<f32>(1.0)));
    ${r4.map((i) => `SH[768u + ((sr * 4u + ${i}u) * 8u + sc) / 4u][sc % 4u] = ps.${"xyzw"[i]};`).join("\n    ")}
    // probabilities, stored [key][row] so phase 3 reads 4 rows as one vec4
    SH[(sc * 4u + 0u) * 16u + sr] = vec4<f32>(e0.x, e1.x, e2.x, e3.x);
    SH[(sc * 4u + 1u) * 16u + sr] = vec4<f32>(e0.y, e1.y, e2.y, e3.y);
    SH[(sc * 4u + 2u) * 16u + sr] = vec4<f32>(e0.z, e1.z, e2.z, e3.z);
    SH[(sc * 4u + 3u) * 16u + sr] = vec4<f32>(e0.w, e1.w, e2.w, e3.w);
    workgroupBarrier();
    var bs = vec4<f32>(0.0);
    ${r4.map((i) => `{ let a = SH[768u + (sr * 4u + ${i}u) * 2u]; let b = SH[768u + (sr * 4u + ${i}u) * 2u + 1u];
      bs.${"xyzw"[i]} = dot(a, vec4<f32>(1.0)) + dot(b, vec4<f32>(1.0)); }`).join("\n    ")}
    let alpha = exp(m - mn);
    l = l * alpha + bs;
    m = mn;
    ${rescale.join("\n    ")}
    workgroupBarrier(); // partials consumed before the V chunk overwrites them

    // ---- phase 3: O += P V, V streamed in 64-wide chunks
    ${[...Array(VC).keys()].map((vc) => `{
      let kr = t / 4u;
      let d16 = (t % 4u) * 16u;
      var a = vec4<f32>(); var b = vec4<f32>(); var c = vec4<f32>(); var e = vec4<f32>();
      if (k0 + kr < P.Lk) {
        let gi = ((k0 + kr) * P.ldv + P.vOff + kvh * ${D}u + ${vc * 64}u + d16) >> 2u;
        a = V[gi]; b = V[gi + 1u]; c = V[gi + 2u]; e = V[gi + 3u];
      }
      let base = 512u + kr * 16u + d16 / 4u;
      SH[base] = a; SH[base + 1u] = b; SH[base + 2u] = c; SH[base + 3u] = e;
      workgroupBarrier();
      for (var k = 0u; k < 32u; k++) {
        let pv = SH[k * 16u + sr];
        let v0 = SH[512u + k * 16u + sc];
        let v1 = SH[512u + k * 16u + 8u + sc];
        ${pvFma(vc)}
      }
      workgroupBarrier();
    }`).join("\n    ")}
  }

  ${store.join("\n  ")}
}`;
});

// Audio-token embeddings: Y[row0 + r] = sum_c TABLE[ids[r][c] + c * vocab] over 8 codebooks.
// TABLE is bf16 [8 * vocab, D] packed in u32 pairs; one workgroup per row.
export const audioEmbedShader = () => memo("aemb", () => /* wgsl */ `
struct Params { rows: u32, D: u32, vocab: u32, row0: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> TABLE: array<u32>;
@group(0) @binding(2) var<storage, read> IDS: array<u32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let r = wg.y * P.nx + wg.x;
  if (r >= P.rows) { return; }
  let half = P.D / 2u;
  for (var p = t; p < half; p += 256u) {
    var lo = 0.0; var hi = 0.0;
    for (var c = 0u; c < 8u; c++) {
      let w = TABLE[(IDS[r * 8u + c] + c * P.vocab) * half + p];
      lo += bitcast<f32>(w << 16u);
      hi += bitcast<f32>(w & 0xffff0000u);
    }
    let o = (P.row0 + r) * P.D + 2u * p;
    Y[o] = lo; Y[o + 1u] = hi;
  }
}`);

// Codec input: Y[t] = bias + sum_c CB_c[code[t][c]] (codebooks folded through project_out and fc2,
// f32 [8][1024][D]). One thread per output value.
export const codebookSumShader = () => memo("cbsum", () => /* wgsl */ `
struct Params { T: u32, D: u32, codes: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> CB: array<f32>;
@group(0) @binding(2) var<storage, read> BIAS: array<f32>;
@group(0) @binding(3) var<storage, read> IDS: array<u32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i >= P.T * P.D) { return; }
  let t = i / P.D;
  let d = i % P.D;
  var v = BIAS[d];
  for (var c = 0u; c < 8u; c++) { v += CB[(c * P.codes + IDS[t * 8u + c]) * P.D + d]; }
  Y[i] = v;
}`);

// Classifier-free guidance + greedy token choice for one (codebook, frame) per workgroup.
// Logits rows: [0, T) conditional, [T, 2T) unconditional, each [8 * V].
//   lc = log_softmax(cond), lu = log_softmax(uncond)
//   lp = log_softmax(lc + g * (lc - lu)), lp[mask] = -inf
//   out: token = argmax lp, score = max lp
export const cfgScoreShader = () => memo("cfg", () => /* wgsl */ `
struct Params { T: u32, V: u32, mask: u32, g: f32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> L: array<f32>;
@group(0) @binding(2) var<storage, read_write> TOK: array<u32>;
@group(0) @binding(3) var<storage, read_write> SCORE: array<f32>;
var<workgroup> rm: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
fn rmax(t: u32, v: f32) -> f32 {
  rm[t] = v; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { rm[t] = max(rm[t], rm[t + s]); } workgroupBarrier(); }
  let r = rm[0]; workgroupBarrier(); return r;
}
fn rsum(t: u32, v: f32) -> f32 {
  rm[t] = v; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { rm[t] = rm[t] + rm[t + s]; } workgroupBarrier(); }
  let r = rm[0]; workgroupBarrier(); return r;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let id = wg.y * P.nx + wg.x;       // id = c * T + frame
  if (id >= 8u * P.T) { return; }
  let c = id / P.T;
  let f = id % P.T;
  let rowC = f * 8u * P.V + c * P.V;
  let rowU = (P.T + f) * 8u * P.V + c * P.V;
  var mc = -3.0e38; var mu = -3.0e38;
  for (var v = t; v < P.V; v += 256u) { mc = max(mc, L[rowC + v]); mu = max(mu, L[rowU + v]); }
  mc = rmax(t, mc); mu = rmax(t, mu);
  var sc = 0.0; var su = 0.0;
  for (var v = t; v < P.V; v += 256u) { sc += exp(L[rowC + v] - mc); su += exp(L[rowU + v] - mu); }
  let lsc = mc + log(rsum(t, sc));
  let lsu = mu + log(rsum(t, su));
  // combined logits z = (1 + g) * lc - g * lu
  var mz = -3.0e38;
  for (var v = t; v < P.V; v += 256u) {
    let z = (1.0 + P.g) * (L[rowC + v] - lsc) - P.g * (L[rowU + v] - lsu);
    mz = max(mz, z);
  }
  mz = rmax(t, mz);
  var sz = 0.0;
  var best = -3.0e38; var bi = 0u;
  for (var v = t; v < P.V; v += 256u) {
    let z = (1.0 + P.g) * (L[rowC + v] - lsc) - P.g * (L[rowU + v] - lsu);
    sz += exp(z - mz);
    if (v != P.mask && z > best) { best = z; bi = v; }
  }
  let lsz = mz + log(rsum(t, sz));
  rm[t] = best; ri[t] = bi; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) {
      let o = rm[t + s]; let oi = ri[t + s];
      if (o > rm[t] || (o == rm[t] && oi < ri[t])) { rm[t] = o; ri[t] = oi; }
    }
    workgroupBarrier();
  }
  if (t == 0u) { TOK[id] = ri[0]; SCORE[id] = rm[0] - lsz; }
}`);

// LoRA merge, one workgroup per output row n (K <= 3072):
//   w[k] = W[n][k] + s * sum_j Bm[n][j] * Am[j][k]
// bf16: written back in place (round to nearest even). i8: dequantized with the row scale, merged
// and requantized in place with a new row scale. w4: decoded from its int8 grid and written as a
// new int8 row (to OUT/OS). Am is pre-rotated for ConvRot weights, so the update lands in the
// same rotated space as the stored weight.
export const mergeShader = (kind) => memo("merge" + kind, () => {
  const src = kind === "bf16" ? "var<storage, read_write> W: array<u32>;"
    : kind === "i8" ? "var<storage, read_write> W: array<u32>;\n@group(0) @binding(4) var<storage, read_write> RS: array<f32>;"
      : `var<storage, read> W: array<u32>;
@group(0) @binding(4) var<storage, read> RS: array<f32>;
@group(0) @binding(5) var<storage, read> SR: array<u32>;
@group(0) @binding(6) var<storage, read> CB: array<f32>;
@group(0) @binding(7) var<storage, read_write> OUT: array<u32>;
@group(0) @binding(8) var<storage, read_write> OS: array<f32>;`;
  const load = kind === "bf16" ? `
    let w = W[(n * P.K + k) >> 1u];
    if ((k & 1u) == 0u) { v = bitcast<f32>(w << 16u); } else { v = bitcast<f32>(w & 0xffff0000u); }`
    : kind === "i8" ? `
    let w = W[(n * P.K + k) >> 2u];
    v = f32(extractBits(i32(w), (k & 3u) * 8u, 8u)) * RS[n];`
      : `
    let codes = W[(n * P.K + k) >> 3u];
    let gi = (n * P.K + k) >> 4u;
    let s = fp8e4m3((SR[gi >> 2u] >> ((gi & 3u) * 8u)) & 255u);
    v = clamp(round(CB[(codes >> ((k & 7u) * 4u)) & 15u] * s), -127.0, 127.0) * RS[n];`;
  const store = kind === "bf16" ? `
  for (var p = t; p < P.K / 2u; p += 256u) {
    W[n * P.K / 2u + p] = bf16(S[2u * p]) | (bf16(S[2u * p + 1u]) << 16u);
  }`
    : `
  var mx = 0.0;
  for (var k = t; k < P.K; k += 256u) { mx = max(mx, abs(S[k])); }
  red[t] = mx; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { red[t] = max(red[t], red[t + s]); } workgroupBarrier(); }
  let scale = max(red[0] / 127.0, 1e-30);
  workgroupBarrier();
  for (var p = t; p < P.K / 4u; p += 256u) {
    var word = 0u;
    for (var j = 0u; j < 4u; j++) {
      let q = i32(clamp(round(S[4u * p + j] / scale), -128.0, 127.0));
      word = word | ((u32(q) & 255u) << (8u * j));
    }
    ${kind === "i8" ? "W" : "OUT"}[n * P.K / 4u + p] = word;
  }
  if (t == 0u) { ${kind === "i8" ? "RS" : "OS"}[n] = scale; }`;
  return /* wgsl */ `
struct Params { N: u32, K: u32, r: u32, nx: u32, s: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) ${src}
@group(0) @binding(2) var<storage, read> Am: array<f32>;
@group(0) @binding(3) var<storage, read> Bm: array<f32>;
var<workgroup> S: array<f32, 3072>;
var<workgroup> red: array<f32, 256>;
var<workgroup> b: array<f32, 64>;
fn bf16(x: f32) -> u32 {
  let u = bitcast<u32>(x);
  return (u + 0x7fffu + ((u >> 16u) & 1u)) >> 16u;
}
fn fp8e4m3(q: u32) -> f32 {
  let e = (q >> 3u) & 15u;
  let m = f32(q & 7u);
  var v: f32;
  if (e == 0u) { v = m * 0.001953125; } else { v = (1.0 + m * 0.125) * exp2(f32(e) - 7.0); }
  if ((q & 128u) != 0u) { v = -v; }
  return v;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let n = wg.y * P.nx + wg.x;
  if (n >= P.N) { return; }
  if (t < P.r) { b[t] = Bm[n * P.r + t] * P.s; }
  workgroupBarrier();
  for (var k = t; k < P.K; k += 256u) {
    var v = 0.0;${load}
    var d = 0.0;
    for (var j = 0u; j < P.r; j++) { d += b[j] * Am[j * P.K + k]; }
    S[k] = v + d;
  }
  workgroupBarrier();${store}
}`;
});

export const copyShader = () => memo("copy", () => /* wgsl */ `
struct Params { n: u32, nx: u32, srcOff: u32, dstOff: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i < P.n) { Y[P.dstOff + i] = A[P.srcOff + i]; }
}`);


// LayerNorm over rows of `cols` with weight and bias.
export const layernormShader = () => memo("ln", () => /* wgsl */ `
struct Params { rows: u32, cols: u32, nx: u32, eps: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
@group(0) @binding(3) var<storage, read> Bs: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;
fn rsum(t: u32, v: f32) -> f32 {
  red[t] = v; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { red[t] = red[t] + red[t + s]; } workgroupBarrier(); }
  let r = red[0]; workgroupBarrier(); return r;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let row = wg.y * P.nx + wg.x;
  if (row >= P.rows) { return; }
  let base = row * P.cols;
  var s = 0.0;
  for (var c = t; c < P.cols; c += 256u) { s += X[base + c]; }
  let mean = rsum(t, s) / f32(P.cols);
  var v = 0.0;
  for (var c = t; c < P.cols; c += 256u) { let d = X[base + c] - mean; v += d * d; }
  let r = inverseSqrt(rsum(t, v) / f32(P.cols) + P.eps);
  for (var c = t; c < P.cols; c += 256u) { Y[base + c] = (X[base + c] - mean) * r * Wt[c] + Bs[c]; }
}`);

// GroupNorm with one group per channel on a time-major [T, C] signal (each column normalized
// over time), affine, then GELU. One workgroup per channel.
export const channelNormGeluShader = () => memo("cngelu", () => /* wgsl */ `
struct Params { T: u32, C: u32, eps: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
@group(0) @binding(3) var<storage, read> Bs: array<f32>;
var<workgroup> red: array<f32, 256>;
fn rsum(t: u32, v: f32) -> f32 {
  red[t] = v; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { red[t] = red[t] + red[t + s]; } workgroupBarrier(); }
  let r = red[0]; workgroupBarrier(); return r;
}
fn erf_(x: f32) -> f32 {
  let s = sign(x);
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return s * y;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let c = wg.x;
  var s = 0.0;
  for (var i = t; i < P.T; i += 256u) { s += X[i * P.C + c]; }
  let mean = rsum(t, s) / f32(P.T);
  var v = 0.0;
  for (var i = t; i < P.T; i += 256u) { let d = X[i * P.C + c] - mean; v += d * d; }
  let r = inverseSqrt(rsum(t, v) / f32(P.T) + P.eps);
  for (var i = t; i < P.T; i += 256u) {
    let y = (X[i * P.C + c] - mean) * r * Wt[c] + Bs[c];
    X[i * P.C + c] = 0.5 * y * (1.0 + erf_(y * 0.7071067811865476));
  }
}`);

// Y += a * X over n elements
export const axpyShader = () => memo("axpy", () => /* wgsl */ `
struct Params { n: u32, nx: u32, a: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i < P.n) { Y[i] = Y[i] + P.a * X[i]; }
}`);

// Y[r][dstOff + c] = X[r * srcLd + c] for c < cols (concatenating feature columns)
export const copyColsShader = () => memo("copycols", () => /* wgsl */ `
struct Params { rows: u32, cols: u32, srcLd: u32, dstLd: u32, dstOff: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i >= P.rows * P.cols) { return; }
  let r = i / P.cols;
  let c = i % P.cols;
  Y[r * P.dstLd + P.dstOff + c] = X[r * P.srcLd + c];
}`);

// 3x3 conv, stride 2, pad 1 on an NHWC image X[H][W][Cin] -> Y[Ho][Wo][C]: dw = depthwise
// (Cin = C, one filter per channel), otherwise a single input channel. W: [C][9], + bias, opt. ReLU.
export const conv3x3s2Shader = (dw, relu) => memo(`c3s2${dw}${relu}`, () => /* wgsl */ `
struct Params { H: u32, W: u32, C: u32, Ho: u32, Wo: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
@group(0) @binding(3) var<storage, read> Bs: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i >= P.Ho * P.Wo * P.C) { return; }
  let c = i % P.C;
  let x = (i / P.C) % P.Wo;
  let y = i / (P.C * P.Wo);
  var s = Bs[c];
  for (var ky = 0u; ky < 3u; ky++) {
    let iy = i32(2u * y + ky) - 1;
    if (iy < 0 || iy >= i32(P.H)) { continue; }
    for (var kx = 0u; kx < 3u; kx++) {
      let ix = i32(2u * x + kx) - 1;
      if (ix < 0 || ix >= i32(P.W)) { continue; }
      let pix = u32(iy) * P.W + u32(ix);
      s += Wt[c * 9u + ky * 3u + kx] * X[${dw ? "pix * P.C + c" : "pix"}];
    }
  }
  Y[i] = ${relu ? "max(s, 0.0)" : "s"};
}`);

// Depthwise conv over time on X[T][C] (kernel k, "same" padding), + bias, then SiLU.
export const dwConvSiluShader = () => memo("dwsilu", () => /* wgsl */ `
struct Params { T: u32, C: u32, k: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
@group(0) @binding(3) var<storage, read> Bs: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i >= P.T * P.C) { return; }
  let c = i % P.C;
  let t = i32(i / P.C);
  let half = i32(P.k / 2u);
  var s = Bs[c];
  for (var j = 0; j < i32(P.k); j++) {
    let tt = t + j - half;
    if (tt >= 0 && tt < i32(P.T)) { s += Wt[c * P.k + u32(j)] * X[u32(tt) * P.C + c]; }
  }
  Y[i] = s / (1.0 + exp(-s));
}`);

// From a fused QKV buffer [T][3*D]: QU = q + u, QV = q + v (per-feature biases of length D).
export const qBiasShader = () => memo("qbias", () => /* wgsl */ `
struct Params { T: u32, D: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> QKV: array<f32>;
@group(0) @binding(2) var<storage, read> U: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> QU: array<f32>;
@group(0) @binding(5) var<storage, read_write> QV: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i >= P.T * P.D) { return; }
  let d = i % P.D;
  let q = QKV[(i / P.D) * 3u * P.D + d];
  QU[i] = q + U[d];
  QV[i] = q + V[d];
}`);

// Relative-position attention scores (Transformer-XL / NeMo rel_shift), softmax in place on AC:
//   S[h][i][j] = (AC[h][i][j] + BD[h][i][T - 1 - i + j]) * scale, BD: [H][T][2T - 1]
export const relSoftmaxShader = () => memo("relsm", () => /* wgsl */ `
struct Params { H: u32, T: u32, nx: u32, scale: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> AC: array<f32>;
@group(0) @binding(2) var<storage, read> BD: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let row = wg.y * P.nx + wg.x;  // h * T + i
  if (row >= P.H * P.T) { return; }
  let h = row / P.T;
  let i = row % P.T;
  let L2 = 2u * P.T - 1u;
  let base = row * P.T;
  let bdBase = (h * P.T + i) * L2 + (P.T - 1u - i);
  var mx = -3.0e38;
  for (var j = t; j < P.T; j += 256u) {
    let s = (AC[base + j] + BD[bdBase + j]) * P.scale;
    AC[base + j] = s;
    mx = max(mx, s);
  }
  red[t] = mx; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { red[t] = max(red[t], red[t + s]); } workgroupBarrier(); }
  mx = red[0]; workgroupBarrier();
  var sm = 0.0;
  for (var j = t; j < P.T; j += 256u) { let e = exp(AC[base + j] - mx); AC[base + j] = e; sm += e; }
  red[t] = sm; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (t < s) { red[t] = red[t] + red[t + s]; } workgroupBarrier(); }
  let inv = 1.0 / red[0];
  for (var j = t; j < P.T; j += 256u) { AC[base + j] = AC[base + j] * inv; }
}`);
