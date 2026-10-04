// Tiled GEMM generator (from Anima Studio's engine, extended for 1D convolutions).
//
// C[z][m][n] = epilogue( alpha * sum_k A(z,m,k) * B(z,n,k) )
//
// A modes : "rows"  A[aOff + z*aBatch + m*lda + k]
//           "conv"  implicit im2col of a time-major signal X[t][cin]: output frame m reads input
//                   frame m*stride - pad + tap*dil, k = tap*cin + ci (zero outside [0, tin))
//           "convT" one output phase z of a stride-s transposed conv (kernel 2s): output frame
//                   m*s + z reads input frame m + (z + pad) / s - a, k = a*cin + ci
//           snake: Snake activation x + sin(alpha x)^2 / alpha (alpha per input channel) applied
//                  to every loaded A value (conv modes), fusing DAC's pre-conv activations
// B fmts  : "f32"   B[bOff + z/bDiv*bBatch + n*ldb + k]    ("f32t": B[... + k*ldb + n])
//           "bf16"  [N,K] bf16 pairs packed in u32
//           "i8"    [N,K] int8 packed 4 per u32            (per-row scale in epilogue)
//           "w4"    [N,K] 4-bit codes packed 8 per u32, fp8-e4m3 group scale per 16,
//                   16-entry codebook, decoded to the int8 grid (per-row scale in epilogue)
//           Weight rows of batch z start at row z*bz (per-phase weights of a transposed conv).
// Epilogue: *rowScale[n], +bias[n], act (gelu|silu|swiglu), then store or C += v (resid).
//           swiglu: B rows interleave (gate_j, up_j); C[m][j] = silu(gate_j) * up_j, ldc = N/2.
//
// Tiles (256 threads, 16 KB of workgroup memory in both):
//   R = 4: 64x64,   TK = 32, 4x4 outputs per thread  (small problems)
//   R = 8: 128x128, TK = 16, 8x8 outputs per thread  (everything large)
// Every thread stages 8 consecutive k of one A row and one B row per k-tile, and issues the
// next tile's global loads before computing on the current one (register prefetch).
//
// All register arrays are indexed with constants only (the generator unrolls every loop that
// touches them): shader compilers spill dynamically indexed arrays to local memory.

const ACT = /* wgsl */ `
fn erf_(x: f32) -> f32 {
  let s = sign(x);
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return s * y;
}
fn gelu(x: f32) -> f32 { return 0.5 * x * (1.0 + erf_(x * 0.7071067811865476)); }
fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }
`;

const range = (n) => [...Array(n).keys()];
const cache = new Map();

// Snake: x + (alpha + 1e-9)^-1 * sin(alpha * x)^2
const SNAKE = /* wgsl */ `
fn snake(x: f32, ci: u32) -> f32 {
  let a = ALPHA[ci];
  let s = sin(a * x);
  return x + s * s / (a + 1e-9);
}`;

// Loaders return V8 { lo, hi }: 8 consecutive k values.
function loaderA(a, vec, snake) {
  if (a === "rows") {
    if (vec) {
      return `
fn loadA8(z: u32, m: u32, k: u32) -> V8 {
  if (m >= P.M || k >= P.K) { return V8(); }
  let q = (P.aOff + z * P.aBatch + m * P.lda + k) >> 2u;
  return V8(A[q], A[q + 1u]);
}`;
    }
    const e = range(8).map((j) => `select(0.0, A[base + ${j}u], k + ${j}u < P.K)`);
    return `
fn loadA8(z: u32, m: u32, k: u32) -> V8 {
  if (m >= P.M) { return V8(); }
  let base = P.aOff + z * P.aBatch + m * P.lda + k;  // out-of-range lanes are masked (robust access keeps reads safe)
  return V8(vec4<f32>(${e.slice(0, 4).join(", ")}), vec4<f32>(${e.slice(4).join(", ")}));
}`;
  }
  // conv / convT: input frame of (m, tap) and channel ci for k
  const frame = a === "conv"
    ? "i32(m * P.stride + (k / P.cin) * P.dil) - i32(P.pad)"
    : "i32(m + (z + P.pad) / P.stride) - i32(k / P.cin)";
  const sn = (v, c) => (snake ? `snake(${v}, ${c})` : v);
  if (vec) {
    // the 8 k share one tap (cin % 8 == 0)
    return `${snake ? SNAKE : ""}
fn loadA8(z: u32, m: u32, k: u32) -> V8 {
  if (m >= P.M || k >= P.K) { return V8(); }
  let f = ${frame};
  if (f < 0 || f >= i32(P.tin)) { return V8(); }
  let ci = k % P.cin;
  let q = (P.aOff + u32(f) * P.cin + ci) >> 2u;
  let a = A[q]; let b = A[q + 1u];
  return V8(vec4<f32>(${range(4).map((j) => sn(`a.${"xyzw"[j]}`, `ci + ${j}u`)).join(", ")}),
            vec4<f32>(${range(4).map((j) => sn(`b.${"xyzw"[j]}`, `ci + ${j + 4}u`)).join(", ")}));
}`;
  }
  return `${snake ? SNAKE : ""}
fn loadA1(z: u32, m: u32, k: u32) -> f32 {
  if (k >= P.K) { return 0.0; }
  let f = ${frame};
  if (f < 0 || f >= i32(P.tin)) { return 0.0; }
  let ci = k % P.cin;
  return ${sn("A[P.aOff + u32(f) * P.cin + ci]", "ci")};
}
fn loadA8(z: u32, m: u32, k: u32) -> V8 {
  if (m >= P.M) { return V8(); }
  return V8(vec4<f32>(${range(4).map((j) => `loadA1(z, m, k + ${j}u)`).join(", ")}), vec4<f32>(${range(4).map((j) => `loadA1(z, m, k + ${j + 4}u)`).join(", ")}));
}`;
}

function loaderB(b, vec) {
  if (b === "f32") {
    if (vec) {
      return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  let q = (P.bOff + (z / P.bDiv) * P.bBatch + n * P.ldb + k) >> 2u;
  return V8(B[q], B[q + 1u]);
}`;
    }
    const e = range(8).map((j) => `select(0.0, B[base + ${j}u], k + ${j}u < P.K)`);
    return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N) { return V8(); }
  let base = P.bOff + (z / P.bDiv) * P.bBatch + n * P.ldb + k;
  return V8(vec4<f32>(${e.slice(0, 4).join(", ")}), vec4<f32>(${e.slice(4).join(", ")}));
}`;
  }
  if (b === "f32t") {
    const e = range(8).map((j) => `select(0.0, B[min(base + (k + ${j}u) * P.ldb, last)], k + ${j}u < P.K)`);
    return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N) { return V8(); }
  let base = P.bOff + (z / P.bDiv) * P.bBatch + n;
  let last = arrayLength(&B) - 1u;
  return V8(vec4<f32>(${e.slice(0, 4).join(", ")}), vec4<f32>(${e.slice(4).join(", ")}));
}`;
  }
  const row = "(n + z * P.bz)";
  if (b === "bf16") {
    const lo = (w) => `bitcast<f32>(${w} << 16u)`;
    const hi = (w) => `bitcast<f32>(${w} & 0xffff0000u)`;
    if (vec) {
      return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  let w = B[(${row} * P.K + k) >> 3u];
  return V8(vec4<f32>(${lo("w.x")}, ${hi("w.x")}, ${lo("w.y")}, ${hi("w.y")}), vec4<f32>(${lo("w.z")}, ${hi("w.z")}, ${lo("w.w")}, ${hi("w.w")}));
}`;
    }
    // K even; a partial last chunk is zero-masked
    const w = range(4).map((j) => `let w${j} = select(0u, B[base + ${j}u], k + ${2 * j}u < P.K);`);
    return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  let base = (${row} * P.K + k) >> 1u;
  ${w.join("\n  ")}
  return V8(vec4<f32>(${lo("w0")}, ${hi("w0")}, ${lo("w1")}, ${hi("w1")}), vec4<f32>(${lo("w2")}, ${hi("w2")}, ${lo("w3")}, ${hi("w3")}));
}`;
  }
  if (b === "i8") {
    const load = vec
      ? `let w = B[(${row} * P.K + k) >> 3u];\n  let w0 = i32(w.x);\n  let w1 = i32(w.y);`
      : `let base = (${row} * P.K + k) >> 2u;\n  let w0 = i32(B[base]);\n  let w1 = i32(B[base + 1u]);`;
    const x = (w) => range(4).map((j) => `f32(extractBits(${w}, ${8 * j}u, 8u))`).join(", ");
    return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  ${load}
  return V8(vec4<f32>(${x("w0")}), vec4<f32>(${x("w1")}));
}`;
  }
  if (b === "w4") {
    const lv = (j) => `clamp(round(CB[(codes >> ${4 * j}u) & 15u] * s), -127.0, 127.0)`;
    return `
fn fp8e4m3(b: u32) -> f32 {
  let e = (b >> 3u) & 15u;
  let m = f32(b & 7u);
  var v: f32;
  if (e == 0u) { v = m * 0.001953125; } else { v = (1.0 + m * 0.125) * exp2(f32(e) - 7.0); }
  if ((b & 128u) != 0u) { v = -v; }
  return v;
}
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  let codes = B[(${row} * P.K + k) >> 3u];
  let gi = (${row} * P.K + k) >> 4u;
  let s = fp8e4m3((SR[gi >> 2u] >> ((gi & 3u) * 8u)) & 255u);
  return V8(vec4<f32>(${range(4).map(lv).join(", ")}), vec4<f32>(${range(4).map((j) => lv(j + 4)).join(", ")}));
}`;
  }
  throw new Error(`unknown B format ${b}`);
}

// vecA / vecB: vectorized global loads; the caller guarantees K % 8 == 0 and 4-aligned offsets
// and strides (for the weight formats: K % 8 == 0; conv modes: cin % 8 == 0).
export function matmulShader({ a = "rows", b = "f32", bias = false, act = "none", resid = false, R = 8, vecA = false, vecB = false, snake = false }) {
  const key = JSON.stringify([a, b, bias, act, resid, R, vecA, vecB, snake]);
  if (cache.has(key)) return cache.get(key);

  const scaled = b === "i8" || b === "w4";
  const bindings = [];
  const bind = (decl) => bindings.push(`@group(0) @binding(${bindings.length + 1}) ${decl};`);
  bind(`var<storage, read> A: array<${vecA ? "vec4<f32>" : "f32"}>`);
  const bType = b === "f32" ? (vecB ? "vec4<f32>" : "f32")
    : b === "f32t" ? "f32"
      : b === "bf16" ? (vecB ? "vec4<u32>" : "u32")
        : b === "i8" ? (vecB ? "vec2<u32>" : "u32") : "u32";
  bind(`var<storage, read> B: array<${bType}>`);
  bind("var<storage, read_write> C: array<f32>");
  if (scaled) bind("var<storage, read> RS: array<f32>");
  if (b === "w4") { bind("var<storage, read> SR: array<u32>"); bind("var<storage, read> CB: array<f32>"); }
  if (bias) bind("var<storage, read> BIAS: array<f32>");
  if (snake) bind("var<storage, read> ALPHA: array<f32>");

  // value of output (m, n) before the activation
  const pre = (v, n) => `${v} * P.alpha${scaled ? ` * RS[${n}]` : ""}${bias ? ` + BIAS[${n}]` : ""}`;
  let epi = "";
  if (act === "gelu") epi = " r = gelu(r);";
  if (act === "silu") epi = " r = silu(r);";
  const store = resid ? "C[ci] = C[ci] + r;" : "C[ci] = r;";

  const T = 16 * R; // tile edge
  const TK = R === 8 ? 16 : 32;
  const V = T / 4; // vec4 per k-row of a tile
  const G = R / 4; // 64-wide row/column groups per thread
  const perRow = TK / 8; // threads per staged row

  // staging: 8 k values of row lr -> As[(lk + j) * V + (lr >> 2)][lr & 3]
  const stage = range(8).map((j) => {
    const src = j < 4 ? `lo.${"xyzw"[j]}` : `hi.${"xyzw"[j - 4]}`;
    return `As[(lk + ${j}u) * ${V}u + sIdx][sLane] = av.${src};\n      Bs[(lk + ${j}u) * ${V}u + sIdx][sLane] = bv.${src};`;
  });

  const fma = [];
  for (let g = 0; g < G; g++) fma.push(`let a${g} = As[kk * ${V}u + ${g * 16}u + tm];`);
  for (let h = 0; h < G; h++) fma.push(`let b${h} = Bs[kk * ${V}u + ${h * 16}u + tn];`);
  for (let g = 0; g < G; g++) {
    ["x", "y", "z", "w"].forEach((c, i) => {
      for (let h = 0; h < G; h++) fma.push(`acc${(g * 4 + i) * G + h} += a${g}.${c} * b${h};`);
    });
  }

  // epilogue: fully unrolled over the thread's R x R outputs
  const out = [];
  for (let g = 0; g < G; g++) {
    for (let i = 0; i < 4; i++) {
      out.push(`{ let m = m0 + ${g * 64 + i}u + tm * 4u;\n    if (m < P.M) {`);
      for (let h = 0; h < G; h++) {
        const acc = `acc${(g * 4 + i) * G + h}`;
        if (act === "swiglu") {
          // columns (n, n+1) = (gate_j, up_j) with j = n/2; N is even, so both are in range together
          for (const j of [0, 2]) {
            const [cg, cu] = ["xyzw"[j], "xyzw"[j + 1]];
            out.push(`      { let n = n0 + ${h * 64 + j}u + tn * 4u; if (n < P.N) { let ci = P.cOff + z * P.cBatch + m * P.ldc + n / 2u;` +
              ` let g = ${pre(`${acc}.${cg}`, "n")}; let u = ${pre(`${acc}.${cu}`, "n + 1u")}; let r = silu(g) * u; ${store} } }`);
          }
        } else {
          for (let j = 0; j < 4; j++) {
            out.push(`      { let n = n0 + ${h * 64 + j}u + tn * 4u; if (n < P.N) { let ci = P.cOff + z * P.cBatch + m * P.ldc + n; var r = ${pre(`${acc}.${"xyzw"[j]}`, "n")};${epi} ${store} } }`);
          }
        }
      }
      out.push("    } }");
    }
  }

  const code = /* wgsl */ `
struct Params {
  M: u32, N: u32, K: u32, alpha: f32,
  lda: u32, aBatch: u32, aOff: u32, ldb: u32,
  bBatch: u32, bDiv: u32, bOff: u32, ldc: u32,
  cBatch: u32, cOff: u32, bz: u32, cin: u32,
  tin: u32, stride: u32, dil: u32, pad: u32,
};
struct V8 { lo: vec4<f32>, hi: vec4<f32> };
@group(0) @binding(0) var<uniform> P: Params;
${bindings.join("\n")}
${act !== "none" ? ACT : ""}
${loaderA(a, vecA, snake)}
${loaderB(b, vecB)}

const TK = ${TK}u;
var<workgroup> As: array<vec4<f32>, ${V * TK}>;  // [TK][T/4]
var<workgroup> Bs: array<vec4<f32>, ${V * TK}>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let z = wg.z;
  let m0 = wg.y * ${T}u;
  let n0 = wg.x * ${T}u;
  let tm = t & 15u;
  let tn = t >> 4u;
  let lr = t / ${perRow}u;
  let lk = (t % ${perRow}u) * 8u;
  let sIdx = lr >> 2u;
  let sLane = lr & 3u;
  ${range(R * G).map((i) => `var acc${i} = vec4<f32>();`).join("\n  ")}
  var av = loadA8(z, m0 + lr, lk);
  var bv = loadB8(z, n0 + lr, lk);

  for (var k0 = 0u; k0 < P.K; k0 += TK) {
    ${stage.join("\n    ")}
    workgroupBarrier();
    if (k0 + TK < P.K) {
      av = loadA8(z, m0 + lr, k0 + TK + lk);
      bv = loadB8(z, n0 + lr, k0 + TK + lk);
    }
    for (var kk = 0u; kk < TK; kk++) {
      ${fma.join("\n      ")}
    }
    workgroupBarrier();
  }

  ${out.join("\n  ")}
}`;
  cache.set(key, code);
  return code;
}
