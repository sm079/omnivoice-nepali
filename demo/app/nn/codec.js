// Higgs Audio v2 tokenizer, decoder side: 8 codebook indices per 40 ms frame -> 24 kHz audio.
//
//   x = fc2(sum_c project_out_c(codebook_c[code_c]))      folded offline into 8 [1024, 256] tables
//   DAC decoder: conv7 256->1024, five upsampling blocks (Snake, transposed conv x8/x5/x4/x2/x3,
//   three residual units with dilations 1/3/9), Snake, conv7 32->1 (no tanh in this variant)
//
// Activations are time-major [T, C]; every conv is an implicit GEMM with the preceding Snake
// fused into its input loads (see gpu/gemm.js).

import * as ops from "../gpu/ops.js";

const UP = [8, 5, 4, 2, 3];
export const HOP = 960;
export const SAMPLE_RATE = 24000;

async function convWeight(gpu, st, base, N) {
  const t = st.info(base + ".weight");
  const kind = t.dtype === "BF16" ? "bf16" : "f32";
  const buf = kind === "bf16" ? await st.bf16(gpu, base + ".weight") : await st.vector(gpu, base + ".weight");
  const K = t.shape.slice(1).reduce((a, b) => a * b, 1);
  return { kind, N: N ?? t.shape[0], K, buf, bias: await st.vector(gpu, base + ".bias") };
}

export class CodecDecoder {
  static async load(gpu, st) {
    const d = new CodecDecoder(gpu);
    const cb = [];
    for (let c = 0; c < 8; c++) cb.push(await st.f32(`codebook.${c}`));
    const all = new Float32Array(cb.reduce((a, x) => a + x.length, 0));
    let off = 0;
    for (const x of cb) { all.set(x, off); off += x.length; }
    d.codebooks = gpu.upload(all);
    d.cbBias = await st.vector(gpu, "codebook.bias");
    d.dim = st.info("codebook.bias").shape[0];
    d.conv1 = await convWeight(gpu, st, "conv1");
    d.blocks = [];
    for (let b = 0; b < UP.length; b++) {
      const p = `block.${b}.`;
      const up = st.info(p + "up.weight").shape; // [s, cout, 2cin]
      const blk = {
        s: UP[b],
        cin: up[2] / 2,
        cout: up[1],
        snake: await st.vector(gpu, p + "snake.alpha"),
        up: await convWeight(gpu, st, p + "up", up[1]),
        res: [],
      };
      blk.up.K = up[2];
      for (const u of [1, 2, 3]) {
        const r = `${p}res${u}.`;
        blk.res.push({
          dil: [1, 3, 9][u - 1],
          snake1: await st.vector(gpu, r + "snake1.alpha"),
          conv1: await convWeight(gpu, st, r + "conv1"),
          snake2: await st.vector(gpu, r + "snake2.alpha"),
          conv2: await convWeight(gpu, st, r + "conv2"),
        });
      }
      d.blocks.push(blk);
    }
    d.snake = await st.vector(gpu, "snake.alpha");
    d.conv2 = await convWeight(gpu, st, "conv2");
    return d;
  }

  constructor(gpu) {
    this.gpu = gpu;
  }

  // codes: Uint32Array [T * 8] (frame-major) -> Float32Array [T * 960]
  async decode(codes, T) {
    const gpu = this.gpu;
    const ids = gpu.fromArray(codes, [T * 8]);
    let x = ops.codebookSum(gpu, this.codebooks, this.cbBias, ids, T, this.dim);
    ids.release();
    let y = ops.conv1d(gpu, x, this.conv1, { tin: T, cin: this.dim, k: 7, pad: 3 });
    x.release();
    let t = T;
    for (const b of this.blocks) {
      const z = ops.convT1d(gpu, y, b.up, { tin: t, cin: b.cin, s: b.s, snake: b.snake });
      y.release();
      y = z;
      t *= b.s;
      for (const r of b.res) {
        const h = ops.conv1d(gpu, y, r.conv1, { tin: t, cin: b.cout, k: 7, dil: r.dil, pad: 3 * r.dil, snake: r.snake1 });
        ops.conv1d(gpu, h, r.conv2, { tin: t, cin: b.cout, k: 1, pad: 0, snake: r.snake2, into: y });
        h.release();
      }
      await gpu.sync(); // keep the queue short (and memory bounded) on long inputs
    }
    const out = ops.conv1d(gpu, y, this.conv2, { tin: t, cin: this.blocks.at(-1).cout, k: 7, pad: 3, snake: this.snake });
    y.release();
    const audio = await gpu.read(out, t);
    out.release();
    return audio;
  }
}
