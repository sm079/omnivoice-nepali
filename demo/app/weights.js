// Reads safetensors files (from a Blob/File, e.g. the OPFS cache) and uploads weights.
//
// Linear weight objects: { kind: "bf16" | "i8", N, K, buf, scale?, bias? }. Quantized weights
// follow the ComfyUI comfy_quant layout written by tools/build_assets.py (int8 + ConvRot).

const BYTES = { F32: 4, BF16: 2, F16: 2, I8: 1, U8: 1, F8_E4M3: 1 };

export function bf16ToF32(u8) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset, u8.byteLength / 2);
  const out = new Float32Array(u16.length);
  const o32 = new Uint32Array(out.buffer);
  for (let i = 0; i < u16.length; i++) o32[i] = u16[i] << 16;
  return out;
}

function f16ToF32(u8) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset, u8.byteLength / 2);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 31;
    const m = h & 1023;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

export class SafeTensors {
  static async open(blob, prefix = "") {
    const head = new DataView(await blob.slice(0, 8).arrayBuffer());
    const n = Number(head.getBigUint64(0, true));
    const header = JSON.parse(new TextDecoder().decode(await blob.slice(8, 8 + n).arrayBuffer()));
    delete header.__metadata__;
    return new SafeTensors(blob, header, 8 + n, prefix);
  }

  constructor(blob, header, dataStart, prefix) {
    this.blob = blob;
    this.header = header;
    this.dataStart = dataStart;
    this.prefix = prefix;
  }

  has(name) {
    return this.prefix + name in this.header;
  }

  info(name) {
    const t = this.header[this.prefix + name];
    if (!t) throw new Error(`missing tensor ${this.prefix + name}`);
    return t;
  }

  async bytes(name) {
    const t = this.info(name);
    const [a, b] = t.data_offsets;
    return new Uint8Array(await this.blob.slice(this.dataStart + a, this.dataStart + b).arrayBuffer());
  }

  async f32(name) {
    const t = this.info(name);
    const u8 = await this.bytes(name);
    if (t.dtype === "F32") return new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4);
    if (t.dtype === "BF16") return bf16ToF32(u8);
    if (t.dtype === "F16") return f16ToF32(u8);
    const buf = u8.slice().buffer; // copy: typed-array views need an aligned offset
    const other = { F64: Float64Array, I64: BigInt64Array, I32: Int32Array, I16: Int16Array, I8: Int8Array, U8: Uint8Array }[t.dtype];
    if (other) return Float32Array.from(new other(buf), Number);
    throw new Error(`unsupported dtype ${t.dtype} for ${name}`);
  }

  // Gather rows of a 2D table (embeddings) without reading the whole tensor.
  async rows(name, ids) {
    const base = name.slice(0, -"weight".length);
    if (this.has(base + "comfy_quant")) return this.int8Rows(base, ids);
    const t = this.info(name);
    const [, dim] = t.shape;
    const rowBytes = dim * BYTES[t.dtype];
    const out = new Float32Array(ids.length * dim);
    await Promise.all(ids.map(async (id, i) => {
      const off = this.dataStart + t.data_offsets[0] + id * rowBytes;
      const u8 = new Uint8Array(await this.blob.slice(off, off + rowBytes).arrayBuffer());
      out.set(t.dtype === "BF16" ? bf16ToF32(u8) : t.dtype === "F16" ? f16ToF32(u8) : new Float32Array(u8.buffer), i * dim);
    }));
    return out;
  }

  // Per-row int8 table (tools/quant.py quantize_int8_rows): value = q * weight_scale[row].
  async int8Rows(base, ids) {
    const t = this.info(base + "weight");
    const dim = t.shape[1];
    this._scales ||= {};
    const scales = this._scales[base] || (this._scales[base] = await this.f32(base + "weight_scale"));
    const out = new Float32Array(ids.length * dim);
    await Promise.all(ids.map(async (id, i) => {
      const off = this.dataStart + t.data_offsets[0] + id * dim;
      const q = new Int8Array(await this.blob.slice(off, off + dim).arrayBuffer());
      const s = scales[id];
      for (let j = 0; j < dim; j++) out[i * dim + j] = q[j] * s;
    }));
    return out;
  }

  async vector(gpu, name) {
    return gpu.upload(await this.f32(name));
  }

  // Raw bytes of a bf16 tensor, uploaded as is (bf16 pairs packed in u32).
  async bf16(gpu, name) {
    const t = this.info(name);
    if (t.dtype !== "BF16") throw new Error(`expected bf16 for ${name}, got ${t.dtype}`);
    return gpu.upload(await this.bytes(name));
  }

  quantFormat(base) {
    return this.has(base + "comfy_quant") ? "i8" : "bf16";
  }

  // Several linears that share an input, stacked into one [sum N, K] weight so one GEMM replaces
  // several. interleave: rows alternate between the parts (gate_0, up_0, gate_1, up_1, ...), the
  // layout the fused SwiGLU epilogue reads. Returns the weight plus each part's row placement
  // ({ name, off, n, stride }), which LoRA merging needs.
  async fused(gpu, bases, { interleave = false } = {}) {
    const kinds = bases.map((b) => this.quantFormat(b));
    if (kinds.some((k) => k !== kinds[0])) throw new Error(`mixed precisions in ${bases.join(", ")}`);
    const kind = kinds[0];
    const infos = bases.map((b) => this.info(b + "weight"));
    if (infos.some((t) => t.dtype !== (kind === "i8" ? "I8" : "BF16"))) {
      const bad = infos.find((t) => t.dtype !== (kind === "i8" ? "I8" : "BF16"));
      throw new Error(`unsupported weight format ${bad.dtype} in ${bases.join(", ")} (this build needs bf16 or int8)`);
    }
    const K = infos[0].shape[1];
    const Ns = infos.map((t) => t.shape[0]);
    const N = Ns.reduce((a, b) => a + b, 0);
    const el = kind === "i8" ? 1 : 2;
    const rowBytes = K * el;
    const parts = await Promise.all(bases.map((b) => this.bytes(b + "weight")));
    const data = new Uint8Array(N * rowBytes);
    const layout = [];
    if (interleave) {
      if (Ns.some((n) => n !== Ns[0])) throw new Error("interleaved parts must have equal rows");
      const P = parts.length;
      for (let i = 0; i < Ns[0]; i++) {
        for (let p = 0; p < P; p++) data.set(parts[p].subarray(i * rowBytes, (i + 1) * rowBytes), (i * P + p) * rowBytes);
      }
      bases.forEach((b, p) => layout.push({ name: b.replace(/\.$/, ""), off: p, n: Ns[p], stride: P }));
    } else {
      let off = 0;
      bases.forEach((b, p) => {
        data.set(parts[p], off * rowBytes);
        layout.push({ name: b.replace(/\.$/, ""), off, n: Ns[p], stride: 1 });
        off += Ns[p];
      });
    }
    const W = { kind, N, K, buf: gpu.upload(data), parts: layout };
    if (kind === "i8") {
      const scales = await Promise.all(bases.map((b) => this.f32(b + "weight_scale")));
      const s = new Float32Array(N);
      for (const [p, l] of layout.entries()) for (let i = 0; i < l.n; i++) s[l.off + i * l.stride] = scales[p][i];
      W.scale = gpu.upload(s);
    }
    return W;
  }

  async linear(gpu, base) {
    return this.fused(gpu, [base]);
  }
}
