// OmniVoice text-to-speech pipeline: download-once assets -> WebGPU models -> iterative masked
// decoding -> codec decoder -> post-processing. A port of OmniVoice.generate (voice cloning with a
// prepared voice prompt, voice design or auto voice; one item at a time).

import { GPU } from "./gpu/device.js";
import * as ops from "./gpu/ops.js";
import { SafeTensors } from "./weights.js";
import { cachedFile, requestPersistence } from "./store.js";
import { TextTokenizer, estimateFrames, chunkText, addPunctuation } from "./text.js";
import { OmniLLM, CFG, ropeTable } from "./nn/llm.js";
import { loadAdapter } from "./nn/lora.js";
import { CodecDecoder, SAMPLE_RATE } from "./nn/codec.js";
import { removeSilence, fadeAndPad, crossFade, peak } from "./audio.js";

export { SAMPLE_RATE };
const FRAME_RATE = 25;

export const DEFAULTS = {
  steps: 32, guidance: 2.0, tShift: 0.1, layerPenalty: 5.0, positionTemperature: 5.0,
  denoise: true, postprocess: true, chunkDuration: 15, chunkThreshold: 30,
};

export async function fetchManifest(baseUrl) {
  const res = await fetch(new URL("manifest.json", baseUrl), { cache: "no-cache" });
  if (!res.ok) throw new Error(`could not load ${new URL("manifest.json", baseUrl)} (${res.status}). Build the assets with demo/tools/build_assets.py or pass ?models=<url>.`);
  const m = await res.json();
  if (m.version !== 1) throw new Error("manifest.json is from a different build; rerun demo/tools/build_assets.py.");
  return m;
}

// Files a selection needs: { llm, decoder, adapter? } with { url, name, size }
export function resolveFiles(baseUrl, manifest, { precision, model }) {
  const llm = manifest.llm[precision];
  if (!llm) throw new Error(`no ${precision} build of the model`);
  const m = manifest.models.find((x) => x.id === model);
  if (!m) throw new Error(`unknown model ${model}`);
  const file = (f) => ({ url: new URL(f.path, baseUrl).href, name: f.cacheName || f.path.split("/").pop(), size: f.size });
  const out = { llm: file(llm), decoder: file(manifest.decoder) };
  if (m.adapter) out.adapter = file(m.adapter);
  return out;
}

// Small seeded RNG (xoshiro128**) for the position noise; uniform in (0, 1)
class Rng {
  constructor(seed) {
    let s = seed >>> 0 || 1;
    const sm = () => { s = (s + 0x9e3779b9) >>> 0; let z = s; z = Math.imul(z ^ (z >>> 16), 0x85ebca6b); z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35); return (z ^ (z >>> 16)) >>> 0; };
    this.s = [sm(), sm(), sm(), sm()];
  }
  next() {
    const s = this.s;
    const r = Math.imul(((Math.imul(s[1], 5) << 7) | (Math.imul(s[1], 5) >>> 25)), 9) >>> 0;
    const t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
    s[3] = ((s[3] << 11) | (s[3] >>> 21)) >>> 0;
    return (r + 0.5) / 4294967296;
  }
}

// Step k of the shifted schedule unmasks this many of `total` tokens (OmniVoice._generate_iterative)
function schedule(total, steps, shift) {
  const f = Math.fround;
  const ts = [];
  for (let i = 0; i <= steps; i++) {
    const t = f(i / steps);
    ts.push(f(f(shift * t) / f(1 + f((shift - 1) * t))));
  }
  const out = [];
  let rem = total;
  for (let s = 0; s < steps; s++) {
    const n = s === steps - 1 ? rem : Math.min(Math.ceil(total * (ts[s + 1] - ts[s])), rem);
    out.push(n);
    rem -= n;
  }
  return out;
}

export class OmniVoicePipeline {
  constructor(gpuOptions = {}) {
    this.gpuOptions = gpuOptions;
    this.gpu = null;
    this.loaded = {}; // llm: "<llm file>|<model id>", decoder: file name
    this.llm = null;
    this.decoder = null;
  }

  isLoaded(files, model) {
    return this.loaded.llm === `${files.llm.name}|${model}` && this.loaded.decoder === files.decoder.name;
  }

  // Downloads (first time only) and loads the selection. Switching models reloads the backbone
  // from the browser cache and merges the new adapter into it; the codec stays on the GPU.
  async load(baseUrl, manifest, selection, { onStatus = () => {}, signal, token = "" } = {}) {
    await requestPersistence();
    this.gpu ||= await GPU.create(this.gpuOptions);
    const gpu = this.gpu;
    const hfHeaders = (url) => (token && new URL(url).hostname === "huggingface.co" ? { Authorization: `Bearer ${token}` } : {});
    const fetchJson = async (p) => {
      const url = new URL(p, baseUrl).href;
      const res = await fetch(url, { headers: hfHeaders(url) });
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
      return res.json();
    };
    this.tokenizer ||= await TextTokenizer.load(fetchJson, manifest.tokenizer);
    const want = resolveFiles(baseUrl, manifest, selection);
    const model = manifest.models.find((m) => m.id === selection.model);

    const llmKey = `${want.llm.name}|${model.id}`;
    const parts = [];
    if (this.loaded.llm !== llmKey) parts.push("llm", ...(want.adapter ? ["adapter"] : []));
    if (this.loaded.decoder !== want.decoder.name) parts.push("decoder");
    const total = parts.reduce((a, k) => a + want[k].size, 0);
    const got = Object.fromEntries(parts.map((k) => [k, 0]));
    const files = {};
    for (const k of parts) {
      const f = want[k];
      files[k] = await cachedFile(f.url, f.name, f.size, (done) => {
        got[k] = done;
        onStatus({ phase: "download", file: f.name, done: parts.reduce((a, p) => a + got[p], 0), total });
      }, signal, hfHeaders(f.url));
    }

    if (parts.includes("decoder")) {
      this.unloadPart("decoder");
      onStatus({ phase: "load", what: "voice decoder", frac: 0 });
      this.decoder = await CodecDecoder.load(gpu, await SafeTensors.open(files.decoder));
      this.loaded.decoder = want.decoder.name;
    }
    if (parts.includes("llm")) {
      this.unloadPart("llm");
      let lora = null;
      if (files.adapter) {
        onStatus({ phase: "load", what: model.label, frac: 0 });
        lora = await loadAdapter(files.adapter, model.adapter.config);
      }
      const what = lora ? `${model.label} (merging its adapter)` : model.label;
      this.llm = await OmniLLM.load(gpu, await SafeTensors.open(files.llm), {
        lora, onProgress: (f) => onStatus({ phase: "load", what, frac: f }),
      });
      this.loaded.llm = llmKey;
    }
    await gpu.sync();
    this.selection = { ...selection };
    onStatus({ phase: "ready" });
  }

  unloadPart(k) {
    const destroy = (o) => {
      if (!o || typeof o !== "object") return;
      if (o instanceof GPUBuffer) { o.destroy(); return; }
      for (const v of Array.isArray(o) ? o : Object.values(o)) {
        if (v && typeof v === "object" && v !== this.gpu && !(v instanceof SafeTensors)) destroy(v);
      }
    };
    if (this[k]) {
      this.gpu.flush();
      destroy(this[k]);
    }
    this[k] = null;
    delete this.loaded[k];
  }

  unload() {
    this.unloadPart("llm");
    this.unloadPart("decoder");
    this.gpu?.pool.trim();
  }

  // The packed decoding problem for one item. voice: { tokens?: Int32Array [8][Tr], frames?, text?, instruct? }
  // Rows: text tokens (style + text) | reference frames | conditional target, then the unconditional target.
  async prepareItem(text, voice, T, denoise = true) {
    const gpu = this.gpu;
    const C = CFG.codebooks;
    const hasRef = !!voice.tokens;
    const Tr = hasRef ? voice.frames : 0;
    const textIds = [...this.tokenizer.style(hasRef, voice.instruct, denoise), ...this.tokenizer.text(text, hasRef ? voice.text : null)];
    const Lt = textIds.length;
    const cLen = Lt + Tr + T;
    // audio rows in packed order: reference frames, conditional target, unconditional target
    const ids = new Uint32Array((Tr + 2 * T) * C);
    for (let f = 0; f < Tr; f++) for (let c = 0; c < C; c++) ids[f * C + c] = voice.tokens[c * Tr + f];
    ids.fill(CFG.mask, Tr * C);
    return {
      T, Tr, Lt, cLen, L: cLen + T, textIds, ids,
      textX: gpu.fromArray(await this.llm.textEmbeddings(textIds), [Lt, CFG.hidden]),
      cs: ropeTable(gpu, cLen),
      idsBuf: gpu.empty([ids.length]),
      tokBuf: gpu.empty([C * T]),
      scoreBuf: gpu.empty([C * T]),
    };
  }

  releaseItem(it) {
    for (const k of ["textX", "cs", "idsBuf", "tokBuf", "scoreBuf"]) it[k].release();
  }

  // Sets frame f of codebook c of the target (conditional and unconditional rows) to `id`.
  setToken(it, c, f, id) {
    it.ids[(it.Tr + f) * CFG.codebooks + c] = id;
    it.ids[(it.Tr + it.T + f) * CFG.codebooks + c] = id;
  }

  // One forward pass on the current tokens -> guided greedy { pred, score } per (codebook, frame).
  // keep: also return the target rows' hidden states and logits (for tools/check.html).
  async step(it, guidance, keep = false) {
    const gpu = this.gpu;
    const D = CFG.hidden;
    const { T, Lt, Tr, cLen, L } = it;
    gpu.flush();
    gpu.write(it.idsBuf.buf, it.ids);
    const x = gpu.empty([L, D]);
    ops.copy(gpu, it.textX, x, Lt * D);
    ops.audioEmbed(gpu, this.llm.audioTable, it.idsBuf, x, { rows: Tr + 2 * T, D, vocab: CFG.vocab, row0: Lt });
    const h = this.llm.forward(x, { L, seg: cLen, cs: it.cs, r0: cLen - T, n: 2 * T });
    const logits = this.llm.logits(h);
    ops.cfgScore(gpu, logits, it.tokBuf, it.scoreBuf, { T, V: CFG.vocab, mask: CFG.mask, g: guidance });
    const [pred, score, hidden, lg] = await Promise.all([
      gpu.read(it.tokBuf), gpu.read(it.scoreBuf), keep ? gpu.read(h) : null, keep ? gpu.read(logits) : null,
    ]);
    h.release();
    logits.release();
    return { pred: new Uint32Array(pred.buffer), score, hidden, logits: lg };
  }

  // Masked iterative decoding of one item -> Int32Array [8][T] audio tokens.
  async generateTokens(text, voice, T, opts, rng, report) {
    const C = CFG.codebooks;
    const it = await this.prepareItem(text, voice, T, opts.denoise);
    const tokens = new Int32Array(C * T).fill(CFG.mask);
    const sched = schedule(C * T, opts.steps, opts.tShift);
    const scores = new Float64Array(C * T);
    try {
      for (let step = 0; step < opts.steps; step++) {
        opts.check();
        const { pred: predIds, score } = await this.step(it, opts.guidance);
        const k = sched[step];
        if (k > 0) {
          // confidence with an earlier-codebook bonus and Gumbel noise; only masked slots compete
          for (let i = 0; i < C * T; i++) {
            if (tokens[i] !== CFG.mask) { scores[i] = -Infinity; continue; }
            let s = score[i] - Math.floor(i / T) * opts.layerPenalty;
            if (opts.positionTemperature > 0) s = s / opts.positionTemperature - Math.log(-Math.log(rng.next() + 1e-10) + 1e-10);
            scores[i] = s;
          }
          const order = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]);
          for (let j = 0; j < k; j++) {
            const i = order[j];
            tokens[i] = predIds[i];
            this.setToken(it, Math.floor(i / T), i % T, predIds[i]);
          }
        }
        report(step + 1);
      }
    } finally {
      this.releaseItem(it);
    }
    return tokens;
  }

  // tokens [8][T] -> 24 kHz audio
  async decodeTokens(tokens, T) {
    const codes = new Uint32Array(T * 8);
    for (let f = 0; f < T; f++) for (let c = 0; c < 8; c++) codes[f * 8 + c] = tokens[c * T + f];
    return this.decoder.decode(codes, T);
  }

  // opts: { text, voice: { tokens?, frames?, text?, rms?, instruct? }, steps, guidance, speed, seed,
  //         onProgress, signal }  ->  { audio: Float32Array, sampleRate, timings }
  async generate(o) {
    const opts = { ...DEFAULTS, ...o };
    const { voice = {}, signal, onProgress = () => {} } = opts;
    opts.check = () => { if (signal?.aborted) throw new DOMException("Generation cancelled", "AbortError"); };
    const gpu = this.gpu;
    const t0 = performance.now();
    const text = opts.text.trim();
    if (!text) throw new Error("Nothing to say: the text is empty.");
    const rng = new Rng(opts.seed ?? 0);
    const speed = opts.speed || 1;
    const hasRef = !!voice.tokens;
    const refText = hasRef ? addPunctuation(voice.text || "") : null;
    const v = hasRef ? { ...voice, text: refText } : voice;

    // long text: chunks of ~15 s, as in OmniVoice._generate_chunked
    const total = estimateFrames(text, refText, voice.frames, speed);
    let jobs = [{ text, T: total }];
    if (total > opts.chunkThreshold * FRAME_RATE) {
      const perChar = total / [...text].length;
      const chunkLen = Math.floor((opts.chunkDuration * FRAME_RATE) / perChar);
      jobs = chunkText(text, chunkLen, 3).map((c) => ({ text: c, T: 0 }));
    }
    const steps = jobs.length * opts.steps;
    let done = 0;
    const tokenList = [];
    let first = null;
    for (const [i, job] of jobs.entries()) {
      // without a reference voice, later chunks continue in the first chunk's voice
      const jv = i > 0 && !hasRef ? { ...voice, tokens: first.tokens, frames: first.T, text: first.text } : v;
      const T = job.T || estimateFrames(job.text, jv.tokens ? jv.text : null, jv.frames, speed);
      const tokens = await this.generateTokens(job.text, jv, T, opts, rng, (s) => {
        onProgress({ phase: "sample", step: done + s, steps, chunk: i, chunks: jobs.length, frac: (done + s) / steps });
      });
      done += opts.steps;
      tokenList.push({ tokens, T });
      if (i === 0) first = { tokens, T, text: job.text };
    }
    const tSample = performance.now();
    onProgress({ phase: "decode" });
    let audio;
    try {
      const chunks = [];
      for (const { tokens, T } of tokenList) {
        opts.check();
        chunks.push(await this.decodeTokens(tokens, T));
      }
      audio = crossFade(chunks, SAMPLE_RATE);
    } finally {
      await gpu.sync().catch(() => {});
      gpu.pool.trim();
    }
    if (opts.postprocess) audio = removeSilence(audio, SAMPLE_RATE, { midSil: 500, leadSil: 100, trailSil: 100 });
    // loudness follows the reference voice; without one, peak-normalise to 0.5
    const refRms = hasRef ? voice.rms ?? null : null;
    if (refRms != null && refRms < 0.1) {
      for (let i = 0; i < audio.length; i++) audio[i] *= refRms / 0.1;
    } else if (refRms == null) {
      const p = peak(audio);
      if (p > 1e-6) for (let i = 0; i < audio.length; i++) audio[i] *= 0.5 / p;
    }
    audio = fadeAndPad(audio, SAMPLE_RATE);
    const t1 = performance.now();
    return {
      audio,
      sampleRate: SAMPLE_RATE,
      tokens: tokenList,
      timings: { sample: tSample - t0, decode: t1 - tSample, total: t1 - t0, perStep: (tSample - t0) / steps, seconds: audio.length / SAMPLE_RATE },
    };
  }
}
