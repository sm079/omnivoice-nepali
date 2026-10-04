// Page-side handle on the engine. Runs the pipeline in a Web Worker (worker.js) so the UI stays
// responsive; falls back to running it in the page when WebGPU isn't exposed to workers.

import { OmniVoicePipeline, resolveFiles } from "./pipeline.js";
import { requestPersistence } from "./store.js";

class WorkerEngine {
  constructor(worker) {
    this.worker = worker;
    this.next = 1;
    this.calls = new Map();
    this.loaded = {};
    this.gpu = { name: "", peak: 0 };
    worker.onmessage = (e) => {
      const m = e.data;
      const call = this.calls.get(m.id);
      if (!call) return;
      if (m.type === "status") call.onStatus?.(m.status);
      else if (m.type === "progress") call.onProgress?.(m.progress);
      else {
        this.calls.delete(m.id);
        if (m.type === "error") {
          const err = new Error(m.message);
          err.name = m.name;
          err.stack = m.stack;
          call.reject(err);
        } else call.resolve(m);
      }
    };
  }

  call(msg, hooks = {}) {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.calls.set(id, { resolve, reject, ...hooks });
      if (hooks.signal) {
        if (hooks.signal.aborted) this.worker.postMessage({ type: "cancel", target: id });
        hooks.signal.addEventListener("abort", () => this.worker.postMessage({ type: "cancel", target: id }), { once: true });
      }
      this.worker.postMessage({ id, ...msg });
    });
  }

  isLoaded(base, manifest, sel) {
    const f = resolveFiles(base, manifest, sel);
    return this.loaded.llm === `${f.llm.name}|${sel.model}` && this.loaded.decoder === f.decoder.name;
  }

  async load(base, manifest, selection, { onStatus, signal, token } = {}) {
    const r = await this.call({ type: "load", base: base.href, manifest, selection, token }, { onStatus, signal });
    this.loaded = r.loaded;
    this.gpu = r.gpu;
  }

  async generate({ onProgress, signal, ...opts }) {
    const r = await this.call({ type: "generate", opts }, { onProgress, signal });
    this.gpu = r.gpu;
    return { audio: r.audio, sampleRate: r.sampleRate, timings: r.timings };
  }

  unload() {
    this.loaded = {};
    return this.call({ type: "unload" });
  }

  loadCloning(base, manifest, part, { onStatus, signal, token } = {}) {
    return this.call({ type: "loadCloning", base: base.href, manifest, part, token }, { onStatus, signal });
  }

  async prepareReference(samples, sr) {
    const { wav, rms, seconds } = await this.call({ type: "prepareReference", samples, sr });
    return { wav, rms, seconds };
  }

  async transcribe(wav) {
    return (await this.call({ type: "transcribe", wav })).text;
  }

  async encodeReference(wav) {
    const { tokens, frames } = await this.call({ type: "encodeReference", wav });
    return { tokens, frames };
  }
}

class LocalEngine {
  constructor() {
    this.pipe = new OmniVoicePipeline();
  }
  get loaded() { return this.pipe.loaded; }
  get gpu() {
    const i = this.pipe.gpu?.info || {};
    return { name: [i.vendor, i.architecture, i.description].filter(Boolean).join(" "), peak: this.pipe.gpu?.pool.peak || 0 };
  }
  isLoaded(base, manifest, sel) { return this.pipe.isLoaded(resolveFiles(base, manifest, sel), sel.model); }
  load(base, manifest, selection, opts) { return this.pipe.load(base, manifest, selection, opts); }
  generate(opts) { return this.pipe.generate(opts); }
  unload() { this.pipe.unload(); }
  loadCloning(base, manifest, part, opts) { return this.pipe.loadCloning(base, manifest, part, opts); }
  async prepareReference(samples, sr) { return this.pipe.prepareReference(samples, sr); }
  transcribe(wav) { return this.pipe.transcribe(wav); }
  encodeReference(wav) { return this.pipe.encodeReference(wav); }
}

// inPage: force the in-page engine (debugging; ?engine=page)
export async function createEngine({ inPage = false } = {}) {
  requestPersistence(); // Window-only API; the worker can't ask for it
  if (inPage) return new LocalEngine();
  try {
    const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    const engine = new WorkerEngine(worker);
    const { ok } = await Promise.race([
      engine.call({ type: "init" }),
      new Promise((_, rej) => setTimeout(() => rej(new Error("worker did not start")), 10000)),
    ]);
    if (ok) return engine;
    worker.terminate();
    console.warn("WebGPU is not available in workers here; running the engine on the page");
  } catch (e) {
    console.warn("engine worker unavailable, running on the page:", e.message);
  }
  return new LocalEngine();
}
