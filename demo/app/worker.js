// Engine worker: hosts the pipeline so weight loading, command encoding and the CPU side of
// decoding never block the page. Protocol (see engine.js):
//   in : { id, type: "init" | "load" | "generate" | "loadCloning" | "prepareReference" |
//          "transcribe" | "encodeReference" | "cancel" | "unload", ... }
//   out: { id, type: "status" | "progress" | "result" | "error", ... }

import { OmniVoicePipeline } from "./pipeline.js";

let pipe = null;
const aborts = new Map();

const gpuInfo = () => {
  const i = pipe?.gpu?.info || {};
  return { name: [i.vendor, i.architecture, i.description].filter(Boolean).join(" "), peak: pipe?.gpu?.pool.peak || 0 };
};

async function handle(msg) {
  const { id, type } = msg;
  const reply = (m, transfer = []) => self.postMessage({ id, ...m }, transfer);
  if (type === "init") {
    reply({ type: "result", ok: !!self.navigator.gpu });
    return;
  }
  if (type === "cancel") {
    aborts.get(msg.target)?.abort();
    return;
  }
  if (type === "unload") {
    pipe?.unload();
    reply({ type: "result" });
    return;
  }
  const ac = new AbortController();
  aborts.set(id, ac);
  try {
    if (type === "load") {
      pipe ||= new OmniVoicePipeline();
      await pipe.load(new URL(msg.base), msg.manifest, msg.selection, { signal: ac.signal, token: msg.token, onStatus: (s) => reply({ type: "status", status: s }) });
      reply({ type: "result", loaded: { ...pipe.loaded }, gpu: gpuInfo() });
    } else if (type === "loadCloning") {
      pipe ||= new OmniVoicePipeline();
      await pipe.loadCloning(new URL(msg.base), msg.manifest, msg.part, { signal: ac.signal, token: msg.token, onStatus: (s) => reply({ type: "status", status: s }) });
      reply({ type: "result" });
    } else if (type === "prepareReference") {
      pipe ||= new OmniVoicePipeline();
      const r = pipe.prepareReference(msg.samples, msg.sr);
      reply({ type: "result", ...r }, [r.wav.buffer]);
    } else if (type === "transcribe") {
      reply({ type: "result", text: await pipe.transcribe(msg.wav) });
    } else if (type === "encodeReference") {
      const r = await pipe.encodeReference(msg.wav);
      reply({ type: "result", tokens: r.tokens, frames: r.frames }, [r.tokens.buffer]);
    } else if (type === "generate") {
      const res = await pipe.generate({ ...msg.opts, signal: ac.signal, onProgress: (p) => reply({ type: "progress", progress: p }) });
      reply({ type: "result", audio: res.audio, sampleRate: res.sampleRate, timings: res.timings, gpu: gpuInfo() }, [res.audio.buffer]);
    }
  } catch (e) {
    reply({ type: "error", name: e.name, message: e.message, stack: e.stack });
  } finally {
    aborts.delete(id);
  }
}

self.onmessage = (e) => { handle(e.data); };
