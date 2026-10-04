// Compares the WebGPU engine with the PyTorch reference dumps of tools/reference.py.
//   tools/check.html?dump=../out/dump_int8                     (llm precision and adapter from meta.json)
//   &only=llm|codec                                            one part only

import { GPU } from "../app/gpu/device.js";
import { SafeTensors } from "../app/weights.js";
import { TextTokenizer } from "../app/text.js";
import { OmniLLM, CFG } from "../app/nn/llm.js";
import { loadAdapter } from "../app/nn/lora.js";
import { CodecDecoder } from "../app/nn/codec.js";
import { OmniVoicePipeline } from "../app/pipeline.js";

const params = new URLSearchParams(location.search);
const DUMP = new URL((params.get("dump") || "../out/dump_int8") + "/", location.href);
const MODELS = new URL(params.get("models") || "../models/", location.href);
const only = params.get("only");
const log = (s) => { document.getElementById("log").textContent += s + "\n"; console.log(s); };

const bin = async (name, T) => new T(await (await fetch(new URL(name, DUMP))).arrayBuffer());
const blob = async (url) => (await fetch(url)).blob();
const fetchJson = async (p) => (await fetch(new URL(p, MODELS))).json();

function relErr(a, b) {
  let num = 0, den = 0;
  for (let i = 0; i < b.length; i++) { const d = a[i] - b[i]; num += d * d; den += b[i] * b[i]; }
  return Math.sqrt(num / den);
}

async function main() {
  const meta = await (await fetch(new URL("meta.json", DUMP))).json();
  const manifest = await fetchJson("manifest.json");
  const gpu = await GPU.create();
  log(`GPU: ${[gpu.info.vendor, gpu.info.architecture, gpu.info.description].filter(Boolean).join(" ")}`);
  const pipe = new OmniVoicePipeline();
  pipe.gpu = gpu;
  pipe.tokenizer = await TextTokenizer.load(fetchJson, manifest.tokenizer);

  // ---- tokenizer
  const style = pipe.tokenizer.style(true, null);
  const text = pipe.tokenizer.text(meta.text, meta.ref_text);
  const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
  log(`tokenizer: style ids ${same(style, meta.style_ids) ? "match" : "DIFFER " + JSON.stringify([style, meta.style_ids])}, ` +
    `text ids ${same(text, meta.text_ids) ? "match" : "DIFFER\n  js " + text.join(",") + "\n  py " + meta.text_ids.join(",")}`);

  if (only !== "codec") {
    let lora = null;
    if (meta.adapter) {
      const name = meta.adapter.replace(/\\/g, "/").split("/").at(-2);
      lora = await loadAdapter(await blob(new URL(`adapters/${name}.safetensors`, MODELS)), { lora_alpha: 32, r: 16 });
      log(`adapter: ${name}`);
    }
    const t0 = performance.now();
    pipe.llm = await OmniLLM.load(gpu, await SafeTensors.open(await blob(new URL(manifest.llm[meta.precision].path, MODELS))), { lora });
    log(`llm ${meta.precision} loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s`);

    const { T, Tr } = meta;
    const ref = await bin("ref_tokens.i32", Int32Array);
    const state = await bin("state.i32", Int32Array);
    const it = await pipe.prepareItem(meta.text, { tokens: ref, frames: Tr, text: meta.ref_text }, T);
    if (it.cLen !== meta.c_len) log(`LAYOUT DIFFERS: c_len ${it.cLen} vs ${meta.c_len}`);
    for (let c = 0; c < 8; c++) for (let f = 0; f < T; f++) pipe.setToken(it, c, f, state[c * T + f]);
    await pipe.step(it, meta.guidance); // warm-up (shader compilation)
    const t1 = performance.now();
    const out = await pipe.step(it, meta.guidance, true);
    const ms = performance.now() - t1;
    const hidden = await bin("hidden.f32", Float32Array);
    const logits = await bin("logits.f32", Float32Array);
    const tok = await bin("cfg_tok.i32", Int32Array);
    const score = await bin("cfg_score.f32", Float32Array);
    let agree = 0;
    for (let i = 0; i < tok.length; i++) agree += out.pred[i] === tok[i];
    log(`one step (${it.L} rows): ${ms.toFixed(0)} ms`);
    log(`  hidden rel err ${relErr(out.hidden, hidden).toExponential(2)}`);
    log(`  logits rel err ${relErr(out.logits, logits).toExponential(2)}`);
    log(`  guided tokens agree ${agree}/${tok.length}, score rel err ${relErr(out.score, score).toExponential(2)}`);
    pipe.releaseItem(it);
  }

  if (only !== "llm") {
    pipe.decoder = await CodecDecoder.load(gpu, await SafeTensors.open(await blob(new URL(manifest.decoder.path, MODELS))));
    const codes = await bin("codes.i32", Int32Array);
    const Tc = meta.Tc;
    await pipe.decodeTokens(codes, Tc); // warm-up
    const t2 = performance.now();
    const audio = await pipe.decodeTokens(codes, Tc);
    const ms = performance.now() - t2;
    const ref = await bin("audio.f32", Float32Array);
    log(`codec decode (${(Tc / 25).toFixed(2)} s of audio): ${ms.toFixed(0)} ms, ${audio.length} vs ${ref.length} samples, rel err ${relErr(audio, ref).toExponential(2)}`);
  }
  await gpu.sync();
  log(`peak pooled activations ${(gpu.pool.peak / 2 ** 20).toFixed(0)} MiB`);
  log("done");
}

main().catch((e) => { log("ERROR " + (e.stack || e)); });
