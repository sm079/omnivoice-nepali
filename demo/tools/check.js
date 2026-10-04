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
import { CodecEncoder } from "../app/nn/encoder.js";
import { resample } from "../app/resample.js";
import * as ops from "../app/gpu/ops.js";
import { NepaliASR, logMel } from "../app/nn/asr.js";

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

// ?asr=../out/asr: the browser ASR against tools/reference_asr.py
async function checkAsr(dir) {
  const D = new URL(dir + "/", location.href);
  const get = async (n) => new Float32Array(await (await fetch(new URL(n, D))).arrayBuffer());
  const meta = await (await fetch(new URL("meta.json", D))).json();
  const manifest = await fetchJson("manifest.json");
  const prec = params.get("precision") || "int8";
  const cfg = await fetchJson(manifest.asr.config);
  const gpu = await GPU.create();
  const wav = await get("wav16.f32");
  const t0 = performance.now();
  const { feats, T } = logMel(wav, cfg);
  log(`mel: ${T} vs ${meta.T} frames in ${(performance.now() - t0).toFixed(0)} ms, rel err ${relErr(feats, await get("mel.f32")).toExponential(2)}`);
  const asr = await NepaliASR.load(gpu, await SafeTensors.open(await blob(new URL(manifest.asr[prec].path, MODELS))), cfg);
  await asr.transcribe(wav); // warm-up
  const t1 = performance.now();
  const r = await asr.transcribe(wav);
  const ms = performance.now() - t1;
  asr.stages = {};
  const { x, L } = asr.encode(feats, T, 80);
  const stage = async (k, f, scale = 1) => { const a = await gpu.read(asr.stages[k]); if (scale !== 1) for (let i = 0; i < a.length; i++) a[i] *= scale; return relErr(a, await get(f)).toExponential(2); };
  log(`  pre ${await stage("pre", "mid_pre_encode_out_Add_output_0.f32", 1 / 32)}, ff1 ${await stage("ff1", "mid_layers0_Add_output_0.f32")}, att ${await stage("att", "mid_layers0_Add_1_output_0.f32")}, conv ${await stage("conv", "mid_layers0_Add_2_output_0.f32")}, out ${await stage("out", "mid_layers0_norm_out_Add_1_output_0.f32")}`);
  log(`asr ${prec}: ${(wav.length / 16000).toFixed(2)} s in ${ms.toFixed(0)} ms, ${L} vs ${meta.L} frames`);
  log(`  encoder rel err ${relErr(await gpu.read(x), await get("enc.f32")).toExponential(2)}, ctc logits rel err ${relErr(r.logits, await get("ctc.f32")).toExponential(2)}`);
  log(`  js: ${r.text}
  py: ${meta.text}
  ${r.text === meta.text ? "same text" : "TEXT DIFFERS"}`);
  log("done");
}

async function main() {
  if (params.get("asr")) return checkAsr(params.get("asr"));
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

  if (!only || only === "llm") {
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

  if (!only || only === "codec") {
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
  if ((!only || only === "encoder") && meta.enc_T) {
    const enc = await CodecEncoder.load(gpu, await SafeTensors.open(await blob(new URL(manifest.encoder.path, MODELS))));
    const wav = await bin("enc_wav.f32", Float32Array);
    await enc.encode(wav); // warm-up
    const t3 = performance.now();
    const { codes, T } = await enc.encode(wav);
    const ms = performance.now() - t3;
    const ref = await bin("enc_codes.i32", Int32Array);
    // the pre-quantization embedding, for a numeric comparison
    enc.keepFeatures = true;
    const sem = enc.semantic(resample(wav, 24000, 16000));
    const sum = await gpu.read(enc.features);
    const feats = new Float32Array(sem.T * 768);
    for (let t = 0; t < sem.T; t++) for (let d = 0; d < 768; d++) feats[t * 768 + d] = sum[2 * t * 768 + d] / 13;
    for (const k of ["conv", "res0", "res1"]) log(`  semantic ${k} rel err ${relErr(await gpu.read(enc.stages[k]), await bin(`enc_s_${k}.f32`, Float32Array)).toExponential(2)}`);
    log(`  hubert features rel err ${relErr(feats, await bin("enc_hubert.f32", Float32Array)).toExponential(2)} (${sum.length / 768} frames)`);
    const ac = enc.acoustic(wav, sem.T);
    const cat = gpu.empty([T, 1024]);
    ops.copyCols(gpu, ac.t, cat, { rows: T, cols: 256, dstLd: 1024 });
    ops.copyCols(gpu, sem.t, cat, { rows: T, cols: 768, dstLd: 1024, dstOff: 256 });
    const emb = await gpu.read(ops.linear(gpu, cat, enc.fc), T * 1024);
    log(`  semantic rel err ${relErr(await gpu.read(sem.t, T * 768), await bin("enc_sem.f32", Float32Array)).toExponential(2)}, ` +
      `acoustic rel err ${relErr(await gpu.read(ac.t, T * 256), await bin("enc_ac.f32", Float32Array)).toExponential(2)}`);
    let agree = [];
    for (let c = 0; c < 8; c++) {
      let n = 0;
      for (let t = 0; t < T; t++) n += codes[c * T + t] === ref[c * meta.enc_T + t];
      agree.push(n);
    }
    log(`codec encode (${(wav.length / 24000).toFixed(2)} s): ${ms.toFixed(0)} ms, ${T} vs ${meta.enc_T} frames, ` +
      `embedding rel err ${relErr(emb, await bin("enc_emb.f32", Float32Array)).toExponential(2)}, codes agree per codebook ${agree.join("/")} of ${T}`);
  }
  await gpu.sync();
  log(`peak pooled activations ${(gpu.pool.peak / 2 ** 20).toFixed(0)} MiB`);
  log("done");
}

main().catch((e) => { log("ERROR " + (e.stack || e)); });
