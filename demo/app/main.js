import { fetchManifest, resolveFiles } from "./pipeline.js";
import { createEngine } from "./engine.js";
import { listCached, clearCache } from "./store.js";
import { estimateFrames } from "./text.js";
import { encodeWav } from "./audio.js";
import { saveVoiceClip, voiceClip, removeVoiceClip } from "./store.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
// Where the converted model files live. Files are cached by name in the browser, so a hosted copy
// should be pinned to one commit (…/resolve/<commit>/); ?models=<url> overrides it.
const MODELS_URL = "./models/";
const BASE = new URL(params.get("models") || MODELS_URL, location.href);
if (!BASE.pathname.endsWith("/")) BASE.pathname += "/";

// ------------------------------------------------------------------ choices (plain language)

const PRESETS = [
  { id: "bf16", label: "Best quality", desc: "The original weights." },
  { id: "int8", label: "Balanced", desc: "Sounds the same as Best in our checks.", tag: "Recommended" },
];
const SPEEDS = [
  { id: 0.85, label: "Slower" },
  { id: 1, label: "Normal" },
  { id: 1.15, label: "Faster" },
];
const QUALITY = [
  { id: 16, label: "Fast", tip: "16 decoding steps: about twice as fast" },
  { id: 32, label: "Best", tip: "32 decoding steps (OmniVoice's default)" },
];
const VOICE_MODES = [
  { id: "preset", label: "Presets", tip: "Ready-made synthetic voices" },
  { id: "mine", label: "Yours", tip: "Clone a voice from a recording" },
  { id: "design", label: "Describe", tip: "Describe a voice: gender, age, pitch" },
  { id: "auto", label: "Random", tip: "The model picks a voice (the seed decides which)" },
];
const EXAMPLES = [
  "नमस्ते! म तपाईंको आफ्नै कम्प्युटरमा चल्ने नेपाली आवाज हुँ।",
  "आज काठमाडौंमा बिहानदेखि नै हल्का पानी परिरहेको छ, त्यसैले छाता बोकेर निस्कनुहोला।",
  "एक समयको कुरा हो, हिमालको फेदीमा एउटा सानो गाउँ थियो, जहाँ एक जना बूढी हजुरआमा बस्नुहुन्थ्यो।",
  "कृपया ध्यान दिनुहोस्, काठमाडौंबाट पोखरा जाने बस दस मिनेटमा छुट्नेछ।",
  "पढाइ भनेको परीक्षा पास गर्नु मात्र होइन, जीवनलाई बुझ्ने बाटो पनि हो।",
];
const ICONS = {
  play: '<svg viewBox="0 0 24 24"><path d="M8 5.5v13l10.5-6.5z" class="fill"/></svg>',
  pause: '<svg viewBox="0 0 24 24"><rect x="6.5" y="5.5" width="4" height="13" rx="1" class="fill"/><rect x="13.5" y="5.5" width="4" height="13" rx="1" class="fill"/></svg>',
  dl: '<svg viewBox="0 0 24 24"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>',
  reuse: '<svg viewBox="0 0 24 24"><path d="M4 12a8 8 0 0 1 14-5.3L20 9M20 4v5h-5M20 12a8 8 0 0 1-14 5.3L4 15M4 20v-5h5"/></svg>',
  x: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>',
};

// ------------------------------------------------------------------ state

const store = {
  get(k, d) { try { const v = localStorage.getItem("nepali-voice." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("nepali-voice." + k, JSON.stringify(v)); } catch { /* private mode */ } },
};
const ui = {
  precision: "int8", model: "run2", voiceMode: "preset", voice: null,
  design: { gender: "female", age: "young adult", pitch: "moderate pitch" },
  speed: 1, steps: 32, guidance: 2,
  ...store.get("ui", {}),
};
let manifest = null;
let voices = [];
let myVoices = []; // cloned: { id, label, text, frames, tokens: number[], rms }
let engine = null;
let phase = "starting"; // starting | welcome | loading | ready | busy
let loadAbort = null;
const jobs = []; // waiting
let running = null; // { ...job, abort, frac }
const clips = []; // finished, newest first
let current = null; // clip in the player
const audioEl = new Audio();
let exampleIdx = 0;

const saveUi = () => store.set("ui", ui);
const fmtGB = (n) => (n >= 2 ** 30 ? `${(n / 2 ** 30).toFixed(1)} GB` : n >= 2 ** 20 ? `${Math.round(n / 2 ** 20)} MB` : n > 0 ? `${Math.max(1, Math.round(n / 1024))} KB` : "0 KB");
const fmtTime = (ms) => {
  const s = Math.max(1, Math.round(ms / 1000));
  return s >= 60 ? `${Math.floor(s / 60)} min ${s % 60 ? `${s % 60} s` : ""}`.trim() : `${s} s`;
};
const fmtClock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const newSeed = () => Math.floor(Math.random() * 2 ** 31);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pressed = (id) => $(id).getAttribute("aria-pressed") === "true";
const modelInfo = (id) => manifest.models.find((m) => m.id === id);
function rateMeter(windowMs = 20000) {
  const samples = [];
  return (done) => {
    const now = performance.now();
    samples.push([now, done]);
    while (samples.length > 2 && now - samples[0][0] > windowMs) samples.shift();
    const [t0, d0] = samples[0];
    return now - t0 > 1500 ? ((done - d0) / (now - t0)) * 1000 : 0;
  };
}

// ------------------------------------------------------------------ small UI helpers

function setStatus(text, kind = "idle") {
  $("statusText").textContent = text;
  $("statusDot").className = "dot " + ({ ok: "ok", busy: "busy", err: "err" }[kind] || "");
}

function showPanel(which) {
  for (const id of ["welcome", "loading", "empty", "player"]) $(id).hidden = id !== which;
}

function showError(msg) {
  const el = $("errorCard");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(showError.t);
  showError.t = setTimeout(() => { el.hidden = true; }, 9000);
}

function friendlyError(e) {
  const m = String(e?.message || e);
  if (/HTTP 40[13]/.test(m)) {
    return store.get("hfToken", "")
      ? "Hugging Face refused the download. Check your token in settings: it needs read access to the model repo."
      : "This model's files need a Hugging Face sign-in. Add a read token in settings, then try again.";
  }
  if (/device lost|out of memory|OOM|allocation/i.test(m)) return "Your graphics card ran out of memory. Try a shorter text, or the Balanced download in settings.";
  if (/HTTP|fetch|network|Failed to fetch/i.test(m)) return "The download was interrupted. Check your connection and try again.";
  if (/quota|storage|space/i.test(m)) return "Not enough storage space in this browser.";
  return "Something went wrong: " + m;
}

function radioGroup(el, items, isOn, onPick) {
  el.innerHTML = "";
  for (const it of items) {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(isOn(it)));
    if (it.tip) b.title = it.tip;
    b.textContent = it.label;
    b.onclick = () => onPick(it);
    el.append(b);
  }
}

// ------------------------------------------------------------------ controls

let cached = new Map(); // file name -> size, refreshed after downloads

async function refreshCached() {
  cached = new Map((await listCached()).filter((f) => !f.partial).map((f) => [f.name, f.size]));
}

const filesFor = (sel) => resolveFiles(BASE, manifest, { precision: sel.precision ?? ui.precision, model: sel.model ?? ui.model });
const missingBytes = (sel) => Object.values(filesFor(sel)).reduce((a, f) => a + (cached.get(f.name) === f.size ? 0 : f.size), 0);

function renderModels() {
  const el = $("models");
  el.innerHTML = "";
  for (const m of manifest.models) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "model";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(m.id === ui.model));
    const need = m.adapter && cached.get(m.adapter.cacheName) !== m.adapter.size ? `+${fmtGB(m.adapter.size)}` : "";
    b.innerHTML = `<span class="radio"></span><b>${esc(m.label)}</b><span class="tag">${need}</span><small>${esc(m.desc || "")}</small>`;
    b.onclick = () => { ui.model = m.id; saveUi(); renderControls(); };
    el.append(b);
  }
}

function voiceById(id) {
  return voices.find((v) => v.id === id) || voices[0];
}

let previewing = null;
function renderVoices() {
  radioGroup($("voiceMode"), VOICE_MODES.filter((m) => m.id !== "preset" || voices.length), (m) => m.id === ui.voiceMode, (m) => { ui.voiceMode = m.id; saveUi(); renderControls(); });
  const el = $("voices");
  el.hidden = ui.voiceMode !== "preset" && ui.voiceMode !== "mine";
  $("design").hidden = ui.voiceMode !== "design";
  el.innerHTML = "";
  if (ui.voiceMode === "mine") renderMyVoices(el);
  const sel = voiceById(ui.voice);
  for (const v of ui.voiceMode === "preset" ? voices : []) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "voice";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(v === sel));
    const [gender, age] = v.instruct.split(", ");
    b.innerHTML = `<b>${esc(v.label)}</b><small>${esc(gender)} · ${esc(age)}</small>`;
    b.onclick = () => { ui.voice = v.id; saveUi(); renderControls(); };
    if (v.preview) {
      const h = document.createElement("span");
      h.className = "hear";
      h.setAttribute("role", "button");
      h.title = `Hear ${v.label}`;
      h.innerHTML = previewing === v.id ? ICONS.pause : ICONS.play;
      h.onclick = (e) => {
        e.stopPropagation();
        if (previewing === v.id) { audioEl.pause(); return; }
        playUrl(new URL(v.preview, BASE).href, v.id);
      };
      b.append(h);
    }
    el.append(b);
  }
  $("dGender").value = ui.design.gender;
  $("dAge").value = ui.design.age;
  $("dPitch").value = ui.design.pitch;
  const tuned = !!modelInfo(ui.model)?.adapter;
  $("voiceHelp").textContent = ui.voiceMode === "preset"
    ? "Synthetic voices, designed with OmniVoice. Every model speaks in the chosen voice."
    : ui.voiceMode === "mine"
      ? (myVoices.length ? "Voices you cloned. They're kept in this browser." : "Clone a voice from a short recording: yours, or one you have permission to use.")
    : ui.voiceMode === "design"
      ? (tuned ? "Voice descriptions work best with the Base model; the fine-tuned models were trained without them." : "The voice follows your description; the seed picks one such voice.")
      : "The model picks a voice. The same seed gives the same voice.";
}

function renderMyVoices(el) {
  const sel = myVoices.find((v) => v.id === ui.myVoice) || myVoices[0];
  for (const v of myVoices) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "voice mine";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(v === sel));
    b.innerHTML = `<b>${esc(v.label)}</b><small>${(v.frames / 25).toFixed(1)} s reference</small>`;
    b.onclick = () => { ui.myVoice = v.id; saveUi(); renderControls(); };
    const h = document.createElement("span");
    h.className = "hear";
    h.setAttribute("role", "button");
    h.title = `Hear the recording of ${v.label}`;
    h.innerHTML = previewing === v.id ? ICONS.pause : ICONS.play;
    h.onclick = async (e) => {
      e.stopPropagation();
      if (previewing === v.id) { audioEl.pause(); return; }
      const f = await voiceClip(v.id);
      if (f) playUrl(URL.createObjectURL(f), v.id);
    };
    const x = document.createElement("span");
    x.className = "x";
    x.setAttribute("role", "button");
    x.title = `Delete ${v.label}`;
    x.innerHTML = ICONS.x;
    x.onclick = (e) => {
      e.stopPropagation();
      myVoices = myVoices.filter((o) => o !== v);
      store.set("myVoices", myVoices);
      removeVoiceClip(v.id);
      renderControls();
    };
    b.append(h, x);
    el.append(b);
  }
  const add = document.createElement("button");
  add.type = "button";
  add.className = "voice add";
  add.textContent = "+ New voice";
  add.onclick = openClone;
  el.append(add);
}

function renderControls() {
  renderModels();
  renderVoices();
  radioGroup($("speed"), SPEEDS, (s) => s.id === ui.speed, (s) => { ui.speed = s.id; saveUi(); renderControls(); });
  radioGroup($("quality"), QUALITY, (q) => q.id === ui.steps, (q) => { ui.steps = q.id; saveUi(); renderControls(); });
  renderChips();
  renderHint();
  renderGo();
}

function renderChips() {
  const el = $("advChips");
  el.innerHTML = "";
  const chips = [["steps", `<b>${ui.steps}</b> steps`], ["guidance", `guidance <b>${ui.guidance}</b>`]];
  for (const [key, html] of chips) {
    const b = document.createElement("button");
    b.type = "button";
    b.innerHTML = html;
    b.title = "Change";
    b.onclick = (e) => { e.stopPropagation(); openPopover(key, b); };
    el.append(b);
  }
}

function openPopover(key, anchor) {
  const pop = $("advPop");
  if (!pop.hidden && pop.dataset.key === key) { closePopover(); return; }
  pop.dataset.key = key;
  pop.innerHTML = "";
  const title = document.createElement("span");
  title.className = "label";
  const note = document.createElement("p");
  note.className = "help";
  const range = (min, max, step, value, onInput) => {
    const row = document.createElement("div");
    row.className = "row";
    const r = document.createElement("input");
    Object.assign(r, { type: "range", min, max, step, value });
    const o = document.createElement("output");
    o.textContent = value;
    r.oninput = () => { o.textContent = r.value; onInput(+r.value); };
    row.append(r, o);
    return row;
  };
  if (key === "steps") {
    title.textContent = "Decoding steps";
    note.textContent = "The model fills in the audio over this many passes. More is slower and usually a little cleaner. OmniVoice's default is 32.";
    pop.append(title, range(8, 64, 1, ui.steps, (v) => { ui.steps = v; saveUi(); renderControls(); }), note);
  } else {
    title.textContent = "Guidance";
    note.textContent = "How strongly the speech follows the text and voice. Higher is more careful but can sound strained. Default 2.";
    pop.append(title, range(0, 4, 0.1, ui.guidance, (v) => { ui.guidance = v; saveUi(); renderControls(); }), note);
  }
  pop.hidden = false;
  const r = anchor.getBoundingClientRect();
  const left = Math.min(Math.max(8, r.left), innerWidth - pop.offsetWidth - 8);
  let top = r.bottom + 6;
  if (top + pop.offsetHeight > innerHeight - 8) top = r.top - pop.offsetHeight - 6;
  pop.style.left = `${left}px`;
  pop.style.top = `${Math.max(8, top)}px`;
}

function closePopover() {
  $("advPop").hidden = true;
  $("advPop").dataset.key = "";
}

function currentVoice() {
  if (ui.voiceMode === "preset" && voices.length) {
    const v = voiceById(ui.voice);
    return { kind: "preset", label: v.label, tokens: v.tokens, frames: v.frames, text: v.text, rms: v.rms };
  }
  if (ui.voiceMode === "mine") {
    const v = myVoices.find((o) => o.id === ui.myVoice) || myVoices[0];
    if (v) return { kind: "clone", label: v.label, tokens: v.tokens, frames: v.frames, text: v.text, rms: v.rms };
  }
  if (ui.voiceMode === "design") {
    const { gender, age, pitch } = ui.design;
    return { kind: "design", label: `${gender}, ${age}, ${pitch.replace(" pitch", "")} pitch`, instruct: `${gender}, ${age}, ${pitch}` };
  }
  return { kind: "auto", label: "surprise voice" };
}

// seconds of speech the model will aim for
function speechSeconds(text = $("text").value.trim(), voice = currentVoice()) {
  if (!text) return 0;
  return estimateFrames(text, voice.tokens ? voice.text : null, voice.frames, ui.speed) / 25;
}

function renderHint() {
  const text = $("text").value.trim();
  const n = [...text].length;
  const s = speechSeconds(text);
  $("textHint").textContent = n ? `${n} characters · about ${Math.max(1, Math.round(s))} s of speech${s > 30 ? " · read in parts" : ""}` : "";
}

// generation time measured on this device: ms per decoding step per sequence row, ms per audio second
function estimateMs(seconds = speechSeconds(), steps = ui.steps) {
  const sp = store.get("speed", null);
  if (!sp || !seconds) return null;
  const T = seconds * 25;
  const rows = 2.6 * T + 60; // cond + uncond target, text, reference
  return steps * rows * sp.stepRow + seconds * sp.decodeSec;
}

function renderGo() {
  const busy = phase === "busy";
  $("goLabel").textContent = busy ? "Add to queue" : "Speak";
  const est = phase === "ready" || busy ? estimateMs() : null;
  $("estimate").textContent = est ? `about ${fmtTime(est)}` : "";
  $("stopBtn").hidden = !busy;
  const ok = (phase === "ready" || busy) && $("text").value.trim().length > 0;
  $("goBtn").disabled = !ok;
  $("compareBtn").disabled = !ok || manifest?.models.length < 2;
  if (busy) setStatus(jobs.length ? `Speaking… (${jobs.length} queued)` : "Speaking…", "busy");
}

function setRandom(on) {
  $("randomBtn").setAttribute("aria-pressed", String(on));
  $("randomBtn").title = on ? "A new random seed for every clip (click to keep the seed)" : "Keeping this seed (click for a new random seed every clip)";
  store.set("randomSeed", on);
}

// ------------------------------------------------------------------ download size presets

function presetBytes(p) {
  return Object.values(filesFor({ precision: p.id })).reduce((a, f) => a + f.size, 0);
}

function isSaved(p) {
  return Object.values(filesFor({ precision: p.id })).every((f) => cached.get(f.name) === f.size);
}

function renderPresets(el, onPick) {
  el.innerHTML = "";
  for (const p of PRESETS) {
    if (!manifest.llm[p.id]) continue;
    const bytes = presetBytes(p);
    const b = document.createElement("button");
    b.type = "button";
    b.className = "preset";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(ui.precision === p.id));
    const vram = manifest.llm[p.id].size + 0.5 * 2 ** 30;
    b.innerHTML = `${p.tag ? `<span class="tag">${p.tag}</span>` : ""}<b>${esc(p.label)}</b><span class="size">${fmtGB(bytes)} download</span><span class="desc">${esc(p.desc)} Needs about ${fmtGB(vram)} of graphics memory.</span>${isSaved(p) ? '<span class="saved">✓ Downloaded</span>' : ""}`;
    b.onclick = () => onPick(p);
    el.append(b);
  }
}

function chipText() {
  const p = PRESETS.find((x) => x.id === ui.precision);
  $("gpuChip").textContent = p ? p.label : ui.precision;
  $("gpuChip").hidden = false;
}

// ------------------------------------------------------------------ loading

async function start() {
  await refreshCached();
  renderControls();
  const sel = { precision: ui.precision, model: ui.model };
  if (missingBytes(sel) === 0) return loadModel(sel);
  showWelcome();
}

function showWelcome() {
  phase = "welcome";
  showPanel("welcome");
  setStatus("Choose a download to begin");
  const pick = (p) => { ui.precision = p.id; saveUi(); renderPresets($("welcomePresets"), pick); renderNote(); };
  const renderNote = () => {
    const need = missingBytes({});
    $("welcomeNote").textContent = need ? `Downloads ${fmtGB(need)} once, then starts from this browser's storage.` : "Already downloaded.";
  };
  renderPresets($("welcomePresets"), pick);
  renderNote();
  $("welcomeToken").value = store.get("hfToken", "");
  $("welcomeGo").onclick = () => {
    store.set("hfToken", $("welcomeToken").value.trim());
    loadModel({ precision: ui.precision, model: ui.model });
  };
}

// Loads (downloading first if needed) the selection. Used at start, from settings and before a job
// on another model. Throws when cancelled or failed (the caller decides what to show).
async function ensureLoaded(sel, { quiet = false } = {}) {
  if (engine.isLoaded(BASE, manifest, sel)) return;
  const meter = rateMeter();
  const label = modelInfo(sel.model).label;
  loadAbort = new AbortController();
  if (!quiet) {
    showPanel("loading");
    $("loadingTitle").textContent = "Getting ready…";
    $("loadingBar").style.width = "0%";
    $("loadingText").textContent = "";
  }
  const say = (title, text, frac) => {
    if (quiet) {
      $("progress").hidden = false;
      $("progressText").textContent = `${title}${text ? " · " + text : ""}`;
      $("progressBar").style.width = `${frac * 100}%`;
    } else {
      $("loadingTitle").textContent = title;
      $("loadingText").textContent = text;
      $("loadingBar").style.width = `${frac * 100}%`;
    }
  };
  try {
    await engine.load(BASE, manifest, sel, {
      signal: loadAbort.signal,
      token: store.get("hfToken", ""),
      onStatus: (s) => {
        if (s.phase === "download") {
          $("loadingCancel").hidden = quiet;
          const rate = meter(s.done);
          say("Downloading the model…", `${fmtGB(s.done)} of ${fmtGB(s.total)}${rate > 0 ? ` · ${fmtGB(rate)}/s · ${fmtTime(((s.total - s.done) / rate) * 1000)} left` : ""}`, s.done / s.total);
          setStatus(`Downloading ${Math.round((100 * s.done) / s.total)}%`, "busy");
        } else if (s.phase === "load") {
          $("loadingCancel").hidden = true;
          say(`Loading ${s.what}…`, "Preparing the graphics card", s.frac);
          setStatus(`Loading ${label}…`, "busy");
        }
      },
    });
  } finally {
    loadAbort = null;
    $("loadingCancel").hidden = true;
    await refreshCached();
    renderModels();
  }
}

async function loadModel(sel) {
  phase = "loading";
  renderGo();
  try {
    await ensureLoaded(sel);
    engine.loadedModel = sel.model;
    toReady();
  } catch (e) {
    console.error(e);
    if (e.name === "AbortError") { showWelcome(); return; }
    showWelcome();
    $("welcomeNote").textContent = friendlyError(e);
    setStatus("Couldn't load the model", "err");
  }
}

function toReady() {
  phase = "ready";
  chipText();
  showPanel(current ? "player" : "empty");
  setStatus(`Ready · ${modelInfo(engine.loadedModel || ui.model).label}${engine.gpu?.name ? " · " + engine.gpu.name : ""}`, "ok");
  renderGo();
}

// ------------------------------------------------------------------ queue

function makeJob(model, seed) {
  const voice = currentVoice();
  return {
    id: Math.random().toString(36).slice(2),
    text: $("text").value.trim(), model, voice, seed,
    steps: ui.steps, guidance: ui.guidance, speed: ui.speed,
  };
}

function nextSeed() {
  if (pressed("randomBtn")) {
    const s = newSeed();
    $("seed").value = s;
    return s;
  }
  return Math.max(0, parseInt($("seed").value, 10) || 0);
}

function enqueue(list) {
  jobs.push(...list);
  renderClips();
  renderGo();
  if (!running) runQueue();
}

async function runQueue() {
  phase = "busy";
  renderGo();
  while (jobs.length) {
    const job = jobs.shift();
    running = { ...job, abort: new AbortController(), frac: 0 };
    renderClips();
    const sel = { precision: ui.precision, model: job.model };
    try {
      if (!engine.isLoaded(BASE, manifest, sel)) {
        await ensureLoaded(sel, { quiet: true });
        engine.loadedModel = job.model;
      }
      if (running.abort.signal.aborted) throw new DOMException("cancelled", "AbortError");
      $("progress").hidden = false;
      setProgress(0, "Starting…");
      const res = await engine.generate({
        text: job.text,
        voice: job.voice.tokens ? { tokens: Int32Array.from(job.voice.tokens), frames: job.voice.frames, text: job.voice.text, rms: job.voice.rms } : { instruct: job.voice.instruct },
        steps: job.steps, guidance: job.guidance, speed: job.speed, seed: job.seed,
        signal: running.abort.signal,
        onProgress: (p) => {
          if (p.phase === "sample") {
            running.frac = 0.95 * p.frac;
            setProgress(running.frac, `${p.chunks > 1 ? `Part ${p.chunk + 1} of ${p.chunks} · ` : ""}step ${p.step} of ${p.steps}`);
          } else setProgress(0.97, "Turning tokens into sound…");
          updateRunningClip();
        },
      });
      recordSpeed(res.timings, job);
      const clip = { ...job, audio: res.audio, sampleRate: res.sampleRate, timings: res.timings };
      clip.url = URL.createObjectURL(encodeWav(res.audio, res.sampleRate));
      clips.unshift(clip);
      if (!current || running.follow !== false) select(clip, true);
    } catch (e) {
      if (e.name !== "AbortError") {
        console.error(e);
        showError(friendlyError(e));
      }
    } finally {
      running = null;
      $("progress").hidden = true;
      renderClips();
    }
  }
  phase = "ready";
  toReady();
}

function setProgress(frac, text) {
  $("progressBar").style.width = `${frac * 100}%`;
  $("progressText").textContent = text;
}

function recordSpeed(t, job) {
  const T = t.seconds * 25;
  const rows = 2.6 * T + 60;
  const sp = { stepRow: t.perStep / rows, decodeSec: t.decode / Math.max(1, t.seconds) };
  const old = store.get("speed", null);
  store.set("speed", old ? { stepRow: (old.stepRow + sp.stepRow) / 2, decodeSec: (old.decodeSec + sp.decodeSec) / 2 } : sp);
  renderGo();
}

function stopAll() {
  jobs.length = 0;
  loadAbort?.abort();
  running?.abort.abort();
  renderClips();
}

// ------------------------------------------------------------------ player

function playUrl(url, previewId = null) {
  audioEl.src = url;
  audioEl.play().catch(() => {});
  previewing = previewId;
  renderVoices();
}

audioEl.onplay = audioEl.onpause = audioEl.onended = () => {
  if (audioEl.paused && previewing) { previewing = null; renderVoices(); }
  $("playIcon").innerHTML = !audioEl.paused && current && audioEl.src === current.url ? ICONS.pause.replace(/^<svg[^>]*>|<\/svg>$/g, "") : ICONS.play.replace(/^<svg[^>]*>|<\/svg>$/g, "");
  renderClips();
};

function select(clip, autoplay = false) {
  current = clip;
  showPanel("player");
  $("said").textContent = clip.text;
  const m = modelInfo(clip.model);
  $("info").innerHTML = `<span class="meta"><b>${esc(m.label)}</b><span class="sep">·</span>${esc(clip.voice.label)}<span class="sep">·</span>${clip.timings.seconds.toFixed(1)} s of speech<span class="sep">·</span>made in ${fmtTime(clip.timings.total)}<span class="sep">·</span>seed ${clip.seed}</span>
    <span class="actions"><button type="button" id="reuseBtn" title="Put this clip's text, voice and seed back into the settings">${ICONS.reuse}<span>Reuse</span></button><a href="${clip.url}" download="nepali-voice-${clip.model}-${clip.seed}.wav">${ICONS.dl}<span>Download</span></a></span>`;
  $("reuseBtn").onclick = () => reuse(clip);
  drawWave();
  if (autoplay) { previewing = null; audioEl.src = clip.url; audioEl.play().catch(() => {}); }
  else if (audioEl.src !== clip.url) { audioEl.pause(); audioEl.src = clip.url; }
  renderClips();
}

function reuse(clip) {
  $("text").value = clip.text;
  store.set("text", clip.text);
  ui.model = clip.model;
  if (clip.voice.kind === "preset") { ui.voiceMode = "preset"; ui.voice = voices.find((v) => v.label === clip.voice.label)?.id || ui.voice; }
  else if (clip.voice.kind === "clone") { ui.voiceMode = "mine"; ui.myVoice = myVoices.find((v) => v.label === clip.voice.label)?.id || ui.myVoice; }
  else ui.voiceMode = clip.voice.kind;
  ui.steps = clip.steps; ui.guidance = clip.guidance; ui.speed = clip.speed;
  $("seed").value = clip.seed;
  setRandom(false);
  saveUi();
  renderControls();
}

// min/max envelope per pixel column, played part in the accent colour
function drawWave() {
  const cv = $("wave");
  if (!current || cv.offsetParent === null) return;
  const dpr = devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const x = current.audio;
  let peak = 1e-6;
  for (let i = 0; i < x.length; i += 7) peak = Math.max(peak, Math.abs(x[i]));
  const cs = getComputedStyle(document.documentElement);
  const played = audioEl.src === current.url ? audioEl.currentTime / (x.length / current.sampleRate) : 0;
  const bar = 3, gap = 2;
  const n = Math.floor(w / (bar + gap));
  for (let c = 0; c < n; c++) {
    const a = Math.floor((c * x.length) / n), b = Math.floor(((c + 1) * x.length) / n);
    let s = 0;
    for (let i = a; i < b; i += 4) s = Math.max(s, Math.abs(x[i]));
    const bh = Math.max(2, (s / peak) * (h - 8));
    g.fillStyle = c / n < played ? cs.getPropertyValue("--accent").trim() : cs.getPropertyValue("--line-2").trim();
    g.beginPath();
    g.roundRect(c * (bar + gap), (h - bh) / 2, bar, bh, 1.5);
    g.fill();
  }
  const dur = x.length / current.sampleRate;
  $("time").textContent = `${fmtClock(audioEl.src === current.url ? audioEl.currentTime : 0)} / ${fmtClock(dur)}`;
}

(function tick() {
  if (current && !audioEl.paused) drawWave();
  requestAnimationFrame(tick);
})();

// ------------------------------------------------------------------ session list

function ring(frac) {
  const c = 2 * Math.PI * 12;
  return `<svg class="ring" viewBox="0 0 30 30"><circle class="bg" cx="15" cy="15" r="12"/><circle class="fg" cx="15" cy="15" r="12" stroke-dasharray="${c}" stroke-dashoffset="${c * (1 - frac)}"/></svg>`;
}

function clipRow(item, kind) {
  const li = document.createElement("li");
  const m = modelInfo(item.model);
  li.className = "clip" + (kind !== "done" ? " pending" : "");
  const sub = kind === "done" ? `${item.voice.label} · ${item.timings.seconds.toFixed(1)} s` : kind === "running" ? `${item.voice.label} · speaking…` : `${item.voice.label} · waiting`;
  const lead = kind === "done"
    ? `<button class="mini" type="button" aria-label="Play">${!audioEl.paused && audioEl.src === item.url ? ICONS.pause : ICONS.play}</button>`
    : ring(kind === "running" ? item.frac : 0);
  li.innerHTML = `${lead}<span class="txt"><span lang="ne">${esc(item.text)}</span><small>${esc(sub)}</small></span><span class="badge ${m.adapter ? "tuned" : ""}">${esc(m.label)}</span>`;
  if (kind === "done") {
    li.setAttribute("aria-current", String(item === current));
    li.onclick = () => select(item);
    li.querySelector(".mini").onclick = (e) => {
      e.stopPropagation();
      if (!audioEl.paused && audioEl.src === item.url) audioEl.pause();
      else select(item, true);
    };
  } else if (kind === "queued") {
    const x = document.createElement("button");
    x.className = "x";
    x.type = "button";
    x.title = "Remove from the queue";
    x.innerHTML = ICONS.x;
    x.onclick = (e) => { e.stopPropagation(); jobs.splice(jobs.indexOf(item), 1); renderClips(); renderGo(); };
    li.querySelector(".badge").after(x);
    li.style.gridTemplateColumns = "30px minmax(0, 1fr) auto auto";
  }
  if (kind === "running") li.id = "runningClip";
  return li;
}

function renderClips() {
  const el = $("clips");
  el.innerHTML = "";
  for (const j of [...jobs].reverse()) el.append(clipRow(j, "queued"));
  if (running) el.append(clipRow(running, "running"));
  for (const c of clips) el.append(clipRow(c, "done"));
  $("session").hidden = !el.children.length;
}

function updateRunningClip() {
  const r = $("runningClip")?.querySelector(".ring .fg");
  if (r) r.style.strokeDashoffset = String(2 * Math.PI * 12 * (1 - running.frac));
}

// ------------------------------------------------------------------ settings

async function openSettings() {
  await refreshCached();
  const pick = async (p) => {
    if (p.id === ui.precision) return;
    $("settings").close();
    const prev = ui.precision;
    ui.precision = p.id;
    saveUi();
    if (phase === "ready") {
      phase = "loading";
      try {
        await ensureLoaded({ precision: ui.precision, model: ui.model });
        engine.loadedModel = ui.model;
        toReady();
      } catch (e) {
        ui.precision = prev;
        saveUi();
        if (e.name !== "AbortError") showError(friendlyError(e));
        toReady();
      }
    } else if (phase === "welcome") showWelcome();
  };
  renderPresets($("presets"), pick);
  $("hfToken").value = store.get("hfToken", "");
  const total = [...cached.values()].reduce((a, b) => a + b, 0);
  $("storageText").textContent = total ? `${fmtGB(total)} of model files.` : "Nothing yet.";
  $("settings").showModal();
}

// ------------------------------------------------------------------ voice cloning

const clone = { wav: null, rms: 0, url: null, rec: null, busy: false };

function cloneStatus(text, kind = "") {
  $("cloneStatus").textContent = text;
  $("cloneStatus").style.color = kind === "err" ? "var(--bad)" : kind === "ok" ? "var(--good)" : "";
}

function cloneProgress(frac) {
  $("cloneBar").hidden = frac == null;
  if (frac != null) $("cloneBar").firstElementChild.style.width = `${frac * 100}%`;
}

function cloneRender() {
  const has = !!clone.wav;
  $("cloneClip").hidden = !has;
  $("cloneTextField").hidden = !has;
  $("cloneNameField").hidden = !has;
  const idle = !clone.busy && (phase === "ready" || phase === "welcome");
  $("cloneSave").disabled = !has || !idle || !$("cloneText").value.trim() || !$("cloneName").value.trim();
  $("transcribeBtn").disabled = !has || !idle;
  $("recBtn").disabled = clone.busy;
  const asr = manifest.asr?.int8;
  $("transcribeBtn").hidden = !asr;
  if (asr) $("transcribeBtn").textContent = cached.get(asr.path) !== asr.size ? `Transcribe automatically (${fmtGB(asr.size)} download)` : "Transcribe automatically";
}

function openClone() {
  if (phase === "busy" || phase === "loading") { showError("Wait for the current clip to finish first."); return; }
  Object.assign(clone, { wav: null, busy: false });
  $("cloneText").value = "";
  $("cloneName").value = `Voice ${myVoices.length + 1}`;
  $("cloneTextHelp").textContent = "";
  $("recLabel").textContent = "Record";
  cloneStatus("");
  cloneProgress(null);
  cloneRender();
  $("cloneDlg").showModal();
}

// Downloads (first time) and loads a cloning model, with progress shown in the dialog.
async function cloneStep(part, label) {
  await engine.loadCloning(BASE, manifest, part, {
    token: store.get("hfToken", ""),
    onStatus: (s) => {
      if (s.phase === "download") { cloneStatus(`Downloading the ${label}… ${fmtGB(s.done)} of ${fmtGB(s.total)}`); cloneProgress(s.done / s.total); }
      else if (s.phase === "load") { cloneStatus(`Loading the ${label}…`); cloneProgress(s.frac); }
    },
  });
  await refreshCached();
  cloneProgress(null);
}

async function useRecording(blob) {
  clone.busy = true;
  cloneRender();
  cloneStatus("Reading the recording…");
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
    ctx.close();
    const mono = new Float32Array(buf.length);
    for (let c = 0; c < buf.numberOfChannels; c++) {
      const d = buf.getChannelData(c);
      for (let i = 0; i < d.length; i++) mono[i] += d[i] / buf.numberOfChannels;
    }
    const r = await engine.prepareReference(mono, buf.sampleRate);
    Object.assign(clone, { wav: r.wav, rms: r.rms });
    if (clone.url) URL.revokeObjectURL(clone.url);
    clone.url = URL.createObjectURL(encodeWav(r.wav, 24000));
    $("cloneAudio").src = clone.url;
    const long = buf.duration > 20 ? ` (cut from ${Math.round(buf.duration)} s at a pause)` : "";
    $("cloneInfo").textContent = `${r.seconds.toFixed(1)} s of speech after trimming silences${long}.${r.seconds < 3 ? " That's short: 3–10 s works best." : ""}`;
    cloneStatus("");
    clone.busy = false;
    cloneRender();
    const asr = manifest.asr?.int8;
    if (asr && (cached.get(asr.path) === asr.size || store.get("autoTranscribe", false))) await transcribeClone();
    else $("cloneTextHelp").textContent = "Type what is said, or let the browser transcribe it.";
  } catch (e) {
    console.error(e);
    clone.busy = false;
    cloneRender();
    cloneStatus(e.name === "EncodingError" || /decod/i.test(e.message || "") ? "Couldn't read that file. Try a WAV, MP3, M4A or OGG recording." : friendlyError(e), "err");
  }
}

async function transcribeClone() {
  if (!clone.wav || clone.busy) return;
  clone.busy = true;
  cloneRender();
  try {
    await cloneStep("asr", "speech recognizer");
    store.set("autoTranscribe", true);
    cloneStatus("Transcribing…");
    $("cloneText").value = await engine.transcribe(clone.wav);
    $("cloneTextHelp").textContent = "Check the transcript and fix any mistakes: the closer it is to what's said, the better the clone.";
    cloneStatus("");
  } catch (e) {
    console.error(e);
    cloneStatus(friendlyError(e), "err");
  } finally {
    clone.busy = false;
    cloneProgress(null);
    cloneRender();
  }
}

async function saveClone() {
  clone.busy = true;
  cloneRender();
  try {
    await cloneStep("encoder", "voice encoder");
    cloneStatus("Learning the voice…");
    const { tokens, frames } = await engine.encodeReference(clone.wav);
    const v = {
      id: `v${Date.now().toString(36)}`, label: $("cloneName").value.trim(), text: $("cloneText").value.trim(),
      frames, tokens: Array.from(tokens), rms: clone.rms,
    };
    await saveVoiceClip(v.id, encodeWav(clone.wav, 24000)).catch(() => {});
    myVoices.push(v);
    store.set("myVoices", myVoices);
    ui.voiceMode = "mine";
    ui.myVoice = v.id;
    saveUi();
    $("cloneDlg").close();
    renderControls();
  } catch (e) {
    console.error(e);
    cloneStatus(friendlyError(e), "err");
  } finally {
    clone.busy = false;
    cloneProgress(null);
    cloneRender();
  }
}

async function toggleRecording() {
  if (clone.rec) { clone.rec.stop(); return; }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: true, autoGainControl: true } });
  } catch {
    cloneStatus("The browser couldn't use a microphone. Allow microphone access, or choose a file instead.", "err");
    return;
  }
  const chunks = [];
  const rec = new MediaRecorder(stream);
  clone.rec = rec;
  const t0 = Date.now();
  const timer = setInterval(() => {
    const s = (Date.now() - t0) / 1000;
    $("recLabel").textContent = `Stop (${s.toFixed(0)} s)`;
    if (s >= 20) rec.stop();
  }, 250);
  rec.ondataavailable = (e) => chunks.push(e.data);
  rec.onstop = () => {
    clearInterval(timer);
    stream.getTracks().forEach((t) => t.stop());
    clone.rec = null;
    $("recBtn").classList.remove("on");
    $("recLabel").textContent = "Record again";
    useRecording(new Blob(chunks, { type: rec.mimeType }));
  };
  rec.start();
  $("recBtn").classList.add("on");
  $("recLabel").textContent = "Stop";
  cloneStatus("Recording… speak naturally for 5–10 seconds.");
}

function setupClone() {
  $("recBtn").onclick = toggleRecording;
  $("cloneFile").onchange = () => { const f = $("cloneFile").files[0]; if (f) useRecording(f); $("cloneFile").value = ""; };
  const drop = $("cloneDrop");
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const f = e.dataTransfer.files[0];
    if (f) useRecording(f);
  };
  $("transcribeBtn").onclick = transcribeClone;
  $("cloneSave").onclick = saveClone;
  $("cloneText").oninput = cloneRender;
  $("cloneName").oninput = cloneRender;
  $("cloneDlg").onclose = () => { clone.rec?.stop(); };
}

// ------------------------------------------------------------------ boot

async function main() {
  if (!navigator.gpu) {
    $("fatal").hidden = false;
    $("fatal").textContent = "This page needs WebGPU, which this browser doesn't offer.\nTry a recent Chrome or Edge on a computer with a graphics card.";
    return;
  }
  try {
    manifest = await fetchManifest(BASE);
  } catch (e) {
    $("fatal").hidden = false;
    $("fatal").textContent = "Couldn't reach the model files.\n" + e.message;
    return;
  }
  if (!manifest.llm[ui.precision]) ui.precision = manifest.llm.int8 ? "int8" : Object.keys(manifest.llm)[0];
  if (!modelInfo(ui.model)) ui.model = manifest.models.at(-1).id;
  try {
    const res = await fetch(new URL("voices.json", BASE));
    if (res.ok) voices = (await res.json()).voices;
  } catch { /* no presets in this build */ }
  if (!voices.length && ui.voiceMode === "preset") ui.voiceMode = "auto";
  myVoices = store.get("myVoices", []);
  if (!manifest.encoder) VOICE_MODES.splice(VOICE_MODES.findIndex((m) => m.id === "mine"), 1);
  if (voices.length && !voices.some((v) => v.id === ui.voice)) ui.voice = voices[0].id;
  if (manifest.loraRepo) $("loraLink").href = manifest.loraRepo;

  $("text").value = store.get("text", EXAMPLES[0]);
  $("text").oninput = () => { store.set("text", $("text").value); renderHint(); renderGo(); };
  $("exampleBtn").onclick = () => {
    exampleIdx = (exampleIdx + 1) % EXAMPLES.length;
    $("text").value = EXAMPLES[exampleIdx];
    store.set("text", $("text").value);
    renderHint();
    renderGo();
  };
  for (const [id, key] of [["dGender", "gender"], ["dAge", "age"], ["dPitch", "pitch"]]) {
    $(id).onchange = () => { ui.design[key] = $(id).value; saveUi(); renderVoices(); };
  }
  $("seed").value = store.get("seed", newSeed());
  $("seed").onchange = () => { store.set("seed", $("seed").value); setRandom(false); };
  setRandom(store.get("randomSeed", true));
  $("randomBtn").onclick = () => setRandom(!pressed("randomBtn"));
  $("rollBtn").onclick = () => { $("seed").value = newSeed(); store.set("seed", $("seed").value); setRandom(false); };
  $("goBtn").onclick = () => enqueue([makeJob(ui.model, nextSeed())]);
  $("compareBtn").onclick = () => {
    const seed = nextSeed();
    enqueue(manifest.models.map((m) => makeJob(m.id, seed)));
  };
  $("stopBtn").onclick = stopAll;
  $("loadingCancel").onclick = () => loadAbort?.abort();
  $("settingsBtn").onclick = openSettings;
  $("gpuChip").onclick = openSettings;
  $("keyToggle").onclick = () => {
    const i = $("hfToken");
    i.type = i.type === "password" ? "text" : "password";
    $("keyToggle").textContent = i.type === "password" ? "Show" : "Hide";
  };
  $("hfToken").onchange = () => store.set("hfToken", $("hfToken").value.trim());
  $("clearBtn").onclick = async () => {
    if (phase === "busy" || phase === "loading") { showError("Wait for the current job to finish first."); return; }
    $("settings").close();
    await engine.unload();
    await clearCache();
    await refreshCached();
    renderModels();
    showWelcome();
  };
  $("playBtn").onclick = () => {
    if (!current) return;
    if (audioEl.src !== current.url) { audioEl.src = current.url; }
    if (audioEl.paused) audioEl.play().catch(() => {}); else audioEl.pause();
  };
  $("wave").onclick = (e) => {
    if (!current) return;
    const r = $("wave").getBoundingClientRect();
    if (audioEl.src !== current.url) audioEl.src = current.url;
    audioEl.currentTime = ((e.clientX - r.left) / r.width) * (current.audio.length / current.sampleRate);
    drawWave();
  };
  addEventListener("resize", drawWave);
  document.addEventListener("click", (e) => { if (!$("advPop").hidden && !$("advPop").contains(e.target)) closePopover(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closePopover();
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && !$("goBtn").disabled) $("goBtn").click();
  });

  setupClone();
  engine = await createEngine({ inPage: params.get("engine") === "page" });
  await start();
}

main().catch((e) => {
  console.error(e);
  $("fatal").hidden = false;
  $("fatal").textContent = "Something went wrong while starting.\n" + (e.message || e);
});
