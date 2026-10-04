"use strict";

/* ============================================================
   helpers
   ============================================================ */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "style") n.style.cssText = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) n.append(kid.nodeType ? kid : document.createTextNode(kid));
  return n;
};
const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return res.json();
}
const postJSON = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

/* ============================================================
   waveform player
   ============================================================ */
let audioCtx = null;
let current = null; // the player that is playing
const peakCache = new Map();

async function peaksFor(url, bars) {
  const key = `${url}|${bars}`;
  if (peakCache.has(key)) return peakCache.get(key);
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  const buf = await (await fetch(url)).arrayBuffer();
  const audio = await audioCtx.decodeAudioData(buf);
  const data = audio.getChannelData(0);
  const step = Math.max(1, Math.floor(data.length / bars));
  const peaks = new Float32Array(bars);
  let max = 1e-6;
  for (let i = 0; i < bars; i++) {
    let m = 0;
    for (let j = i * step, end = Math.min(data.length, j + step); j < end; j++) m = Math.max(m, Math.abs(data[j]));
    peaks[i] = m;
    max = Math.max(max, m);
  }
  for (let i = 0; i < bars; i++) peaks[i] = Math.sqrt(peaks[i] / max);
  const out = { peaks, duration: audio.duration };
  peakCache.set(key, out);
  return out;
}

function player(url, { color = "var(--accent)", compact = false } = {}) {
  const node = $("#tpl-player").content.firstElementChild.cloneNode(true);
  if (compact) node.classList.add("compact");
  node.style.setProperty("--c", color);
  const btn = $(".play", node), wave = $(".wave", node), canvas = $("canvas", node), time = $(".time", node);
  const audio = new Audio(url);
  audio.preload = "none";
  let peaks = null, raf = 0;

  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const resolveColor = () => {
    const c = getComputedStyle(node).getPropertyValue("--c").trim();
    return c.startsWith("var(") ? css(c.slice(4, -1)) : c;
  };
  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const w = wave.clientWidth, h = wave.clientHeight;
    if (!w) return;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const bars = peaks ? peaks.peaks : null;
    const n = bars ? bars.length : Math.floor(w / 4);
    const bw = w / n, prog = audio.duration ? audio.currentTime / audio.duration : 0;
    const fg = resolveColor(), bg = css("--wave");
    for (let i = 0; i < n; i++) {
      const v = bars ? bars[i] : 0.12;
      const bh = Math.max(2, v * (h - 4));
      ctx.fillStyle = i / n < prog ? fg : bg;
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(i * bw + 0.5, (h - bh) / 2, Math.max(1, bw - 1.5), bh, 1.5)
                    : ctx.rect(i * bw + 0.5, (h - bh) / 2, Math.max(1, bw - 1.5), bh);
      ctx.fill();
    }
  }
  async function load() {
    if (peaks) return;
    try {
      const bars = Math.max(40, Math.floor((wave.clientWidth || 300) / 4));
      peaks = await peaksFor(url, bars);
      time.textContent = fmtTime(peaks.duration);
    } catch { /* still playable */ }
    draw();
  }
  const tick = () => { draw(); time.textContent = fmtTime(audio.currentTime || 0); if (!audio.paused) raf = requestAnimationFrame(tick); };
  btn.addEventListener("click", async () => {
    if (audio.paused) {
      if (current && current !== api_) current.stop();
      current = api_;
      await audio.play();
      node.classList.add("playing");
      raf = requestAnimationFrame(tick);
    } else api_.stop(false);
  });
  wave.addEventListener("click", (e) => {
    if (!audio.duration && !peaks) return;
    const r = wave.getBoundingClientRect();
    audio.currentTime = ((e.clientX - r.left) / r.width) * (audio.duration || peaks.duration);
    draw();
  });
  audio.addEventListener("ended", () => { node.classList.remove("playing"); cancelAnimationFrame(raf); audio.currentTime = 0; draw(); time.textContent = fmtTime(peaks ? peaks.duration : 0); });
  const api_ = {
    node,
    stop(reset = true) { audio.pause(); node.classList.remove("playing"); cancelAnimationFrame(raf); if (reset) audio.currentTime = 0; draw(); },
    play: () => btn.click(),
  };
  new ResizeObserver(() => draw()).observe(wave);
  requestAnimationFrame(() => load());
  return node;
}

/* ============================================================
   state
   ============================================================ */
const state = {
  config: null,
  models: new Map(),      // name -> {label, color, description}
  selected: new Set(store.get("models", [])),
  ref: null,              // {ref_id, url, duration, note, text, label}
  filter: "all",
  libSelected: null,
};
const modelColor = (name) => (state.models.get(name) || {}).color || "var(--faint)";
const modelLabel = (name) => (name === "real speech" ? "Real speech" : (state.models.get(name) || {}).label || name);

/* ============================================================
   tabs & theme
   ============================================================ */
function showPage(page) {
  $$(".tab").forEach((x) => x.classList.toggle("active", x.dataset.page === page));
  $$(".page").forEach((p) => p.classList.toggle("active", p.id === `page-${page}`));
  if (page === "bench") loadBench();
  if (page === "history") loadHistory();
}
$$(".tab").forEach((t) => t.addEventListener("click", () => {
  history.replaceState(null, "", `#${t.dataset.page}`);
  showPage(t.dataset.page);
}));
const applyTheme = (t) => { document.documentElement.dataset.theme = t; store.set("theme", t); };
applyTheme(store.get("theme", "dark"));
$("#theme").addEventListener("click", () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

/* ============================================================
   config & status
   ============================================================ */
async function loadConfig() {
  const cfg = await api("/api/config");
  const first = !state.config;
  state.config = cfg;
  cfg.models.forEach((m) => state.models.set(m.name, m));
  if (first) {
    if (!state.selected.size) cfg.models.forEach((m) => state.selected.add(m.name));
    renderModels(); renderLibrary(); renderPresets(); loadTally();
  }
  const s = $("#status");
  s.classList.toggle("ready", cfg.ready);
  s.classList.toggle("loading", !cfg.ready);
  $(".label", s).textContent = cfg.ready ? `${cfg.models.length} models ready` : "Loading models…";
  updateGenerate();
  if (!cfg.ready) setTimeout(loadConfig, 2500);
}

/* ============================================================
   voice: library / upload / record
   ============================================================ */
$$("#voice-mode .seg-btn").forEach((b) => b.addEventListener("click", () => {
  $$("#voice-mode .seg-btn").forEach((x) => x.classList.toggle("active", x === b));
  $$(".voice-pane").forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== b.dataset.mode));
}));
$$("#lib-filters .chip").forEach((c) => c.addEventListener("click", () => {
  state.filter = c.dataset.f;
  $$("#lib-filters .chip").forEach((x) => x.classList.toggle("active", x === c));
  renderLibrary();
}));

const preview = new Audio();
function renderLibrary() {
  const list = $("#lib-list");
  list.innerHTML = "";
  const f = state.filter;
  const clips = state.config.library.filter((c) =>
    f === "all" || (f === "unseen" && !c.seen) || (f === "seen" && c.seen) || c.voice === f);
  if (!clips.length) { list.append(el("div", { class: "empty" }, "No clips match.")); return; }
  for (const c of clips) {
    const mini = el("span", { class: "mini", title: "Preview" },
      el("svg", {}, ));
    mini.innerHTML = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
    mini.addEventListener("click", (e) => {
      e.stopPropagation();
      if (current) current.stop();
      if (!preview.paused && preview.dataset.id === c.id) { preview.pause(); return; }
      preview.src = c.url; preview.dataset.id = c.id; preview.play();
    });
    const item = el("button", { class: `lib-item${state.libSelected === c.id ? " selected" : ""}` },
      mini,
      el("div", { style: "min-width:0" },
        el("div", { class: "lib-meta" },
          el("span", { class: "lib-name" }, c.speaker),
          el("span", { class: `badge ${c.seen ? "seen" : "unseen"}` }, c.seen ? "in training" : "unseen"),
          el("span", { class: "badge" }, c.voice),
          el("span", { class: "badge mono" }, `${c.duration}s`)),
        el("div", { class: "lib-text np" }, c.text)));
    item.addEventListener("click", async () => {
      state.libSelected = c.id;
      renderLibrary();
      preview.pause();
      const ref = await postJSON("/api/reference/library", { clip_id: c.id });
      setReference({ ...ref, label: `${c.speaker} · ${c.seen ? "in training" : "unseen"}` });
    });
    list.append(item);
  }
}

// upload
const dz = $("#dropzone"), fileInput = $("#file-input");
fileInput.addEventListener("change", () => fileInput.files[0] && uploadFile(fileInput.files[0], fileInput.files[0].name));
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) uploadFile(f, f.name); });

async function uploadFile(blob, name) {
  const fd = new FormData();
  fd.append("file", blob, name);
  $("#reference").innerHTML = '<div class="ref-empty">Processing voice…</div>';
  try {
    const ref = await api("/api/reference/upload", { method: "POST", body: fd });
    state.libSelected = null; renderLibrary();
    setReference({ ...ref, label: name });
  } catch (e) {
    $("#reference").innerHTML = `<div class="error">${e.message}</div>`;
  }
}

// record
let recorder = null, recChunks = [], recStart = 0, recTimer = 0, recStream = null, analyser = null;
$("#rec-btn").addEventListener("click", async () => {
  const btn = $("#rec-btn");
  if (recorder && recorder.state === "recording") { recorder.stop(); return; }
  try {
    recStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) { $("#rec-hint").textContent = "Microphone not available: " + e.message; return; }
  audioCtx = audioCtx || new AudioContext();
  analyser = audioCtx.createAnalyser(); analyser.fftSize = 512;
  audioCtx.createMediaStreamSource(recStream).connect(analyser);
  recorder = new MediaRecorder(recStream);
  recChunks = [];
  recorder.ondataavailable = (e) => recChunks.push(e.data);
  recorder.onstop = () => {
    btn.classList.remove("on"); clearInterval(recTimer);
    recStream.getTracks().forEach((t) => t.stop());
    $("#rec-hint").textContent = "Click to record again.";
    uploadFile(new Blob(recChunks, { type: recorder.mimeType }), "recording.webm");
  };
  recorder.start(); recStart = Date.now(); btn.classList.add("on");
  $("#rec-hint").textContent = "Recording… click to stop.";
  const meter = $("#rec-meter"), mctx = meter.getContext("2d"), buf = new Uint8Array(analyser.fftSize);
  recTimer = setInterval(() => {
    const s = (Date.now() - recStart) / 1000;
    $("#rec-time").textContent = fmtTime(s);
    analyser.getByteTimeDomainData(buf);
    let peak = 0; for (const v of buf) peak = Math.max(peak, Math.abs(v - 128) / 128);
    mctx.clearRect(0, 0, meter.width, meter.height);
    const segs = 16;
    for (let i = 0; i < segs; i++) {
      mctx.fillStyle = i / segs < peak * 1.4 ? (i > 12 ? "#ff6b7a" : "#2fc58f") : "rgba(128,128,128,.25)";
      mctx.fillRect(i * 10, 6, 7, 24);
    }
    if (s >= 20) recorder.stop();
  }, 80);
});

function setReference(ref) {
  state.ref = ref;
  const box = $("#reference");
  box.innerHTML = "";
  const textArea = el("textarea", { class: "ref-text np", rows: 3, placeholder: "What is said in the reference (auto-filled for library clips)…" });
  textArea.value = ref.text || "";
  const btn = el("button", { class: "btn" }, "Auto-transcribe");
  btn.addEventListener("click", async () => {
    btn.disabled = true; btn.textContent = "Transcribing…";
    try {
      const r = await postJSON("/api/transcribe", { ref_id: ref.ref_id });
      textArea.value = r.text; state.ref.text = r.text;
      btn.textContent = `Transcribed in ${r.seconds}s`;
    } catch (e) { btn.textContent = "Failed — " + e.message; }
    setTimeout(() => { btn.disabled = false; btn.textContent = "Auto-transcribe"; }, 2500);
  });
  textArea.addEventListener("input", () => { state.ref.text = textArea.value; });
  box.append(
    el("div", { class: "ref-head" }, el("b", {}, "Reference"), el("span", { class: "hint" }, `${ref.label || ""} · ${ref.duration}s`)),
    player(ref.url, { color: "var(--accent)" }),
    ref.note ? el("p", { class: "ref-note" }, ref.note) : null,
    el("div", { class: "ref-text-wrap" },
      el("div", { class: "row" }, el("span", { class: "hint" }, "Transcript"), btn),
      textArea),
  );
  if (!ref.text) btn.click();
  updateGenerate();
}

/* ============================================================
   script, models, options
   ============================================================ */
const textBox = $("#text");
textBox.value = store.get("text", "नमस्ते, आज हामी नेपालको स्वास्थ्य प्रणालीको बारेमा कुरा गर्दैछौं।");
const updateCount = () => { $("#char-count").textContent = `${[...textBox.value].length} characters`; store.set("text", textBox.value); updateGenerate(); };
textBox.addEventListener("input", updateCount);
updateCount();

function renderPresets() {
  const box = $("#presets");
  for (const p of state.config.presets) {
    box.append(el("button", { class: "chip", title: p, onclick: () => { textBox.value = p; updateCount(); } }, p));
  }
}
function renderModels() {
  const box = $("#model-toggles");
  box.innerHTML = "";
  for (const m of state.config.models) {
    const on = state.selected.has(m.name);
    box.append(el("button", {
      class: `model-toggle${on ? " on" : ""}`, style: `--c:${m.color}`,
      onclick: () => { on ? state.selected.delete(m.name) : state.selected.add(m.name); store.set("models", [...state.selected]); renderModels(); updateGenerate(); },
    }, el("span", { class: "box" }), el("span", {}, el("div", { class: "name" }, m.label), el("div", { class: "desc" }, m.description))));
  }
}
$("#dice").addEventListener("click", () => { $("#seed").value = Math.floor(Math.random() * 10000); });
const blindBox = $("#blind");
blindBox.checked = store.get("blind", false);
blindBox.addEventListener("change", () => store.set("blind", blindBox.checked));

function updateGenerate() {
  const g = $("#generate");
  const ok = state.config && state.config.ready && state.ref && textBox.value.trim() && state.selected.size;
  g.disabled = !ok;
  $(".gen-label", g).textContent = !state.config || !state.config.ready ? "Models loading…"
    : !state.ref ? "Choose a voice first" : !state.selected.size ? "Pick at least one model"
    : `Generate with ${state.selected.size} model${state.selected.size > 1 ? "s" : ""}`;
}

/* ============================================================
   generate & vote
   ============================================================ */
$("#generate").addEventListener("click", async () => {
  const g = $("#generate"), err = $("#gen-error");
  err.hidden = true;
  g.classList.add("busy"); g.disabled = true;
  $(".gen-label", g).textContent = "Generating…";
  try {
    const order = state.config.models.map((m) => m.name).filter((n) => state.selected.has(n));
    const round = await postJSON("/api/generate", {
      ref_id: state.ref.ref_id, ref_text: state.ref.text || "", text: textBox.value.trim(),
      seed: Number($("#seed").value) || 0, models: order, blind: blindBox.checked,
    });
    if (round.ref_text && !state.ref.text) { state.ref.text = round.ref_text; const ta = $(".ref-text"); if (ta) ta.value = round.ref_text; }
    renderRound(round, true);
  } catch (e) {
    err.textContent = e.message; err.hidden = false;
  } finally {
    g.classList.remove("busy"); updateGenerate();
  }
});

function renderRound(round, prepend = false, container = $("#results")) {
  const wrap = el("div", { class: "round", "data-round": round.round_id });
  const best = round.vote ? round.vote.best_slot : null;
  const head = el("div", { class: "round-head" },
    el("div", { class: "np" }, round.text),
    el("span", { class: "hint" }, `${round.blind ? (round.revealed ? "blind · revealed" : "blind") : "labelled"} · prompt ${round.prompt_time}s`));
  const grid = el("div", { class: "result-grid" });
  for (const s of round.slots) {
    const known = !!s.model;
    const color = known ? modelColor(s.model) : "var(--faint)";
    const card = el("div", { class: `result${best === s.slot ? " winner" : ""}`, style: `--c:${color}` },
      el("div", { class: "r-head" },
        el("div", { class: "r-title" }, el("span", { class: "sw" }), known ? modelLabel(s.model) : `Sample ${s.slot}`),
        best === s.slot ? el("span", { class: "crown" }, "★ your pick") : known && round.blind ? el("span", { class: "hint" }, `was ${s.slot}`) : null),
      player(s.url, { color: known ? color : "var(--muted)" }),
      el("div", { class: "r-meta" }, el("span", {}, `${s.seconds}s audio`), el("span", {}, `generated in ${s.gen_time}s`)));
    if (round.blind && !round.revealed) {
      card.append(el("div", { class: "vote-row" },
        el("button", { class: "btn", onclick: () => vote(round.round_id, s.slot) }, `${s.slot} is best`)));
    }
    grid.append(card);
  }
  wrap.append(head, grid);
  if (round.blind && !round.revealed) {
    wrap.append(el("div", { class: "round-actions" },
      el("button", { class: "btn", onclick: () => vote(round.round_id, "tie") }, "Can't tell them apart")));
  }
  const old = $(`[data-round="${round.round_id}"]`, container);
  if (old) old.replaceWith(wrap);
  else if (prepend) container.prepend(wrap);
  else container.append(wrap);
  while (container === $("#results") && container.children.length > 4) container.lastChild.remove();
}

async function vote(roundId, best) {
  const r = await postJSON("/api/vote", { round_id: roundId, best });
  renderRound(r.round);
  renderTally(r.tally);
}

async function loadTally() { renderTally(await api("/api/tally")); }
function renderTally(t) {
  const box = $("#tally");
  box.innerHTML = "";
  $("#tally-rounds").textContent = t.rounds ? `${t.rounds} blind round${t.rounds > 1 ? "s" : ""}${t.ties ? ` · ${t.ties} ties` : ""}` : "";
  if (!t.rounds) { box.append(el("div", { class: "empty" }, "Turn on Blind test, generate, and pick the best sample. Wins add up here.")); return; }
  for (const [name, s] of Object.entries(t.models)) {
    if (!s.played) continue;
    box.append(el("div", { class: "tally-row", style: `--c:${modelColor(name)}` },
      el("span", {}, modelLabel(name)),
      el("div", { class: "bar" }, el("span", { style: `width:${(s.win_rate || 0) * 100}%` })),
      el("span", { class: "mono hint" }, `${s.wins}/${s.played} · ${Math.round((s.win_rate || 0) * 100)}%`)));
  }
}

/* ============================================================
   benchmark
   ============================================================ */
const METRICS = [
  { key: "cer", title: "Character error rate", dir: "lower is better", scale: 100, unit: "%", digits: 1,
    explain: "Generated speech transcribed by IndicConformer vs. the text it should say. Mostly measures intelligibility." },
  { key: "wer", title: "Word error rate", dir: "lower is better", scale: 100, unit: "%", digits: 1,
    explain: "Same as CER at the word level. Stricter: one wrong letter makes the whole word wrong." },
  { key: "sim", title: "Speaker similarity", dir: "higher is better", scale: 1, unit: "", digits: 3, min: 0, max: 1,
    explain: "Cosine similarity of WavLM-ECAPA speaker embeddings: output vs. reference voice (SIM-o)." },
  { key: "utmos", title: "Naturalness (UTMOS)", dir: "higher is better", scale: 1, unit: "", digits: 2, min: 1, max: 5,
    explain: "Predicted mean opinion score, 1–5. Trained on English listening tests: compare models, don't read as absolute." },
];
let bench = null, benchGroup = "all", itemFilter = "all";

async function loadBench() {
  if (bench) return;
  bench = await api("/api/bench");
  if (!bench.available) {
    $("#bench-intro").innerHTML = "<h1>No benchmark yet</h1><p>Run <code>python -m tools.tts_bench --work …</code>, then reload this page.</p>";
    return;
  }
  for (const n of Object.keys(bench.models)) if (!state.models.has(n)) state.models.set(n, { label: n, color: "#999" });
  const refs = Object.values(bench.refs), texts = Object.values(bench.texts);
  $("#bench-intro").innerHTML = "";
  $("#bench-intro").append(
    el("h1", {}, "Objective benchmark"),
    el("p", {}, "Every model says every sentence in every reference voice, with the same voice prompt and seed. References come from held-out recordings; most voices were never in anyone's training data. ",
      el("b", {}, "Real speech"), " is the actual recording of each held-out sentence: its CER is the floor set by the ASR itself, its UTMOS what natural speech scores."),
    el("div", { class: "bench-facts" },
      fact(Object.keys(bench.models).length, "models"),
      fact(refs.length, `reference clips · ${new Set(refs.map((r) => r.speaker)).size} voices`),
      fact(texts.length, `sentences (${texts.filter((t) => t.kind === "written").length} written, ${texts.filter((t) => t.kind === "dev").length} held-out)`),
      fact(bench.items.length * Object.keys(bench.models).length, "generated utterances")));
  const seg = $("#bench-group");
  seg.innerHTML = "";
  for (const g of Object.keys(bench.summary.groups)) {
    seg.append(el("button", { class: `seg-btn${g === benchGroup ? " active" : ""}`, onclick: (e) => {
      benchGroup = g; $$("#bench-group .seg-btn").forEach((b) => b.classList.toggle("active", b === e.currentTarget)); renderMetrics();
    } }, g));
  }
  renderMetrics(); renderPaired(); renderItemFilters(); renderItems();
}
const fact = (v, label) => el("div", { class: "fact" }, el("b", {}, String(v)), el("span", {}, label));

function renderMetrics() {
  const grid = $("#metric-grid");
  grid.innerHTML = "";
  const g = bench.summary.groups[benchGroup] || {};
  for (const m of METRICS) {
    const rows = Object.entries(g).filter(([, s]) => s[m.key] && isFinite(s[m.key][0]));
    if (!rows.length) continue;
    const vals = rows.map(([, s]) => s[m.key][2] * m.scale);
    const lo = m.min != null ? m.min * m.scale : 0, hi = m.max != null ? m.max * m.scale : Math.max(...vals) * 1.15;
    const pos = (v) => `${Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100))}%`;
    const candidates = rows.filter(([n]) => n !== "real speech").map(([n, s]) => [n, s[m.key][0]]);
    const bestName = candidates.length ? candidates.sort((a, b) => (m.dir.startsWith("lower") ? a[1] - b[1] : b[1] - a[1]))[0][0] : null;
    const card = el("div", { class: "card metric" },
      el("div", { class: "card-head", style: "margin-bottom:4px" }, el("h3", {}, m.title), el("span", { class: "dir" }, m.dir)),
      el("div", { class: "explain" }, m.explain));
    for (const [name, s] of rows) {
      const [mean, l, h] = s[m.key].map((x) => x * m.scale);
      card.append(el("div", { class: `mrow${name === "real speech" ? " real" : ""}${name === bestName ? " best" : ""}`, style: `--c:${modelColor(name)}` },
        el("span", { class: "name" }, modelLabel(name)),
        el("div", { class: "mbar" }, el("div", { class: "fill", style: `width:${pos(mean)}` }), el("div", { class: "ci", style: `left:${pos(l)};width:calc(${pos(h)} - ${pos(l)})` })),
        el("span", { class: "val" }, `${mean.toFixed(m.digits)}${m.unit}`, el("small", {}, ` ±${((h - l) / 2).toFixed(m.digits)}`))));
    }
    grid.append(card);
  }
}

function renderPaired() {
  const box = $("#paired");
  box.innerHTML = "";
  const p = bench.summary.paired_vs_base || {};
  if (!Object.keys(p).length) { box.append(el("div", { class: "empty" }, "No paired results.")); return; }
  const table = el("table", { class: "paired" },
    el("tr", {}, el("th", {}, "Model vs. base"), ...METRICS.map((m) => el("th", {}, m.title)), el("th", {}, "pairs")));
  for (const [name, e] of Object.entries(p)) {
    const row = el("tr", {}, el("td", {}, el("b", { style: `color:${modelColor(name)}` }, modelLabel(name))));
    for (const m of METRICS) {
      const [d, l, h] = e[m.key].diff.map((x) => x * m.scale);
      const lowerBetter = m.dir.startsWith("lower");
      const sig = l > 0 || h < 0;
      const good = sig && (lowerBetter ? h < 0 : l > 0);
      const sign = d > 0 ? "+" : "";
      row.append(el("td", {}, el("span", { class: `delta${sig ? (good ? " good" : " bad") : ""}` },
        `${sign}${d.toFixed(m.digits)}${m.unit}`, el("small", {}, ` [${l.toFixed(m.digits)}, ${h.toFixed(m.digits)}]`)),
        el("div", { class: "hint" }, `better in ${Math.round(e[m.key].win_rate * 100)}% of pairs`)));
    }
    row.append(el("td", { class: "mono" }, String(e.n)));
    table.append(row);
  }
  box.append(table, el("p", { class: "hint" }, "Green/red: the 95% interval excludes zero (a real difference on this test set). Grey: within noise."));
}

function renderItemFilters() {
  const box = $("#item-filters");
  box.innerHTML = "";
  for (const [k, label] of [["all", "All"], ["unseen", "Unseen voices"], ["seen", "Seen in training"], ["female", "Female"], ["male", "Male"], ["written", "Written text"], ["dev", "Held-out text"]]) {
    box.append(el("button", { class: `chip${itemFilter === k ? " active" : ""}`, onclick: () => { itemFilter = k; renderItemFilters(); renderItems(); } }, label));
  }
}

function renderItems() {
  const box = $("#items");
  box.innerHTML = "";
  const models = Object.keys(bench.models);
  const shown = bench.items.filter((it) => {
    const r = bench.refs[it.ref], t = bench.texts[it.text_id];
    return itemFilter === "all" || (itemFilter === "unseen" && !r.seen) || (itemFilter === "seen" && r.seen)
      || r.voice === itemFilter || t.kind === itemFilter;
  });
  for (const it of shown.slice(0, 80)) {
    const r = bench.refs[it.ref], t = bench.texts[it.text_id];
    const d = el("details", { class: "item" },
      el("summary", {},
        el("div", { class: "np" }, t.text),
        el("div", { class: "tags" },
          el("span", { class: `badge ${r.seen ? "seen" : "unseen"}` }, r.seen ? "in training" : "unseen"),
          el("span", { class: "badge" }, r.voice), el("span", { class: "badge" }, t.kind === "dev" ? "held-out text" : "written"))));
    d.addEventListener("toggle", () => {
      if (!d.open || d.dataset.loaded) return;
      d.dataset.loaded = "1";
      const body = el("div", { class: "item-body" });
      body.append(el("div", { class: "item-cell" }, el("div", { class: "t" }, "Reference voice", el("span", { class: "hint" }, r.speaker)),
        player(`/media/bench-ref/${it.ref}`, { compact: true, color: "var(--muted)" })));
      if (it.real) body.append(cell("real speech", `/media/bench-real/${it.text_id}`, it.real));
      for (const m of models) if (it.scores[m]) body.append(cell(m, `/media/bench/${m}/${it.item}`, it.scores[m]));
      d.append(body);
    });
    box.append(d);
  }
  if (shown.length > 80) box.append(el("div", { class: "hint" }, `Showing 80 of ${shown.length}; filter to narrow.`));
}
function cell(name, url, s) {
  return el("div", { class: "item-cell", style: `--c:${modelColor(name)}` },
    el("div", { class: "t" }, el("span", { style: `color:${modelColor(name)}` }, modelLabel(name))),
    player(url, { compact: true, color: name === "real speech" ? "var(--muted)" : modelColor(name) }),
    el("div", { class: "scores" },
      el("span", {}, `CER ${(s.cer * 100).toFixed(1)}%`),
      s.sim != null ? el("span", {}, `SIM ${s.sim.toFixed(3)}`) : null,
      el("span", {}, `UTMOS ${s.utmos.toFixed(2)}`)),
    el("div", { class: "asr np" }, `ASR: ${s.asr}`));
}

/* ============================================================
   history
   ============================================================ */
async function loadHistory() {
  const box = $("#history");
  box.innerHTML = "";
  const rounds = await api("/api/rounds");
  if (!rounds.length) { box.append(el("div", { class: "card empty" }, "No rounds yet.")); return; }
  for (const r of rounds) {
    const c = el("div", { class: "card hist-round" });
    box.append(c);
    renderRound(r, false, c);
  }
}

loadConfig()
  .then(() => { const page = location.hash.slice(1); if (["bench", "history"].includes(page)) showPage(page); })
  .catch((e) => { $("#status .label").textContent = "Server error: " + e.message; });
