"use strict";

const SPEAKER_COLORS = ["#4c6ef5", "#12b886", "#f59f00", "#ae3ec9", "#1098ad", "#f76707", "#74b816", "#d6336c"];
const GUTTER = 118;
const RULER_H = 26;
const WAVE_H = 74;
const SPK_H = 70;
const OVL_H = 60;
const LANE_GAP = 8;

const $ = (id) => document.getElementById(id);
const audio = $("audio");
const timeline = $("timeline");
const overview = $("overview");
const tooltip = $("tooltip");

const state = {
  data: null,
  view: { t0: 0, t1: 1 },
  filter: "all",
  current: null, // segment being played
  lanes: [],
  segRects: [],
  dirty: true,
  clipMode: false, // true while a segment plays from its exported clip
  version: Date.now(), // cache-buster for clip files, which are rewritten on re-export
};

// ---------- helpers ----------

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function decode(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function fmt(t, decimals = 1) {
  if (!isFinite(t)) t = 0;
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = (t % 60).toFixed(decimals).padStart(decimals ? 3 + decimals : 2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
}

// Segment number: the index in the segment id (e.g. "<recording>_00003" -> 3).
function segNum(seg) {
  return parseInt(seg.id.slice(seg.id.lastIndexOf("_") + 1), 10);
}

function speakerColor(index) {
  return SPEAKER_COLORS[index % SPEAKER_COLORS.length];
}

function withAlpha(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

// first interval whose end is after t (intervals sorted and non-overlapping)
function firstVisible(intervals, t) {
  let lo = 0, hi = intervals.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (intervals[mid][1] <= t) lo = mid + 1; else hi = mid;
  }
  return lo;
}

function setupCanvas(canvas, height) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  canvas.style.height = height + "px";
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

// ---------- data loading ----------

async function loadRecordings() {
  const recordings = await (await fetch("/api/recordings")).json();
  const sel = $("recording-select");
  sel.innerHTML = "";
  // one group per work directory
  const groups = new Map();
  for (const r of recordings) {
    if (!groups.has(r.work)) groups.set(r.work, []);
    groups.get(r.work).push(r);
  }
  for (const [work, recs] of groups) {
    const g = document.createElement("optgroup");
    g.label = work;
    for (const r of recs) {
      const o = document.createElement("option");
      o.value = r.key;
      o.textContent = `${r.title || r.recording_id} · ${r.num_segments ?? "?"} segments`;
      o.title = [r.work, r.title, r.recording_id].filter(Boolean).join(" · ");
      g.appendChild(o);
    }
    sel.appendChild(g);
  }
  sel.onchange = () => loadRecording(sel.value);
  if (recordings.length) loadRecording(recordings[0].key);
  else $("stats").textContent = "No processed recordings found. Run `nepvoice run INPUT_DIR --to select` first.";
}

async function loadRecording(rid) {
  $("stats").textContent = "Loading…";
  const data = await (await fetch(`/api/recording/${encodeURIComponent(rid)}`)).json();
  data.peaksArr = decode(data.peaks);
  data.overlap.pMultiArr = decode(data.overlap.p_multi);
  for (const s of data.speakers) {
    s.probArr = decode(s.prob);
    s.otherArr = decode(s.other_prob);
    s.color = speakerColor(s.index);
  }
  data.speakerByName = Object.fromEntries(data.speakers.map((s) => [s.name, s]));
  data.segments.sort((a, b) => a.start - b.start);
  state.data = data;
  state.filter = "all";
  state.current = null;
  state.clipMode = false;
  clip.pause();
  audio.src = data.audio_url;
  $("dur").textContent = fmt(data.duration, 0);
  state.view = { t0: 0, t1: data.duration };
  const m = location.hash.match(/t=([\d.]+)-([\d.]+)/);
  if (m) setView(parseFloat(m[1]), parseFloat(m[2]));
  renderStats();
  renderLegend();
  renderTabs();
  renderList();
  layoutLanes();
  draw();
}

// ---------- header / list ----------

function renderStats() {
  const d = state.data;
  const kept = d.segments.reduce((a, s) => a + s.duration, 0);
  const ovl = d.overlap.any.reduce((a, [s, e]) => a + (e - s), 0);
  const parts = [
    `<span>audio <b>${fmt(d.duration, 0)}</b></span>`,
    `<span>kept <b>${fmt(kept, 0)}</b> (${((kept / d.duration) * 100).toFixed(1)}%)</span>`,
    `<span><b>${d.segments.length}</b> segments</span>`,
    `<span>overlap (p&gt;0.5) <b>${ovl.toFixed(0)} s</b></span>`,
  ];
  $("stats").innerHTML = parts.join("");
}

function renderLegend() {
  const items = [
    ...state.data.speakers.map((s) => [s.color, `${s.label}${s.target ? "" : " (spurious)"}`]),
    [css("--overlap"), "overlap (≥2 speakers) / rejected speech"],
    ["linear-gradient(#e03131 0 0) center/100% 2px no-repeat", "other voice (DiariZen)"],
  ];
  $("legend").innerHTML = items.map(([c, l]) => `<span><i style="background:${c}"></i>${l}</span>`).join("")
    + "<span>speaker lane: shaded = P(speaking) · red strip = speech rejected · blocks = kept segments</span>";
}

function filteredSegments() {
  const segs = state.data.segments;
  return state.filter === "all" ? segs : segs.filter((s) => s.speaker === state.filter);
}

function renderTabs() {
  const tabs = $("tabs");
  const d = state.data;
  const counts = {};
  for (const s of d.segments) counts[s.speaker] = (counts[s.speaker] || 0) + 1;
  const entries = [["all", "All", null, d.segments.length]];
  for (const s of d.speakers) if (counts[s.name]) entries.push([s.name, s.label, s.color, counts[s.name]]);
  tabs.innerHTML = "";
  for (const [key, label, color, n] of entries) {
    const b = document.createElement("button");
    b.className = "tab";
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", String(state.filter === key));
    b.innerHTML = `${color ? `<span class="dot" style="background:${color}"></span>` : ""}${label} · ${n}`;
    b.onclick = () => { state.filter = key; renderTabs(); renderList(); };
    tabs.appendChild(b);
  }
}

function trimNote(s) {
  const parts = [];
  if (s.trimmed_head > 0) parts.push(`start −${s.trimmed_head.toFixed(2)}s`);
  if (s.trimmed_tail > 0) parts.push(`end −${s.trimmed_tail.toFixed(2)}s`);
  return parts.length ? ` · breath trimmed: ${parts.join(", ")}` : "";
}

function renderList() {
  const list = $("seg-list");
  const segs = filteredSegments();
  const total = segs.reduce((a, s) => a + s.duration, 0);
  $("list-meta").textContent = `${segs.length} segments · ${fmt(total, 0)} of speech`;
  list.innerHTML = "";
  const frag = document.createDocumentFragment();
  for (const s of segs) {
    const spk = state.data.speakerByName[s.speaker];
    const li = document.createElement("li");
    li.className = "seg";
    li.dataset.id = s.id;
    li.innerHTML = `
      <button class="play" aria-label="Play segment">▶</button>
      <div class="head">
        <span class="num mono">#${segNum(s)}</span>
        <span class="spk" style="color:${spk.color}">${spk.label}</span>
        <span class="mono">${fmt(s.start)} – ${fmt(s.end)}</span>
        <span class="mono">${s.duration.toFixed(1)} s</span>
        <a href="${s.clip_url}" download title="Download exported clip">flac</a>
      </div>
      <div class="text"></div>
      <div class="scores mono">peak other-speaker: nemotron ${s.max_nemotron_other.toFixed(3)} · diarizen ${s.max_diarizen_other.toFixed(3)} · P(≥2) ${s.max_diarizen_multi.toFixed(3)}${trimNote(s)}</div>`;
    li.querySelector(".text").textContent = s.text || "—";
    li.onclick = (e) => {
      if (e.target.tagName === "A") return;
      if (state.current === s && isPlaying()) active().pause();
      else playSegment(s);
    };
    frag.appendChild(li);
  }
  list.appendChild(frag);
  highlightCurrent();
}

function highlightCurrent() {
  state.dirty = true;
  for (const li of document.querySelectorAll(".seg")) {
    const active = state.current && li.dataset.id === state.current.id;
    li.classList.toggle("active", !!active);
    li.querySelector(".play").textContent = active && isPlaying() ? "❚❚" : "▶";
  }
}

function scrollToCurrent() {
  if (!state.current) return;
  const li = document.querySelector(`.seg[data-id="${CSS.escape(state.current.id)}"]`);
  if (li) li.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

// ---------- playback ----------

// Segments play from their exported clip file by default, so what you hear is exactly
// what goes into the dataset. Timeline clicks play the full recording.
const clip = new Audio();
clip.preload = "auto";

function active() {
  return state.clipMode ? clip : audio;
}

function isPlaying() {
  return !active().paused;
}

function nowTime() {
  return state.clipMode && state.current ? state.current.start + clip.currentTime : audio.currentTime;
}

function togglePlay() {
  const el = active();
  el.paused ? el.play() : el.pause();
}

function playSegment(seg) {
  state.current = seg;
  if ($("exact-clip").checked) {
    audio.pause();
    state.clipMode = true;
    clip.src = `${seg.clip_url}?v=${state.version}`;
    clip.play();
  } else {
    clip.pause();
    state.clipMode = false;
    audio.currentTime = seg.start;
    audio.play();
  }
  const span = state.view.t1 - state.view.t0;
  if (seg.start < state.view.t0 || seg.end > state.view.t1 || span > 600) {
    const pad = Math.max(2, seg.duration * 0.6);
    setView(seg.start - pad, seg.end + pad);
  }
  highlightCurrent();
  scrollToCurrent();
}

function seekSource(t) {
  clip.pause();
  state.clipMode = false;
  state.current = null;
  audio.currentTime = Math.max(0, t);
  highlightCurrent();
}

function nextAfter(seg) {
  const segs = filteredSegments();
  return segs[segs.indexOf(seg) + 1];
}

function stepSegment(dir) {
  const segs = filteredSegments();
  if (!segs.length) return;
  const t = nowTime();
  let idx;
  if (state.current && segs.includes(state.current)) idx = segs.indexOf(state.current) + dir;
  else idx = dir > 0 ? segs.findIndex((s) => s.start > t) : segs.findLastIndex((s) => s.end < t);
  if (idx >= 0 && idx < segs.length) playSegment(segs[idx]);
}

function playNumber(n) {
  const seg = state.data.segments.find((s) => segNum(s) === n);
  if (!seg) return false;
  if (state.filter !== "all" && state.filter !== seg.speaker) {
    state.filter = "all";
    renderTabs();
    renderList();
  }
  playSegment(seg);
  return true;
}

function tick() {
  const seg = state.current;
  // Source-audio segment playback stops (or advances) at the segment end.
  if (!state.clipMode && seg && !audio.paused && audio.currentTime >= seg.end) {
    if ($("autoplay-next").checked) {
      const next = nextAfter(seg);
      if (next) playSegment(next); else audio.pause();
    } else if ($("stop-at-end").checked) {
      audio.pause();
      audio.currentTime = seg.end;
    }
  }
  const t = nowTime();
  if (isPlaying()) {
    // keep the playhead in view while playing
    const { t0, t1 } = state.view;
    if (t > t1 || t < t0) setView(t - (t1 - t0) * 0.1, t + (t1 - t0) * 0.9);
  }
  $("cur").textContent = fmt(t);
  if (state.dirty || isPlaying()) { state.dirty = false; draw(); }
  requestAnimationFrame(tick);
}

for (const el of [audio, clip]) {
  el.addEventListener("play", () => { $("play").textContent = "❚❚"; highlightCurrent(); });
  el.addEventListener("pause", () => { $("play").textContent = "▶"; highlightCurrent(); });
}
clip.addEventListener("ended", () => {
  const next = state.current && $("autoplay-next").checked ? nextAfter(state.current) : null;
  if (next) playSegment(next);
  else highlightCurrent();
});

// ---------- view ----------

function setView(t0, t1) {
  const d = state.data.duration;
  const span = Math.min(Math.max(t1 - t0, 0.5), d);
  t0 = Math.min(Math.max(t0, 0), d - span);
  state.view = { t0, t1: t0 + span };
  state.dirty = true;
  clearTimeout(setView.timer);
  setView.timer = setTimeout(() => {
    history.replaceState(null, "", `#t=${state.view.t0.toFixed(1)}-${state.view.t1.toFixed(1)}`);
  }, 300);
}

function zoom(factor, center) {
  const { t0, t1 } = state.view;
  const c = center ?? (t0 + t1) / 2;
  setView(c - (c - t0) * factor, c + (t1 - c) * factor);
}

function layoutLanes() {
  const lanes = [{ kind: "ruler", h: RULER_H }, { kind: "wave", h: WAVE_H, label: "Waveform" }];
  for (const s of state.data.speakers) lanes.push({ kind: "speaker", h: s.target ? SPK_H : 40, spk: s, label: s.label });
  lanes.push({ kind: "overlap", h: OVL_H, label: "Overlap" });
  let y = 0;
  for (const l of lanes) { l.y = y; y += l.h + (l.kind === "ruler" ? 0 : LANE_GAP); }
  state.lanes = lanes;
  state.height = y + 4;
}

// ---------- drawing ----------

function drawCurve(ctx, arr, hop, x0, w, y, h, color, fill) {
  const { t0, t1 } = state.view;
  const pxPerSec = w / (t1 - t0);
  ctx.beginPath();
  ctx.moveTo(x0, y + h);
  for (let px = 0; px <= w; px++) {
    const ta = t0 + px / pxPerSec, tb = t0 + (px + 1) / pxPerSec;
    let ia = Math.floor(ta / hop), ib = Math.max(ia + 1, Math.ceil(tb / hop));
    ia = Math.max(0, ia); ib = Math.min(arr.length, ib);
    let m = 0;
    for (let i = ia; i < ib; i++) if (arr[i] > m) m = arr[i];
    ctx.lineTo(x0 + px, y + h - (m / 255) * h);
  }
  ctx.lineTo(x0 + w, y + h);
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (color) { ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.stroke(); }
}

function drawIntervals(ctx, intervals, x, y, h, color, minW = 1) {
  const { t0, t1 } = state.view;
  ctx.fillStyle = color;
  for (let i = firstVisible(intervals, t0); i < intervals.length; i++) {
    const [s, e] = intervals[i];
    if (s > t1) break;
    const xa = x(s), xb = x(e);
    ctx.fillRect(xa, y, Math.max(minW, xb - xa), h);
  }
}

function niceStep(span, px) {
  const target = span / (px / 90);
  const steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
  return steps.find((s) => s >= target) || 3600;
}

function draw() {
  const d = state.data;
  if (!d) return;
  const { ctx, width } = setupCanvas(timeline, state.height);
  const W = width - GUTTER - 8;
  const { t0, t1 } = state.view;
  const x = (t) => GUTTER + ((t - t0) / (t1 - t0)) * W;
  // x clamped to the plot area, for shapes that may start or end outside the view
  const xc = (t) => Math.min(GUTTER + W, Math.max(GUTTER, x(t)));
  const text = css("--text"), muted = css("--muted"), grid = css("--grid");
  state.segRects = [];

  ctx.clearRect(0, 0, width, state.height);
  ctx.font = "12px Inter, system-ui, sans-serif";

  // grid + ruler
  const step = niceStep(t1 - t0, W);
  ctx.textBaseline = "middle";
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
    const xx = Math.round(x(t)) + 0.5;
    ctx.strokeStyle = grid;
    ctx.beginPath(); ctx.moveTo(xx, RULER_H - 6); ctx.lineTo(xx, state.height); ctx.stroke();
    ctx.fillStyle = muted;
    ctx.fillText(fmt(t, step < 1 ? 1 : 0), xx + 4, RULER_H / 2);
  }

  for (const lane of state.lanes) {
    const { y, h } = lane;
    if (lane.kind !== "ruler") {
      ctx.fillStyle = text;
      ctx.font = "500 12px Inter, system-ui, sans-serif";
      ctx.fillText(lane.label, 12, y + 14);
      ctx.font = "11px Inter, system-ui, sans-serif";
      ctx.fillStyle = muted;
    }

    if (lane.kind === "wave") {
      // tint kept segments and overlaps behind the waveform
      for (const s of d.segments) {
        if (s.end < t0 || s.start > t1) continue;
        ctx.fillStyle = withAlpha(d.speakerByName[s.speaker].color, 0.16);
        ctx.fillRect(xc(s.start), y, xc(s.end) - xc(s.start), h);
      }
      drawIntervals(ctx, d.overlap.any, xc, y, h, withAlpha(css("--overlap").startsWith("#") ? css("--overlap") : "#e03131", 0.18));
      const arr = d.peaksArr, hop = 1 / d.peaks_per_second, mid = y + h / 2;
      ctx.fillStyle = css("--wave");
      const pxPerSec = W / (t1 - t0);
      for (let px = 0; px < W; px++) {
        const ia = Math.max(0, Math.floor((t0 + px / pxPerSec) / hop));
        const ib = Math.min(arr.length, Math.max(ia + 1, Math.ceil((t0 + (px + 1) / pxPerSec) / hop)));
        let m = 0;
        for (let i = ia; i < ib; i++) if (arr[i] > m) m = arr[i];
        const a = (m / 255) * (h / 2 - 2);
        ctx.fillRect(GUTTER + px, mid - a, 1, Math.max(1, 2 * a));
      }
    }

    if (lane.kind === "speaker") {
      const s = lane.spk;
      ctx.fillText(`${s.target ? "" : "spurious · "}${s.speech_seconds.toFixed(0)} s speech`, 12, y + 30);
      const actY = y + 4, actH = 30, rejY = actY + actH + 1, keptY = y + 42, keptH = 24;
      ctx.fillStyle = css("--surface-2");
      ctx.fillRect(GUTTER, actY, W, actH);
      // speaker probability (filled) — what Nemotron hears from this speaker
      drawCurve(ctx, s.probArr, d.curve_hop, GUTTER, W, actY, actH, null, withAlpha(s.color, 0.55));
      if (!s.target) continue; // spurious channel: only ever counts as a foreign voice
      // DiariZen's "someone else" activity for this speaker, as a thin line
      drawCurve(ctx, s.otherArr, d.curve_hop, GUTTER, W, actY, actH, withAlpha("#e03131", 0.7), null);
      // own speech that was rejected because a second voice is possible
      drawIntervals(ctx, s.rejected, xc, rejY, 4, css("--overlap"));
      // kept segments
      for (const seg of d.segments) {
        if (seg.speaker !== s.name || seg.end < t0 || seg.start > t1) continue;
        const xa = xc(seg.start), xb = xc(seg.end), active = state.current === seg;
        ctx.fillStyle = active ? text : s.color;
        ctx.fillRect(xa, keptY, Math.max(2, xb - xa), keptH);
        if (xb - xa > 46) {
          ctx.save();
          ctx.beginPath(); ctx.rect(xa, keptY, xb - xa, keptH); ctx.clip();
          ctx.fillStyle = active ? css("--surface") : "#fff";
          ctx.fillText(`#${segNum(seg)} · ${seg.duration.toFixed(1)}s`, xa + 5, keptY + keptH / 2);
          ctx.restore();
        }
        state.segRects.push({ seg, x0: xa, x1: Math.max(xa + 2, xb), y0: keptY, y1: keptY + keptH });
      }
    }

    if (lane.kind === "overlap") {
      const top = y + 4, rowH = 14;
      ctx.fillText("nemotron", 12, top + rowH / 2 + 18);
      ctx.fillText("diarizen P(≥2)", 12, top + 40);
      ctx.fillStyle = css("--surface-2");
      ctx.fillRect(GUTTER, top + 12, W, rowH);
      drawIntervals(ctx, d.overlap.nemotron, xc, top + 12, rowH, css("--overlap"), 2);
      const curveY = top + 30, curveH = h - 34;
      ctx.fillStyle = css("--surface-2");
      ctx.fillRect(GUTTER, curveY, W, curveH);
      drawCurve(ctx, d.overlap.pMultiArr, d.curve_hop, GUTTER, W, curveY, curveH, css("--overlap"), withAlpha("#e03131", 0.25));
    }
  }

  // playhead
  const ph = x(nowTime() || 0);
  if (ph >= GUTTER && ph <= GUTTER + W) {
    ctx.strokeStyle = css("--playhead");
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(ph, 0); ctx.lineTo(ph, state.height); ctx.stroke();
  }

  drawOverview();
}

function drawOverview() {
  const d = state.data;
  const { ctx, width, height } = setupCanvas(overview, 46);
  const x = (t) => (t / d.duration) * width;
  ctx.clearRect(0, 0, width, height);
  const targets = d.speakers.filter((s) => s.target);
  const rowH = Math.max(6, Math.floor((height - 14) / Math.max(1, targets.length)));
  targets.forEach((s, i) => {
    const y = 4 + i * rowH;
    ctx.fillStyle = withAlpha(s.color, 0.18);
    for (const [a, b] of s.activity) ctx.fillRect(x(a), y, Math.max(0.5, x(b) - x(a)), rowH - 2);
    ctx.fillStyle = s.color;
    for (const seg of d.segments) if (seg.speaker === s.name) ctx.fillRect(x(seg.start), y, Math.max(1, x(seg.end) - x(seg.start)), rowH - 2);
  });
  ctx.fillStyle = css("--overlap");
  for (const [a, b] of d.overlap.any) ctx.fillRect(x(a), height - 8, Math.max(1, x(b) - x(a)), 5);
  // viewport
  const { t0, t1 } = state.view;
  ctx.strokeStyle = css("--accent");
  ctx.lineWidth = 2;
  ctx.strokeRect(x(t0) + 1, 1, Math.max(3, x(t1) - x(t0) - 2), height - 2);
  ctx.fillStyle = css("--playhead");
  ctx.fillRect(x(nowTime() || 0), 0, 1.5, height);
}

// ---------- interaction ----------

function timeAt(clientX) {
  const r = timeline.getBoundingClientRect();
  const W = r.width - GUTTER - 8;
  const { t0, t1 } = state.view;
  return t0 + ((clientX - r.left - GUTTER) / W) * (t1 - t0);
}

function segAt(clientX, clientY) {
  const r = timeline.getBoundingClientRect();
  const px = clientX - r.left, py = clientY - r.top;
  return state.segRects.find((s) => px >= s.x0 && px <= s.x1 && py >= s.y0 && py <= s.y1)?.seg;
}

timeline.addEventListener("wheel", (e) => {
  if (!state.data) return;
  e.preventDefault();
  if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
    const span = state.view.t1 - state.view.t0;
    setView(state.view.t0 + (e.deltaX / timeline.clientWidth) * span, state.view.t1 + (e.deltaX / timeline.clientWidth) * span);
  } else {
    zoom(Math.exp(e.deltaY * 0.0015), timeAt(e.clientX));
  }
}, { passive: false });

let drag = null;
timeline.addEventListener("mousedown", (e) => {
  drag = { x: e.clientX, t0: state.view.t0, t1: state.view.t1, moved: false };
});
window.addEventListener("mousemove", (e) => {
  if (drag) {
    const dx = e.clientX - drag.x;
    if (Math.abs(dx) > 3) { drag.moved = true; timeline.classList.add("dragging"); }
    if (drag.moved) {
      const span = drag.t1 - drag.t0;
      const dt = (dx / (timeline.clientWidth - GUTTER - 8)) * span;
      setView(drag.t0 - dt, drag.t1 - dt);
    }
  }
  updateTooltip(e);
});
window.addEventListener("mouseup", (e) => {
  if (drag && !drag.moved && e.target === timeline) {
    const seg = segAt(e.clientX, e.clientY);
    if (seg) playSegment(seg);
    else if (e.clientX - timeline.getBoundingClientRect().left > GUTTER) seekSource(timeAt(e.clientX));
  }
  drag = null;
  timeline.classList.remove("dragging");
});
timeline.addEventListener("mouseleave", () => { tooltip.hidden = true; });

function updateTooltip(e) {
  if (!state.data || e.target !== timeline || drag?.moved) { tooltip.hidden = true; return; }
  const r = timeline.getBoundingClientRect();
  const px = e.clientX - r.left, py = e.clientY - r.top;
  if (px < GUTTER) { tooltip.hidden = true; return; }
  const t = timeAt(e.clientX);
  const seg = segAt(e.clientX, e.clientY);
  let html = `<div class="mono">${fmt(t, 2)}</div>`;
  if (seg) {
    const spk = state.data.speakerByName[seg.speaker];
    html = `<div><b class="mono">#${segNum(seg)}</b> <b style="color:${spk.color}">${spk.label}</b> <span class="mono">${fmt(seg.start)}–${fmt(seg.end)} · ${seg.duration.toFixed(1)}s</span></div>`
      + `<div class="np"></div><div class="mono" style="opacity:.7">click to play</div>`;
  } else {
    const lane = state.lanes.find((l) => py >= l.y && py < l.y + l.h);
    const i = Math.floor(t / state.data.curve_hop);
    if (lane?.kind === "speaker") {
      html += `<div>${lane.spk.label}: P(active) ${(lane.spk.probArr[i] / 255).toFixed(2)}</div>`
        + `<div>other voice (diarizen): ${(lane.spk.otherArr[i] / 255).toFixed(2)}</div>`;
    } else if (lane?.kind === "overlap") {
      html += `<div>P(≥2 speakers): ${(state.data.overlap.pMultiArr[i] / 255).toFixed(2)}</div>`;
    }
  }
  tooltip.innerHTML = html;
  if (seg) tooltip.querySelector(".np").textContent = seg.text || "";
  tooltip.hidden = false;
  const tw = tooltip.offsetWidth;
  tooltip.style.left = Math.min(px + 14, r.width - tw - 4) + "px";
  tooltip.style.top = py + 16 + "px";
}

function overviewSeek(e) {
  const r = overview.getBoundingClientRect();
  const t = ((e.clientX - r.left) / r.width) * state.data.duration;
  const span = state.view.t1 - state.view.t0;
  setView(t - span / 2, t + span / 2);
}
let ovDrag = false;
overview.addEventListener("mousedown", (e) => { if (state.data) { ovDrag = true; overviewSeek(e); } });
window.addEventListener("mousemove", (e) => { if (ovDrag) overviewSeek(e); });
window.addEventListener("mouseup", () => { ovDrag = false; });

$("play").onclick = togglePlay;
$("jump-form").onsubmit = (e) => {
  e.preventDefault();
  const input = $("jump");
  const ok = state.data && playNumber(parseInt(input.value.replace("#", ""), 10));
  input.classList.toggle("miss", !ok);
};
$("prev").onclick = () => stepSegment(-1);
$("next").onclick = () => stepSegment(1);
$("zoom-in").onclick = () => zoom(0.5, nowTime());
$("zoom-out").onclick = () => zoom(2, nowTime());
$("zoom-fit").onclick = () => setView(0, state.data.duration);
$("autoplay-next").onchange = (e) => { if (e.target.checked) $("stop-at-end").checked = true; };

window.addEventListener("keydown", (e) => {
  if (!state.data || e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  else if (e.key === "ArrowRight") { e.preventDefault(); stepSegment(1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); stepSegment(-1); }
  else if (e.key === "+" || e.key === "=") zoom(0.5, nowTime());
  else if (e.key === "-") zoom(2, nowTime());
  else if (e.key === "0") setView(0, state.data.duration);
});

window.addEventListener("resize", () => { state.dirty = true; });
audio.addEventListener("seeked", () => { state.dirty = true; });
loadRecordings();
requestAnimationFrame(tick);
