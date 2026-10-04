// Audio post-processing, ported from omnivoice.utils.audio (which works on pydub segments):
// silence trimming, fades, chunk cross-fades, plus WAV encoding for downloads.
// Mono Float32Array signals; positions in pydub's units (milliseconds) where it matters.

const MAX_AMP = 32768;

// pydub's view of a signal: 16-bit samples, millisecond slicing
class Seg {
  constructor(x, sr) {
    this.x = x;
    this.sr = sr;
    this.len = Math.round((1000 * x.length) / sr);
  }
  frame(ms) {
    return Math.floor((Math.min(ms, this.len) * this.sr) / 1000);
  }
  // audioop.rms of the int16 samples in [a, b) ms
  rms(a, b) {
    const s = this.frame(a), e = this.frame(b);
    if (e <= s) return 0;
    let sum = 0;
    for (let i = s; i < e; i++) {
      const v = Math.max(-32768, Math.min(32767, Math.trunc(this.x[i] * 32768)));
      sum += v * v;
    }
    return Math.floor(Math.sqrt(sum / (e - s)));
  }
  dbfs(a, b) {
    const r = this.rms(a, b);
    return r === 0 ? -Infinity : 20 * Math.log10(r / MAX_AMP);
  }
  slice(a, b) {
    return this.x.subarray(this.frame(a), this.frame(b));
  }
}

function detectSilence(seg, minLen, thresh, step) {
  if (seg.len < minLen) return [];
  const abs = 10 ** (thresh / 20) * MAX_AMP;
  const last = seg.len - minLen;
  const starts = [];
  for (let i = 0; i <= last; i += step) starts.push(i);
  if (last % step) starts.push(last);
  const silent = starts.filter((i) => seg.rms(i, i + minLen) <= abs);
  if (!silent.length) return [];
  const ranges = [];
  let prev = silent.shift();
  let cur = prev;
  for (const s of silent) {
    const continuous = s === prev + step;
    const gap = s > prev + minLen;
    if (!continuous && gap) {
      ranges.push([cur, prev + minLen]);
      cur = s;
    }
    prev = s;
  }
  ranges.push([cur, prev + minLen]);
  return ranges;
}

function detectNonsilent(seg, minLen, thresh, step) {
  const silent = detectSilence(seg, minLen, thresh, step);
  if (!silent.length) return [[0, seg.len]];
  if (silent[0][0] === 0 && silent[0][1] === seg.len) return [];
  const out = [];
  let prevEnd = 0;
  let end = 0;
  for (const [s, e] of silent) {
    out.push([prevEnd, s]);
    prevEnd = e;
    end = e;
  }
  if (end !== seg.len) out.push([prevEnd, seg.len]);
  if (out[0][0] === 0 && out[0][1] === 0) out.shift();
  return out;
}

function leadingSilence(seg, thresh, chunk = 10) {
  let t = 0;
  while (seg.dbfs(t, t + chunk) < thresh && t < seg.len) t += chunk;
  return Math.min(t, seg.len);
}

const concat = (parts) => {
  const out = new Float32Array(parts.reduce((a, p) => a + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
};

// Shorten middle silences longer than midSil ms to midSil and trim the edges, keeping leadSil /
// trailSil ms (omnivoice.utils.audio.remove_silence, threshold -50 dBFS).
export function removeSilence(x, sr, { midSil = 300, leadSil = 100, trailSil = 300 } = {}) {
  let y = x;
  if (midSil > 0) {
    const seg = new Seg(y, sr);
    const ranges = detectNonsilent(seg, midSil, -50, 10).map(([s, e]) => [s - midSil, e + midSil]);
    for (let i = 0; i + 1 < ranges.length; i++) {
      if (ranges[i + 1][0] < ranges[i][1]) {
        ranges[i][1] = Math.floor((ranges[i][1] + ranges[i + 1][0]) / 2);
        ranges[i + 1][0] = ranges[i][1];
      }
    }
    y = concat(ranges.map(([s, e]) => seg.slice(Math.max(s, 0), Math.min(e, seg.len))));
  }
  let seg = new Seg(y, sr);
  const start = Math.max(0, leadingSilence(seg, -50) - leadSil);
  y = seg.slice(start, seg.len);
  const rev = y.slice().reverse();
  seg = new Seg(rev, sr);
  const tail = Math.max(0, leadingSilence(seg, -50) - trailSil);
  return seg.slice(tail, seg.len).slice().reverse();
}

// Trim audio longer than `threshold` s to at most maxDur s, cutting at a pause
// (omnivoice.utils.audio.trim_long_audio).
export function trimLongAudio(x, sr, { maxDur = 15, minDur = 3, threshold = 20 } = {}) {
  if (x.length / sr <= threshold) return x;
  const seg = new Seg(x, sr);
  const ns = detectNonsilent(seg, 100, -40, 10);
  if (!ns.length) return x;
  const maxMs = maxDur * 1000, minMs = minDur * 1000;
  let best = 0;
  for (const [s, e] of ns) {
    if (s > best && s <= maxMs) best = s;
    if (e > maxMs) break;
  }
  if (best < minMs) best = Math.min(maxMs, seg.len);
  return seg.slice(0, best).slice();
}

const linspace = (a, b, n) => Float32Array.from({ length: n }, (_, i) => (n === 1 ? a : a + ((b - a) * i) / (n - 1)));

export function fadeAndPad(x, sr, { pad = 0.1, fade = 0.1 } = {}) {
  if (!x.length) return x;
  const y = x.slice();
  const k = Math.min(Math.floor(fade * sr), Math.floor(y.length / 2));
  if (k > 0) {
    const w = linspace(0, 1, k);
    for (let i = 0; i < k; i++) { y[i] *= w[i]; y[y.length - k + i] *= w[k - 1 - i]; }
  }
  const p = Math.floor(pad * sr);
  if (p <= 0) return y;
  const out = new Float32Array(y.length + 2 * p);
  out.set(y, p);
  return out;
}

// Concatenate chunks with a short gap and cross-fades (omnivoice.utils.audio.cross_fade_chunks).
export function crossFade(chunks, sr, silence = 0.3) {
  if (chunks.length === 1) return chunks[0];
  const fadeN = Math.floor(Math.floor(silence * sr) / 3);
  let merged = chunks[0].slice();
  for (const c of chunks.slice(1)) {
    const fo = Math.min(fadeN, merged.length);
    const wo = linspace(1, 0, fo);
    for (let i = 0; i < fo; i++) merged[merged.length - fo + i] *= wo[i];
    const fin = c.slice();
    const fi = Math.min(fadeN, fin.length);
    const wi = linspace(0, 1, fi);
    for (let i = 0; i < fi; i++) fin[i] *= wi[i];
    merged = concat([merged, new Float32Array(fadeN), fin]);
  }
  return merged;
}

export const rms = (x) => {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return x.length ? Math.sqrt(s / x.length) : 0;
};

export function peak(x) {
  let m = 0;
  for (let i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i]));
  return m;
}

// 16-bit PCM WAV
export function encodeWav(x, sr) {
  const buf = new ArrayBuffer(44 + x.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + x.length * 2, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, x.length * 2, true);
  for (let i = 0; i < x.length; i++) v.setInt16(44 + i * 2, Math.max(-32768, Math.min(32767, Math.round(x[i] * 32767))), true);
  return new Blob([buf], { type: "audio/wav" });
}
