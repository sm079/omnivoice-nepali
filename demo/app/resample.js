// Band-limited resampling, a port of torchaudio.functional.resample (sinc_interp_hann,
// lowpass_filter_width 6, rolloff 0.99), so references are prepared as OmniVoice prepares them.

const gcd = (a, b) => (b ? gcd(b, a % b) : a);

export function resample(x, from, to) {
  if (from === to) return x;
  const g = gcd(from, to);
  const orig = from / g, neu = to / g;
  const lw = 6;
  const base = Math.min(orig, neu) * 0.99;
  const width = Math.ceil((lw * orig) / base);
  const taps = 2 * width + orig;
  // kernels[p][j]: output phase p, input offset j
  const kernels = new Float32Array(neu * taps);
  for (let p = 0; p < neu; p++) {
    for (let j = 0; j < taps; j++) {
      let t = ((-p / neu) + (j - width) / orig) * base;
      t = Math.max(-lw, Math.min(lw, t));
      const w = Math.cos((t * Math.PI) / lw / 2) ** 2;
      const tp = t * Math.PI;
      kernels[p * taps + j] = (tp === 0 ? 1 : Math.sin(tp) / tp) * w * (base / orig);
    }
  }
  const n = x.length;
  const outLen = Math.ceil((neu * n) / orig);
  const y = new Float32Array(outLen);
  // padded input: width zeros in front; frame i starts at input i*orig - width
  for (let i = 0, o = 0; o < outLen; i++) {
    const start = i * orig - width;
    for (let p = 0; p < neu && o < outLen; p++, o++) {
      let s = 0;
      const k = p * taps;
      const j0 = Math.max(0, -start), j1 = Math.min(taps, n - start);
      for (let j = j0; j < j1; j++) s += kernels[k + j] * x[start + j];
      y[o] = s;
    }
  }
  return y;
}
