// Text side of OmniVoice inference, ported from omnivoice (models/omnivoice.py, utils/text.py,
// utils/duration.py): prompt tokens, duration estimate and long-text chunking.

import { Tokenizer } from "./vendor/tokenizers.min.mjs";

export const LANGUAGE = "npi"; // OmniVoice's id for Nepali

const NONVERBAL = /\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|surprise-yo|dissatisfaction-hnn)\]/g;

export class TextTokenizer {
  static async load(fetchJson, path) {
    const dir = path.replace(/[^/]+$/, "");
    const [json, config] = await Promise.all([fetchJson(path), fetchJson(dir + "tokenizer_config.json")]);
    return new TextTokenizer(new Tokenizer(json, config));
  }

  constructor(tok) {
    this.tok = tok;
  }

  ids(text) {
    return this.tok.encode(text, { add_special_tokens: false }).ids;
  }

  // <|denoise|> (with a reference) + language + instruct
  style(hasRef, instruct, denoise = true) {
    let s = denoise && hasRef ? "<|denoise|>" : "";
    s += `<|lang_start|>${LANGUAGE}<|lang_end|>`;
    s += `<|instruct_start|>${instruct || "None"}<|instruct_end|>`;
    return this.ids(s);
  }

  // <|text_start|>{ref text + text}<|text_end|>, non-verbal tags tokenized on their own
  text(text, refText) {
    const wrapped = `<|text_start|>${combineText(text, refText)}<|text_end|>`;
    const out = [];
    let last = 0;
    for (const m of wrapped.matchAll(NONVERBAL)) {
      if (m.index > last) out.push(...this.ids(wrapped.slice(last, m.index)));
      out.push(...this.ids(m[0]));
      last = m.index + m[0].length;
    }
    if (last < wrapped.length) out.push(...this.ids(wrapped.slice(last)));
    return out;
  }
}

export function combineText(text, refText) {
  let s = refText ? refText.trim() + " " + text.trim() : text.trim();
  s = s.replace(/[\r\n]+/g, "");
  s = s.replace(/（/g, "(").replace(/）/g, ")");
  s = s.replace(/[ \t]+/g, " ");
  s = s.replace(/(?<=[一-鿿])\s+|\s+(?=[一-鿿])/g, "");
  return s;
}

// --------------------------------------------------------------------------- duration

const WEIGHTS = {
  cjk: 3.0, hangul: 2.5, kana: 2.2, ethiopic: 3.0, yi: 3.0, indic: 1.8, thai_lao: 1.5, khmer_myanmar: 1.8,
  arabic: 1.5, hebrew: 1.5, latin: 1.0, cyrillic: 1.0, greek: 1.0, armenian: 1.0, georgian: 1.0,
  punctuation: 0.5, space: 0.2, digit: 3.5, mark: 0.0, default: 1.0,
};
// [last code point, script], searched in order
const RANGES = [
  [0x02af, "latin"], [0x03ff, "greek"], [0x052f, "cyrillic"], [0x058f, "armenian"], [0x05ff, "hebrew"], [0x077f, "arabic"],
  [0x089f, "arabic"], [0x08ff, "arabic"], [0x097f, "indic"], [0x09ff, "indic"], [0x0a7f, "indic"], [0x0aff, "indic"],
  [0x0b7f, "indic"], [0x0bff, "indic"], [0x0c7f, "indic"], [0x0cff, "indic"], [0x0d7f, "indic"], [0x0dff, "indic"],
  [0x0eff, "thai_lao"], [0x0fff, "indic"], [0x109f, "khmer_myanmar"], [0x10ff, "georgian"], [0x11ff, "hangul"], [0x137f, "ethiopic"],
  [0x139f, "ethiopic"], [0x13ff, "default"], [0x167f, "default"], [0x169f, "default"], [0x16ff, "default"], [0x171f, "default"],
  [0x173f, "default"], [0x175f, "default"], [0x177f, "default"], [0x17ff, "khmer_myanmar"], [0x18af, "default"], [0x18ff, "default"],
  [0x194f, "indic"], [0x19df, "indic"], [0x19ff, "khmer_myanmar"], [0x1a1f, "indic"], [0x1aaf, "indic"], [0x1b7f, "indic"],
  [0x1bbf, "indic"], [0x1bff, "indic"], [0x1c4f, "indic"], [0x1c7f, "indic"], [0x1c8f, "cyrillic"], [0x1cbf, "georgian"],
  [0x1ccf, "indic"], [0x1cff, "indic"], [0x1d7f, "latin"], [0x1dbf, "latin"], [0x1dff, "default"], [0x1eff, "latin"],
  [0x309f, "kana"], [0x30ff, "kana"], [0x312f, "cjk"], [0x318f, "hangul"], [0x9fff, "cjk"], [0xa4cf, "yi"],
  [0xa4ff, "default"], [0xa63f, "default"], [0xa69f, "cyrillic"], [0xa6ff, "default"], [0xa7ff, "latin"], [0xa82f, "indic"],
  [0xa87f, "default"], [0xa8df, "indic"], [0xa8ff, "indic"], [0xa92f, "indic"], [0xa95f, "indic"], [0xa97f, "hangul"],
  [0xa9df, "indic"], [0xa9ff, "khmer_myanmar"], [0xaa5f, "indic"], [0xaa7f, "khmer_myanmar"], [0xaadf, "indic"], [0xaaff, "indic"],
  [0xab2f, "ethiopic"], [0xab6f, "latin"], [0xabbf, "default"], [0xabff, "indic"], [0xd7af, "hangul"], [0xfaff, "cjk"],
  [0xfdff, "arabic"], [0xfe6f, "default"], [0xfeff, "arabic"], [0xffef, "latin"],
];

function charWeight(ch) {
  const code = ch.codePointAt(0);
  if ((code >= 65 && code <= 90) || (code >= 97 && code <= 122)) return WEIGHTS.latin;
  if (code === 32) return WEIGHTS.space;
  if (code === 0x0640) return WEIGHTS.mark;
  if (/\p{M}/u.test(ch)) return WEIGHTS.mark;
  if (/[\p{P}\p{S}]/u.test(ch)) return WEIGHTS.punctuation;
  if (/\p{Z}/u.test(ch)) return WEIGHTS.space;
  if (/\p{N}/u.test(ch)) return WEIGHTS.digit;
  const r = RANGES.find(([end]) => code <= end);
  if (r) return WEIGHTS[r[1]];
  return code > 0x20000 ? WEIGHTS.cjk : WEIGHTS.default;
}

const totalWeight = (s) => [...s].reduce((a, c) => a + charWeight(c), 0);

// Audio frames (25 per second) for `text`, scaled from a reference's text and frame count
// (RuleDurationEstimator with OmniVoice's defaults; without a reference, "Nice to meet you." = 25).
export function estimateFrames(text, refText, refFrames, speed = 1) {
  if (!refText || !refFrames) { refText = "Nice to meet you."; refFrames = 25; }
  const rw = totalWeight(refText);
  let est = rw ? totalWeight(text) / (rw / refFrames) : 0;
  if (est < 50) est = 50 * (est / 50) ** (1 / 3);
  if (speed > 0 && speed !== 1) est /= speed;
  return Math.max(1, Math.floor(est));
}

// --------------------------------------------------------------------------- chunking

// The danda (। ॥) ends Nepali sentences; upstream splits only on Latin/CJK punctuation.
const SPLIT = new Set(".,;:!?。，；：！？।॥");
const CLOSING = new Set("\"'“”‘’）]》>」】");
const END = new Set([";", ":", ",", ".", "!", "?", "…", ")", "]", "}", '"', "'", "“", "”", "‘", "’", "；", "：", "，", "。", "！", "？", "、", "）", "】", "।", "॥"]);
const ABBR = new Set(["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "No.", "vs.", "e.g.", "i.e."]);

// Split text into chunks of about chunkLen characters at punctuation (chunk_text_punctuation).
export function chunkText(text, chunkLen, minLen = 3) {
  const sentences = [];
  let cur = [];
  for (const ch of [...text]) {
    if (!cur.length && sentences.length && (SPLIT.has(ch) || CLOSING.has(ch))) {
      sentences.at(-1).push(ch);
      continue;
    }
    cur.push(ch);
    if (SPLIT.has(ch)) {
      let abbr = false;
      if (ch === ".") {
        const w = cur.join("").trim().split(/\s+/).at(-1);
        abbr = ABBR.has(w);
      }
      if (!abbr) { sentences.push(cur); cur = []; }
    }
  }
  if (cur.length) sentences.push(cur);
  const merged = [];
  let chunk = [];
  for (const s of sentences) {
    if (chunk.length + s.length <= chunkLen) chunk.push(...s);
    else {
      if (chunk.length) merged.push(chunk);
      chunk = [...s];
    }
  }
  if (chunk.length) merged.push(chunk);
  const out = [];
  const firstShort = merged.length > 0 && merged[0].length < minLen;
  merged.forEach((c, i) => {
    if (i === 1 && firstShort) out.at(-1).push(...c);
    else if (c.length >= minLen || !out.length) out.push(c);
    else out.at(-1).push(...c);
  });
  return out.map((c) => c.join("").trim()).filter(Boolean);
}

export function addPunctuation(text) {
  const t = text.trim();
  if (!t) return t;
  return END.has([...t].at(-1)) ? t : t + "।";
}
