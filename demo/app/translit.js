// Romanized Nepali -> Devanagari, ported from the nepali-romanized Python package (rules.py,
// translit.py, lexicon.py). Strict mode is fixed rules ("paanii" -> पानी); with a Lexicon, casual
// spelling is ranked by word and word-pair frequencies ("pani" -> पनि, "nepal" -> नेपाल).

// ------------------------------------------------------------------ rules

const HALANT = "्";

const CONSONANTS = {
  k: "क", q: "क", kh: "ख", g: "ग", gh: "घ", NG: "ङ",
  c: "च", ch: "च", chh: "छ", Ch: "छ", j: "ज", z: "ज",
  jh: "झ", NY: "ञ",
  T: "ट", Th: "ठ", D: "ड", Dh: "ढ", N: "ण",
  ".D": "ड़", ".Dh": "ढ़",
  t: "त", th: "थ", d: "द", dh: "ध", n: "न",
  p: "प", ph: "फ", f: "फ", b: "ब", bh: "भ", m: "म",
  y: "य", r: "र", l: "ल", w: "व", v: "व",
  sh: "श", Sh: "ष", S: "ष", s: "स", h: "ह",
  x: "क्ष", ksh: "क्ष", kSh: "क्ष",
  gy: "ज्ञ", gny: "ज्ञ", jny: "ज्ञ",
};
// romanization -> [independent form, dependent sign]
const VOWELS = {
  a: ["अ", ""],
  aa: ["आ", "ा"], A: ["आ", "ा"],
  i: ["इ", "ि"],
  ii: ["ई", "ी"], ee: ["ई", "ी"], I: ["ई", "ी"],
  u: ["उ", "ु"],
  uu: ["ऊ", "ू"], oo: ["ऊ", "ू"], U: ["ऊ", "ू"],
  R: ["ऋ", "ृ"],
  e: ["ए", "े"],
  ai: ["ऐ", "ै"], ei: ["ऐ", "ै"],
  o: ["ओ", "ो"],
  au: ["औ", "ौ"], ou: ["औ", "ौ"],
};
const MODIFIERS = { M: "ं", "~": "ँ", H: "ः" };
const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
const isKey = (k) => has(CONSONANTS, k) || has(VOWELS, k) || has(MODIFIERS, k);
const MAX_KEY_LEN = Math.max(...[CONSONANTS, VOWELS, MODIFIERS].flatMap((o) => Object.keys(o).map((k) => k.length)));
const DEVA_DIGITS = "०१२३४५६७८९";

const ESCAPE = /\{([^{}]*)\}/g;
// "." ends a sentence -> "।" unless it sits inside a number (3.5) or "..."
const FULL_STOP = /(?<![0-9.])\.(?![0-9.])/g;

// longest table key at text[i:], exact case first
function match(text, i) {
  for (const fold of [false, true]) {
    for (let size = Math.min(MAX_KEY_LEN, text.length - i); size > 0; size--) {
      let chunk = text.slice(i, i + size);
      if (fold) chunk = chunk.toLowerCase();
      if (isKey(chunk)) return [chunk, size];
    }
  }
  return [null, 1];
}

function convertPlain(text, digits, danda) {
  const out = [];
  let pending = false; // last emitted char is a consonant still carrying the inherent "a"
  let i = 0;
  while (i < text.length) {
    const [key, size] = match(text, i);
    i += size;
    if (key !== null && has(CONSONANTS, key)) {
      if (pending) out.push(HALANT);
      out.push(CONSONANTS[key]);
      pending = true;
    } else if (key !== null && has(VOWELS, key)) {
      const [independent, sign] = VOWELS[key];
      out.push(pending ? sign : independent);
      pending = false;
    } else if (key !== null) {
      out.push(MODIFIERS[key]);
      pending = false;
    } else {
      const ch = text[i - size];
      if (ch === "_") continue; // only breaks key matching; keeps conjunct state
      if (ch === "\\") {
        if (pending) out.push(HALANT);
      } else if (ch === "|") out.push("।");
      else out.push(ch);
      pending = false;
    }
  }
  let result = out.join("").replaceAll("।।", "॥");
  if (danda) result = result.replace(FULL_STOP, "।");
  if (digits) result = result.replace(/[0-9]/g, (d) => DEVA_DIGITS[d]);
  return result;
}

// Fixed-rule conversion; text inside {braces} is copied through unchanged.
export function transliterate(text, { digits = true, danda = true } = {}) {
  const parts = [];
  let last = 0;
  for (const m of text.matchAll(ESCAPE)) {
    parts.push(convertPlain(text.slice(last, m.index), digits, danda), m[1]);
    last = m.index + m[0].length;
  }
  parts.push(convertPlain(text.slice(last), digits, danda));
  return parts.join("");
}

// ------------------------------------------------------------------ Devanagari -> rough romanization

const R_CONS = {
  "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
  "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
  "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
  "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
  "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
  "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "w",
  "श": "sh", "ष": "sh", "स": "s", "ह": "h",
};
const R_MATRAS = {
  "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "ृ": "ri",
  "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॅ": "e", "ॉ": "o",
};
const R_VOWELS = {
  "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
  "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
};
// chandrabindu is often typed as "n" (aaunchhu = आउँछु); the lexicon also indexes words without it
const R_SIGNS = { "ं": "n", "ँ": "n", "ः": "h", "ऽ": "", "़": "" };
const NUKTA = "़";

// only good enough to build lookup keys
export function romanize(word) {
  const out = [];
  const n = word.length;
  let i = 0;
  while (i < n) {
    const ch = word[i];
    if (has(R_CONS, ch)) {
      let nxt = word[i + 1] ?? "";
      if (nxt === NUKTA && i + 2 < n) {
        i += 1;
        nxt = word[i + 1];
      }
      if (ch === "ज" && word.slice(i + 1, i + 3) === HALANT + "ञ") { // ज्ञ
        out.push("gy");
        i += 3;
        nxt = word[i] ?? "";
        if (has(R_MATRAS, nxt)) {
          out.push(R_MATRAS[nxt]);
          i += 1;
        } else if (nxt !== HALANT) out.push("a");
        continue;
      }
      out.push(R_CONS[ch]);
      if (has(R_MATRAS, nxt)) {
        out.push(R_MATRAS[nxt]);
        i += 1;
      } else if (nxt === HALANT) i += 1;
      else out.push("a");
    } else {
      out.push(R_VOWELS[ch] || R_SIGNS[ch] || "");
    }
    i += 1;
  }
  return out.join("");
}

// ------------------------------------------------------------------ lossy keys

const KEY_RULES = [
  [/chh|ch|c/g, "c"],
  [/ph/g, "f"],
  [/[vw]/g, "b"],
  [/z/g, "j"],
  [/q/g, "k"],
  [/x/g, "ks"],
  [/sh/g, "s"],
  [/ee|ii/g, "i"],
  [/oo|uu/g, "u"],
  [/ou/g, "au"],
  [/ei/g, "ai"],
  [/m(?=[pbf])/g, "n"],
  [/(.)\1+/g, "$1"],
];

export function phoneticKey(latin) {
  let key = latin.toLowerCase();
  for (const [re, repl] of KEY_RULES) key = key.replace(re, repl);
  if (key.length > 1 && key.endsWith("a")) key = key.slice(0, -1);
  return key;
}

// key without inner "a"s: catches schwa the writer did or didn't type
const looseKey = (key) => key.slice(0, 1) + key.slice(1).replaceAll("a", "");

// make a typed word's case-sensitive signs comparable to romanize()
const inputLatin = (word) =>
  word.replaceAll("R", "ri").replaceAll("M", "n").replaceAll("H", "h")
    .replaceAll("~", "").replaceAll("_", "").replaceAll("\\", "");

const LIGHT_RULES = [
  [/x/g, "ksh"],
  [/f/g, "ph"],
  [/[vw]/g, "b"],
  [/z/g, "j"],
  [/q/g, "k"],
  [/c(?!h)/g, "ch"],
  [/ee/g, "ii"],
  [/oo/g, "uu"],
];

// spelling-variant cleanup that keeps vowel length and aspiration
function light(latin) {
  latin = latin.toLowerCase();
  for (const [re, repl] of LIGHT_RULES) latin = latin.replace(re, repl);
  return latin;
}

// Casual romanization mostly *omits* information (long vowels, schwa, the h of sh/chh), so a
// candidate adding letters the user didn't type is cheap, one contradicting typed letters is not.
const INSERT_A_COST = 0.5;
const INSERT_COST = 1.0;
const DELETE_COST = 5.0;
const SUBSTITUTE_COST = 3.0;

// weighted edit distance from what was typed to a candidate's romanization
export function typingDistance(typed, cand) {
  let prev = [0];
  for (const cb of cand) prev.push(prev[prev.length - 1] + (cb === "a" ? INSERT_A_COST : INSERT_COST));
  for (const ca of typed) {
    const cur = [prev[0] + DELETE_COST];
    for (let j = 1; j <= cand.length; j++) {
      const cb = cand[j - 1];
      cur.push(Math.min(
        prev[j] + DELETE_COST,
        cur[j - 1] + (cb === "a" ? INSERT_A_COST : INSERT_COST),
        prev[j - 1] + (ca === cb ? 0 : SUBSTITUTE_COST),
      ));
    }
    prev = cur;
  }
  return prev[prev.length - 1];
}

// ------------------------------------------------------------------ lexicon + transliterator

// a romanized word; ".D"/".Dh" spell ड़/ढ़
const WORD = /(?:\.D|[A-Za-z~_\\])+/g;
// markers that mean "I'm spelling this exactly": skip the lexicon
const EXPLICIT = ["_", "\\", ".D"];
// pseudo-count added to observed and expected pair counts
const PAIR_PRIOR = 1.0;
// candidates per word kept for ranking in context
const BEAM = 8;

export class Lexicon {
  // words: Devanagari words; counts: their frequencies; pairs: Map "a\tb" -> count.
  // Use Lexicon.build(), which fills the lookup index without blocking the page for long.
  constructor(words, counts, pairs = new Map()) {
    this.freqs = new Map();
    let total = 0;
    words.forEach((w, i) => { this.freqs.set(w, counts[i]); total += counts[i]; });
    this.pairs = pairs;
    this.norm = total + this.freqs.size + 1;
    this.byKey = new Map();
    this.byLoose = new Map();
  }

  static async build(words, counts, pairs) {
    const lex = new Lexicon(words, counts, pairs);
    const add = (map, key, w) => {
      const list = map.get(key);
      if (!list) map.set(key, [w]);
      else if (!list.includes(w)) list.push(w);
    };
    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      const keys = new Set([phoneticKey(romanize(w)), phoneticKey(romanize(w.replaceAll("ँ", "")))]);
      for (const k of keys) {
        add(lex.byKey, k, w);
        add(lex.byLoose, looseKey(k), w);
      }
      if (i % 2000 === 1999) await new Promise((r) => setTimeout(r));
    }
    return lex;
  }

  // lexicon.txt: word<TAB>count per line; bigrams.txt: i<TAB>j<TAB>count with i, j line numbers
  // in lexicon.txt
  static async parse(lexiconText, bigramText = "") {
    const words = [], counts = [];
    for (const line of lexiconText.split("\n")) {
      const [w, c] = line.split("\t");
      if (w) { words.push(w); counts.push(parseInt(c, 10) || 1); }
    }
    const pairs = new Map();
    const lines = bigramText.split("\n");
    for (let i = 0; i < lines.length; i++) {
      const f = lines[i].split("\t");
      if (f.length === 3) pairs.set(words[+f[0]] + "\t" + words[+f[1]], +f[2]);
      if (i % 20000 === 19999) await new Promise((r) => setTimeout(r));
    }
    return Lexicon.build(words, counts, pairs);
  }

  lookup(key) {
    let found = this.byKey.get(key) || [];
    if (found.length < 3) found = found.concat((this.byLoose.get(looseKey(key)) || []).filter((w) => !found.includes(w)));
    return found;
  }

  // log P(word), adjusted by how much more (or less) often than chance (prev, word) was seen
  logp(word, prev = null) {
    const pWord = ((this.freqs.get(word) || 0) + 1) / this.norm;
    if (prev === null || !this.pairs.size) return Math.log(pWord);
    const pair = this.pairs.get(prev + "\t" + word) || 0;
    const expected = (this.freqs.get(prev) || 0) * pWord;
    return Math.log(pWord) + Math.log((pair + PAIR_PRIOR) / (expected + PAIR_PRIOR));
  }
}

export class Transliterator {
  constructor(lexicon = null, { digits = true, danda = true, distanceWeight = 1.0 } = {}) {
    this.lexicon = lexicon;
    this.digits = digits;
    this.danda = danda;
    this.distanceWeight = distanceWeight;
  }

  // [[candidate, typing penalty]] for one word, best context-free first
  options(word) {
    if (/^[A-Z][^A-Z]*[a-z][^A-Z]*$/.test(word)) word = word.toLowerCase(); // "Nepal": capitalisation, not "N" = ण
    const strict = transliterate(word, { digits: false, danda: false });
    if (!this.lexicon || EXPLICIT.some((m) => word.includes(m))) return [[strict, 0]];
    const typed = light(inputLatin(word));
    const options = new Set(this.lexicon.lookup(phoneticKey(inputLatin(word))));
    options.add(strict);
    const scored = [...options].map((c) => [c, this.distanceWeight * typingDistance(typed, light(romanize(c)))]);
    scored.sort((a, b) => (this.lexicon.logp(b[0]) - b[1]) - (this.lexicon.logp(a[0]) - a[1]));
    return scored.slice(0, BEAM);
  }

  // ranked Devanagari options for one romanized word; prev: the Devanagari word before it
  candidates(word, { limit = 6, prev = null } = {}) {
    const options = this.options(word);
    if (this.lexicon && prev) {
      const lex = this.lexicon;
      options.sort((a, b) => (lex.logp(b[0], prev) - b[1]) - (lex.logp(a[0], prev) - a[1]));
    }
    return options.slice(0, limit).map(([c]) => c);
  }

  // Convert a stretch of text word by word, each ranked after the word before it (punctuation and
  // other non-space text break the context). fixed: {lowercase romanized word: Devanagari} to force.
  // Returns [{ src, out, kind: "word" | "escape" | "other" }].
  tokens(text, { prev = null, fixed = {} } = {}) {
    const tokens = [];
    const other = (s) => {
      if (!s) return;
      tokens.push({ src: s, out: transliterate(s, { digits: this.digits, danda: this.danda }), kind: "other" });
      if (!/^\s+$/.test(s)) prev = null;
    };
    let last = 0;
    for (const esc of [...text.matchAll(ESCAPE), null]) {
      const plain = text.slice(last, esc ? esc.index : text.length);
      let pos = 0;
      for (const m of plain.matchAll(WORD)) {
        other(plain.slice(pos, m.index));
        const src = m[0];
        const out = fixed[src.toLowerCase()] ?? this.candidates(src, { limit: 1, prev })[0];
        tokens.push({ src, out, kind: "word" });
        prev = out;
        pos = m.index + src.length;
      }
      other(plain.slice(pos));
      if (esc) {
        tokens.push({ src: esc[0], out: esc[1], kind: "escape" });
        prev = null;
        last = esc.index + esc[0].length;
      }
    }
    return tokens;
  }

  transliterate(text, opts) {
    return this.tokens(text, opts).map((t) => t.out).join("");
  }
}
