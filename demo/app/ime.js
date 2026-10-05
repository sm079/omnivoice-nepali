// Romanized typing for the Nepali text boxes: a word typed in Latin letters turns into Devanagari
// when it's finished (space, Enter, leaving the box), like a phone keyboard's transliteration.
// While a word is being typed, its ranked readings show under the box: Alt+1–6 or a click picks
// one, Esc keeps the Latin letters, Backspace right after a conversion brings the Latin back.

import { Lexicon, Transliterator } from "./translit.js";

const store = {
  get(k, d) { try { const v = localStorage.getItem("nepali-voice." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("nepali-voice." + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

// romanized text still waiting to be converted, at the end of the text before the caret: ASCII
// without spaces, plus {kept as typed} which may contain spaces
const PENDING = /(?:\{[^{}\n]*\}|[!-z|~])+$/;
const OPEN_BRACE = /\{[^{}\n]*$/;
// the Devanagari word right before a position, for ranking in context
const PREV_WORD = /([ऀ-ॣॱ-ॿ]+)[ \t]*$/;
const HAS_LATIN = /[A-Za-z0-9.|~]/;
const LIMIT = 6;

let enabled = store.get("romanized", true);
let picks = store.get("romanPicks", {}); // lowercase romanized word -> chosen Devanagari
let tr = new Transliterator(); // rules only until the word list is in
let loading = null;
const toggles = [];
const boxes = [];

function loadLexicon(baseUrl) {
  loading ??= (async () => {
    const get = async (name) => {
      const res = await fetch(new URL("translit/" + name, baseUrl));
      if (!res.ok) throw new Error(`${name}: ${res.status}`);
      return res.text();
    };
    const [lex, pairs] = await Promise.all([get("lexicon.txt"), get("bigrams.txt").catch(() => "")]);
    tr = new Transliterator(await Lexicon.parse(lex, pairs));
    renderToggles();
  })().catch((e) => console.warn("romanized typing: no word list, using spelling rules only", e));
  return loading;
}

function renderToggles() {
  const words = tr.lexicon ? `, checked against ${Math.round(tr.lexicon.freqs.size / 1000)}k common words` : "";
  for (const b of toggles) {
    b.setAttribute("aria-pressed", enabled);
    b.title = enabled
      ? `Romanized typing is on: "mero naam" becomes "मेरो नाम" as you type${words}. Alt+1–6 picks a suggestion, Esc keeps the Latin letters, {braces} keep text as typed.`
      : "Romanized typing is off. Click to type Nepali in Latin letters.";
  }
}

function setEnabled(on) {
  enabled = on;
  store.set("romanized", on);
  renderToggles();
  for (const b of boxes) {
    if (on) loadLexicon(b.baseUrl);
    b.update();
  }
}

// Attach romanized typing to a textarea. toggle: an optional on/off button (shared state).
// baseUrl: where translit/lexicon.txt and translit/bigrams.txt live.
export function romanizedInput(ta, { toggle, baseUrl }) {
  if (toggle) {
    toggles.push(toggle);
    toggle.onclick = () => { setEnabled(!enabled); ta.focus(); };
    renderToggles();
  }

  const wrap = document.createElement("div");
  wrap.className = "ime";
  ta.replaceWith(wrap);
  const bar = document.createElement("div");
  bar.className = "ime-bar";
  bar.hidden = true;
  wrap.append(ta, bar);

  let pending = null; // { start, end, src, word, options, prev } at the caret
  let skip = null; // { start, src }: Esc'd, leave as typed
  let last = null; // { start, out, src, tail }: the latest conversion, for Backspace
  let replacing = false;

  // Write text over [start, end) keeping the browser's undo history where it can.
  function replace(start, end, text, caret) {
    replacing = true;
    ta.focus();
    ta.setSelectionRange(start, end);
    if (!document.execCommand?.("insertText", false, text)) {
      ta.setRangeText(text, start, end, "end");
      ta.dispatchEvent(new Event("input", { bubbles: true }));
    }
    ta.setSelectionRange(caret, caret);
    replacing = false;
  }

  // the pending romanized stretch ending at `end`
  function pendingAt(end) {
    const before = ta.value.slice(0, end);
    if (OPEN_BRACE.test(before)) return null;
    const m = before.match(PENDING);
    if (!m || !HAS_LATIN.test(m[0])) return null;
    const start = end - m[0].length;
    if (skip && skip.start === start && m[0].startsWith(skip.src)) return null;
    const prev = ta.value.slice(0, start).match(PREV_WORD)?.[1] ?? null;
    return { start, end, src: m[0], prev };
  }

  function convert(p, fixed = {}) {
    return tr.tokens(p.src, { prev: p.prev, fixed: { ...picks, ...fixed } }).map((t) => t.out).join("");
  }

  // convert the stretch ending at `end`; tail: what was typed after it (the space or newline)
  function commit(end, tail = "") {
    const p = pendingAt(end);
    if (!p) return false;
    const out = convert(p);
    if (out === p.src) return false;
    replace(p.start, p.end, out, p.start + out.length + tail.length);
    last = { start: p.start, out, src: p.src, tail };
    return true;
  }

  // ranked readings of the last word in the pending stretch at the caret
  function update() {
    pending = null;
    if (enabled && document.activeElement === ta && ta.selectionStart === ta.selectionEnd) {
      const p = pendingAt(ta.selectionEnd);
      const toks = p ? tr.tokens(p.src, { prev: p.prev, fixed: picks }) : [];
      const i = toks.findLastIndex((t) => t.kind === "word");
      if (i >= 0) {
        const prev = toks.slice(0, i).findLast((t) => t.kind === "word")?.out ?? p.prev;
        const word = toks[i].src;
        const options = tr.candidates(word, { limit: LIMIT, prev });
        const chosen = toks[i].out;
        if (!options.includes(chosen)) options.unshift(chosen);
        pending = { ...p, word, chosen, options: options.slice(0, LIMIT) };
      }
    }
    render();
  }

  function render() {
    bar.hidden = !pending;
    wrap.classList.toggle("open", !!pending);
    if (!pending) return;
    bar.replaceChildren();
    const src = document.createElement("span");
    src.className = "ime-src";
    src.textContent = pending.word;
    const opts = document.createElement("div");
    opts.className = "ime-opts";
    bar.append(src, opts);
    pending.options.forEach((o, k) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "ime-opt";
      b.lang = "ne";
      b.setAttribute("aria-pressed", o === pending.chosen);
      b.title = `Alt+${k + 1}`;
      b.innerHTML = `<small>${k + 1}</small>`;
      b.append(o);
      b.onclick = () => pick(o);
      opts.append(b);
    });
    const keep = document.createElement("button");
    keep.type = "button";
    keep.className = "ime-keep";
    keep.textContent = "Esc";
    keep.title = "Keep these Latin letters as typed (Esc)";
    keep.onclick = keepLatin;
    bar.append(keep);
  }

  function pick(option) {
    if (!pending) return;
    const key = pending.word.toLowerCase();
    picks[key] = option;
    store.set("romanPicks", picks);
    const p = pending;
    const out = convert(p, { [key]: option });
    replace(p.start, p.end, out, p.start + out.length);
    last = null;
    update();
  }

  function keepLatin() {
    if (!pending) return;
    skip = { start: pending.start, src: pending.src };
    ta.focus();
    update();
  }

  ta.addEventListener("keydown", (e) => {
    if (!enabled) return;
    const digit = /^Digit([1-9])$/.exec(e.code);
    if (e.altKey && !e.ctrlKey && !e.metaKey && digit && pending?.options[digit[1] - 1]) {
      e.preventDefault();
      pick(pending.options[digit[1] - 1]);
    } else if (e.key === "Escape" && pending) {
      e.preventDefault(); // also keeps a surrounding <dialog> open
      e.stopPropagation();
      keepLatin();
    } else if (e.key === "Backspace" && last && !e.ctrlKey && !e.altKey && ta.selectionStart === ta.selectionEnd) {
      const { start, out, src, tail } = last;
      const end = start + out.length + tail.length;
      if (ta.selectionEnd === end && ta.value.slice(start, end) === out + tail) {
        e.preventDefault();
        replace(start, end, src, start + src.length);
        last = null;
        update();
      }
    } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      commit(ta.selectionEnd); // before the page's Ctrl+Enter shortcut reads the text
    }
  });

  ta.addEventListener("input", (e) => {
    if (replacing || e.isComposing) return;
    last = null;
    if (enabled && /^insert(Text|LineBreak|Paragraph|CompositionText)$/.test(e.inputType || "")) {
      const end = ta.selectionEnd;
      const ch = ta.value[end - 1];
      if (ch && /\s/.test(ch) && ta.selectionStart === end) {
        commit(end - 1, ch);
        skip = null;
      }
    }
    update();
  });
  for (const ev of ["keyup", "click", "focus"]) ta.addEventListener(ev, () => { if (!replacing) update(); });
  // a word left unfinished when the box loses focus is converted too
  ta.addEventListener("blur", () => {
    if (enabled && !replacing && ta.selectionStart === ta.selectionEnd && pendingAt(ta.selectionEnd)) {
      const p = pendingAt(ta.selectionEnd);
      const out = convert(p);
      if (out !== p.src) {
        ta.setRangeText(out, p.start, p.end, "end");
        ta.dispatchEvent(new Event("input", { bubbles: true }));
      }
    }
    pending = skip = last = null;
    render();
  });
  bar.addEventListener("mousedown", (e) => e.preventDefault()); // keep focus in the box

  boxes.push({ update, baseUrl });
  if (enabled) loadLexicon(baseUrl);
}
