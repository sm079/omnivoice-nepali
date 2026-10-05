"""Word list for the demo's romanized-Nepali input (app/translit.js).

Packs the lexicon and word-pair counts built by nepali-romanized's scripts/build_lexicon.py
(Leipzig nep_news_2020, CC BY 4.0) into two plain-text files the browser parses quickly:

  translit/lexicon.txt   word<TAB>count, most frequent first
  translit/bigrams.txt   i<TAB>j<TAB>count, i and j being line numbers in lexicon.txt

  python demo/tools/build_translit.py --lexicon ../nepali-romanized/data/lexicon.tsv   # -> demo/models/translit/
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", required=True, help="lexicon.tsv (word<TAB>count)")
    ap.add_argument("--bigrams", help="bigrams.tsv (word<TAB>word<TAB>count); default: next to the lexicon")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "models", "translit"))
    args = ap.parse_args()
    bigrams = args.bigrams or os.path.join(os.path.dirname(args.lexicon), "bigrams.tsv")

    freqs: dict[str, int] = {}
    with open(args.lexicon, encoding="utf-8") as fh:
        for line in fh:
            word, _, count = line.rstrip("\n").partition("\t")
            if word:
                freqs[word] = int(count or 1)
    words = sorted(freqs, key=lambda w: (-freqs[w], w))
    index = {w: i for i, w in enumerate(words)}

    pairs = []
    if os.path.exists(bigrams):
        with open(bigrams, encoding="utf-8") as fh:
            for line in fh:
                fields = line.rstrip("\n").split("\t")
                if len(fields) == 3 and fields[0] in index and fields[1] in index:
                    pairs.append((index[fields[0]], index[fields[1]], int(fields[2])))
        pairs.sort()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "lexicon.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(f"{w}\t{freqs[w]}\n" for w in words)
    with open(os.path.join(args.out, "bigrams.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(f"{a}\t{b}\t{c}\n" for a, b, c in pairs)
    print(f"{len(words)} words, {len(pairs)} pairs -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
