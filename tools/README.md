# tools

Optional web UIs and benchmarks on top of the pipeline's work, run and benchmark
directories. Nothing in the pipeline depends on them.

```bash
uv sync --extra tools
```

## segment_viewer

```bash
uv run python -m tools.segment_viewer --work dataset/preprocessed1                       # http://127.0.0.1:8765/
uv run python -m tools.segment_viewer --work dataset/preprocessed1 dataset/preprocessed2  # several at once
```

The dropdown groups recordings by work directory.

Every processed recording on its original timeline:

* **waveform**, tinted where segments were kept (speaker colour) and where overlap was detected (red);
* one **lane per speaker**: Nemotron's P(speaking) shaded, DiariZen's "someone else" activity as a red
  line, a red strip under speech that was rejected, and the kept segments as blocks (click to play);
* an **overlap lane**: Nemotron frames with two or more active speakers, and DiariZen's P(≥2 speakers);
* a segment list per speaker with transcript, peak detector scores and the exported clip.

By default a segment plays from its exported clip, so you hear exactly what goes into the
dataset; untick "play exported clip" to hear it in context. Scroll to zoom, drag to pan,
Space plays/pauses, ←/→ step through segments, `+`/`-`/`0` zoom. Re-running `select`
shows up on reload.

## voice_studio

```bash
uv run python -m tools.voice_studio --work dataset/preprocessed1 --out studio \
  --model base=k2-fsa/OmniVoice --model run1=models/run1 --bench bench/bench1   # http://127.0.0.1:7861/
```

`--model` takes a Hub id, a merged model folder or a packaged training run (merged into
`<out>/models/<name>` first). Reference voices come from the dev split of `--work`;
rounds and uploaded references are kept in `--out`.

* **Compare**: pick a held-out voice (marked *unseen* or *in training*), upload a file or
  record from the microphone; the reference is trimmed to ≤15 s at a pause and
  transcribed with IndicConformer (editable). Every model gets the same voice prompt and
  seed, so only the weights differ. **Blind test** shows "Sample A/B/C" until you pick
  the best; picks build a running preference tally.
* **Benchmark**: the `tts_bench` results with confidence intervals, per-group views,
  the paired head-to-head table and every test item playable.
* **History**: past rounds with their audio and votes.

All models stay on the GPU together and share one audio tokenizer (their Higgs-audio
tokenizers are identical), so each extra 0.6B model costs little memory.

## tts_bench

```bash
uv run python -m tools.tts_bench --work dataset/preprocessed1 --out bench/bench1 \
  --model base=k2-fsa/OmniVoice --model run1=models/run1
```

`--model` takes a Hub id, a merged model folder or a packaged training run, which is merged
into `<out>/models/<name>` first. The test set is built once from the dev split of
`--work` and saved as `<out>/testset.json`; later runs (and other models) reuse it, so
`--work` is only needed the first time.

```
bench/bench1/
  testset.json      reference clips, texts, real recordings
  models.json       the models of the last run, as given
  models/<name>/    merged packaged runs
  audio/<model>/    generated clips, <reference id>__<text id>.wav
  results.jsonl     per item: ASR transcript, CER/WER, SIM, UTMOS
  summary.json      means with 95% bootstrap intervals, per group, paired vs. base
```

Every model says the same texts (written sentences plus held-out transcripts) in the same
reference voices from the dev split, with one voice prompt and seed per item:

| metric | measures | model |
| --- | --- | --- |
| CER / WER | intelligibility: ASR of the output vs. the text | AI4Bharat IndicConformer 600M |
| SIM | voice cloning: speaker-embedding cosine, output vs. reference | WavLM-Large ECAPA-TDNN (SIM-o) |
| UTMOS | naturalness: predicted MOS (1–5) | UTMOS22-strong |

The real recordings of the held-out sentences are scored as "real speech". Results come
with 95% bootstrap intervals and paired differences against the base model. UTMOS was
trained on English listening tests: use it to compare models, not as an absolute MOS.

## overlap_bench

```bash
uv run python -m tools.overlap_bench RECORDING_ID --work dataset/preprocessed1 --seed 0
```

Builds a 15-minute synthetic session from a processed recording's own clean clips: turns
of the main speaker alternate with a second speaker, and ~120 short snippets of the
second speaker (0.15–2 s, 0 to −24 dB) are mixed *under* the main one. Both detection
models run on it, and each selection configuration reports how many snippets leak into
selected clips and how much untouched speech survives. The session and report are written
to `<work>/overlap_bench/<id>_seed<seed>/`.
