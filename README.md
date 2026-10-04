# omnivoice-nepali

Nepali TTS fine-tuning: long recordings with rough transcripts → **strictly single-speaker,
overlap-free** clips → LoRA fine-tuning of [OmniVoice](https://github.com/k2-fsa/OmniVoice)
on a single small GPU.

```bash
uv sync
uv run nepvoice run dataset/input --work dataset/preprocessed1          # data preparation
uv run nepvoice run --work dataset/preprocessed1 --run models/run1 \
  --config dataset/preprocessed1/config.toml --from splits             # training
uv run nepvoice merge models/run1 --out merged/run1                     # standalone model
```

Every step caches its output, so an interrupted run resumes and changed settings redo only
what they affect.

## Input

Recordings paired by name with a transcript; subfolders are kept in the output.

```
input/
  talk-01.webm     audio: anything ffmpeg reads
  talk-01.json3    transcript: .json3 / .srt / .vtt (timed) or .txt (whole file)
  talk-01.json     optional metadata, e.g. {"title": ...}
```

Timed transcripts let a long recording be cut into many clips (word timings may be rough);
a `.txt` file becomes one clip or is rejected. Recordings are identified by file name, which
must be unique; moving files keeps what was computed for them.

## Pipeline

`nepvoice run [INPUT] --work DIR [--run DIR]` runs these steps; `--from`, `--to`, `--only`
pick a part. Data preparation writes the work directory, training the run directory (only
with `--run`).

| step | | output |
| --- | --- | --- |
| `ingest` | register recordings and transcripts | `recordings/<sub>/<id>/source.json` |
| `diarize` | who speaks when ([Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)) | `diarization.npy` |
| `osd` | overlapped speech ([DiariZen](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2)) | `osd.npz` |
| `select` | overlap-free single-speaker segments with their words | `segments.jsonl` |
| `export` | clips cut from the source, 24 kHz, peak-normalised | `clips/*.flac` |
| `embed` | speaker embedding per clip ([ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)) | `speaker_emb.npz` |
| `speakers` | the same person across recordings, voice class | `speakers.json` |
| `manifest` | OmniVoice manifests, whole recordings held out as dev | `dataset/{train,dev}.jsonl` |
| `splits` | female voices in their own, oversampled manifest | `scratch/manifests/` |
| `tokenize` | audio tokens | `scratch/tokens/` |
| `train` | LoRA fine-tuning, resumable | `scratch/exp/checkpoint-*` |
| `evaluate` | dev loss of base and every checkpoint | `scratch/exp/eval.json` |
| `package` | keep the adapters, delete `scratch/` | `*.safetensors` |

```
<work>/                          <run>/
  config.toml                      config.toml                 data prep + training settings
  recordings/<sub>/<id>/           adapter_config.json
  speakers.json, voiceprints.npz   adapter_model.safetensors   final adapter
  dataset/train.jsonl, dev.jsonl   checkpoint-<step>.safetensors
  failures.jsonl                   eval.json                   when evaluated
```

Source audio is not copied: a recording whose audio changes (size or modification time) is
processed again; a changed transcript keeps the model outputs. A packaged run is final and
a standard PEFT adapter directory; `nepvoice merge RUN --out DIR [--checkpoint checkpoint-N]`
makes a standalone OmniVoice model from it.

## Configuration

Defaults are documented in [`configs/default.toml`](configs/default.toml). Override with
`--config FILE.toml` (repeatable, only the changed keys) and `--set section.key=value`;
unknown keys are errors. `nepvoice config ...` prints the resolved configuration, and
[`configs/smoke.toml`](configs/smoke.toml) is a few-minute end-to-end check:

```bash
uv run nepvoice run input/ --work work-smoke --run run-smoke --config configs/smoke.toml
```

## Data preparation

**Single speaker.** A frame is foreign to target speaker *k* when any of these exceeds its
threshold (default 0.03): another Nemotron speaker, DiariZen's P(≥2 speakers), or a DiariZen
local speaker not matched to *k*. Foreign frames are padded by 0.3 s and clips are cut only
at real pauses or gaps in *k*'s activity (`cut_mode = "speaker_gap"`: only at gaps, dropping
any utterance an interjection touches). On a synthetic test (`tools/overlap_bench`), no
second voice of 0.5 s or longer leaked at any level.

**Levels.** Clips are peak-normalised to 0.9 and OmniVoice's tokenizer does the same, as in
upstream OmniVoice training. Alternatively `export.loudness_normalize = true` scales clips to
`loudness_lufs` (true peak ≤ `peak_dbfs`); set `train.peak_normalize = false` to train on
those levels, since OmniVoice scales its output to the reference voice's level.
`select.trim_breaths`, `export.fade_ms` and `export.pad_ms` trim edge inhales, fade and pad.

**Speakers.** Each (recording, speaker) gets a voiceprint; voiceprints with ≥ 10 s of speech
are clustered across recordings (cosine ≥ 0.8; the same person scored 0.93–0.97, different
people ≤ 0.71) into stable ids `S0001`, …. A wav2vec2 classifier estimates P(female) (pitch
alone misclassifies many male speakers). `manifest.max_hours_per_speaker` caps a person
round-robin across recordings; `manifest.only_speakers` keeps chosen speakers of a recording.

## Training

OmniVoice's LoRA recipe (rank 16 on all attention and MLP projections, audio embeddings and
heads fully trained), with 2048 tokens × 4 accumulation steps instead of 8192 × 1, 16 instead
of 64 clips per micro-batch, and 3000 steps. Gradient checkpointing (on by default) trades a
little compute for much less memory. Female voices are repeated `train.female_repeat` times
per pass. Re-running continues from the newest checkpoint; a run never resumes with
different settings or data. `train.final_checkpoint` (`last`, `best` by dev loss, or
`checkpoint-N`) becomes `adapter_model.safetensors`. Windows works: DataLoader workers get a
picklable `length_fn`, and shard paths are written as `file:C:/...`.

## Tools

Optional (`uv sync --extra tools`), see [`tools/`](tools/README.md): `segment_viewer`
(recordings on their timeline with detector signals and kept speech), `voice_studio` (blind
A/B listening), `tts_bench` (CER/WER, speaker similarity and UTMOS on held-out recordings),
`overlap_bench` (synthetic leak test).

## Development

```bash
uv sync --all-extras
uv run pytest
uv run --with ruff ruff check src tools tests
```

DiariZen's weights are CC BY-NC 4.0; IndicConformer (tools) is gated on Hugging Face.
