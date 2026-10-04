# Omnivoice Nepali Web Demo

Nepali text-to-speech with [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) running entirely in
the browser on WebGPU (hand-written WGSL kernels, no ONNX runtime). Three models:

| model | what it is |
|---|---|
| Base | OmniVoice as released |
| More natural v1 / v2 | the run1 / run2 LoRA adapters from [sm079/omnivoice-nepali-lora](https://huggingface.co/sm079/omnivoice-nepali-lora) |

The adapters are not shipped as merged models: the app downloads the PEFT adapter and merges
`W + (alpha/r)·B·A` into the backbone on the GPU while loading (int8 weights are dequantized in their
ConvRot-rotated space, merged and requantized per row). Switching models reloads the backbone from
the browser cache and merges the other adapter (a few seconds).

```bash
python demo/tools/build_assets.py --adapter run1=local/models/run1/adapter_model.safetensors \
  --adapter run2=local/models/run2/adapter_model.safetensors   # -> demo/models/
python demo/tools/build_voices.py                              # synthetic preset voices
uv run --with onnx python demo/tools/build_asr.py              # Nepali ASR for cloning (gated model)
python demo/tools/serve.py --port 8090                          # http://127.0.0.1:8090/
```

- **Downloads**: backbone bf16 (1.2 GB) or int8 + ConvRot (600 MB), codec decoder 47 MB, adapters 103 MB each,
  cached in OPFS with resumable downloads.
- **Engine**: Qwen3 backbone run bidirectionally with cond/uncond sequences packed into one batch, fused
  q/k/v and SwiGLU GEMMs, GQA flash attention, CFG scoring on the GPU; Higgs/DAC decoder as implicit-GEMM
  1D convs with Snake fused into the loads and transposed convs split into output phases.
- **Checked** against the upstream PyTorch model on the same weights (`tools/reference.py` +
  `tools/check.html`): hidden states 7e-7 (int8), 4e-6 (int8 + run2 merge), 7e-7 (bf16 + run1), guided
  tokens 479–480/480 agree; codec 1.6e-3 vs. the fp32 original.
- **Speed** (laptop GPU, int8): one decoding step ~95 ms at 250 rows; 4 s of speech in ~5 s at 32 steps.
- **Voices**: synthetic presets made with OmniVoice voice design (best of 12 takes by ASR CER and UTMOS),
  voice description, a random voice, or **your own**: record or upload 3–15 s of speech, the browser
  transcribes it (editable) and turns it into a voice prompt, prepared as OmniVoice's
  `create_voice_clone_prompt` does (loudness, trimming at a pause, silence removal). Saved voices stay in
  the browser and work with all three models.
- **Cloning models** (downloaded only when first used): the Higgs codec encoder (HuBERT + DAC encoder +
  residual quantizer, 314 MB bf16; embeddings within 2e-3 of the original, 96–100% of the codes identical),
  and [IndicConformer 600M](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) with its
  CTC head restricted to Nepali (599 MB int8; same transcripts as the ONNX original in our checks). Of the
  Nepali ASR models compared in our dataset work, it was the most accurate (median CER 0.068 with CTC
  decoding vs 0.111 for the next Whisper fine-tune). It is MIT-licensed; downloading the original needs
  its terms accepted on Hugging Face.
- **Laptops with two GPUs**: on Windows the browser picks the GPU itself (it ignores WebGPU's
  `powerPreference`), often the integrated one, which was about 3× slower here. Choosing "High performance"
  for the browser under Windows Settings → System → Display → Graphics makes it use the dedicated GPU.
