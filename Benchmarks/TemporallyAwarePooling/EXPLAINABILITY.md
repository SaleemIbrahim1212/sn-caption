# Caption explainability: implementation and run commands

This document describes **decoder** and **transformer encoder** attention tooling. Training code in `src/` is unchanged; logic lives under **`analysis/`**.

---

## 1. Implemented files

| File | Role |
|------|------|
| `Benchmarks/TemporallyAwarePooling/analysis/attention_decode.py` | Greedy decode while collecting softmax weights over encoder time (single `DecoderRNN` or `DualModalDecoderRNN`). |
| `Benchmarks/TemporallyAwarePooling/analysis/explain_attention.py` | CLI: decoder cross-attention → JSON (and optional PNG). |
| `Benchmarks/TemporallyAwarePooling/analysis/encoder_self_attention.py` | Capture encoder self-attention via analysis-only runtime patch (see §2.2). |
| `Benchmarks/TemporallyAwarePooling/analysis/explain_encoder_self_attention.py` | CLI: encoder self-attention → JSON (and optional PNG). |

`SOS_TOKEN` / `EOS_TOKEN` in `attention_decode.py` match `dataset.py` so `torchtext` is not required to import that module alone. Running the full CLI still needs the same environment as caption training (**torch**, **torchtext**, **spaCy** for `SoccerNetCaptions`, etc.). **`matplotlib`** is optional; use it with `--plot`.

---

## 2. Two kinds of “attention” (interpretation)

### 2.1 Decoder cross-attention (what the scripts export)

The LSTM decoder uses a **query** from its hidden state, dot-products with **encoder outputs** over time, and **softmax** → one weight per encoder timestep per decode step.

- **Transformer video (and fused multimodal single decoder):** encoder sequence length **T > 1** → non-trivial heatmaps.
- **Baseline (`Video2Caption`):** pooled vector is repeated as a **single** key (**T = 1**). The script uses the same geometry as teacher forcing in training (`pooled.unsqueeze(1)`). Weights are **degenerate**; JSON includes a `meta.note` explaining this.
- **Dual LSTM (`--dual_lstm_decoder`):** two weight vectors per step → JSON fields `attention_audio` and `attention_video`, and a two-row PNG with `--plot`.

### 2.2 Transformer encoder self-attention (implemented in `analysis/`)

PyTorch’s `TransformerEncoderLayer` calls `MultiheadAttention` with **`need_weights=False`**, and a fused fast path can skip Python `forward` entirely—so a plain **forward hook** on `self_attn` usually sees **`None`** for weight tensors.

The analysis code instead:

1. Temporarily sets **`torch.backends.mha.set_fastpath_enabled(False)`** for the capture forward (restored afterward).
2. Replaces each layer’s **`_sa_block`** bound method with a wrapper that calls **`self_attn(..., need_weights=True, average_attn_weights=True)`** once per block, matching the original dropout path.

That only mutates **in-memory** layer instances while the script runs; **`src/transformer.py` is not edited.**

**NetVLAD baseline:** there is **no** sequence self-attention. To compare “pooling vs temporal mixing,” pair **encoder** self-attention heatmaps (transformer) with **decoder** or **input-gradient** views for the baseline (see §6), and state the architectural difference explicitly.

---

## 3. Outputs

Default directory: `models/<model_name>/explainability/` (under repo root).

### Decoder (`explain_attention.py`)

| Output | Description |
|--------|-------------|
| `attention_sample<index>.json` | Reference and generated caption, token ids, `attention` (or `attention_audio` / `attention_video`), `meta` (e.g. `encoder_seq_len`, `pool`, `dual_decoder`). |
| `attention_sample<index>.png` | If `--plot`: heatmap(s) of decode step × encoder index. |

### Encoder (`explain_encoder_self_attention.py`)

| Output | Description |
|--------|-------------|
| `encoder_self_attn_sample<index>.json` | `video` and/or `audio` → `layers`: list of `{ "layer": k, "attention": [[...]] }` with shape **T×T** (query × key positions), **heads averaged** per PyTorch default. |
| `encoder_self_attn_sample<index>.png` | If `--plot`: last-layer **T×T** heatmap per captured stream (side-by-side if both). |

---

## 4. Run commands (repository root)

Placeholders: **`DATA`**, **`MAP`**, **`FEAT`**, **`MODEL`** (same `model_name` as training), **`AUDIO`** (for audio/both/dual).

### 4.1 Transformer, video only (A1-style)

```text
python Benchmarks/TemporallyAwarePooling/analysis/explain_attention.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --model_name MODEL ^
  --caption_type Transformer ^
  --transformer_modality video ^
  --split valid ^
  --sample_index 0 ^
  --plot ^
  --GPU 0
```

### 4.2 Baseline (A0-style)

Match `--pool` to training (e.g. `NetVLAD++`).

```text
python Benchmarks/TemporallyAwarePooling/analysis/explain_attention.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --model_name MODEL ^
  --caption_type Baseline ^
  --pool NetVLAD++ ^
  --split valid ^
  --sample_index 0 ^
  --plot ^
  --GPU 0
```

### 4.3 Transformer, video + audio, single fused decoder (A2-style)

```text
python Benchmarks/TemporallyAwarePooling/analysis/explain_attention.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --master_audio_dir AUDIO ^
  --model_name MODEL ^
  --caption_type Transformer ^
  --transformer_modality both ^
  --split valid ^
  --sample_index 0 ^
  --plot ^
  --GPU 0
```

### 4.4 Transformer, video + audio, dual LSTM (A5-style)

```text
python Benchmarks/TemporallyAwarePooling/analysis/explain_attention.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --master_audio_dir AUDIO ^
  --model_name MODEL ^
  --caption_type Transformer ^
  --transformer_modality both ^
  --dual_lstm_decoder ^
  --split valid ^
  --sample_index 0 ^
  --plot ^
  --GPU 0
```

### 4.5 Transformer, audio only (A3-style)

```text
python Benchmarks/TemporallyAwarePooling/analysis/explain_attention.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --master_audio_dir AUDIO ^
  --model_name MODEL ^
  --caption_type Transformer ^
  --transformer_modality audio ^
  --split valid ^
  --sample_index 0 ^
  --plot ^
  --GPU 0
```

### 4.6 Contrastive init (A4a-style)

Use the **same** `--contrastive_weights_path`, `--freeze_contrastive_encoder`, and `--unfreeze_contrastive_projection` flags as training so state dict keys align.

```text
python Benchmarks/TemporallyAwarePooling/analysis/explain_attention.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --model_name MODEL ^
  --caption_type Transformer ^
  --transformer_modality video ^
  --contrastive_weights_path CKPT ^
  --freeze_contrastive_encoder ^
  --unfreeze_contrastive_projection ^
  --split valid ^
  --sample_index 0 ^
  --plot ^
  --GPU 0
```

### 4.7 Overrides

| Flag | Use |
|------|-----|
| `--checkpoint PATH` | Load a specific `.pth.tar` instead of `models/<model_name>/caption/model.pth.tar`. |
| `--out_dir DIR` | Write JSON/PNG somewhere other than `models/<model_name>/explainability/`. |
| `--max_seq_length N` | Cap decoding length (default 70). |
| `--split train valid test` | Any split accepted by `SoccerNetCaptions` (one or more names). |
| `--device cpu` | Force CPU. |

On bash/zsh, replace line-ending `^` with `\`.

---

## 5. Encoder self-attention run commands (`explain_encoder_self_attention.py`)

**Requires** a checkpoint whose encoder is `Transformer_Video`, `Transformer_Audio`, or multimodal `Transformer`. Baseline **NetVLAD** exits with an error (no encoder self-attention). Use the **same** architecture flags as training; add **`--dual_lstm_decoder`** if the checkpoint used it.

### 5.1 Transformer video (A1)

```text
python Benchmarks/TemporallyAwarePooling/analysis/explain_encoder_self_attention.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --model_name MODEL ^
  --transformer_modality video ^
  --split valid ^
  --sample_index 0 ^
  --plot ^
  --GPU 0
```

### 5.2 Transformer audio (A3)

```text
python Benchmarks/TemporallyAwarePooling/analysis/explain_encoder_self_attention.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --master_audio_dir AUDIO ^
  --model_name MODEL ^
  --transformer_modality audio ^
  --split valid ^
  --sample_index 0 ^
  --plot ^
  --GPU 0
```

### 5.3 Multimodal: both encoders (A2 / A5)

```text
python Benchmarks/TemporallyAwarePooling/analysis/explain_encoder_self_attention.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --master_audio_dir AUDIO ^
  --model_name MODEL ^
  --transformer_modality both ^
  --encoder_stream both ^
  --dual_lstm_decoder ^
  --split valid ^
  --sample_index 0 ^
  --plot ^
  --GPU 0
```

Omit `--dual_lstm_decoder` if the checkpoint did not use dual decoders. Use `--encoder_stream video` or `audio` to capture one tower only.

### 5.4 Contrastive init (A4a)

Match training contrastive flags, as for decoder explainability.

---

## 6. Aligning weights to a particular frame

**Decoder cross-attention:** time index `i` is the **i-th** vector in the sequence passed into the decoder (same ordering as the memmap clip after `SoccerNetVideoProcessor` for video). Map to wall-clock time using **`--framerate`** and **`--window_size_caption`**. For frame **k**, use column **k** in the decoder heatmap or entry **k** in that step’s weight vector.

**Encoder self-attention:** rows and columns index the **T** positions after projection + positional encoding in `Transformer_Video` / `Transformer_Audio` (or the audio stream’s analogue). For video, **i** matches the same temporal order as features in the clip tensor.

---

## 7. What you can claim in writing

- **Decoder** weights: which encoder time positions the decoder emphasized per output token.
- **Encoder** weights: how much each position attended to every other position **inside the transformer stack** (averaged over heads in the export).
- Neither is a full causal proof; use **alignment** / **mixing** language. For baseline vs transformer, combine encoder plots (transformer only) with an honest baseline alternative (e.g. input saliency) where needed.

---

## 8. Checklist

- [ ] `model_name` and architecture flags match the trained checkpoint (`ABLATION_RUNS.md`).
- [ ] For multimodal runs, `--master_audio_dir` is set and matches training.
- [ ] For **A5**, include `--dual_lstm_decoder` if the checkpoint was trained with it.
- [ ] Baseline figures do not over-interpret **T = 1** decoder attention as temporal selectivity.
- [ ] Encoder script: MHA fastpath is toggled only during capture; do not rely on it for other jobs in the same process if you embed this in a larger notebook without isolation.

---

## 9. Code references (training / model definition)

- `DecoderRNN` / `DualModalDecoderRNN`: `Benchmarks/TemporallyAwarePooling/src/model.py`
- `Transformer_Video`: `Benchmarks/TemporallyAwarePooling/src/transformer.py`
- Training entry: `Benchmarks/TemporallyAwarePooling/src/captioning.py`
