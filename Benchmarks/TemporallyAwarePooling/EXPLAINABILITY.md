# Explainability (focused) + run commands

This document is intentionally narrow:

1. NetVLAD baseline vs Transformer aggregation
2. Contrastive initialization
3. Dual vs single LSTM decoding

Explainability scripts live in `Benchmarks/TemporallyAwarePooling/analysis/`.
Expected checkpoint path is `models/<model_name>/caption/model.pth.tar`.

Run from repository root (`sn-caption`). On Windows cmd, use `^` for line continuation. On bash/zsh, use `\`.

Placeholders used below:

- `DATA`: SoccerNet root
- `MAP`: mapping JSON path
- `FEAT`: feature memmap path
- `MODEL`: model name used during training
- `AUDIO`: multimodal audio feature directory
- `CKPT`: contrastive checkpoint path

---

## 1. What to explain in writing

### 1.1 NetVLAD vs Transformer aggregation

- **Baseline (NetVLAD-style pooling):** summarizes evidence into a pooled vector before decoding.
- In exported decoder-attention geometry, baseline behaves as **T = 1** (degenerate temporal axis), so do not over-interpret it as fine-grained temporal selection.
- **Transformer:** keeps temporal sequence positions (**T > 1**) and mixes them through encoder self-attention.
- Decoder cross-attention then reflects which time positions are emphasized for each generated token.

Use alignment/mixing language rather than causal claims.

### 1.2 Contrastive initialization

- Contrastive pretraining changes the quality of input representations (feature-space structure).
- It is not a separate aggregation mechanism; it changes what the aggregator receives.

### 1.3 Single vs dual LSTM decoder

- **Single (fused) decoder:** modalities are fused, then one decoder generates tokens.
- **Dual LSTM decoder:** modality-specific decoder streams (audio/video) are kept longer before fusion.
- Explainability outputs include separate attention tracks per modality in dual mode.

---

## 2. Decoder cross-attention commands (`explain_attention.py`)

Default output directory: `models/<model_name>/explainability/` (JSON, plus PNG with `--plot`).

### 2.1 Baseline (NetVLAD++)

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

### 2.2 Transformer, video only

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

### 2.3 Transformer, video + audio (single fused decoder)

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

### 2.4 Transformer, video + audio + dual LSTM

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

### 2.5 Transformer + contrastive init (match training flags)

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

### 2.6 Useful overrides

| Flag | Use |
|------|-----|
| `--checkpoint PATH` | Load a specific `.pth.tar` |
| `--out_dir DIR` | Write outputs to a custom directory |
| `--max_seq_length N` | Cap generation length |
| `--device cpu` | Force CPU |

---

## 3. Encoder self-attention commands (`explain_encoder_self_attention.py`)

This applies to Transformer checkpoints only. Baseline NetVLAD has no encoder self-attention map.

### 3.1 Transformer, video only

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

### 3.2 Transformer, multimodal (both streams)

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

Omit `--dual_lstm_decoder` if the checkpoint was not trained with dual decoders.
Use `--encoder_stream video` or `--encoder_stream audio` to capture one tower only.
For contrastive models, reuse the same contrastive flags as in the decoder command.

---

## 4. Implemented analysis files

| File | Role |
|------|------|
| `Benchmarks/TemporallyAwarePooling/analysis/attention_decode.py` | Greedy decode while collecting decoder cross-attention weights |
| `Benchmarks/TemporallyAwarePooling/analysis/explain_attention.py` | Decoder attention CLI (`JSON` and optional `PNG`) |
| `Benchmarks/TemporallyAwarePooling/analysis/encoder_self_attention.py` | Encoder self-attention capture logic |
| `Benchmarks/TemporallyAwarePooling/analysis/explain_encoder_self_attention.py` | Encoder self-attention CLI (`JSON` and optional `PNG`) |

---

## 5. Minimal checklist

- [ ] `model_name` and architecture flags match the checkpoint.
- [ ] Multimodal runs set `--master_audio_dir`.
- [ ] Dual LSTM checkpoints include `--dual_lstm_decoder` at analysis time.
- [ ] Baseline (`T = 1`) attention is not presented as temporal selectivity.
- [ ] Claims use alignment/mixing language, not causal proof.
