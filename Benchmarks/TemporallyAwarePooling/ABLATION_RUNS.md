# Captioning ablation: training and inference runs

This document is **documentation only** for planning runs. Training uses `src/captioning.py`; reference/generated caption export uses `src/export_split_captions.py` (§8); explainability tooling uses `analysis/` (see `EXPLAINABILITY.md`).

**Entry script (train / eval):** `Benchmarks/TemporallyAwarePooling/src/captioning.py`  
**Best checkpoint (after training):** `models/<model_name>/caption/model.pth.tar`

Run commands from the **repository root** (`sn-caption`) unless noted.

Use **`^`** at end of line on Windows **cmd**; use **`\`** on bash/zsh.

---

## 1. What must stay fixed across compared runs

If two runs are meant to be comparable, keep these identical unless the ablation is explicitly about them.

| Area | Flags | Notes |
|------|--------|--------|
| Data root | `--SoccerNet_path` | Same dataset root |
| Video features | `--features`, `--mapping_json`, `--feature_file` | Same memmap / feature setup |
| Temporal window | `--window_size_caption`, `--framerate` | Same clip geometry |
| Splits | `--split_train`, `--split_valid`, `--split_test` | Same protocol |
| Reproducibility | `--seed` | Same seed when comparing stochastic training |
| Optimization (optional to vary) | `--batch_size`, `--LR`, `--patience`, `--max_epochs` | Fix across rows unless studying training recipe |

Give every distinct configuration its own **`--model_name`** so logs and checkpoints never overwrite.

---

## 2. How model type maps to flags

| Goal | `--caption_type` | Additional flags |
|------|-------------------|------------------|
| Baseline (NetVLAD-style encoder, etc.) | `Baseline` | `--pool` (default `NetVLAD++`) |
| Transformer, video stream only | `Transformer` | `--transformer_modality video` |
| Transformer, audio stream only | `Transformer` | `--transformer_modality audio` + `--master_audio_dir` |
| Transformer, video + audio | `Transformer` | `--transformer_modality both` + `--master_audio_dir` |
| Multimodal, two LSTM decoders | `Transformer` | `--transformer_modality both` + `--dual_lstm_decoder` |

Audio or multimodal runs require `--master_audio_dir` pointing at a directory that contains `audio_mapping.json` and `audio_features.dat`.

---

## 3. Ablation run IDs

| ID | Description |
|----|-------------|
| **A0** | Baseline captioning (`NetVLAD++` or other `--pool`) |
| **A1** | Transformer, video only |
| **A2** | Transformer, video + audio |
| **A3** | Transformer, audio only |
| **A4a** | Transformer video + contrastive init (real `--contrastive_weights_path`) |
| **A4b** | Transformer video, no contrastive init (missing path skips load; see command) |
| **A5** | Same as **A2** + `--dual_lstm_decoder` |

---

## 4. Training command per ablation

Replace `DATA`, `MAP`, `FEAT`, `AUDIO`, `CKPT` with your paths. Keep one `model_name` per row.

### A0 — Baseline (train)

```text
python Benchmarks/TemporallyAwarePooling/src/captioning.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --model_name abl-A0-baseline-nvlad ^
  --caption_type Baseline ^
  --pool NetVLAD++ ^
  --split_train train --split_valid valid --split_test test ^
  --GPU 0 --device cuda
```

### A1 — Transformer, video only (train)

```text
python Benchmarks/TemporallyAwarePooling/src/captioning.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --model_name abl-A1-tfm-video ^
  --caption_type Transformer ^
  --transformer_modality video ^
  --split_train train --split_valid valid --split_test test ^
  --GPU 0 --device cuda
```

### A2 — Transformer, video + audio (train)

```text
python Benchmarks/TemporallyAwarePooling/src/captioning.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --master_audio_dir AUDIO ^
  --model_name abl-A2-tfm-both ^
  --caption_type Transformer ^
  --transformer_modality both ^
  --split_train train --split_valid valid --split_test test ^
  --GPU 0 --device cuda
```

### A3 — Transformer, audio only (train)

```text
python Benchmarks/TemporallyAwarePooling/src/captioning.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --master_audio_dir AUDIO ^
  --model_name abl-A3-tfm-audio ^
  --caption_type Transformer ^
  --transformer_modality audio ^
  --split_train train --split_valid valid --split_test test ^
  --GPU 0 --device cuda
```

### A4a — Transformer video + contrastive checkpoint (train)

```text
python Benchmarks/TemporallyAwarePooling/src/captioning.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --model_name abl-A4a-tfm-video-contrastive ^
  --caption_type Transformer ^
  --transformer_modality video ^
  --contrastive_weights_path CKPT ^
  --freeze_contrastive_encoder ^
  --unfreeze_contrastive_projection ^
  --split_train train --split_valid valid --split_test test ^
  --GPU 0 --device cuda
```

Use `--no_freeze_contrastive_encoder` or `--no_unfreeze_contrastive_projection` as needed for sub-ablations.

### A4b — Transformer video, skip contrastive load (train)

`load_contrastive_video_weights` skips if the file is missing. Pointing to a non-existent path avoids editing code:

```text
python Benchmarks/TemporallyAwarePooling/src/captioning.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --model_name abl-A4b-tfm-video-scratch ^
  --caption_type Transformer ^
  --transformer_modality video ^
  --contrastive_weights_path "__skip_contrastive__.pth" ^
  --split_train train --split_valid valid --split_test test ^
  --GPU 0 --device cuda
```

### A5 — Dual LSTM decoder (train; requires multimodal)

```text
python Benchmarks/TemporallyAwarePooling/src/captioning.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --master_audio_dir AUDIO ^
  --model_name abl-A5-tfm-both-dual-lstm ^
  --caption_type Transformer ^
  --transformer_modality both ^
  --dual_lstm_decoder ^
  --split_train train --split_valid valid --split_test test ^
  --GPU 0 --device cuda
```

---

## 5. Inference / evaluation command per ablation

Use the **same** architecture and data flags as training, the matching **`--model_name`**, and **`--test_only`**. The script loads `models/<model_name>/caption/model.pth.tar` and runs `validate_captioning` for each split in `--split_test`.

### A0 — Baseline (inference only)

```text
python Benchmarks/TemporallyAwarePooling/src/captioning.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --model_name abl-A0-baseline-nvlad ^
  --caption_type Baseline ^
  --pool NetVLAD++ ^
  --split_test test ^
  --test_only ^
  --GPU 0 --device cuda
```

### A1 — Transformer video (inference only)

```text
python Benchmarks/TemporallyAwarePooling/src/captioning.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --model_name abl-A1-tfm-video ^
  --caption_type Transformer ^
  --transformer_modality video ^
  --split_test test ^
  --test_only ^
  --GPU 0 --device cuda
```

### A2 — Transformer both (inference only)

```text
python Benchmarks/TemporallyAwarePooling/src/captioning.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --master_audio_dir AUDIO ^
  --model_name abl-A2-tfm-both ^
  --caption_type Transformer ^
  --transformer_modality both ^
  --split_test test ^
  --test_only ^
  --GPU 0 --device cuda
```

### A3 — Transformer audio only (inference only)

```text
python Benchmarks/TemporallyAwarePooling/src/captioning.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --master_audio_dir AUDIO ^
  --model_name abl-A3-tfm-audio ^
  --caption_type Transformer ^
  --transformer_modality audio ^
  --split_test test ^
  --test_only ^
  --GPU 0 --device cuda
```

### A4a — Same as training contrastive flags (inference only)

```text
python Benchmarks/TemporallyAwarePooling/src/captioning.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --model_name abl-A4a-tfm-video-contrastive ^
  --caption_type Transformer ^
  --transformer_modality video ^
  --contrastive_weights_path CKPT ^
  --freeze_contrastive_encoder ^
  --unfreeze_contrastive_projection ^
  --split_test test ^
  --test_only ^
  --GPU 0 --device cuda
```

### A4b — No contrastive file (inference only)

```text
python Benchmarks/TemporallyAwarePooling/src/captioning.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --model_name abl-A4b-tfm-video-scratch ^
  --caption_type Transformer ^
  --transformer_modality video ^
  --contrastive_weights_path "__skip_contrastive__.pth" ^
  --split_test test ^
  --test_only ^
  --GPU 0 --device cuda
```

### A5 — Dual LSTM (inference only)

```text
python Benchmarks/TemporallyAwarePooling/src/captioning.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --master_audio_dir AUDIO ^
  --model_name abl-A5-tfm-both-dual-lstm ^
  --caption_type Transformer ^
  --transformer_modality both ^
  --dual_lstm_decoder ^
  --split_test test ^
  --test_only ^
  --GPU 0 --device cuda
```

If `nlgeval` (or related) is not installed, caption validation may be skipped; see the log.

---

## 6. Automatic evaluation after training

When you run **without** `--test_only`, training ends by loading the best checkpoint and evaluating on `--split_test` the same way as above.

---

## 7. Dense captioning (DVC-style)

`dvc(args)` in `captioning.py` runs dense metrics (`test_captioning`). The module’s `__main__` currently calls `main(args)` only. If you need dense evaluation, invoke `dvc` from a small driver or notebook, with spotting outputs under `models/<model_name>/outputs/<split>/` as required by `test_captioning`.

---

## 8. What to record for the ablation table

For each `model_name`:

- Metrics from validation (BLEU, METEOR, CIDEr, ROUGE-L, etc.) and W&B if used.
- **Reference vs generated captions (for figures / external viz):** `captioning.py` inference computes generations for metrics but does not save them. Use **`src/export_split_captions.py`** to write the same `model.sample(...)` outputs to disk.

**Default output directory:** `models/<model_name>/predictions/` (repo root). Override with `--out_dir`.

**Example (transformer video, test split):**

```text
python Benchmarks/TemporallyAwarePooling/src/export_split_captions.py ^
  --SoccerNet_path DATA ^
  --features baidu_soccer_embeddings.npy ^
  --mapping_json MAP ^
  --feature_file FEAT ^
  --model_name abl-A1-tfm-video ^
  --caption_type Transformer ^
  --transformer_modality video ^
  --split test ^
  --format json ^
  --GPU 0
```

Use the **same** architecture and data flags as training (`--pool`, `--master_audio_dir`, `--dual_lstm_decoder`, contrastive flags, etc.) so the checkpoint loads. Each record includes `game_id`, `caption_id`, `game_name`, `reference_caption`, `generated_caption`. For JSONL, set `--format jsonl`.

---

## 9. Minimal paper-style comparison sets

| Question | Train / load | Inference |
|----------|----------------|-----------|
| Pooling vs temporal transformer | **A0** vs **A1** | **A0** / **A1** `--test_only` on same `--split_test` |
| Adding audio | **A1** vs **A2** | **A1** / **A2** `--test_only` |
| Contrastive init | **A4a** vs **A4b** | Matching `--test_only` commands |
| Single fused vs dual LSTM | **A2** vs **A5** | **A2** / **A5** `--test_only` |

