# Transformer Training Guide (Captioning)

> **Note: This file is partially outdated.** Audio and multimodal (`both`) training are now implemented. The section below describing them as "not implemented" no longer applies. Refer to [`RUNNING_MODEL.md`](RUNNING_MODEL.md) and [`ABLATION_RUNS.md`](ABLATION_RUNS.md) for the current, authoritative training guide and all run commands.

This file documents how to run transformer-based caption training in this benchmark.

## What's new and what this runs

- Runs **captioning training only** (no spotting stage) using:
  - `Benchmarks/TemporallyAwarePooling/src/captioning.py`
- Uses the new config:
  - `--caption_type Transformer`
  - `--transformer_modality video|audio|both`

## Current Support Status

- `video`: supported for training.
- `audio`: supported for training (requires `--master_audio_dir`).
- `both`: supported for training (requires `--master_audio_dir`).

## Prerequisites

1. Install required packages from project requirements.
2. Ensure your SoccerNet data path is correct (examples below use `data`).

## Basic GPU Command (Video Transformer Only)

Run from repository root:

```powershell
python Benchmarks/TemporallyAwarePooling/src/captioning.py --SoccerNet_path data --features baidu_soccer_embeddings.npy --model_name transformer-video-run --caption_type Transformer --transformer_modality video --teacher_forcing_ratio 1 --max_epochs 50 --evaluation_frequency 999999 --batch_size 16 --max_num_worker 0 --split_train train --split_valid valid --split_test test --GPU 0 --device cuda --loglevel INFO
```

Notes:
- `--evaluation_frequency 999999` avoids calling caption metric evaluation too often.
- `--max_num_worker 0` is safe for memory issues (if mem is too tight)

