# Transformer Training Guide (Captioning)

This file documents how to run transformer-based caption training in this benchmark

## What's new and what this runs

- Runs **captioning training only** (no spotting stage) using:
  - `Benchmarks/TemporallyAwarePooling/src/captioning.py`
- Uses the new config:
  - `--caption_type Transformer`
  - `--transformer_modality video|audio|both`

## Current Support Status

- `video`: supported for training.
- `audio`: not implemented in training loop yet.
- `both`: not implemented in training loop yet.

If you set `audio` or `both`, training will hit `NotImplementedError` in `src/train.py`.

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

