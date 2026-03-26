# Running Model (Caption Side)

This is the main runbook for the caption-side workflow on this benchmark branch.

## Scope

- Caption side only.
- Primary entrypoint: `Benchmarks/TemporallyAwarePooling/src/captioning.py`.
- The team workflow here does not use `src/main.py`.

## Prerequisites

- Python 3.8 environment.
- Install dependencies from repo root:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

- Optional NLG caption metrics dependency (`nlgeval`) for full validation metrics:
  - Follow the note in `requirements.txt` (`scripts/install_nlg_eval.sh`).
  - If not installed, caption validation metrics are skipped with a warning.

## Data Requirements

You need:

- A valid SoccerNet root path (passed as `--SoccerNet_path`).
- Feature name used for download/loading (for example `baidu_soccer_embeddings.npy`).
- Mapping and memmap files used by the caption dataset:
  - `mapping.json` (pass with `--mapping_json`)
  - `features.dat` (pass with `--feature_file`)

Defaults for mapping/memmap flags are:

- `--mapping_json mapping.json`
- `--feature_file features.dat`

These paths are resolved from your current working directory when running commands.

## Recommended Working Directory

Run commands from repository root.

Windows (PowerShell) examples in this document are written from repo root.

## Training (Baseline Caption Model)

This uses default training values unless you override them:

- `max_epochs=1000`
- `batch_size=256`
- `evaluation_frequency=10`

```powershell
python Benchmarks/TemporallyAwarePooling/src/captioning.py `
  --SoccerNet_path "C:/path/to/SoccerNet" `
  --features baidu_soccer_embeddings.npy `
  --mapping_json mapping.json `
  --feature_file features.dat `
  --model_name baseline-caption `
  --caption_type Baseline `
  --pool NetVLAD++ `
  --GPU 0 `
  --device cuda
```

## Training (Transformer Video Caption Model)

Current support status for caption transformer:

- `video`: implemented
- `audio`: not implemented in training path
- `both`: not implemented in training path

```powershell
python Benchmarks/TemporallyAwarePooling/src/captioning.py `
  --SoccerNet_path "C:/path/to/SoccerNet" `
  --features baidu_soccer_embeddings.npy `
  --mapping_json mapping.json `
  --feature_file features.dat `
  --model_name transformer-video-caption `
  --caption_type Transformer `
  --transformer_modality video `
  --pool NetVLAD++ `
  --GPU 0 `
  --device cuda
```

## Inference / Test-Only (Caption Side)

Runs the caption pipeline in test-only mode with your saved model checkpoint.

```powershell
python Benchmarks/TemporallyAwarePooling/src/captioning.py `
  --SoccerNet_path "C:/path/to/SoccerNet" `
  --features baidu_soccer_embeddings.npy `
  --mapping_json mapping.json `
  --feature_file features.dat `
  --model_name transformer-video-caption `
  --caption_type Transformer `
  --transformer_modality video `
  --test_only `
  --GPU 0 `
  --device cuda
```

## Common Useful Flags

- Splits:
  - `--split_train train`
  - `--split_valid valid`
  - `--split_test test challenge`
- Reproducibility:
  - `--seed 0`
- Logging:
  - `--loglevel INFO`
  - `--log_every_n_batches 50`

## Outputs

Model checkpoints and logs:

- `models/<model_name>/caption/model.pth.tar`
- `models/<model_name>/caption/checkpoint_last.pth.tar`
- `models/<model_name>/*.log`

Dense caption prediction outputs:

- `models/<model_name>/outputs/<split>/...`
- `models/<model_name>/results_dense_captioning_<split>.zip`

## Troubleshooting

- `Invalid log level`:
  - Ensure `--loglevel` is one of Python logging levels (for example `INFO`).
- `No mapping entry found for game ...`:
  - Confirm `mapping.json` matches the selected split and feature memmap.
- `Cannot infer memmap shape ...`:
  - Confirm `features.dat` is the expected file for the selected mapping/features.
- `Skipping caption validation: ...`:
  - Install the optional NLG evaluation dependencies for full caption metrics.
- `NotImplementedError` with transformer modality:
  - Use `--transformer_modality video` for current training/inference support.

## Team Workflow Notes

- Keep caption-side runs on `src/captioning.py`.
- Keep command lines in this file as the source of truth for team usage.
