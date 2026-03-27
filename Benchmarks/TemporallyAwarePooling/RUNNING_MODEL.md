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
- Shared contrastive checkpoint for transformer caption runs:
  - Request the `sbertcontrastive` model checkpoint from Saleem.
  - Pass it with `--contrastive_weights_path`.

## Data Requirements

You need:

- A valid SoccerNet root path (passed as `--SoccerNet_path`).
- Feature name used for download/loading (for example `baidu_soccer_embeddings.npy`).
- Mapping and memmap files used by the caption dataset (for the **45 @ 1 fps** dense pipeline, paths will look like a folder containing `mapping.json` and `features.dat`):
  - `--mapping_json` → path to `mapping.json`
  - `--feature_file` → path to `features.dat`

**Transformer + `sbertcontrastive`:** clip length in frames must match the checkpoint and memmap. Use **`--window_size_caption 45`** and **`--framerate 1`** so `window_size_caption × framerate = 45`.

**CLI tip:** Do not put a space before the path after a flag (e.g. use `--SoccerNet_path C:/path` or `--SoccerNet_path=C:/path`). A space can make the value start with `=` and break paths.

Defaults for mapping/memmap (if files sit in the current working directory):

- `--mapping_json mapping.json`
- `--feature_file features.dat`

## Recommended Working Directory

Run commands from repository root.

Windows (PowerShell) examples below use backtick line continuation and are written from repo root.

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

Use Saleem’s `sbertcontrastive` checkpoint and the **45 @ 1 fps** memmap bundle.
The parser defaults in `src/captioning.py` are now set to match the "full working model" training recipe, so you can run with fewer flags.

```powershell
python Benchmarks/TemporallyAwarePooling/src/captioning.py \
  --SoccerNet_path /kaggle/input/datasets/salzeem/soccernet/data \
  --features baidu_soccer_embeddings.npy \
  --mapping_json /kaggle/input/datasets/salzeem/soccernet-densefile-at-45-1fps/mapping.json \
  --feature_file /kaggle/input/datasets/salzeem/soccernet-densefile-at-45-1fps/features.dat \
  --contrastive_weights_path /kaggle/input/models/salzeem/sbertcontrastive/pytorch/default/1/best.pth \
  --model_name NetVLAD-Transformer-memapfixed \
  --transformer_modality video
```

Current defaults for this transformer recipe (if not explicitly provided):

- `--SoccerNet_path /kaggle/input/datasets/salzeem/soccernet/data`
- `--features baidu_soccer_embeddings.npy`
- `--mapping_json /kaggle/input/datasets/salzeem/soccernet-densefile-at-45-1fps/mapping.json`
- `--feature_file /kaggle/input/datasets/salzeem/soccernet-densefile-at-45-1fps/features.dat`
- `--model_name NetVLAD-Transformer-memapfixed`
- `--caption_type Transformer`
- `--transformer_modality video`
- `--contrastive_weights_path /kaggle/input/models/salzeem/sbertcontrastive/pytorch/default/1/best.pth`
- `--freeze_contrastive_encoder` (enabled by default)
- `--unfreeze_contrastive_projection` (enabled by default)
- `--teacher_forcing_ratio 1.0`
- `--window_size_caption 45`
- `--word_dropout 0.01`
- `--framerate 1`
- `--max_epochs 100`
- `--evaluation_frequency 1`
- `--log_every_n_batches 20`
- `--max_num_worker 2`
- `--num_layers 2`
- `--split_train train`
- `--split_valid valid`
- `--split_test test`
- `--GPU 0`
- `--device cuda`
- `--loglevel INFO`

## Inference / Test-Only (Caption Side)

Runs the caption pipeline in test-only mode with your saved model checkpoint. Keep the same `--window_size_caption`, `--framerate`, and data paths as training.

```powershell
python Benchmarks/TemporallyAwarePooling/src/captioning.py `
  --SoccerNet_path "C:/path/to/SoccerNet" `
  --features baidu_soccer_embeddings.npy `
  --mapping_json "C:/path/to/soccernet-densefile-at-45-1fps/mapping.json" `
  --feature_file "C:/path/to/soccernet-densefile-at-45-1fps/features.dat" `
  --model_name transformer-video-caption `
  --caption_type Transformer `
  --transformer_modality video `
  --contrastive_weights_path "C:/path/to/sbertcontrastive/best.pth" `
  --freeze_contrastive_encoder `
  --pool NetVLAD `
  --GPU 0 `
  --window_size_caption 45 `
  --framerate 1 `
  --test_only `
  --device cuda
```

## Common Useful Flags

- Splits:
  - `--split_train train`
  - `--split_valid valid`
  - `--split_test test`
- Reproducibility:
  - `--seed 0`
- Logging:
  - `--loglevel INFO`
  - `--log_every_n_batches 50`
- Contrastive preload:
  - `--contrastive_weights_path "C:/path/to/sbertcontrastive/best.pth"`
  - `--freeze_contrastive_encoder` to keep pretrained encoder frozen
  - `--no_freeze_contrastive_encoder` to fine-tune encoder
  - `--unfreeze_contrastive_projection` to unfreeze projection when encoder is frozen

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
- `embedding_video` size mismatch (e.g. 45 vs 30):
  - Use `--window_size_caption 45 --framerate 1` with `sbertcontrastive` and the 45–1 fps memmap.
- Paths starting with `=`:
  - Fix shell quoting so values are real paths, not `=/path/...`.
- `No mapping entry found for game ...`:
  - Confirm `mapping.json` matches the selected split and feature memmap.
- `Cannot infer memmap shape ...`:
  - Confirm `features.dat` is the expected file for the selected mapping/features.
- `Skipping caption validation: ...`:
  - Install the optional NLG evaluation dependencies for full caption metrics.
- `NotImplementedError` with transformer modality:
  - Use `--transformer_modality video` for current training/inference support.
- `Could not find the pretrained aggregator so skipping preload.`:
  - Verify `--contrastive_weights_path` points to the shared `sbertcontrastive` checkpoint from Saleem.

## Team Workflow Notes

- Keep caption-side runs on `src/captioning.py`.
- Keep command lines in this file as the source of truth for team usage.
- For transformer runs with Saleem’s `sbertcontrastive` weights, use `--window_size_caption 45 --framerate 1` and the matching memmap bundle.
