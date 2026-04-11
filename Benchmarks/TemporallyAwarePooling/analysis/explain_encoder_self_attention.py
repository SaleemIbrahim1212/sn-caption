#!/usr/bin/env python3
"""
Export transformer **encoder** self-attention (per layer, averaged over heads).

Uses analysis-only runtime patches (see ``encoder_self_attention.py``); does not
modify ``src/transformer.py``.

Run from repository root::

  python Benchmarks/TemporallyAwarePooling/analysis/explain_encoder_self_attention.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ANALYSIS_DIR = Path(__file__).resolve().parent
_BENCH_ROOT = _ANALYSIS_DIR.parent
_SRC = _BENCH_ROOT / "src"
_REPO_ROOT = _BENCH_ROOT.parent.parent


def _build_parser():
    p = argparse.ArgumentParser(
        description="Export transformer encoder self-attention for one caption clip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--SoccerNet_path", type=str, required=True)
    p.add_argument("--features", type=str, default="baidu_soccer_embeddings.npy")
    p.add_argument("--mapping_json", type=str, default="mapping.json")
    p.add_argument("--feature_file", type=str, default="features.dat")
    p.add_argument("--master_audio_dir", type=str, default=None)
    p.add_argument("--version", type=int, default=2)
    p.add_argument("--framerate", type=int, default=1)
    p.add_argument("--window_size_caption", type=int, default=45)
    p.add_argument("--caption_type", type=str, choices=["Transformer", "transformer"], default="Transformer")
    p.add_argument("--transformer_modality", type=str, choices=["video", "audio", "both"], default="video")
    p.add_argument("--dual_lstm_decoder", action="store_true",
                   help="Must match training if the checkpoint used multimodal dual decoders.")
    p.add_argument("--encoder_stream", type=str, choices=["video", "audio", "both"], default="both",
                   help="For multimodal (transformer_modality=both): which encoder(s) to capture.")
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--teacher_forcing_ratio", type=float, default=1.0)
    p.add_argument("--word_dropout", type=float, default=0.01)
    p.add_argument("--freeze_encoder", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False)
    p.add_argument("--weights_encoder", type=str, default=None)
    p.add_argument("--feature_dim", type=int, default=None)
    p.add_argument("--contrastive_weights_path", type=str, default=None)
    p.add_argument("--freeze_contrastive_encoder", dest="freeze_contrastive_encoder", action="store_true")
    p.add_argument("--no_freeze_contrastive_encoder", dest="freeze_contrastive_encoder", action="store_false")
    p.add_argument("--unfreeze_contrastive_projection", dest="unfreeze_contrastive_projection", action="store_true")
    p.add_argument("--no_unfreeze_contrastive_projection", dest="unfreeze_contrastive_projection", action="store_false")
    p.set_defaults(freeze_contrastive_encoder=True, unfreeze_contrastive_projection=True)

    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--split", nargs="+", default=["valid"])
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--plot", action="store_true", help="Save last-layer encoder heatmap(s) (matplotlib)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--GPU", type=int, default=0)
    p.add_argument("--loglevel", type=str, default="INFO")
    return p


def _maybe_plot_encoder(payload: dict, out_path: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    def one_heatmap(attn_layers, title, ax):
        if not attn_layers:
            return
        last = attn_layers[-1]["attention"]
        arr = np.array(last, dtype=float)
        im = ax.imshow(arr, aspect="equal", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")
        plt.colorbar(im, ax=ax, fraction=0.046)

    ref = payload.get("reference_caption", "")

    has_v = payload.get("video") and payload["video"].get("layers")
    has_a = payload.get("audio") and payload["audio"].get("layers")
    if has_v and has_a:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        one_heatmap(payload["video"]["layers"], "Video encoder (last layer)", axes[0])
        one_heatmap(payload["audio"]["layers"], "Audio encoder (last layer)", axes[1])
    elif has_v:
        fig, ax = plt.subplots(figsize=(6, 6))
        one_heatmap(payload["video"]["layers"], "Video encoder (last layer)", ax)
    elif has_a:
        fig, ax = plt.subplots(figsize=(6, 6))
        one_heatmap(payload["audio"]["layers"], "Audio encoder (last layer)", ax)
    else:
        return

    fig.subplots_adjust(top=0.88, bottom=0.12)
    fig.text(0.5, 0.04, f"Reference: {ref}", ha="center", va="bottom", fontsize=9,
             style="italic", transform=fig.transFigure)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    if str(_ANALYSIS_DIR) not in sys.path:
        sys.path.insert(0, str(_ANALYSIS_DIR))

    import numpy as np
    import torch
    from encoder_self_attention import (
        FastpathGuard,
        capture_multimodal_encoder_attention,
        capture_to_jsonable,
        capture_transformer_audio_encoder_attention,
        capture_transformer_video_encoder_attention,
    )
    from explain_attention import (
        caption_dataset_kw,
        resolve_caption_feature_dims,
        resolve_device,
        _build_model,
        _load_checkpoint,
        _unpack_sample_row,
    )
    from dataset import SoccerNetCaptions

    if args.device is None:
        args.device = "cuda" if args.GPU >= 0 and torch.cuda.is_available() else "cpu"
    device = resolve_device(args, torch)

    d_kw = caption_dataset_kw(args)
    dataset = SoccerNetCaptions(split=args.split, **d_kw)
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise SystemExit(f"sample_index {args.sample_index} out of range [0, {len(dataset)})")

    resolve_caption_feature_dims(args, dataset)

    from model import SoccerNetTransformerCaption, Video2Caption

    model = _build_model(args, dataset.vocab_size, device, SoccerNetTransformerCaption, Video2Caption)
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "pool"):
        raise SystemExit("Expected caption model with transformer encoder.")
    pool = model.encoder.pool
    if pool not in ("Transformer_Video", "Transformer_Audio", "Transformer"):
        raise SystemExit(
            f"Encoder self-attention applies to transformer encoders only (pool={pool}). "
            "Baseline NetVLAD has no sequence self-attention; use decoder explainability or input saliency."
        )

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = _REPO_ROOT / "models" / args.model_name / "caption" / "model.pth.tar"
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    _load_checkpoint(model, ckpt_path, device, torch)

    row = dataset[args.sample_index]
    cap_mod = str(args.transformer_modality).strip().lower()
    feats_v, feats_a = _unpack_sample_row(row, cap_mod, np, torch)

    ref_caption = row[-1]
    game_id, cap_id = row[-3], row[-2]

    pl = model.encoder.pooling_layer
    payload = {
        "model_name": args.model_name,
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "sample_index": args.sample_index,
        "game_id": int(game_id),
        "caption_id": int(cap_id),
        "reference_caption": ref_caption,
        "pool": pool,
        "method": (
            "Temporary TransformerEncoderLayer._sa_block wrapper with need_weights=True; "
            "torch.backends.mha fast path disabled for the forward pass (see encoder_self_attention.py)."
        ),
        "video": None,
        "audio": None,
    }

    with torch.no_grad():
        if pool == "Transformer_Video":
            if feats_v is None:
                raise SystemExit("Missing video features for this sample.")
            fv = feats_v.unsqueeze(0).to(device)
            with FastpathGuard():
                cap, _, _ = capture_transformer_video_encoder_attention(pl, fv)
            payload["video"] = {"layers": capture_to_jsonable(cap)}
        elif pool == "Transformer_Audio":
            if feats_a is None:
                raise SystemExit("Missing audio features for this sample.")
            fa = feats_a.unsqueeze(0).to(device)
            with FastpathGuard():
                cap, _, _ = capture_transformer_audio_encoder_attention(pl, fa)
            payload["audio"] = {"layers": capture_to_jsonable(cap)}
        else:
            stream = args.encoder_stream
            if stream == "video" and feats_v is None:
                raise SystemExit("encoder_stream=video requires video features.")
            if stream == "audio" and feats_a is None:
                raise SystemExit("encoder_stream=audio requires audio features.")
            if stream == "both" and (feats_v is None or feats_a is None):
                raise SystemExit("encoder_stream=both requires video and audio features.")
            fv = feats_v.unsqueeze(0).to(device) if feats_v is not None else None
            fa = feats_a.unsqueeze(0).to(device) if feats_a is not None else None
            with FastpathGuard():
                mm = capture_multimodal_encoder_attention(pl, fa, fv, streams=stream)
            if mm.get("video"):
                payload["video"] = {"layers": capture_to_jsonable(mm["video"]["capture"])}
            if mm.get("audio"):
                payload["audio"] = {"layers": capture_to_jsonable(mm["audio"]["capture"])}

    out_dir = Path(args.out_dir) if args.out_dir else _REPO_ROOT / "models" / args.model_name / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"encoder_self_attn_sample{args.sample_index}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logging.info("Wrote %s", json_path)

    if args.plot:
        plot_path = out_dir / f"encoder_self_attn_sample{args.sample_index}.png"
        _maybe_plot_encoder(payload, plot_path)
        logging.info("Wrote %s", plot_path)


if __name__ == "__main__":
    main()
