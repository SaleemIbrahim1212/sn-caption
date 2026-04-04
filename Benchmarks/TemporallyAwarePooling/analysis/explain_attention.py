#!/usr/bin/env python3
"""
Decode one caption clip and save decoder attention weights (and optional heatmap).

Run from repository root (so models/<model_name>/... resolves):

  python Benchmarks/TemporallyAwarePooling/analysis/explain_attention.py --help

Heavy imports (torch, dataset, model) load only after CLI parsing so ``--help`` works
without the full training environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo layout: analysis/ under TemporallyAwarePooling; models/ under sn-caption root
# ---------------------------------------------------------------------------
_ANALYSIS_DIR = Path(__file__).resolve().parent
_BENCH_ROOT = _ANALYSIS_DIR.parent
_SRC = _BENCH_ROOT / "src"
_REPO_ROOT = _BENCH_ROOT.parent.parent


def resolve_device(args, torch_mod):
    if getattr(args, "device", None) is not None:
        return torch_mod.device(args.device)
    if getattr(args, "GPU", -1) >= 0 and torch_mod.cuda.is_available():
        return torch_mod.device("cuda")
    return torch_mod.device("cpu")


def resolve_caption_pool(args):
    caption_type = str(getattr(args, "caption_type", "baseline")).strip().lower()
    if caption_type != "transformer":
        return args.pool
    modality = str(getattr(args, "transformer_modality", "video")).strip().lower()
    modality_to_pool = {
        "video": "Transformer_Video",
        "audio": "Transformer_Audio",
        "both": "Transformer",
    }
    if modality not in modality_to_pool:
        raise ValueError(f"Incorrect modality --transformer_modality='{modality}'.")
    return modality_to_pool[modality]


def caption_dataset_kw(args):
    cap_mod = (
        str(args.transformer_modality).strip().lower()
        if str(args.caption_type).strip().lower() == "transformer"
        else "video"
    )
    mad = getattr(args, "master_audio_dir", None)
    if isinstance(mad, str) and mad.strip() == "":
        mad = None
    if cap_mod in ("audio", "both") and not mad:
        raise ValueError(
            "Set --master_audio_dir to the folder with audio_mapping.json and audio_features.dat "
            "when using --transformer_modality audio or both."
        )
    return dict(
        path=args.SoccerNet_path,
        features=args.features,
        version=args.version,
        framerate=args.framerate,
        window_size=args.window_size_caption,
        mapping_json=args.mapping_json,
        feature_file=args.feature_file,
        caption_modality=cap_mod,
        master_audio_dir=mad if cap_mod in ("audio", "both") else None,
    )


def resolve_caption_feature_dims(args, dataset_Test):
    s0 = dataset_Test[0]
    if str(args.caption_type).strip().lower() == "transformer":
        mod = str(args.transformer_modality).strip().lower()
        if mod == "both":
            if args.feature_dim is None:
                args.feature_dim = s0[0].shape[-1]
            args.audio_feature_dim = s0[1].shape[-1]
        elif mod == "audio":
            if args.feature_dim is None:
                args.feature_dim = s0[0].shape[-1]
            args.audio_feature_dim = args.feature_dim
        else:
            if args.feature_dim is None:
                args.feature_dim = s0[0].shape[-1]
            args.audio_feature_dim = None
    else:
        if args.feature_dim is None:
            args.feature_dim = s0[0].shape[-1]
        args.audio_feature_dim = None
    logging.info("feature_dim: %s audio_feature_dim: %s", args.feature_dim, getattr(args, "audio_feature_dim", None))


def _build_parser():
    p = argparse.ArgumentParser(
        description="Export decoder attention weights for one SoccerNet-Caption sample.",
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
    p.add_argument("--caption_type", type=str, choices=["Transformer", "Baseline", "transformer", "baseline"], default="Transformer")
    p.add_argument("--transformer_modality", type=str, choices=["video", "audio", "both"], default="video")
    p.add_argument("--dual_lstm_decoder", action="store_true")
    p.add_argument("--pool", type=str, default="NetVLAD++")
    p.add_argument("--vlad_k", type=int, default=64)
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

    p.add_argument("--model_name", type=str, required=True, help="Checkpoint under models/<model_name>/caption/model.pth.tar")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path (default: models/<model_name>/caption/model.pth.tar under repo root)",
    )
    p.add_argument("--split", nargs="+", default=["valid"], help="Dataset split to index into (e.g. valid test)")
    p.add_argument("--sample_index", type=int, default=0, help="Index into SoccerNetCaptions for this split")
    p.add_argument("--max_seq_length", type=int, default=70)
    p.add_argument("--out_dir", type=str, default=None, help="Output directory (default: models/<model_name>/explainability)")
    p.add_argument("--plot", action="store_true", help="Save attention heatmap PNG (requires matplotlib)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--GPU", type=int, default=0)
    p.add_argument("--loglevel", type=str, default="INFO")
    return p


def _build_model(args, vocab_size: int, device, SoccerNetTransformerCaption, Video2Caption):
    caption_pool = resolve_caption_pool(args)
    if str(args.caption_type).strip().lower() == "transformer":
        model = SoccerNetTransformerCaption(
            vocab_size=vocab_size,
            weights=None,
            input_size=args.feature_dim,
            window_size=args.window_size_caption,
            framerate=args.framerate,
            pool=caption_pool,
            num_layers=args.num_layers,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            word_dropout=args.word_dropout,
            freeze_encoder=args.freeze_encoder,
            weights_encoder=args.weights_encoder,
            contrastive_weights_path=args.contrastive_weights_path,
            freeze_contrastive_encoder=args.freeze_contrastive_encoder,
            unfreeze_contrastive_projection=args.unfreeze_contrastive_projection,
            audio_input_size=getattr(args, "audio_feature_dim", None),
            use_dual_lstm_decoder=getattr(args, "dual_lstm_decoder", False),
        ).to(device)
    else:
        model = Video2Caption(
            vocab_size=vocab_size,
            weights=None,
            input_size=args.feature_dim,
            window_size=args.window_size_caption,
            vlad_k=args.vlad_k,
            framerate=args.framerate,
            pool=args.pool,
            num_layers=args.num_layers,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            word_dropout=args.word_dropout,
            freeze_encoder=args.freeze_encoder,
            weights_encoder=args.weights_encoder,
        ).to(device)
    return model


def _load_checkpoint(model, ckpt_path: Path, device, torch):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()


def _unpack_sample_row(row, caption_modality: str, np, torch):
    if caption_modality == "both":
        vfeats, afeats, _, _, _, _ = row
        return torch.as_tensor(np.asarray(vfeats, dtype=np.float32)), torch.as_tensor(np.asarray(afeats, dtype=np.float32))
    if caption_modality == "audio":
        (afeats,), _, _, _, _ = row
        return None, torch.as_tensor(np.asarray(afeats, dtype=np.float32))
    (vfeats,), _, _, _, _ = row
    return torch.as_tensor(np.asarray(vfeats, dtype=np.float32)), None


def _tensor_list_to_jsonable(attn_list):
    return [t.detach().cpu().float().numpy().tolist() for t in attn_list]


def _maybe_plot(payload: dict, out_path: Path, dual: bool):
    import matplotlib.pyplot as plt
    import numpy as np

    if dual:
        aa = np.array([np.array(x) for x in payload["attention_audio"]], dtype=float)
        av = np.array([np.array(x) for x in payload["attention_video"]], dtype=float)
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
        im0 = axes[0].imshow(aa, aspect="auto", cmap="viridis")
        axes[0].set_title("Decoder attention (audio stream)")
        axes[0].set_ylabel("decode step")
        fig.colorbar(im0, ax=axes[0], fraction=0.02)
        im1 = axes[1].imshow(av, aspect="auto", cmap="viridis")
        axes[1].set_title("Decoder attention (video stream)")
        axes[1].set_ylabel("decode step")
        axes[1].set_xlabel("encoder time index")
        fig.colorbar(im1, ax=axes[1], fraction=0.02)
    else:
        att = np.array([np.array(x) for x in payload["attention"]], dtype=float)
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(att, aspect="auto", cmap="viridis")
        ax.set_title("Decoder attention over encoder time")
        ax.set_ylabel("decode step")
        ax.set_xlabel("encoder time index")
        fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
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
    from attention_decode import run_explain_baseline, run_explain_transformer
    from dataset import SoccerNetCaptions
    from model import SoccerNetTransformerCaption, Video2Caption

    if args.device is None:
        args.device = "cuda" if args.GPU >= 0 and torch.cuda.is_available() else "cpu"
    device = resolve_device(args, torch)

    d_kw = caption_dataset_kw(args)
    dataset = SoccerNetCaptions(split=args.split, **d_kw)
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise SystemExit(f"sample_index {args.sample_index} out of range [0, {len(dataset)})")

    resolve_caption_feature_dims(args, dataset)

    model = _build_model(args, dataset.vocab_size, device, SoccerNetTransformerCaption, Video2Caption)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = _REPO_ROOT / "models" / args.model_name / "caption" / "model.pth.tar"
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    _load_checkpoint(model, ckpt_path, device, torch)

    row = dataset[args.sample_index]
    cap_mod = (
        str(args.transformer_modality).strip().lower()
        if str(args.caption_type).strip().lower() == "transformer"
        else "video"
    )
    feats_v, feats_a = _unpack_sample_row(row, cap_mod, np, torch)

    ref_caption = row[-1]
    game_id, cap_id = row[-3], row[-2]

    if str(args.caption_type).strip().lower() == "baseline":
        ids, attn, enc_t = run_explain_baseline(model, feats_v, device, args.max_seq_length)
        meta = {
            "encoder_seq_len": enc_t,
            "pool": "baseline",
            "note": "Attention is over a single pooled vector (T=1); weights are degenerate.",
        }
        payload = {
            "model_name": args.model_name,
            "checkpoint": str(ckpt_path),
            "split": args.split,
            "sample_index": args.sample_index,
            "game_id": int(game_id),
            "caption_id": int(cap_id),
            "reference_caption": ref_caption,
            "generated_token_ids": ids,
            "generated_caption": dataset.detokenize(ids),
            "attention": _tensor_list_to_jsonable(attn),
            "meta": meta,
        }
        dual = False
    else:
        ids, att_data, meta = run_explain_transformer(model, feats_v, feats_a, device, args.max_seq_length)
        payload = {
            "model_name": args.model_name,
            "checkpoint": str(ckpt_path),
            "split": args.split,
            "sample_index": args.sample_index,
            "game_id": int(game_id),
            "caption_id": int(cap_id),
            "reference_caption": ref_caption,
            "generated_token_ids": ids,
            "generated_caption": dataset.detokenize(ids),
            "meta": meta,
        }
        if meta.get("dual_decoder"):
            aa, av = att_data
            payload["attention_audio"] = _tensor_list_to_jsonable(aa)
            payload["attention_video"] = _tensor_list_to_jsonable(av)
            dual = True
        else:
            payload["attention"] = _tensor_list_to_jsonable(att_data)
            dual = False

    out_dir = Path(args.out_dir) if args.out_dir else _REPO_ROOT / "models" / args.model_name / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"attention_sample{args.sample_index}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logging.info("Wrote %s", json_path)

    if args.plot:
        plot_path = out_dir / f"attention_sample{args.sample_index}.png"
        _maybe_plot(payload, plot_path, dual=dual)
        logging.info("Wrote %s", plot_path)


if __name__ == "__main__":
    main()
