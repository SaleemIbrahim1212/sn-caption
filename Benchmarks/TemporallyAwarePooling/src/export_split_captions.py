#!/usr/bin/env python3
"""
Export reference vs generated captions for full SoccerNet-Caption splits.

Same inference protocol as validate_captioning (model.sample per clip).
Default output: models/<model_name>/predictions/ (use --out_dir to override).

Run from repository root, e.g.:
  python Benchmarks/TemporallyAwarePooling/src/export_split_captions.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# sn-caption/Benchmarks/TemporallyAwarePooling/src/this_file.py -> repo root is 4 parents up from file path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _build_parser():
    p = argparse.ArgumentParser(
        description="Export reference + generated captions for a SoccerNet-Caption split.",
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

    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--split", nargs="+", default=["test"], help="Splits to export (e.g. test valid)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_num_worker", type=int, default=2)
    p.add_argument(
        "--format",
        type=str,
        choices=["json", "jsonl"],
        default="json",
        help="json: one file per split with records array; jsonl: one JSON object per line",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: models/<model_name>/predictions under repo root)",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--GPU", type=int, default=0)
    p.add_argument("--loglevel", type=str, default="INFO")
    return p


def _build_model(args, vocab_size: int, device, SoccerNetTransformerCaption, Video2Caption, resolve_caption_pool):
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


def _normalize_indices(cap_id_batch, torch):
    """collate_fn_padd returns list of (game_id, caption_id) per sample."""
    out = []
    for item in cap_id_batch:
        if isinstance(item, torch.Tensor):
            gi, ci = int(item[0].item()), int(item[1].item())
        else:
            gi, ci = int(item[0]), int(item[1])
        out.append((gi, ci))
    return out


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))

    import torch
    from captioning import (
        caption_dataset_kw,
        resolve_caption_feature_dims,
        resolve_caption_pool,
        resolve_device,
    )
    from dataset import SoccerNetCaptions, collate_fn_padd
    from model import SoccerNetTransformerCaption, Video2Caption
    from train import _unpack_caption_batch_feats
    from tqdm import tqdm

    if args.device is None:
        args.device = "cuda" if args.GPU >= 0 and torch.cuda.is_available() else "cpu"
    device = resolve_device(args)

    d_kw = caption_dataset_kw(args)
    out_dir = Path(args.out_dir) if args.out_dir else _REPO_ROOT / "models" / args.model_name / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = _REPO_ROOT / "models" / args.model_name / "caption" / "model.pth.tar"
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    for split_name in args.split:
        split_kw = {**d_kw, "split": [split_name]}
        dataset = SoccerNetCaptions(**split_kw)
        resolve_caption_feature_dims(args, dataset)

        model = _build_model(args, dataset.vocab_size, device, SoccerNetTransformerCaption, Video2Caption, resolve_caption_pool)
        _load_checkpoint(model, ckpt_path, device, torch)
        model.eval()

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.max_num_worker,
            pin_memory=True,
            collate_fn=collate_fn_padd,
            persistent_workers=(args.max_num_worker > 0),
        )

        pool = getattr(getattr(model, "encoder", None), "pool", "")
        records = []

        with torch.no_grad():
            for (feats_batch, _caption), _lengths, _mask, caption_or, cap_id_batch in tqdm(
                loader, desc=f"Captions {split_name}"
            ):
                feats_v, feats_a = _unpack_caption_batch_feats(feats_batch, pool, device)
                if hasattr(model, "encoder") and pool.startswith("Transformer_Video"):
                    hyps = [
                        dataset.detokenize(list(model.sample(feats_v[idx], None).detach().cpu()))
                        for idx in range(feats_v.shape[0])
                    ]
                elif hasattr(model, "encoder") and pool.startswith("Transformer_Audio"):
                    hyps = [
                        dataset.detokenize(list(model.sample(None, feats_a[idx]).detach().cpu()))
                        for idx in range(feats_a.shape[0])
                    ]
                elif hasattr(model, "encoder") and pool == "Transformer":
                    hyps = [
                        dataset.detokenize(list(model.sample(feats_v[idx], feats_a[idx]).detach().cpu()))
                        for idx in range(feats_v.shape[0])
                    ]
                else:
                    hyps = [
                        dataset.detokenize(list(model.sample(feats_v[idx]).detach().cpu()))
                        for idx in range(feats_v.shape[0])
                    ]

                pairs = _normalize_indices(cap_id_batch, torch)
                for (gi, ci), ref, gen in zip(pairs, caption_or, hyps):
                    gname = dataset.listGames[gi] if gi < len(dataset.listGames) else str(gi)
                    records.append(
                        {
                            "game_id": gi,
                            "caption_id": ci,
                            "game_name": gname,
                            "reference_caption": ref,
                            "generated_caption": gen,
                        }
                    )

        stem = f"captions_{split_name}_{args.model_name.replace('/', '_')}"
        if args.format == "json":
            path = out_dir / f"{stem}.json"
            payload = {
                "model_name": args.model_name,
                "checkpoint": str(ckpt_path),
                "split": [split_name],
                "num_records": len(records),
                "records": records,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logging.info("Wrote %s (%d rows)", path, len(records))
        else:
            path = out_dir / f"{stem}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for r in records:
                    r2 = {
                        **r,
                        "model_name": args.model_name,
                        "checkpoint": str(ckpt_path),
                        "split": split_name,
                    }
                    f.write(json.dumps(r2, ensure_ascii=False) + "\n")
            logging.info("Wrote %s (%d lines)", path, len(records))


if __name__ == "__main__":
    main()
