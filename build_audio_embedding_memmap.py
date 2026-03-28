#!/usr/bin/env python3
"""
Colab: pip install SoccerNet tqdm numpy
Build audio_features.dat + audio_mapping.json (same padding contract as memap_build.ipynb).

Games are listed with ``getListGames(splits, task="caption")`` — same order as
``memap_build.ipynb`` and ``dataset.py`` / ``_build_game_to_mapping_key``.

If either half ``.npy`` is missing for a game, both halves are filled with zeros sized from
your **video** ``mapping.json`` only (``half1_len`` / ``half2_len``, already padded). The
script **never reads** ``features.dat`` — only that small JSON. Copy ``mapping.json`` from
your PC or Kaggle bundle to Drive/Colab and pass ``--reference_mapping_json``, or place it
next to ``--out_feature_file``. Default search also tries
``<parent of audio_root>/master/mapping.json``.

Embedding width is inferred from the **first** game (in split order) that has both half
``.npy`` files. If **no** game has npy files (all zero-filled), ``--audio_feature_dim``
defaults to **128** (VGGish). Override if your vectors use another size.

Running in Google Colab
-----------------------
Fill ``NOTEBOOK_CONFIG`` or use ``!python ...`` with flags. Notebook ``sys.argv`` is
sanitized so ipykernel ``-f`` does not break ``argparse``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from SoccerNet.Downloader import getListGames

# --- Paste-in-Colab: set Path/str or None; use CLI / !python otherwise ---
NOTEBOOK_CONFIG = {
    "soccernet_path": None, # on Colab: "/content/drive/MyDrive/data_sn/caption-2024",
    "audio_root": None, # on Colab: "/content/drive/MyDrive/data_sn/audio_output",
    "out_feature_file": None, # on Colab: "/content/drive/MyDrive/data_sn/audio_memmap/audio_features.dat",
    "out_mapping_json": None, # on Colab: "/content/drive/MyDrive/data_sn/audio_memmap/audio_mapping.json",
    # optional: "reference_mapping_json": Path(...),  # video mapping.json only (not features.dat) on Colab: '/content/drive/MyDrive/data_sn/video_mapping.json',
    # optional: "audio_feature_dim": 256,  # override default 128 if no npy to infer from
    # optional: "dry_run": True,
}

def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def _argv_for_argparse(argv: list[str]) -> list[str]:
    """Drop ipykernel's ``-f connection.json`` so argparse does not choke."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "-f" and i + 1 < len(argv):
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def _apply_notebook_config(args: argparse.Namespace) -> None:
    for key, val in NOTEBOOK_CONFIG.items():
        if val is None:
            continue
        if key in (
            "soccernet_path",
            "audio_root",
            "out_feature_file",
            "out_mapping_json",
            "audio_emb_subdir",
            "reference_mapping_json",
        ):
            val = Path(val)
        setattr(args, key, val)


def audio_npy_paths_for_game(
    game: str,
    audio_root: Path,
    emb_subdir: Path,
    half1_name: str,
    half2_name: str,
) -> tuple[Path, Path]:
    p = Path(game)
    parts = p.parts
    if len(parts) < 3:
        raise ValueError(
            f"Expected game with at least league/season/match, got {game!r}"
        )
    league, season = parts[0], parts[1]
    match_folder = Path(*parts[2:])
    base = audio_root / league / season / emb_subdir / match_folder
    return base / half1_name, base / half2_name


def resolve_video_mapping_path(
    explicit: Path | None,
    out_feature_file: Path,
    audio_root: Path,
) -> Path:
    """Pick video memmap mapping.json: explicit, then out dir, then <audio_root.parent>/master/."""
    if explicit is not None:
        return explicit
    candidates = [
        out_feature_file.parent / "mapping.json",
        audio_root.parent / "master" / "mapping.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


def infer_feature_dim_from_first_audio_npy(
    games: list[str],
    audio_root: Path,
    emb_subdir: Path,
    half1_name: str,
    half2_name: str,
) -> int | None:
    """Last dimension of the first available half-1 npy in ``games`` order, or None."""
    for game in games:
        p1, p2 = audio_npy_paths_for_game(
            game, audio_root, emb_subdir, half1_name, half2_name
        )
        if p1.is_file() and p2.is_file():
            arr = np.load(p1)
            arr = np.asarray(arr).reshape(-1, arr.shape[-1])
            return int(arr.shape[-1])
    return None


def load_half(path: Path, feature_dim: int | None) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr).reshape(-1, arr.shape[-1])
    if feature_dim is not None and int(arr.shape[-1]) != feature_dim:
        raise ValueError(
            f"{path}: expected feature_dim {feature_dim}, got {arr.shape[-1]}"
        )
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audio embedding memmap builder (see NOTEBOOK_CONFIG for Colab paste)."
    )
    parser.add_argument("--soccernet_path", type=Path, default=None)
    parser.add_argument("--audio_root", type=Path, default=None)
    parser.add_argument(
        "--audio_emb_subdir",
        type=Path,
        default=Path("embeddings/vggish_1fps/raw"),
    )
    parser.add_argument("--half1_name", default="1_224p_raw.npy")
    parser.add_argument("--half2_name", default="2_224p_raw.npy")
    parser.add_argument("--out_feature_file", type=Path, default=None)
    parser.add_argument("--out_mapping_json", type=Path, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
    )
    parser.add_argument("--framerate", type=int, default=1)
    parser.add_argument("--window_size_seconds", type=int, default=45)
    parser.add_argument("--pad_mode", default="edge")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only scan games and print total_rows / feature_dim",
    )
    parser.add_argument(
        "--reference_mapping_json",
        type=Path,
        default=None,
        help="Video memmap mapping.json only (not features.dat). Same getListGames order. Default: next to out_feature_file or audio_root/../master/mapping.json",
    )
    parser.add_argument(
        "--audio_feature_dim",
        type=int,
        default=128,
        help="Fallback embedding width when no .npy exists in the split (all zero-fill). Default 128 (VGGish).",
    )

    if _in_notebook():
        filtered = _argv_for_argparse(sys.argv[1:])
        args = parser.parse_args(filtered)
    else:
        args = parser.parse_args()

    _apply_notebook_config(args)

    required = [
        ("soccernet_path", args.soccernet_path),
        ("audio_root", args.audio_root),
        ("out_feature_file", args.out_feature_file),
        ("out_mapping_json", args.out_mapping_json),
    ]
    missing = [name for name, val in required if val is None]
    if missing:
        parser.error(
            "Missing required paths: "
            + ", ".join(missing)
            + ". Set NOTEBOOK_CONFIG or CLI flags."
        )

    if not args.soccernet_path.is_dir():
        print(
            "Warning: soccernet_path is not a directory (metadata / downloader sanity).",
            file=sys.stderr,
        )

    window_size_frame = args.framerate * args.window_size_seconds
    l_pad = window_size_frame // 2 + window_size_frame % 2
    r_pad = window_size_frame // 2

    games = getListGames(args.splits, task="caption")
    if not games:
        sys.exit("No games for splits " + str(args.splits))

    ref_path = resolve_video_mapping_path(
        args.reference_mapping_json,
        args.out_feature_file,
        args.audio_root,
    )
    args.reference_mapping_json = ref_path
    if not ref_path.is_file():
        parser.error(
            f"Video mapping.json not found at {ref_path}\n"
            "You only need the small mapping.json from your video memmap build (not features.dat). "
            "Copy it from your machine or Kaggle dataset, then pass --reference_mapping_json, "
            "or put mapping.json next to --out_feature_file, or under <parent of audio_root>/master/."
        )
    with open(ref_path, encoding="utf-8") as f:
        ref_mapping: dict[str, dict[str, int]] = json.load(f)
    print(f"Using video mapping for missing-audio zero-fill: {ref_path}", file=sys.stderr)

    inferred = infer_feature_dim_from_first_audio_npy(
        games,
        args.audio_root,
        args.audio_emb_subdir,
        args.half1_name,
        args.half2_name,
    )
    feature_dim = inferred if inferred is not None else int(args.audio_feature_dim)
    if inferred is None:
        print(
            f"No audio npy found in split order; using --audio_feature_dim={feature_dim} for zero-fill.",
            file=sys.stderr,
        )
    else:
        print(f"Inferred embedding width {feature_dim} from first available .npy", file=sys.stderr)

    total_rows = 0
    n_zero_games = 0

    def load_halves_for_index(
        game: str,
        idx: int,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return (f1, f2, zero_fill). Zero blocks are final row counts; real halves are raw before pad."""
        p1, p2 = audio_npy_paths_for_game(
            game,
            args.audio_root,
            args.audio_emb_subdir,
            args.half1_name,
            args.half2_name,
        )
        if p1.is_file() and p2.is_file():
            f1 = load_half(p1, feature_dim)
            f2 = load_half(p2, feature_dim)
            return f1, f2, False
        key = str(idx)
        if key not in ref_mapping:
            raise KeyError(
                f"reference_mapping_json missing key {key!r} for game {game!r}"
            )
        e = ref_mapping[key]
        z1 = np.zeros((int(e["half1_len"]), feature_dim), dtype=np.float32)
        z2 = np.zeros((int(e["half2_len"]), feature_dim), dtype=np.float32)
        return z1, z2, True

    for idx, game in enumerate(tqdm(games, desc="Pass1 scan")):
        f1, f2, zf = load_halves_for_index(game, idx)
        if zf:
            n_zero_games += 1
            total_rows += int(f1.shape[0] + f2.shape[0])
        else:
            total_rows += int(f1.shape[0] + l_pad + r_pad)
            total_rows += int(f2.shape[0] + l_pad + r_pad)

    if n_zero_games:
        print(
            f"Note: {n_zero_games} game(s) used zero audio (aligned to video mapping half lengths).",
            file=sys.stderr,
        )

    print(
        "Pass1:",
        "games=",
        len(games),
        "total_rows=",
        total_rows,
        "feature_dim=",
        feature_dim,
        "window_size_frame=",
        window_size_frame,
        "l_pad=",
        l_pad,
        "r_pad=",
        r_pad,
        "zero_filled_games=",
        n_zero_games,
    )
    if args.dry_run:
        return

    args.out_feature_file.parent.mkdir(parents=True, exist_ok=True)
    mem = np.memmap(
        args.out_feature_file,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, feature_dim),
    )
    mapping: dict[str, dict[str, int]] = {}
    cursor = 0

    for idx, game in enumerate(tqdm(games, desc="Pass2 write")):
        f1, f2, zf = load_halves_for_index(game, idx)
        f1 = f1.astype(np.float32, copy=False)
        f2 = f2.astype(np.float32, copy=False)
        if not zf:
            f1 = np.pad(f1, ((l_pad, r_pad), (0, 0)), mode=args.pad_mode)
            f2 = np.pad(f2, ((l_pad, r_pad), (0, 0)), mode=args.pad_mode)

        h1_start = cursor
        h1_len = int(f1.shape[0])
        mem[h1_start : h1_start + h1_len] = f1
        cursor += h1_len

        h2_start = cursor
        h2_len = int(f2.shape[0])
        mem[h2_start : h2_start + h2_len] = f2
        cursor += h2_len

        mapping[str(idx)] = {
            "half1_start": h1_start,
            "half1_len": h1_len,
            "half2_start": h2_start,
            "half2_len": h2_len,
        }

    mem.flush()
    with open(args.out_mapping_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print("Wrote", args.out_feature_file, "and", args.out_mapping_json)


if __name__ == "__main__":
    main()