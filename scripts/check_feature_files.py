#!/usr/bin/env python3
"""
Check SoccerNet feature .npy files for corruption (e.g. truncated files
where header shape does not match data size). Prints paths of bad files so
you can re-download them.

Usage (from project root or from Benchmarks/TemporallyAwarePooling):
  python scripts/check_feature_files.py --SoccerNet_path=/path/to/SoccerNet --features=baidu_soccer_embeddings.npy

Or from Benchmarks/TemporallyAwarePooling/src (so SoccerNet is importable):
  python -c "import sys; sys.path.insert(0, '..'); exec(open('../scripts/check_feature_files.py').read())"
  # Better: run from repo root with PYTHONPATH including the package that has SoccerNet.
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np

# Allow running from repo root or from TemporallyAwarePooling
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAP_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(TAP_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from SoccerNet.Downloader import getListGames


def check_file(path):
    """Load .npy file; return (True, shape) if OK, (False, error_msg) if corrupt."""
    if not os.path.isfile(path):
        return False, "file not found"
    try:
        arr = np.load(path)
        return True, arr.shape
    except ValueError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Find corrupted SoccerNet feature .npy files")
    parser.add_argument("--SoccerNet_path", required=True, type=str, help="Path to SoccerNet (caption) dataset root")
    parser.add_argument("--features", type=str, default="baidu_soccer_embeddings.npy", help="Feature filename (e.g. baidu_soccer_embeddings.npy)")
    parser.add_argument("--split", nargs="+", default=["train", "valid", "test"], help="Splits to check")
    parser.add_argument("--out", type=str, default=None, help="Optional: write bad file paths to this file")
    args = parser.parse_args()

    path_root = args.SoccerNet_path
    if not os.path.isdir(path_root):
        print("Error: SoccerNet_path is not a directory:", path_root, file=sys.stderr)
        sys.exit(1)

    bad = []
    checked = 0
    for split in args.split:
        try:
            games = getListGames([split], task="caption")
        except Exception as e:
            print("Warning: could not get games for split", split, ":", e, file=sys.stderr)
            continue
        for game in games:
            for half in ("1", "2"):
                rel = os.path.join(game, half + "_" + args.features)
                full = os.path.join(path_root, rel)
                checked += 1
                ok, result = check_file(full)
                if ok:
                    print("OK  {}  shape={}".format(rel, result))
                else:
                    print("BAD {}  error={}".format(rel, result))
                    bad.append(full)

    print("\n--- Summary ---")
    print("Checked:", checked, "files. Bad:", len(bad))
    if bad:
        print("\nCorrupted or missing files (re-download these):")
        for p in bad:
            print(p)
        if args.out:
            with open(args.out, "w") as f:
                f.write("\n".join(bad))
            print("\nPaths written to:", args.out)
    else:
        print("No corrupted files found.")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
