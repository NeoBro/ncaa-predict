#!/usr/bin/env python3
import argparse
import glob
import os

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob", default="csv/ncaa_games_*.csv",
        help="Glob pattern for yearly games CSV files.")
    parser.add_argument(
        "--out", "-o", default="csv/ncaa_games_all.csv",
        help="Output merged games CSV.")
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise ValueError("No files matched: %s" % args.input_glob)
    frames = [pd.read_csv(path) for path in files]
    merged = pd.concat(frames, ignore_index=True).drop_duplicates()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print("Merged %s files into %s rows at %s" % (len(files), len(merged), args.out))
